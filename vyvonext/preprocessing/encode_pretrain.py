import os
import time
import multiprocessing as mp

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from vyvonext.core.tokens import TokenLayout
from vyvonext.core.audio import MimiCodec, load_clips

# Config (hardcoded; no CLI args). Each (audio, text) row becomes one Orpheus TTS
# sequence; the 8 GPUs each encode a round-robin slice of the source shards.
SRC_REPO = "YT-Data/ja-en-syn"
TARGET_REPO = "YT-Data/ja-en-syn-qwen3-mimi32"
TOKENIZER_NAME = "Qwen/Qwen3-4B"
LOCAL_SRC_ROOT = "/scratch/kadirnar/ja-en-syn"          # data/ + data2/ already here
OUTPUT_ROOT = "/scratch/kadirnar/ja-en-syn-qwen3-mimi32"
OUTPUT_DIR = OUTPUT_ROOT + "/data"
LOG_PATH = OUTPUT_ROOT + "/encode.log"
NUM_GPUS = 8
BATCH_SIZE = 32                 # clips per Mimi forward pass
MIN_AUDIO_SECONDS = 0.2
MAX_AUDIO_SECONDS = 30.0
DELETE_SOURCE_CACHE = True      # drop downloaded shards after encoding
PUSH_TO_HUB = False
TOKEN = os.environ.get("HF_TOKEN")


def _log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + "  " + msg
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def list_shards():
    """All .parquet shard names in the source dataset repo, sorted."""
    api = HfApi(token=TOKEN)
    files = [f for f in api.list_repo_files(SRC_REPO, repo_type="dataset")
             if f.endswith(".parquet")]
    return sorted(files)


def encode_shard(fn, tok, layout, codec):
    """Encode one source shard -> one output parquet; skip if it already exists.

    Returns (status, rows_written, rows_dropped).
    """
    out_path = os.path.join(OUTPUT_DIR, fn.replace("/", "-"))
    if os.path.exists(out_path):
        return "skip", 0, 0

    local = os.path.join(LOCAL_SRC_ROOT, fn)
    downloaded = not os.path.exists(local)
    src = (local if not downloaded
           else hf_hub_download(SRC_REPO, fn, repo_type="dataset", token=TOKEN))

    table = pq.read_table(src, columns=["audio", "text"])
    n_in = table.num_rows
    waves, texts = load_clips(table.column("audio").to_pylist(),
                              table.column("text").to_pylist(),
                              codec.sampling_rate,
                              MIN_AUDIO_SECONDS, MAX_AUDIO_SECONDS)

    rows = []
    for s in range(0, len(waves), BATCH_SIZE):
        chunk, chunk_text = waves[s:s + BATCH_SIZE], texts[s:s + BATCH_SIZE]
        codes, n_frames = codec.encode(chunk)
        for j in range(len(chunk)):
            if n_frames[j] <= 0:
                continue
            text_ids = tok(chunk_text[j], add_special_tokens=False).input_ids
            rows.append(layout.tts_sequence(text_ids, codes[j][:, :n_frames[j]]))

    tmp = out_path + ".tmp"
    pq.write_table(
        pa.table({"input_ids": pa.array(rows, type=pa.list_(pa.int32()))}),
        tmp, compression="snappy")
    os.replace(tmp, out_path)

    if downloaded and DELETE_SOURCE_CACHE:
        for p in (src, os.path.realpath(src)):
            try:
                os.remove(p)
            except OSError:
                pass
    return "ok", len(rows), n_in - len(rows)


def worker(gpu_id, shards):
    """One process per GPU: load the codec once, then encode its shards."""
    device = "cuda:%d" % gpu_id
    torch.cuda.set_device(device)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    layout = TokenLayout.from_tokenizer(tok)
    codec = MimiCodec(device, num_codebooks=layout.num_codebooks)

    _log("gpu%d: start, %d shards (base_vocab=%d)" % (gpu_id, len(shards), layout.base))
    done = rows = dropped = 0
    t0 = time.time()
    for fn in shards:
        try:
            _, n, skipped = encode_shard(fn, tok, layout, codec)
        except Exception as exc:  # noqa: BLE001 - log and move to the next shard
            _log("gpu%d ERROR %s: %r" % (gpu_id, fn, exc))
            continue
        done, rows, dropped = done + 1, rows + n, dropped + skipped
        if done % 50 == 0 or done == len(shards):
            _log("gpu%d: %d/%d shards | %d rows | %d dropped | %.0fs" %
                 (gpu_id, done, len(shards), rows, dropped, time.time() - t0))
    _log("gpu%d: DONE %d shards | %d rows | %d dropped" % (gpu_id, done, rows, dropped))


def push_to_hub():
    """Upload the encoded shards (+ a dataset card) to TARGET_REPO."""
    api = HfApi(token=TOKEN)
    api.create_repo(TARGET_REPO, repo_type="dataset", exist_ok=True)
    readme = (
        "---\nlicense: apache-2.0\ntask_categories:\n- text-to-speech\n"
        "language:\n- en\n- ja\nconfigs:\n- config_name: default\n  data_files:\n"
        "  - split: train\n    path: data/*.parquet\n---\n\n"
        "# ja-en-syn-qwen3-mimi\n\n"
        "`YT-Data/ja-en-syn` encoded for VyvoNext: Qwen3-4B text tokens + Mimi\n"
        "audio tokens in the Orpheus prompt layout. Single column `input_ids`.\n")
    api.upload_file(path_or_fileobj=readme.encode("utf-8"), path_in_repo="README.md",
                    repo_id=TARGET_REPO, repo_type="dataset")
    api.upload_large_folder(repo_id=TARGET_REPO, repo_type="dataset",
                            folder_path=OUTPUT_ROOT, allow_patterns=["data/*.parquet"])
    _log("DONE: pushed to %s" % TARGET_REPO)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    shards = list_shards()
    _log("encoding %d shards -> %s on %d GPUs" % (len(shards), TARGET_REPO, NUM_GPUS))

    procs = [mp.Process(target=worker, args=(g, shards[g::NUM_GPUS]))
             for g in range(NUM_GPUS)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    n_out = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".parquet")])
    _log("all workers finished | %d output shards" % n_out)
    if PUSH_TO_HUB:
        push_to_hub()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
