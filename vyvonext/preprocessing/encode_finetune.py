import os
import time
import multiprocessing as mp

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from vyvonext.core.tokens import TokenLayout
from vyvonext.core.audio import MimiCodec, load_clips

# Config (hardcoded; no CLI args). Single-speaker English set (~2000 clips); each
# (audio, text) row becomes one Orpheus TTS sequence in the SAME layout as the
# pretraining data, so the pretrained checkpoint can be finetuned on it directly.
# Rows are split round-robin across NUM_GPUS workers, each writing its own shard.
SRC_REPO = "Vyvo/ElevenLabs-EN-iP95p4xoKVk53GoZ742B"
SRC_FILE = "data/train-00000-of-00001.parquet"
TOKENIZER_NAME = "Qwen/Qwen3-0.6B"          # same vocab length as the trained model
OUTPUT_ROOT = "/scratch/kadirnar/elevenlabs-en-ft-mimi32"
OUTPUT_DIR = OUTPUT_ROOT + "/data"
NUM_GPUS = 8
BATCH_SIZE = 32                 # clips per Mimi forward pass
MIN_AUDIO_SECONDS = 0.2
MAX_AUDIO_SECONDS = 30.0
TOKEN = os.environ.get("HF_TOKEN")


def worker(gpu_id, audios, texts):
    """One process per GPU: encode its slice of rows into a single parquet shard."""
    out_path = os.path.join(OUTPUT_DIR, "train-%02d.parquet" % gpu_id)
    if os.path.exists(out_path):
        print("[gpu%d] output exists, skip" % gpu_id, flush=True)
        return

    device = "cuda:%d" % gpu_id
    torch.cuda.set_device(device)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    layout = TokenLayout.from_tokenizer(tok)
    codec = MimiCodec(device, num_codebooks=layout.num_codebooks)

    waves, keep_text = load_clips(audios, texts, codec.sampling_rate,
                                  MIN_AUDIO_SECONDS, MAX_AUDIO_SECONDS)
    print("[gpu%d] %d clips kept (base_vocab=%d)" % (gpu_id, len(waves), layout.base),
          flush=True)

    rows, t0 = [], time.time()
    for s in range(0, len(waves), BATCH_SIZE):
        chunk, chunk_text = waves[s:s + BATCH_SIZE], keep_text[s:s + BATCH_SIZE]
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
    print("[gpu%d] DONE %d rows -> %s | %.0fs" %
          (gpu_id, len(rows), out_path, time.time() - t0), flush=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    src = hf_hub_download(SRC_REPO, SRC_FILE, repo_type="dataset", token=TOKEN)
    table = pq.read_table(src, columns=["audio", "text"])
    audios = table.column("audio").to_pylist()
    texts = table.column("text").to_pylist()
    print("loaded %d rows -> %d GPUs" % (len(audios), NUM_GPUS))

    procs = []
    for g in range(NUM_GPUS):
        sub_a, sub_t = audios[g::NUM_GPUS], texts[g::NUM_GPUS]
        if not sub_a:
            continue
        p = mp.Process(target=worker, args=(g, sub_a, sub_t))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    n_rows = sum(pq.read_metadata(os.path.join(OUTPUT_DIR, f)).num_rows
                 for f in os.listdir(OUTPUT_DIR) if f.endswith(".parquet"))
    print("ALL DONE: %d total rows in %s" % (n_rows, OUTPUT_DIR))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
