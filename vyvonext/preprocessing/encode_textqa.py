import os
import time
import multiprocessing as mp

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoTokenizer

from vyvonext.core.tokens import TokenLayout

# Config (hardcoded; no CLI args). Builds the text-QA stream for pretraining so
# the LM keeps its reasoning ability while it learns speech. The QA layout mirrors
# the TTS one but the assistant turn is a TEXT answer instead of audio:
#   TTS : [SOH] text [eot] [EOH] [SOA] [SOS] <audio> [EOS_SPEECH]
#   QA  : [SOH] question [eot] [EOH] [SOA] answer [eot]
# EN/JA shards are written side by side (en-*, ja-*); pretrain.py shuffles them.
# CPU-only (no audio), so this uses a process Pool instead of one-worker-per-GPU.
SOURCES = [
    ("en", "SynDataLab/DeepSeekFlash-3M-en"),
    ("ja", "SynDataLab/DeepSeekFlash-3M-ja"),
]
TOKENIZER_NAME = "Qwen/Qwen3-4B"
OUTPUT_ROOT = "/scratch/kadirnar/textqa-qwen3"
OUTPUT_DIR = OUTPUT_ROOT + "/data"
LOG_PATH = OUTPUT_ROOT + "/encode.log"
NUM_WORKERS = 16
MAX_LEN = 2048                  # drop QA pairs longer than this (tokens)
MIN_LEN = 4
DELETE_SOURCE_CACHE = True
TOKEN = os.environ.get("HF_TOKEN")


def _log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + "  " + msg
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def worker(args):
    """Encode one (lang, repo, shard) into a QA `input_ids` parquet; skip if done.

    Returns (status, rows_written, rows_dropped).
    """
    lang, repo, fn = args
    out_path = os.path.join(OUTPUT_DIR, f"{lang}-{os.path.basename(fn)}")
    if os.path.exists(out_path):
        return "skip", 0, 0

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    layout = TokenLayout.from_tokenizer(tok)
    src = hf_hub_download(repo, fn, repo_type="dataset", token=TOKEN)

    t = pq.read_table(src, columns=["question", "answer"])
    rows, dropped = [], 0
    for q, a in zip(t.column("question").to_pylist(), t.column("answer").to_pylist()):
        if not q or not a:
            dropped += 1
            continue
        q_ids = tok(q, add_special_tokens=False).input_ids
        a_ids = tok(a, add_special_tokens=False).input_ids
        ids = layout.qa_sequence(q_ids, a_ids)
        if MIN_LEN <= len(ids) <= MAX_LEN:
            rows.append(ids)
        else:
            dropped += 1

    tmp = out_path + ".tmp"
    pq.write_table(
        pa.table({"input_ids": pa.array(rows, type=pa.list_(pa.int32()))}),
        tmp, compression="snappy")
    os.replace(tmp, out_path)
    if DELETE_SOURCE_CACHE:
        try:
            os.remove(os.path.realpath(src))
        except OSError:
            pass
    return "ok", len(rows), dropped


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tasks = []
    for lang, repo in SOURCES:
        shards = sorted(f for f in list_repo_files(repo, repo_type="dataset", token=TOKEN)
                        if f.endswith(".parquet"))
        tasks += [(lang, repo, fn) for fn in shards]
    _log("encoding %d shards (%d sources) -> %s with %d workers"
         % (len(tasks), len(SOURCES), OUTPUT_DIR, NUM_WORKERS))

    done = rows = dropped = 0
    t0 = time.time()
    with mp.Pool(NUM_WORKERS) as pool:
        for _, n, d in pool.imap_unordered(worker, tasks):
            done, rows, dropped = done + 1, rows + n, dropped + d
            if done % 10 == 0 or done == len(tasks):
                _log("%d/%d shards | %d rows | %d dropped | %.0fs"
                     % (done, len(tasks), rows, dropped, time.time() - t0))
    n_out = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".parquet")])
    _log("DONE | %d output shards | %d rows total" % (n_out, rows))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
