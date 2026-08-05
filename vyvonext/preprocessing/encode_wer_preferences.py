"""Encode Whisper-ranked audio candidates into SimPO preference parquet.

Input is JSONL with one target per line::

    {"text": "Target sentence.", "ref_text": "Reference transcript.",
     "ref_wav": "audio/ref.wav", "target_wav": "audio/target.wav",
     "candidates": [
       {"wav": "audio/candidate_1.wav", "transcript": "Target sentence."},
       {"wav": "audio/candidate_2.wav", "transcript": "Target sent tense."}]}

Candidates may provide a fractional ``wer`` instead of ``transcript``. The
ground-truth ``target_wav`` is the chosen response, and the closest candidate
above the minimum WER gap is the hard negative. Paths are resolved relative to
the manifest.
"""

import argparse
import json
import math
import os
import string
import unicodedata
from functools import lru_cache
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from vyvonext.core.audio import MimiCodec
from vyvonext.core.tokens import TokenLayout


def normalize_english(text):
    """Apply Seed-TTS English punctuation removal before ASR WER."""
    normalized = (
        unicodedata.normalize("NFKC", str(text)).lower().replace("’", "'")
    )
    characters = []
    for character in normalized:
        is_punctuation = (
            character in string.punctuation
            or unicodedata.category(character).startswith("P")
        )
        if character == "'" or not is_punctuation:
            characters.append(character)
    return " ".join("".join(characters).split())


def word_error_rate(reference, hypothesis):
    """Compute word-level Levenshtein distance divided by reference length."""
    reference_words = normalize_english(reference).split()
    hypothesis_words = normalize_english(hypothesis).split()
    if not reference_words:
        raise ValueError("WER reference is empty after normalization")
    previous = list(range(len(hypothesis_words) + 1))
    for row, reference_word in enumerate(reference_words, start=1):
        current = [row]
        for column, hypothesis_word in enumerate(hypothesis_words, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (reference_word != hypothesis_word),
            ))
        previous = current
    return previous[-1] / len(reference_words)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-wer-gap", type=float, default=0.05)
    parser.add_argument("--min-speaker-similarity", type=float, default=None)
    parser.add_argument("--min-quality", type=float, default=None)
    parser.add_argument("--max-reference-tokens", type=int, default=2400)
    parser.add_argument("--max-total-tokens", type=int, default=16384)
    return parser.parse_args()


def candidate_wer(candidate, reference_text):
    if "wer" in candidate:
        score = float(candidate["wer"])
    else:
        if "transcript" not in candidate:
            raise ValueError("candidate must contain either 'wer' or 'transcript'")
        score = word_error_rate(reference_text, candidate["transcript"])
    if not math.isfinite(score) or score < 0.0:
        raise ValueError(f"candidate WER must be finite and non-negative: {score}")
    return score


def resolve_audio(path, manifest_dir):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return str(path.resolve())


def select_rejected(scored, min_wer_gap):
    """Return the lowest-WER candidate above the minimum gap."""
    if min_wer_gap < 0.0:
        raise ValueError("minimum WER gap must be non-negative")
    for score, candidate in sorted(scored, key=lambda entry: entry[0]):
        if score >= min_wer_gap:
            return score, candidate
    return None


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    layout = TokenLayout.from_tokenizer(tokenizer)
    codec = MimiCodec(args.device, num_codebooks=layout.num_codebooks)
    manifest_dir = args.manifest.resolve().parent
    reference_limit = (
        args.max_reference_tokens // layout.num_codebooks
        * layout.num_codebooks
    )
    if reference_limit < layout.num_codebooks:
        raise ValueError("--max-reference-tokens must fit at least one Mimi frame")

    @lru_cache(maxsize=256)
    def encode_audio(path):
        return tuple(layout.codes_to_ids(codec.encode_file(path)))

    rows = []
    skipped = 0
    with open(args.manifest, encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            target_text = str(item["text"]).strip()
            ref_text = str(item["ref_text"]).strip()
            if not target_text or not ref_text:
                skipped += 1
                continue
            scored = []
            for candidate in item.get("candidates", []):
                if args.min_speaker_similarity is not None and (
                    "speaker_similarity" not in candidate
                    or float(candidate["speaker_similarity"])
                    < args.min_speaker_similarity
                ):
                    continue
                if args.min_quality is not None and (
                    "quality" not in candidate
                    or float(candidate["quality"]) < args.min_quality
                ):
                    continue
                score = candidate_wer(candidate, target_text)
                scored.append((score, candidate))
            preference = select_rejected(scored, args.min_wer_gap)
            if preference is None:
                skipped += 1
                continue
            rejected_wer, rejected = preference

            try:
                ref_path = resolve_audio(item["ref_wav"], manifest_dir)
                chosen_path = resolve_audio(item["target_wav"], manifest_dir)
                rejected_path = resolve_audio(rejected["wav"], manifest_dir)
                reference_audio = list(encode_audio(ref_path))[:reference_limit]
                chosen_audio = list(encode_audio(chosen_path))
                rejected_audio = list(encode_audio(rejected_path))
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"line {line_number}: audio encode failed: {exc}", flush=True)
                skipped += 1
                continue

            combined_text = (ref_text + " " + target_text).strip()
            text_ids = tokenizer(
                combined_text, add_special_tokens=False
            ).input_ids
            prompt_without_audio = layout.voice_clone_prompt(text_ids, [])
            longest_response = max(len(chosen_audio), len(rejected_audio)) + 1
            available_reference = (
                args.max_total_tokens - len(prompt_without_audio) - longest_response
            )
            available_reference = (
                available_reference // layout.num_codebooks
                * layout.num_codebooks
            )
            if available_reference < layout.num_codebooks:
                skipped += 1
                continue
            reference_audio = reference_audio[:available_reference]
            prompt = layout.voice_clone_prompt(text_ids, reference_audio)
            chosen_response = chosen_audio + [layout.eos_speech]
            rejected_response = rejected_audio + [layout.eos_speech]
            rows.append({
                "chosen_input_ids": prompt + chosen_response,
                "chosen_labels": [-100] * len(prompt) + chosen_response,
                "rejected_input_ids": prompt + rejected_response,
                "rejected_labels": [-100] * len(prompt) + rejected_response,
                "wer_gap": rejected_wer,
                "text": target_text,
            })

    if not rows:
        raise RuntimeError("no WER preference pairs were encoded")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    list_type = pa.list_(pa.int32())
    table = pa.table({
        "chosen_input_ids": pa.array(
            [row["chosen_input_ids"] for row in rows], type=list_type),
        "chosen_labels": pa.array(
            [row["chosen_labels"] for row in rows], type=list_type),
        "rejected_input_ids": pa.array(
            [row["rejected_input_ids"] for row in rows], type=list_type),
        "rejected_labels": pa.array(
            [row["rejected_labels"] for row in rows], type=list_type),
        "wer_gap": pa.array(
            [row["wer_gap"] for row in rows], type=pa.float32()),
        "text": pa.array([row["text"] for row in rows], type=pa.string()),
    })
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    pq.write_table(table, temporary, compression="snappy")
    os.replace(temporary, args.output)
    print(
        f"encoded {len(rows)} WER pairs -> {args.output} "
        f"({skipped} rows skipped)",
        flush=True,
    )


if __name__ == "__main__":
    main()
