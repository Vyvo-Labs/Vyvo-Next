"""Lightweight checks for WER-focused SimPO data and loss helpers."""

import torch

from vyvonext.core.data import make_preference_collator
from vyvonext.core.tokens import TokenLayout
from vyvonext.core.trainer import select_semantic_targets
from vyvonext.preprocessing.encode_wer_preferences import (
    normalize_english,
    select_rejected,
    word_error_rate,
)


def main():
    layout = TokenLayout(base=100, eot=99, num_codebooks=2, codebook_size=4)

    collator = make_preference_collator(pad_token=0)
    batch = collator([
        {
            "chosen_input_ids": [1, 2, 3],
            "chosen_labels": [-100, 2, 3],
            "rejected_input_ids": [1, 4],
            "rejected_labels": [-100, 4],
            "wer_gap": 0.5,
        }
    ])
    assert batch["input_ids"].tolist() == [[1, 2, 3], [1, 4, 0]]
    assert batch["wer_gap"].tolist() == [0.5]

    labels = torch.tensor([[-100, 110, 114, 111, 115, layout.eos_speech]])
    hidden = torch.arange(1 * 6 * 3, dtype=torch.float32).reshape(1, 6, 3)
    selected_hidden, selected_labels = select_semantic_targets(
        hidden, labels, layout
    )
    assert selected_labels.tolist() == [[110, 111, layout.eos_speech]]
    assert selected_hidden.tolist() == hidden[:, [0, 2, 4]].tolist()

    assert word_error_rate("Hello, world!", "hello world") == 0.0
    assert word_error_rate("one two", "one too") == 0.5
    assert normalize_english("text-to-speech") == "texttospeech"

    scored = [
        (0.02, {"wav": "near_tie.wav"}),
        (0.10, {"wav": "best_negative.wav"}),
        (0.30, {"wav": "hard_negative.wav"}),
    ]
    rejected = select_rejected(scored, 0.05)
    assert rejected[1]["wav"] == "best_negative.wav"
    rejected = select_rejected(scored, 0.25)
    assert rejected[1]["wav"] == "hard_negative.wav"
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
