# WER-Focused Post-Training

Use this recipe after ordinary voice fine-tuning. It improves intelligibility
with WER-ranked preference pairs while keeping voice and audio quality as
checkpoint-selection guardrails.

## Build Preference Pairs

Create a JSONL manifest from 4–6 on-policy candidates per training prompt:

```json
{
  "text": "The target sentence.",
  "ref_text": "Exact reference text.",
  "ref_wav": "ref.wav",
  "target_wav": "target.wav",
  "candidates": [
    {"wav": "a.wav", "transcript": "The target sentence."},
    {"wav": "b.wav", "transcript": "The target sent tense."}
  ]
}
```

`encode_wer_preferences` uses the ground-truth `target_wav` as the chosen
response and selects the closest on-policy candidate with at least 0.05 WER as
the hard negative. Candidates can supply fractional `wer` instead of
`transcript`, plus optional `speaker_similarity` and `quality` scores. Use
`--min-speaker-similarity` and `--min-quality` to avoid unusable negatives.

Encode the manifest, then launch training with the supplied config:

```bash
python -m vyvonext.preprocessing.encode_wer_preferences \
  --manifest data/wer_candidates.jsonl \
  --output data/wer_preferences/train.parquet

accelerate launch --config_file vyvonext/configs/accelerate_config.yaml \
  -m vyvonext.training.posttrain_simpo \
  --config vyvonext/configs/posttrain_simpo.yaml
```

## Training Recipe

The small-data sweep selected `beta: 2.0`, `gamma: 1.0`, chosen-SFT weight
`0.1`, learning rate `5e-7`, gradient clipping at `0.5`, and two warmup steps.
Start from the English FT checkpoint and evaluate after `0.25` epoch; the same
data already overfit by `0.5` epoch. Preference-only loss, longer training, and
LoRA were unstable in this low-data setting. Only semantic codebook 0 and
speech EOS enter the preference objective, while chosen-SFT regularization
limits quality collapse.

## Evaluation Guardrails

- Keep training/development prompts disjoint from Seed-TTS test prompts.
- Rank candidates with Whisper-Turbo, but report WER with Whisper Large-v3 or
  another independent ASR to reduce scorer overfitting.
- Evaluate at quarter-epoch intervals on small datasets. Select the lowest
  development WER subject to speaker-similarity and speech-quality floors;
  stop if WER or either quality guardrail materially regresses.
- Compare against the same candidate-1 decoding profile. Report Best-of-N
  reranking separately.

The design follows [Seed-TTS reinforcement post-training](https://arxiv.org/abs/2406.02430),
[SimPO](https://arxiv.org/abs/2405.14734), Mimi's
[semantic first codebook](https://github.com/kyutai-labs/moshi#Mimi), and the
cross-entropy stabilization used by
[minimum-WER sequence training](https://arxiv.org/abs/1712.01818).
