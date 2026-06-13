# VyvoNext: Qwen3 + Mimi Text-to-Speech

## Overview

VyvoNext converts text to speech by having a Qwen3 LM generate interleaved
[Mimi](https://huggingface.co/kyutai/mimi) audio-codec tokens, which are decoded
back to a waveform. Give it a reference clip and it clones that voice in any
language (EN/JA). Built on the Orpheus recipe.

## Installation

```bash
pip install -r vyvonext/requirements.txt   # Python 3.10, CUDA 12.x
export HF_TOKEN=...                         # for private dataset/model pulls
```

## Inference

```python
from vyvonext.inference.infer import VyvoNextTTS

engine = VyvoNextTTS(
    checkpoint="checkpoints/checkpoint-171622",
    tokenizer_name="Qwen/Qwen3-0.6B",
    device="cuda:0",
)

audio = engine.clone(
    text="Hello, this is a cloned voice.",
    ref_wav="andrew.wav",
    ref_text="all sorts of things, mainly browsing.",
    output_path="output.wav",
)
```

Batch synthesis — edit `CKPT` and `JOBS` at the top of the script, then:

```bash
python -m vyvonext.inference.infer
```

## Dataset Preparation

Encode raw (audio, text) into `input_ids` parquet. Source repos and output paths
are set at the top of each script:

```bash
python -m vyvonext.preprocessing.encode_textqa     # text-QA stream
python -m vyvonext.preprocessing.encode_pretrain   # speech (TTS) stream
python -m vyvonext.preprocessing.encode_finetune   # single-voice set
```

Sanity-check the codec and token round-trip: `python -m tests.test_mimi`.

## Training

```bash
# Pre-training (8-GPU FSDP; knobs in configs/pretrain.yaml)
accelerate launch --config_file vyvonext/configs/accelerate_config.yaml \
                  -m vyvonext.training.pretrain

# Fine-tuning on a single voice (set PRETRAINED_CKPT in training/finetune.py)
accelerate launch --config_file vyvonext/configs/accelerate_config.yaml \
                  -m vyvonext.training.finetune
```

## Acknowledgements

- [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) 
- [Kyutai Mimi](https://huggingface.co/kyutai/mimi)
- [Qwen3](https://huggingface.co/Qwen)
