import io

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoFeatureExtractor, MimiModel

from vyvonext.core.tokens import NUM_CODEBOOKS

MIMI_MODEL_ID = "kyutai/mimi"


def decode_audio_bytes(audio_bytes, target_sr):
    """Encoded audio bytes -> mono float32 waveform resampled to target_sr."""
    try:
        wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        wav = np.asarray(wav, dtype=np.float32)
    except Exception:           # fall back to the ffmpeg/sox backend
        t, sr = torchaudio.load(io.BytesIO(audio_bytes))
        wav = t.mean(0).numpy().astype(np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        t = torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0)
        wav = torchaudio.functional.resample(t, sr, target_sr).squeeze(0).numpy()
    return wav


def load_clips(audios, texts, target_sr, min_seconds=0.2, max_seconds=30.0):
    """Decode a batch of HF audio dicts, dropping unusable / too-short / too-long.

    Returns (waves, texts) kept in parallel. `audios` are HF `{"bytes": ...}`
    dicts; rows with no audio/text or a decode error are silently skipped.
    """
    lo, hi = int(min_seconds * target_sr), int(max_seconds * target_sr)
    waves, kept_text = [], []
    for au, tx in zip(audios, texts):
        if not au or not au.get("bytes") or not tx:
            continue
        try:
            w = decode_audio_bytes(au["bytes"], target_sr)
        except Exception:       # skip a corrupt clip, keep the rest of the shard
            continue
        if lo <= w.size <= hi:
            waves.append(w)
            kept_text.append(tx)
    return waves, kept_text


class MimiCodec:
    """Loads the Mimi neural codec on a device with simple encode/decode helpers."""

    def __init__(self, device, model_id=MIMI_MODEL_ID, num_codebooks=NUM_CODEBOOKS):
        self.device = device
        self.num_codebooks = num_codebooks
        self.model = MimiModel.from_pretrained(model_id).to(device).eval()
        self.fe = AutoFeatureExtractor.from_pretrained(model_id)
        self.sampling_rate = self.fe.sampling_rate
        self.hop = self.sampling_rate / self.model.config.frame_rate   # 24000/12.5 = 1920

    @torch.no_grad()
    def encode(self, waves):
        """List of mono waveforms -> (codes (B, nq, Tmax), valid n_frames per clip).

        Clips are zero-padded to a common length; use the returned `n_frames` to
        trim each clip's codes back to its true duration.
        """
        inputs = self.fe(raw_audio=waves, sampling_rate=self.sampling_rate,
                         return_tensors="pt", padding=True)
        iv = inputs["input_values"].to(self.device)
        mask = inputs.get("padding_mask")
        mask = mask.to(self.device) if mask is not None else None
        codes = self.model.encode(
            iv, padding_mask=mask, num_quantizers=self.num_codebooks).audio_codes
        codes = codes.cpu().numpy()
        n_frames = [min(int(np.ceil(w.size / self.hop)), codes.shape[-1]) for w in waves]
        return codes, n_frames

    @torch.no_grad()
    def encode_file(self, path):
        """Read a wav file from disk and encode it -> trimmed codes (nq, T)."""
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self.sampling_rate:
            wav = torchaudio.functional.resample(
                torch.from_numpy(wav).unsqueeze(0), sr, self.sampling_rate
            ).squeeze(0).numpy()
        codes, n_frames = self.encode([wav])
        return codes[0][:, :n_frames[0]]

    @torch.no_grad()
    def decode(self, codes):
        """Mimi codes (nq, T) -> mono float32 waveform."""
        arr = torch.as_tensor(np.asarray(codes), dtype=torch.long, device=self.device)
        wav = self.model.decode(arr.unsqueeze(0)).audio_values
        return wav.squeeze().cpu().to(torch.float32).numpy()
