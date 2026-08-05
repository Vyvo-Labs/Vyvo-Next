from dataclasses import dataclass

import numpy as np

# Mimi RVQ shape and the reserved-id offset. Changing any of these means
# re-encoding every dataset AND retraining — they are baked into the token ids.
NUM_CODEBOOKS = 32          # full Mimi depth; ~4x longer sequences than 8 codebooks
CODEBOOK_SIZE = 2048        # entries per codebook
AUDIO_OFFSET = 10           # first 10 ids above `base` are reserved for specials

# Special-token offsets relative to `base`.
SOS, EOS_SPEECH, SOH, EOH, SOA = 1, 2, 3, 4, 5


@dataclass(frozen=True)
class TokenLayout:
    """Orpheus token layout — the single source of truth for every stage.

    A speech clip is turned into discrete codes by the Mimi codec, and those
    codes are appended to the language model's vocabulary as extra tokens. The
    LM then learns to *continue* text with audio tokens, which is what makes it
    a TTS model. Everything is an offset from the base text vocab:

        base                = len(text tokenizer)             # e.g. Qwen3-0.6B
        <custom_token_i>    = base + i                         # the added tokens
        SOS / EOS_SPEECH    = base + 1 / base + 2              # start / end of speech
        SOH / EOH / SOA     = base + 3 / base + 4 / base + 5   # turn markers
        audio (cb, code)    = base + AUDIO_OFFSET + cb*CODEBOOK_SIZE + code

    Training-sequence layouts (`eot` is the tokenizer's own end-of-text id):

        TTS : [SOH] text [eot] [EOH] [SOA] [SOS] <audio tokens> [EOS_SPEECH]
        QA  : [SOH] question [eot] [EOH] [SOA] answer [eot]

    Binding the scheme to a tokenizer's vocab size keeps encoding and decoding
    from ever drifting apart.
    """

    base: int                       # len(tokenizer)
    eot: int                        # tokenizer.eos_token_id
    num_codebooks: int = NUM_CODEBOOKS
    codebook_size: int = CODEBOOK_SIZE

    @classmethod
    def from_tokenizer(cls, tokenizer, num_codebooks=NUM_CODEBOOKS):
        return cls(base=len(tokenizer), eot=tokenizer.eos_token_id,
                   num_codebooks=num_codebooks)

    # --- special tokens ---------------------------------------------------
    @property
    def sos(self):        return self.base + SOS
    @property
    def eos_speech(self): return self.base + EOS_SPEECH
    @property
    def soh(self):        return self.base + SOH
    @property
    def eoh(self):        return self.base + EOH
    @property
    def soa(self):        return self.base + SOA

    # --- vocabulary expansion --------------------------------------------
    @property
    def num_added_tokens(self):
        """Count of <custom_token_i> ids (all codes + the reserved specials)."""
        return self.num_codebooks * self.codebook_size + AUDIO_OFFSET

    def added_token_strings(self):
        """Placeholder strings to register with `tokenizer.add_tokens`."""
        return [f"<custom_token_{i}>" for i in range(self.num_added_tokens + 1)]

    # --- audio codes <-> LM token ids ------------------------------------
    def codes_to_ids(self, codes):
        """Mimi codes (num_codebooks, T) -> flat per-frame-interleaved LM ids."""
        codes = np.asarray(codes)
        flat = codes.T.reshape(-1)                      # frame-major order
        cb = np.arange(flat.size) % self.num_codebooks
        ids = flat + AUDIO_OFFSET + cb * self.codebook_size + self.base
        return ids.astype(np.int64).tolist()

    def ids_to_codes(self, ids):
        """Generated LM ids -> Mimi codes (num_codebooks, T); stop at 1st invalid.

        Returns None if not even one full frame of valid codes is present.
        """
        valid = []
        for i, tid in enumerate(ids):
            cb = i % self.num_codebooks
            code = tid - self.base - AUDIO_OFFSET - cb * self.codebook_size
            if 0 <= code < self.codebook_size:
                valid.append(code)
            else:
                break
        n_frames = len(valid) // self.num_codebooks
        if n_frames == 0:
            return None
        arr = np.asarray(valid[: n_frames * self.num_codebooks], dtype=np.int64)
        return arr.reshape(n_frames, self.num_codebooks).T      # (num_codebooks, T)

    # --- training-sequence builders --------------------------------------
    def tts_sequence(self, text_ids, codes):
        """[SOH] text [eot] [EOH] [SOA] [SOS] <audio> [EOS_SPEECH]."""
        return ([self.soh] + list(text_ids)
                + [self.eot, self.eoh, self.soa, self.sos]
                + self.codes_to_ids(codes)
                + [self.eos_speech])

    def qa_sequence(self, question_ids, answer_ids):
        """[SOH] question [eot] [EOH] [SOA] answer [eot]."""
        return ([self.soh] + list(question_ids)
                + [self.eot, self.eoh, self.soa]
                + list(answer_ids) + [self.eot])

    def voice_clone_prompt(self, text_ids, reference_audio_ids):
        """Build the exact prompt consumed by voice-cloning inference."""
        return ([self.soh] + list(text_ids)
                + [self.eot, self.eoh, self.soa, self.sos]
                + list(reference_audio_ids))
