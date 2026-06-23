"""N-best decoding from a frozen wav2vec2 (CTC) model -- a *second ASR family*.

Whisper is an attention encoder-decoder; wav2vec2 is a self-supervised encoder
with a CTC head. Running Phase A (streaming) and Phase B (audio-free n-best
rescoring) on both lets us show our findings are not a Whisper-specific quirk.

CTC n-best with acoustic scores comes from a beam-search decoder (pyctcdecode):
each beam carries a ``logit_score`` (the acoustic beam log-probability), which is
the analogue of Whisper's length-penalized average log-prob. No external LM is
used here -- this is the purely acoustic n-best that Phase B then rescores, so
the comparison to Whisper is apples-to-apples.

Interface matches asr.nbest.NBestEngine: ``nbest(audio, n) -> [Hypothesis]``.
"""
from __future__ import annotations

import numpy as np

from .nbest import Hypothesis   # reuse the same dataclass (text, acoustic_score)

DEFAULT_MODEL = "facebook/wav2vec2-large-960h-lv60-self"


class Wav2Vec2NBestEngine:
    """Frozen wav2vec2-CTC that emits N scored hypotheses per clip."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str | None = None,
                 beam_width: int = 100):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        from pyctcdecode import build_ctcdecoder

        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.beam_width = beam_width
        self.processor = Wav2Vec2Processor.from_pretrained(model)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model, torch_dtype=dtype).to(device).eval()

        # Map the wav2vec2 vocabulary to pyctcdecode conventions: blank token
        # ("<pad>") -> "", word delimiter ("|") -> " ". Other special tokens are
        # left as-is (they essentially never win a beam).
        vocab = self.processor.tokenizer.get_vocab()
        labels = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
        labels = ["" if t == "<pad>" else " " if t == "|" else t for t in labels]
        self.decoder = build_ctcdecoder(labels)
        self.name = model

    def nbest(self, audio: np.ndarray, n: int = 10) -> list[Hypothesis]:
        """Return up to ``n`` CTC beam hypotheses (text, acoustic_score)."""
        iv = self.processor(audio, sampling_rate=16000,
                            return_tensors="pt").input_values.to(self.device)
        if self.device == "cuda":
            iv = iv.half()
        with self._torch.no_grad():
            logits = self.model(iv).logits[0].float().cpu().numpy()

        beams = self.decoder.decode_beams(logits, beam_width=max(n, self.beam_width))
        hyps: list[Hypothesis] = []
        for b in beams[:n]:
            text = b[0].strip().lower()            # beam text (space-joined words)
            logit_score = float(b[-2])             # acoustic beam log-prob
            hyps.append(Hypothesis(text=text, acoustic_score=logit_score))
        return hyps
