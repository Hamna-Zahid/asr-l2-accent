"""N-best decoding from a frozen Whisper, for Phase B audio-free rescoring.

faster-whisper's high-level ``transcribe`` only returns the single best
hypothesis. Phase B needs the top-N candidates *with acoustic scores* so a
text-only domain LM can re-rank them (shallow-fusion-style rescoring on a frozen
decoder — no audio fine-tuning). We get this by driving the underlying
CTranslate2 ``Whisper.generate`` with ``num_hypotheses`` + ``return_scores``,
replicating faster-whisper's own feature/encode/prompt pipeline (v1.2.1).

Each clip is decoded as a single <=30s window (our eval clips are short), which
is exactly what we want for whole-utterance rescoring.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import pad_or_trim
from faster_whisper.tokenizer import Tokenizer

from .engine import _resolve_device


@dataclass
class Hypothesis:
    text: str
    acoustic_score: float   # length-penalized avg log-prob from the decoder


class NBestEngine:
    """Frozen Whisper that emits N scored hypotheses per clip."""

    def __init__(self, model: str = "small.en", device: str = "auto",
                 compute_type: str = "int8", cpu_threads: int = 4,
                 language: str = "en"):
        resolved_device, _ = _resolve_device(device)
        ctype = compute_type if resolved_device == "cpu" else "float16"
        self.fw = WhisperModel(model, device=resolved_device, compute_type=ctype,
                               cpu_threads=cpu_threads)
        self.tokenizer = Tokenizer(
            self.fw.hf_tokenizer, self.fw.model.is_multilingual,
            task="transcribe", language=language,
        )
        self.device = resolved_device

    def nbest(self, audio: np.ndarray, n: int = 10,
              length_penalty: float = 1.0) -> list[Hypothesis]:
        """Return up to ``n`` hypotheses (text, acoustic_score), best-first."""
        features = self.fw.feature_extractor(audio)
        features = pad_or_trim(features)             # -> [n_mels, 3000]
        encoder_output = self.fw.encode(features)
        prompt = self.fw.get_prompt(self.tokenizer, [], without_timestamps=True)

        result = self.fw.model.generate(
            encoder_output,
            [prompt],
            beam_size=max(n, 5),
            num_hypotheses=n,
            length_penalty=length_penalty,
            return_scores=True,
            max_length=self.fw.max_length,
            suppress_blank=True,
            suppress_tokens=[-1],
        )[0]

        hyps: list[Hypothesis] = []
        for ids, score in zip(result.sequences_ids, result.scores):
            text = self.tokenizer.decode(list(ids)).strip()
            hyps.append(Hypothesis(text=text, acoustic_score=float(score)))
        return hyps
