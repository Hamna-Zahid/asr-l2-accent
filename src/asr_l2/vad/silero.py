"""Silero VAD wrapper.

Used in two places:
  - optional pre-segmentation of long audio into utterance-like spans;
  - the "VAD cutoff" error hypothesis in Phase A (a deletion at a span boundary
    is a candidate VAD-induced error, distinct from an accent-induced one).

The model is small and CPU-friendly. We load it once and reuse it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.audio import TARGET_SR


@dataclass
class SpeechSpan:
    start_s: float
    end_s: float


class SileroVad:
    def __init__(self, threshold: float = 0.5, min_speech_ms: int = 250,
                 min_silence_ms: int = 100, sample_rate: int = TARGET_SR):
        # Imported lazily so the rest of the pipeline does not require torch
        # to be present just to import this module.
        from silero_vad import load_silero_vad
        self.model = load_silero_vad()
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.sample_rate = sample_rate

    def segments(self, audio: np.ndarray) -> list[SpeechSpan]:
        """Return detected speech spans (seconds) for a mono 16 kHz array."""
        import torch
        from silero_vad import get_speech_timestamps

        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        ts = get_speech_timestamps(
            wav, self.model, threshold=self.threshold,
            sampling_rate=self.sample_rate,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            return_seconds=True,
        )
        return [SpeechSpan(t["start"], t["end"]) for t in ts]
