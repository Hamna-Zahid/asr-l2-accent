"""faster-whisper wrapper: offline (full-utterance) and simulated streaming.

Streaming model
---------------
Real streaming ASR degrades mainly because the decoder cannot see *future*
audio. We reproduce that here without a true online decoder:

  For each fixed chunk of ``chunk_ms`` covering (t, t+C], we transcribe the
  window [max(0, t+C - left_ctx_ms), t+C] -- i.e. a bounded left context plus
  the current chunk, but NEVER any audio past t+C. We then emit only the words
  whose timestamps fall in the newly revealed region (> last emitted time).
  Concatenating across chunks yields the streaming hypothesis.

This isolates the "no future context" effect (the streaming/chunking penalty)
from accent/disfluency effects, which is exactly the Phase A separation goal.
The algorithmic latency of such a system is ~chunk_ms (no look-ahead).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from faster_whisper import WhisperModel

from ..io.audio import TARGET_SR


@dataclass
class Word:
    word: str
    start: float
    end: float
    prob: float


@dataclass
class AsrResult:
    text: str
    words: list[Word] = field(default_factory=list)
    proc_time_s: float = 0.0       # wall-clock processing time
    audio_dur_s: float = 0.0
    rtf: float = 0.0               # real-time factor = proc_time / audio_dur
    # streaming-only diagnostics:
    mean_chunk_proc_s: float | None = None
    algorithmic_latency_ms: float | None = None


def _resolve_device(device: str) -> tuple[str, str]:
    """Return (device, compute_type_hint). Falls back to CPU if no CUDA."""
    if device == "auto":
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda", "float16"
        except Exception:
            pass
        return "cpu", "int8"
    return device, "int8"


class AsrEngine:
    """Loads a frozen Whisper model once and runs offline / streaming decoding."""

    def __init__(self, model: str = "small.en", device: str = "auto",
                 compute_type: str = "int8", cpu_threads: int = 4,
                 language: str = "en"):
        resolved_device, _hint = _resolve_device(device)
        self.device = resolved_device
        self.language = language
        # On GPU, int8 is not ideal; honor caller's compute_type on CPU.
        ctype = compute_type if resolved_device == "cpu" else "float16"
        self.model = WhisperModel(
            model, device=resolved_device, compute_type=ctype,
            cpu_threads=cpu_threads,
        )

    # ---- offline -----------------------------------------------------------
    def transcribe_offline(self, audio: np.ndarray, beam_size: int = 5) -> AsrResult:
        t0 = time.perf_counter()
        segments, _info = self.model.transcribe(
            audio, language=self.language, beam_size=beam_size,
            word_timestamps=True, vad_filter=False,
        )
        words: list[Word] = []
        for seg in segments:               # generator -> realizes decoding
            for w in (seg.words or []):
                words.append(Word(w.word.strip(), w.start, w.end, w.probability))
        proc = time.perf_counter() - t0
        dur = len(audio) / TARGET_SR
        text = " ".join(w.word for w in words).strip()
        return AsrResult(text=text, words=words, proc_time_s=proc,
                         audio_dur_s=dur, rtf=proc / dur if dur else 0.0)

    # ---- streaming (simulated) --------------------------------------------
    def transcribe_streaming(self, audio: np.ndarray, chunk_ms: int,
                             left_ctx_ms: int, beam_size: int = 5) -> AsrResult:
        sr = TARGET_SR
        chunk = int(chunk_ms / 1000 * sr)
        left = int(left_ctx_ms / 1000 * sr)
        n = len(audio)
        dur = n / sr

        emitted: list[Word] = []
        last_end = 0.0
        chunk_times: list[float] = []
        t_start = time.perf_counter()

        pos = 0
        while pos < n:
            win_end = min(pos + chunk, n)
            win_start = max(0, win_end - left - chunk)
            window = audio[win_start:win_end]
            offset = win_start / sr

            tc = time.perf_counter()
            segments, _ = self.model.transcribe(
                window, language=self.language, beam_size=beam_size,
                word_timestamps=True, vad_filter=False,
            )
            for seg in segments:
                for w in (seg.words or []):
                    abs_start = w.start + offset
                    abs_end = w.end + offset
                    # Emit only newly revealed words inside this chunk region.
                    if abs_start >= last_end - 0.02 and abs_end <= win_end / sr + 0.02:
                        emitted.append(Word(w.word.strip(), abs_start, abs_end,
                                            w.probability))
                        last_end = max(last_end, abs_end)
            chunk_times.append(time.perf_counter() - tc)
            pos = win_end

        proc = time.perf_counter() - t_start
        text = " ".join(w.word for w in emitted).strip()
        return AsrResult(
            text=text, words=emitted, proc_time_s=proc, audio_dur_s=dur,
            rtf=proc / dur if dur else 0.0,
            mean_chunk_proc_s=float(np.mean(chunk_times)) if chunk_times else 0.0,
            algorithmic_latency_ms=float(chunk_ms),
        )
