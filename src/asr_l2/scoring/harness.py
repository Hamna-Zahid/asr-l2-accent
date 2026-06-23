"""Phase A.1 streaming-mismatch harness core.

Runs each utterance through the ASR engine under several decoding *conditions*
(offline, and streaming at each chunk size) and records WER/CER/latency. The
CLI wrapper (`scripts/run_streaming_harness.py`) handles config + I/O; this
module holds the loop so it is unit-testable on a tiny synthetic manifest.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..asr.engine import AsrEngine, AsrResult
from ..io.audio import load_audio
from ..io.manifest import Utterance
from .metrics import score_pair


@dataclass
class Condition:
    name: str               # "offline" | "stream_320ms" | ...
    mode: str               # "offline" | "streaming"
    chunk_ms: int | None    # None for offline
    left_ctx_ms: int | None


def build_conditions(chunk_sizes_ms: list[int], left_ctx_ms: int) -> list[Condition]:
    conds = [Condition("offline", "offline", None, None)]
    for c in chunk_sizes_ms:
        conds.append(Condition(f"stream_{c}ms", "streaming", c, left_ctx_ms))
    return conds


def _decode(engine: AsrEngine, audio, cond: Condition, beam_size: int) -> AsrResult:
    if cond.mode == "offline":
        return engine.transcribe_offline(audio, beam_size=beam_size)
    return engine.transcribe_streaming(audio, chunk_ms=cond.chunk_ms,
                                       left_ctx_ms=cond.left_ctx_ms,
                                       beam_size=beam_size)


def run_one(engine: AsrEngine, utt: Utterance, cond: Condition,
            beam_size: int) -> dict:
    """Decode one utterance under one condition and score it. Returns a row dict."""
    audio = load_audio(utt.wav_path)
    res = _decode(engine, audio, cond, beam_size)
    sc = score_pair(utt.text, res.text)
    return {
        "utt_id": utt.utt_id,
        "dataset": utt.dataset,
        "speaker": utt.speaker,
        "accent": utt.accent,
        "cefr": utt.cefr,
        "condition": cond.name,
        "mode": cond.mode,
        "chunk_ms": cond.chunk_ms,
        "left_ctx_ms": cond.left_ctx_ms,
        "ref_text": utt.text,
        "hyp_text": res.text,
        "duration_s": round(res.audio_dur_s, 3),
        "wer": sc["wer"],
        "cer": sc["cer"],
        "n_ref_words": sc["n_ref_words"],
        "substitutions": sc["substitutions"],
        "deletions": sc["deletions"],
        "insertions": sc["insertions"],
        "proc_time_s": round(res.proc_time_s, 4),
        "rtf": round(res.rtf, 4),
        "mean_chunk_proc_s": (round(res.mean_chunk_proc_s, 4)
                              if res.mean_chunk_proc_s is not None else None),
        "algorithmic_latency_ms": res.algorithmic_latency_ms,
        # Word timings travel alongside the row so the error-CSV builder never
        # has to re-decode (expensive on a slow CPU). The harness CLI persists
        # these to a sidecar for the offline condition and drops the column.
        "_hyp_words": [{"w": w.word, "start": round(w.start, 3),
                        "end": round(w.end, 3), "prob": round(w.prob, 4)}
                       for w in res.words],
    }
