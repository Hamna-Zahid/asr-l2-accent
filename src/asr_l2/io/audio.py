"""Audio loading / conversion helpers — everything normalizes to 16 kHz mono."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16000


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load any audio file as float32 mono at ``target_sr``.

    Returns a 1-D numpy array in [-1, 1]. librosa handles resampling and
    down-mixing; this is the single entry point used across the pipeline so
    sample-rate assumptions never drift between stages.
    """
    audio, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return audio.astype(np.float32)


def save_wav(path: str | Path, audio: np.ndarray, sr: int = TARGET_SR) -> None:
    """Write a mono float array as 16-bit PCM WAV (clipped to [-1, 1])."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(str(path), audio, sr, subtype="PCM_16")


def convert_to_wav16k(src: str | Path, dst: str | Path) -> float:
    """Convert any audio file to 16 kHz mono PCM16 WAV. Returns duration (s)."""
    audio = load_audio(src, TARGET_SR)
    save_wav(dst, audio, TARGET_SR)
    return len(audio) / TARGET_SR
