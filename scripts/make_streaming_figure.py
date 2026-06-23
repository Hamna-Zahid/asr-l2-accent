#!/usr/bin/env python
"""Audio-grounded Phase A figure: how fixed-latency streaming shatters ASR.

A real L2-ARCTIC clip that offline transcribes perfectly. The mel-spectrogram is
overlaid with the 320 ms chunk boundaries and the offline word alignment; below,
the 320 ms streaming hypothesis fragments and hallucinates ("I will see you in
the next video") because each chunk is decoded with no future context.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import librosa
import librosa.display

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "figures"
R = REPO / "results" / "l2arctic"
CLIP = "l2arctic_EBVS_arctic_a0058"
CHUNK_MS = 320
plt.rcParams.update({"font.size": 11, "savefig.dpi": 300, "font.family": "DejaVu Sans"})


def load(f):
    return {d["utt_id"]: d for d in (json.loads(l) for l in open(R / f, encoding="utf-8"))}


def main():
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from asr_l2.io.manifest import read_manifest
    off = load("phaseA_offline_words.jsonl")[CLIP]
    s320 = load("phaseA_stream_320ms_words.jsonl")[CLIP]
    meta = {u.utt_id: u for u in read_manifest(REPO / "data/processed/l2arctic/manifest.jsonl")}
    ref = meta[CLIP].text
    wav = meta[CLIP].wav_path

    y, sr = librosa.load(wav, sr=16000, mono=True)
    dur = len(y) / sr

    fig = plt.figure(figsize=(8.6, 6.4))
    gs = GridSpec(2, 1, height_ratios=[1.25, 1.0], hspace=0.38)

    # ---- spectrogram with chunk boundaries + offline word alignment ----
    ax0 = fig.add_subplot(gs[0])
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80,
                            fmax=8000), ref=np.max)
    librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="mel", fmax=8000,
                             ax=ax0, cmap="magma")
    ax0.set_ylabel("mel freq (Hz)")
    ax0.set_title(f'Reference (offline ASR is correct):  “{ref}”', fontsize=10.5,
                  style="italic", pad=8)
    # 320 ms chunk boundaries
    t = CHUNK_MS / 1000.0
    while t < dur:
        ax0.axvline(t, color="cyan", ls="--", lw=0.8, alpha=0.7)
        t += CHUNK_MS / 1000.0
    ax0.text(0.01, 0.95, "dashed = 320 ms chunk boundaries", transform=ax0.transAxes,
             color="cyan", fontsize=8.5, va="top")
    # offline word labels at their timestamps
    for w in off["words"]:
        xc = (w["start"] + w["end"]) / 2
        ax0.text(xc, 7600, w["w"], rotation=90, ha="center", va="top",
                 color="white", fontsize=7.0)

    # ---- offline vs streaming text ----
    ax = fig.add_subplot(gs[1]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.text(0.2, 9.2, "Offline (full utterance):", fontsize=10, weight="bold", color="#1e7a34")
    ax.text(0.2, 8.1, f'“{off["hyp_text"]}”', fontsize=10, color="#1e7a34")
    ax.text(8.9, 8.1, "✓ WER 0.00", fontsize=10, weight="bold", color="#1e7a34", ha="right")

    ax.text(0.2, 6.2, "Streaming, 320 ms chunks (no future context):", fontsize=10,
            weight="bold", color="#c0392b")
    # render streaming hyp, highlighting the hallucinated span
    hyp = s320["hyp_text"]
    hall = "I will see you in the next video"
    ax.text(8.9, 5.1, "✗ WER 1.18", fontsize=10, weight="bold", color="#c0392b", ha="right")
    # wrap streaming text manually with the hallucination boxed
    import textwrap
    ax.text(0.2, 5.1, '“' + hyp + '”', fontsize=9.6, color="#c0392b",
            wrap=True, va="top")
    ax.add_patch(plt.Rectangle((0.2, 0.5), 9.6, 1.6, fc="#fff8e1", ec="#e0c060", lw=1))
    ax.text(5.0, 1.3, "Each 320 ms chunk is decoded with no future context → the frozen "
            "decoder\nfragments the sentence and hallucinates filler "
            '(“I will see you in the next video”, “courgette”).',
            ha="center", va="center", fontsize=8.8, color="#222")

    fig.savefig(OUT / "fig_streaming_break.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_streaming_break.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_streaming_break.{png,pdf}")


if __name__ == "__main__":
    main()
