#!/usr/bin/env python
"""Flagship audio-grounded figure: how a text LM repairs accented ASR.

A real L2-ARCTIC clip whose top acoustic hypothesis has a homophone error
('where' for 'were') that Whisper cannot resolve from sound, but GPT-2 can from
grammar. Top panel: the clip's mel-spectrogram. Bottom: the n-best re-ranking -
acoustics put the wrong wording at #1; acoustic + lambda*LM promotes the correct
one. n-best is decoded once and cached so the figure is reproducible.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import numpy as np
import librosa
import librosa.display

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "figures"
CACHE = REPO / "results" / "figure_data"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

CLIP = "l2arctic_ABA_arctic_a0022"
LAMBDA = 0.1
plt.rcParams.update({"font.size": 11, "savefig.dpi": 300, "font.family": "DejaVu Sans"})


def get_nbest():
    cf = CACHE / f"{CLIP}_nbest.json"
    if cf.exists():
        return json.loads(cf.read_text())
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from asr_l2.asr.nbest import NBestEngine
    from asr_l2.lm.neural import NeuralLM
    from asr_l2.io.audio import load_audio
    from asr_l2.io.manifest import read_manifest
    meta = {u.utt_id: u for u in read_manifest(REPO / "data/processed/l2arctic/manifest.jsonl")}
    u = meta[CLIP]
    eng = NBestEngine(model="small.en", device="cpu", compute_type="int8")
    lm = NeuralLM("gpt2", device="cpu")
    hyps = eng.nbest(load_audio(u.wav_path), n=8)
    data = {"ref": u.text, "wav": u.wav_path,
            "hyps": [{"text": h.text, "ac": h.acoustic_score,
                      "lm": lm.logprob_per_word(h.text)} for h in hyps]}
    cf.write_text(json.dumps(data, indent=2))
    return data


def wer(ref, hyp):
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from asr_l2.scoring.metrics import score_pair
    return score_pair(ref, hyp)["wer"]


def shorten(text, ref):
    """Abbreviate a hypothesis (first 3 + last 3 words), UPPERCASING any word
    that differs from the reference so the discriminative error pops."""
    rwords = ref.lower().strip().strip('."').replace(",", "").split()
    t = text.strip().strip('."').replace(",", "")
    words = t.split()

    def mark(w, i):
        rw = rwords[i] if i < len(rwords) else ""
        return w.upper() if w.lower() != rw else w

    marked = [mark(w, i) for i, w in enumerate(words)]
    if len(marked) > 7:
        return " ".join(marked[:3]) + " … " + " ".join(marked[-3:])
    return " ".join(marked)


def main():
    d = get_nbest()
    ref = d["ref"]
    hyps = d["hyps"]
    for h in hyps:
        h["comb"] = h["ac"] + LAMBDA * h["lm"]
        h["wer"] = wer(ref, h["text"])
    ac_order = sorted(range(len(hyps)), key=lambda i: -hyps[i]["ac"])
    cb_order = sorted(range(len(hyps)), key=lambda i: -hyps[i]["comb"])
    ac_top, cb_top = ac_order[0], cb_order[0]

    fig = plt.figure(figsize=(8.4, 7.2))
    gs = GridSpec(2, 1, height_ratios=[1.0, 1.45], hspace=0.32)

    # ---- top: mel-spectrogram ----
    ax0 = fig.add_subplot(gs[0])
    y, sr = librosa.load(d["wav"], sr=16000, mono=True)
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80,
                            fmax=8000), ref=np.max)
    librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="mel", fmax=8000,
                             ax=ax0, cmap="magma")
    ax0.set_title(f'Reference:  “{ref}”', fontsize=11, style="italic", pad=8)
    ax0.set_ylabel("mel freq (Hz)")
    ax0.text(0.01, 0.92, "real L2-accented speech (Arabic L1)", transform=ax0.transAxes,
             color="white", fontsize=8.5, va="top")

    # ---- bottom: n-best re-ranking ----
    ax = fig.add_subplot(gs[1]); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    # ensure the rescored-correct hypothesis is shown on the left even if its
    # acoustic rank is low (that low rank is the whole point of the figure).
    nshow = max(6, ac_order.index(cb_top) + 1) if cb_top in ac_order else 6
    nshow = min(nshow, len(hyps))
    yr = np.linspace(8.4, 1.3, nshow)
    Lx, Rx, bw = 0.2, 5.5, 4.3

    def box(x, y, text, color, ec, bold=False):
        ax.add_patch(FancyBboxPatch((x, y - 0.42), bw, 0.84,
                     boxstyle="round,pad=0.03", fc=color, ec=ec, lw=1.6 if bold else 0.8))
        ax.text(x + 0.12, y, text, va="center", ha="left", fontsize=8.6,
                weight="bold" if bold else "normal")

    ax.text(Lx + bw/2, 9.2, "1.  Whisper n-best\n(ranked by acoustic score)",
            ha="center", fontsize=10, weight="bold")
    ax.text(Rx + bw/2, 9.2, "2.  After GPT-2 rescoring\n(acoustic + λ·LM)",
            ha="center", fontsize=10, weight="bold")

    pos_left = {}
    for rank, i in enumerate(ac_order[:nshow]):
        y = yr[rank]; pos_left[i] = y
        correct = hyps[i]["wer"] == 0
        is_top = (i == ac_top)
        mark = "✗ " if (is_top and not correct) else ("✓ " if correct else "")
        col = "#fde0e0" if (is_top and not correct) else ("#e3f4e3" if correct else "#f4f4f4")
        ec = "#c0392b" if (is_top and not correct) else ("#27ae60" if correct else "#bbb")
        box(Lx, y, f"{mark}{shorten(hyps[i]['text'], ref)}", col, ec, bold=is_top)

    pos_right = {}
    for rank, i in enumerate(cb_order[:nshow]):
        y = yr[rank]; pos_right[i] = y
        correct = hyps[i]["wer"] == 0
        is_top = (i == cb_top)
        mark = "✓ " if (is_top and correct) else ""
        col = "#e3f4e3" if (is_top and correct) else "#f4f4f4"
        ec = "#27ae60" if (is_top and correct) else "#bbb"
        box(Rx, y, f"{mark}{shorten(hyps[i]['text'], ref)}", col, ec, bold=is_top)

    # arrow: the correct hypothesis moves from its low acoustic rank to #1
    if cb_top in pos_left:
        ax.add_patch(FancyArrowPatch((Lx + bw + 0.08, pos_left[cb_top]),
                     (Rx - 0.08, pos_right[cb_top]), connectionstyle="arc3,rad=-0.35",
                     arrowstyle="-|>", mutation_scale=22, lw=2.4, color="#1e9e4a"))
        ax.text(5.0, (pos_left[cb_top] + pos_right[cb_top]) / 2,
                "LM promotes\nrank 6 → 1", ha="center", fontsize=8.8,
                color="#1e7a34", style="italic", weight="bold")

    ax.text(5.0, 0.35, "A homophone the acoustics cannot resolve — "
            "“where” ranks #1 by sound, “were” ranks #1 by the LM.   WER 0.08 → 0.00",
            ha="center", fontsize=9, color="#222",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e1", ec="#e0c060"))

    fig.savefig(OUT / "fig_hero_rescore.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_hero_rescore.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_hero_rescore.{png,pdf}")
    print("acoustic #1:", repr(hyps[ac_top]["text"]), "WER", hyps[ac_top]["wer"])
    print("rescored #1:", repr(hyps[cb_top]["text"]), "WER", hyps[cb_top]["wer"])


if __name__ == "__main__":
    main()
