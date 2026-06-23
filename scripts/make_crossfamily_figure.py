#!/usr/bin/env python
"""Cross-family figure: LM strength decides the rescoring gain on Whisper (which
has an internal LM and a lexically rich n-best) but not on wav2vec2 (CTC, no
internal LM, thin n-best). Uses the headline Table-8 numbers so it is consistent
with the text by construction.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures"
plt.rcParams.update({"font.size": 11, "savefig.dpi": 300, "font.family": "DejaVu Sans"})

# (family, oracle %, 4-gram rel %, GPT-2 rel %)
DATA = [("Whisper-small\n(attn. enc-dec)", 50, +0.8, -8.0),
        ("wav2vec2-large\n(CTC)",          26, -4.6, -3.0)]


def main():
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    x = np.arange(len(DATA)); w = 0.34
    weak = [d[2] for d in DATA]; strong = [d[3] for d in DATA]
    b1 = ax.bar(x - w/2, weak,   w, label="weak LM (4-gram)",  color="#1f77b4")
    b2 = ax.bar(x + w/2, strong, w, label="strong LM (GPT-2)", color="#d62728")
    ax.axhline(0, color="#333", lw=0.8)
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.text(r.get_x()+r.get_width()/2, h + (0.3 if h >= 0 else -0.3),
                    f"{h:+.1f}%", ha="center", va="bottom" if h >= 0 else "top",
                    fontsize=9.5, weight="bold")
    # annotate the oracle head-room under each family
    for i, d in enumerate(DATA):
        ax.text(i, -10.6, f"oracle: {d[1]}% recoverable", ha="center",
                fontsize=8.8, color="#555")
    ax.annotate("LM strength\ndecisive", (0, -8.0), xytext=(0.34, -6.0),
                fontsize=8.6, color="#a02020", ha="left",
                arrowprops=dict(arrowstyle="->", color="#a02020"))
    ax.annotate("LM strength\nbarely matters", (1, -3.0), xytext=(1.18, -1.2),
                fontsize=8.6, color="#1f4e79", ha="left",
                arrowprops=dict(arrowstyle="->", color="#1f4e79"))
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in DATA])
    ax.set_ylabel("Test WER change vs baseline (%)")
    ax.set_title("LM strength decides the gain on Whisper, but not on wav2vec2")
    ax.set_ylim(-11.2, 3.4)
    ax.legend(loc="upper center", ncol=2, fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_crossfamily.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_crossfamily.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_crossfamily.{png,pdf}")


if __name__ == "__main__":
    main()
