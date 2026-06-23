#!/usr/bin/env python
"""Oracle head-room figure (Phase B analysis).

Shows that ~half the baseline error is *recoverable by reranking alone* (the
correct words are already in the n-best), yet a text-only LM captures only a
small slice of it -- localizing the residual as acoustic/accent-bound error that
sits in the list but is invisible to a language model. Reads phaseB_oracle.json
(produced by oracle_bound.py) and the GPT-2 summary.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "figures"
R = REPO / "results" / "l2arctic_24spk"
plt.rcParams.update({"font.size": 11, "savefig.dpi": 300, "font.family": "DejaVu Sans"})


def main():
    o = json.loads((R / "phaseB_oracle.json").read_text())
    base = o["test_wer_baseline_top1"]
    oracle = o["test_wer_oracle"]
    gpt2 = o["test_wer_gpt2_adapted"]
    depth = {int(k): v for k, v in o["oracle_by_depth"].items()}
    cap = o["gpt2_fraction_of_headroom_captured"]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ds = sorted(depth)
    ax.plot(ds, [depth[d] for d in ds], "o-", color="#1f3b73", lw=2.2,
            label="oracle (best in top-N)", zorder=5)
    for d in ds:
        ax.annotate(f"{depth[d]:.3f}", (d, depth[d]), textcoords="offset points",
                    xytext=(6, 6), fontsize=8, color="#1f3b73")

    # baseline (top-1) and GPT-2 reference lines
    ax.axhline(base, ls="--", color="#c0392b", lw=1.6, label=f"baseline top-1 ({base:.3f})")
    ax.axhline(gpt2, ls="-.", color="#1e7a34", lw=1.6,
               label=f"GPT-2 rescoring ({gpt2:.3f})")

    # shade the recoverable head-room (baseline -> oracle@10)
    ax.axhspan(oracle, base, xmin=0.0, xmax=1.0, color="#1f3b73", alpha=0.06)
    ax.annotate("", xy=(9.4, oracle), xytext=(9.4, base),
                arrowprops=dict(arrowstyle="<->", color="#555", lw=1.3))
    ax.text(9.0, (base + oracle) / 2,
            f"50% of error\nrecoverable by\nreranking", ha="right", va="center",
            fontsize=9, color="#333")

    # what GPT-2 captures
    ax.annotate("", xy=(1.9, gpt2), xytext=(1.9, base),
                arrowprops=dict(arrowstyle="<->", color="#1e7a34", lw=1.3))
    ax.text(2.15, (base + gpt2) / 2, f"GPT-2 captures\nonly {cap*100:.0f}% of it",
            ha="left", va="center", fontsize=9, color="#1e7a34")

    ax.set_xlabel("n-best depth N (hypotheses considered)")
    ax.set_ylabel("Test WER (L2-ARCTIC, 24 speakers)")
    ax.set_title("Half the error is in the n-best; a text LM reaches little of it")
    ax.set_xticks(ds)
    ax.set_ylim(0.03, 0.092)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_oracle_headroom.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_oracle_headroom.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_oracle_headroom.{png,pdf}")
    print(f"  baseline {base:.4f}  gpt2 {gpt2:.4f}  oracle {oracle:.4f}  captured {cap*100:.1f}%")


if __name__ == "__main__":
    main()
