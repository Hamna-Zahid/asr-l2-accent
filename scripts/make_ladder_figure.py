#!/usr/bin/env python
"""LM-strength ladder figure (Phase B analysis).

Plots the realized rescoring gain against LM strength (word-level perplexity on
held-out test references, stronger = lower ppl, plotted rightward). The gain
climbs with LM strength then SATURATES far short of the oracle ceiling: scaling
the LM 4x past GPT-2-medium adds nothing, and the plateau captures only ~a
quarter of the recoverable head-room. The remaining gap is the accent/acoustic
error -- recoverable hypotheses exist in the n-best but are invisible to any text
LM. A fine-tuned 124M model reaches the same plateau as a 1.5B general model.
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
    d = json.loads((R / "phaseB_lm_ladder.json").read_text())
    base, oracle = d["test_wer_baseline"], d["test_wer_oracle"]
    oracle_rel = 100 * (oracle - base) / base          # ~ -50%

    scaled, ft = [], None
    for r in d["ladder"]:
        pt = (r["ppl_word"], r["delta_rel_pct"], r["label"])
        if "fine-tuned" in r["label"]:
            ft = pt
        else:
            scaled.append(pt)
    scaled.sort(key=lambda x: -x[0])                    # weak -> strong (left->right)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    xs = [p[0] for p in scaled]; ys = [p[1] for p in scaled]
    ax.plot(xs, ys, "o-", color="#1f3b73", lw=2.2, ms=7, zorder=5,
            label="general LMs (4-gram -> GPT-2-xl)")
    for ppl, dr, lab in scaled:
        short = lab.split(" (")[0].replace("GPT-2", "G2")
        ax.annotate(short, (ppl, dr), textcoords="offset points",
                    xytext=(0, 9 if dr > -11 else -14), ha="center", fontsize=7.6,
                    color="#1f3b73")

    if ft:
        ax.scatter([ft[0]], [ft[1]], s=210, marker="*", color="#1e9e4a",
                   edgecolor="black", lw=0.8, zorder=6,
                   label="GPT-2 fine-tuned (124M, in-domain)")
        ax.annotate("fine-tuned 124M\n= 3x-bigger general LM", (ft[0], ft[1]),
                    textcoords="offset points", xytext=(14, -6), fontsize=8,
                    color="#1e7a34")

    # oracle ceiling + the unreachable acoustic band
    ax.axhline(oracle_rel, ls="--", color="#c0392b", lw=1.6)
    ax.axhline(0, color="#888", ls=":", alpha=0.7)
    plateau = min(p[1] for p in scaled)
    ax.axhspan(oracle_rel, plateau, color="#c0392b", alpha=0.06)
    ax.text(min(xs) * 0.85, (oracle_rel + plateau) / 2,
            "accent / acoustic bound\n(in the n-best, unreachable by any text LM)",
            ha="left", va="center", fontsize=8.6, color="#a02020")
    ax.text(min(xs) * 0.85, oracle_rel + 1.2,
            f"oracle ceiling ({oracle_rel:.0f}%): best hyp in top-10",
            ha="left", va="bottom", fontsize=8.2, color="#c0392b")

    ax.set_xscale("log")
    ax.invert_xaxis()                                  # stronger LM (lower ppl) -> right
    ax.set_xlabel("LM strength  =  word-level perplexity on held-out references  (lower = stronger ->)")
    ax.set_ylabel("Test WER change vs baseline (%)")
    ax.set_title("Rescoring gain scales with LM strength, then saturates far below the ceiling")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="center right", fontsize=8.5)
    ax.set_ylim(oracle_rel - 4, 5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_lm_ladder.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_lm_ladder.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_lm_ladder.{png,pdf}")
    print(f"  oracle ceiling {oracle_rel:.1f}%  plateau {plateau:.1f}%  "
          f"(captures {plateau/oracle_rel*100:.0f}% of recoverable)")


if __name__ == "__main__":
    main()
