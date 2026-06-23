#!/usr/bin/env python
"""Generate publication-quality figures for the paper into paper/figures/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
OUT = REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 300, "savefig.dpi": 300,
                     "font.family": "DejaVu Sans", "axes.titlesize": 12})


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10, 4.2)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.2)

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                     fc=color, ec="#333", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                     arrowstyle="-|>", mutation_scale=14, lw=1.3, color="#333"))

    box(0.2, 2.6, 1.5, 1.0, "L2 accented\nspeech", "#dbe9f6")
    box(2.0, 2.6, 1.6, 1.0, "Silero VAD\n(segmentation)", "#dbe9f6")
    box(3.9, 2.6, 1.9, 1.0, "Whisper decoder\n(frozen, faster-whisper)", "#cfe8cf")
    box(6.1, 3.0, 1.7, 0.9, "offline /\nstreaming", "#f6e3c5")
    box(6.1, 1.9, 1.7, 0.9, "N-best +\nacoustic scores", "#f6e3c5")
    for (x1, y1, x2, y2) in [(1.7, 3.1, 2.0, 3.1), (3.6, 3.1, 3.9, 3.1),
                             (5.8, 3.2, 6.1, 3.4), (5.8, 3.0, 6.1, 2.4)]:
        arrow(x1, y1, x2, y2)
    # Phase A
    box(8.1, 3.0, 1.7, 0.95, "Phase A:\nerror taxonomy\n(stream/accent/model)", "#e8d5e8")
    arrow(7.8, 3.45, 8.1, 3.45)
    # Phase B
    box(8.1, 1.7, 1.7, 0.95, "Phase B:\ntext-only LM\nn-best rescoring", "#e8d5e8")
    arrow(7.8, 2.35, 8.1, 2.35)
    box(3.9, 0.5, 1.9, 0.8, "domain text corpus\n(leakage-guarded)", "#f3cccc")
    arrow(4.85, 1.3, 7.0, 1.9)
    ax.text(5.0, 4.0, "Cascaded ASR pipeline", ha="center", fontsize=12, weight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig1_pipeline.png", bbox_inches="tight"); fig.savefig(OUT / "fig1_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


def _cond_summary(ds):
    df = pd.read_parquet(RES / ds / "phaseA_streaming_results.parquet")
    order = ["offline", "stream_1280ms", "stream_640ms", "stream_320ms"]
    g = df.groupby("condition").wer.mean()
    return [g[c] for c in order]


def fig_latency_wer():
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    nice = {"svarah": "Svarah", "l2arctic": "L2-ARCTIC"}
    for ds, mk, col in [("svarah", "o", "#1f77b4"), ("l2arctic", "s", "#d62728")]:
        wers = _cond_summary(ds)            # [offline, 1280, 640, 320]
        ax.plot([1280, 640, 320], wers[1:], marker=mk, color=col, lw=2,
                label=f"{nice[ds]} (streaming)")
        ax.axhline(wers[0], ls="--", alpha=0.6, color=col,
                   label=f"{nice[ds]} (offline)")
        # annotate the 320 ms degradation factor vs offline
        if wers[0]:
            ax.annotate(f"{wers[3]/wers[0]:.1f}x offline", (320, wers[3]),
                        textcoords="offset points", xytext=(10, -4), fontsize=8.5,
                        color=col, weight="bold")
    ax.set_xlabel("Chunk size = algorithmic latency (ms)")
    ax.set_ylabel("Mean WER")
    ax.set_title("Streaming latency vs WER")
    ax.set_xticks([1280, 640, 320])
    ax.invert_xaxis(); ax.legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "fig2_latency_wer.png", bbox_inches="tight"); fig.savefig(OUT / "fig2_latency_wer.pdf", bbox_inches="tight"); plt.close(fig)


def fig_error_categories():
    cats = ["accent_phoneme", "model_lexical", "hallucination",
            "normalization", "deletion_other", "reference_error"]
    labels = ["accent\n(sound)", "model\nlexical", "halluc.", "normaliz.",
              "deletion", "ref.\nerror"]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    width = 0.38
    x = np.arange(len(cats))
    nice = {"l2arctic": "L2-ARCTIC", "svarah": "Svarah"}
    for i, ds in enumerate(["l2arctic", "svarah"]):
        s = json.loads((RES / ds / "phaseA_error_categories.json").read_text())
        props = [s["proportions"].get(c, 0) * 100 for c in cats]
        ax.bar(x + (i - 0.5) * width, props, width, label=nice[ds],
               color=("#1f77b4" if ds == "l2arctic" else "#ff7f0e"))
    # Svarah has no phoneme labels -> no accent share; mark it explicitly
    ax.annotate("n/a\n(no phoneme\nlabels)", (x[0] + 0.5 * width, 1.5),
                ha="center", va="bottom", fontsize=7.5, color="#b25a00")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("% of errors"); ax.set_title("Error category composition (offline)")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig3_error_categories.png", bbox_inches="tight"); fig.savefig(OUT / "fig3_error_categories.pdf", bbox_inches="tight"); plt.close(fig)


def fig_phaseB():
    """Dev WER vs fusion weight lambda, with the realized TEST outcome at each
    LM's dev-chosen lambda overlaid. Both LMs lower DEV WER, but only the strong
    LM's gain GENERALIZES: the weak n-gram's dev-optimal lambda raises TEST WER
    (overfit), while GPT-2's lowers it. Tuning on dev / reporting on test is the
    honest, non-peeking protocol, so test is a single marker, not a curve."""
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    sources = [("phaseB_summary_ngram.json", "4-gram LM (weak)", "o", "#1f77b4"),
               ("phaseB_summary_gpt2.json", "GPT-2 LM (strong)", "s", "#d62728")]
    # Prefer the definitive 24-speaker set if present.
    base_dir = "l2arctic_24spk" if (RES / "l2arctic_24spk" / "phaseB_summary_gpt2.json").exists() else "l2arctic"
    for fname, label, mk, color in sources:
        p = RES / base_dir / fname
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        lams = sorted(float(k) for k in s["dev_wer_by_lambda"])
        wers = [s["dev_wer_by_lambda"][str(l)] for l in lams]
        w0 = wers[0] if wers[0] else 1
        ax.plot(lams, [w / w0 for w in wers], marker=mk, color=color, alpha=0.85,
                label=f"{label} - dev")
        # realized test outcome at the dev-selected lambda (relative to test base)
        lam = s["best_lambda"]
        tb, ta = s["test_wer_baseline"], s["test_wer_adapted"]
        if tb:
            trel = ta / tb
            ax.scatter([lam], [trel], s=170, marker="*", color=color,
                       edgecolor="black", linewidth=0.8, zorder=6)
            ax.annotate(f"test {(trel-1)*100:+.1f}%", (lam, trel),
                        textcoords="offset points", xytext=(8, -2 if trel > 1 else 8),
                        fontsize=8.5, color=color, weight="bold")
    ax.axhline(1.0, color="#888", ls=":", alpha=0.7)
    ax.text(1.0, 1.002, "baseline (no LM)", ha="right", va="bottom", fontsize=8, color="#666")
    ax.set_xlabel("LM fusion weight  lambda")
    ax.set_ylabel("WER relative to lambda=0")
    ax.set_title("Strong-LM dev gain generalizes to test; weak-LM does not")
    ax.scatter([], [], s=120, marker="*", color="gray", edgecolor="black",
               label="test outcome (dev-tuned lambda)")
    ax.legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "fig4_phaseB.png"); fig.savefig(OUT / "fig4_phaseB.pdf"); plt.close(fig)


def fig_phaseB_categories():
    s = json.loads((RES / "l2arctic" / "phaseB_category_delta.json").read_text())
    fixed = s["fixed_by_category"]
    reg = s["regressed_by_category"]
    cats = sorted(set(fixed) | set(reg), key=lambda c: -(fixed.get(c, 0)))
    pos = [fixed.get(c, 0) for c in cats]
    neg = [-reg.get(c, 0) for c in cats]
    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x, pos, 0.6, color="#2e7d32", label="errors fixed")
    ax.bar(x, neg, 0.6, color="#c62828", label="errors introduced")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([c.replace("_", "\n") for c in cats])
    ax.set_ylabel("net errors removed (L2-ARCTIC)")
    ax.set_title("What audio-free rescoring fixes, by error category")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig5_phaseB_categories.png"); fig.savefig(OUT / "fig5_phaseB_categories.pdf"); plt.close(fig)


def fig_model_robustness():
    """GPT-2 relative WER reduction across ASR model sizes (Gap 2)."""
    mr = RES / "model_robustness"
    if not (mr / "summary_tiny.json").exists():
        return
    rows = []
    for m, label in [("tiny", "tiny.en\n(39M)"), ("base", "base.en\n(74M)")]:
        s = json.loads((mr / f"summary_{m}.json").read_text())
        rel = 100 * s["test_wer_delta"] / s["test_wer_baseline"]
        rows.append((label, rel))
    # small.en on the SAME 6-speaker set as tiny/base (comparable across models)
    s = json.loads((RES / "l2arctic" / "phaseB_summary_gpt2.json").read_text())
    rows.append(("small.en\n(244M)", 100 * s["test_wer_delta"] / s["test_wer_baseline"]))
    labels, rels = zip(*rows)
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.bar(labels, rels, color="#2e7d32", width=0.62)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_ylabel("Relative WER change from GPT-2 (%)")
    ax.set_title("GPT-2 gain holds across ASR model size", fontsize=12)
    ax.set_ylim(min(rels) * 1.18, 1.5)
    for i, r in enumerate(rels):
        ax.text(i, r / 2, f"{r:.1f}%", ha="center", va="center", color="white",
                fontsize=11, weight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig5_model_robustness.png", bbox_inches="tight"); fig.savefig(OUT / "fig5_model_robustness.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_pipeline(); fig_latency_wer(); fig_error_categories(); fig_phaseB()
    fig_model_robustness()
    print("wrote figures to", OUT)
    for p in sorted(OUT.glob("*.png")):
        print("  ", p.name)
