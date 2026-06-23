"""Phase A plots: latency vs WER trade-off and per-condition error mix.

All functions take the per-utterance results DataFrame (from the streaming
harness) and write PNGs into a directory. Matplotlib only (no seaborn) to keep
the dependency surface small for the 8GB target.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / no display on the workstation
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["dataset", "condition", "mode", "algorithmic_latency_ms"],
                       dropna=False)
              .agg(mean_wer=("wer", "mean"), mean_cer=("cer", "mean"),
                   mean_rtf=("rtf", "mean"), mean_proc_s=("proc_time_s", "mean"),
                   n=("utt_id", "count"))
              .reset_index())


def plot_latency_wer_tradeoff(df: pd.DataFrame, out_dir: str | Path) -> Path:
    """Mean WER vs algorithmic latency (chunk size), per dataset.

    Offline (no streaming look-ahead) is drawn as a dashed horizontal baseline.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summ = _condition_summary(df)

    fig, ax = plt.subplots(figsize=(7, 5))
    for ds, g in summ.groupby("dataset"):
        offline = g[g["mode"] == "offline"]
        stream = g[g["mode"] == "streaming"].sort_values("algorithmic_latency_ms")
        if not stream.empty:
            ax.plot(stream["algorithmic_latency_ms"], stream["mean_wer"],
                    marker="o", label=f"{ds} (streaming)")
        if not offline.empty:
            ax.axhline(offline["mean_wer"].iloc[0], linestyle="--", alpha=0.6,
                       label=f"{ds} (offline baseline)")
    ax.set_xlabel("Algorithmic latency = chunk size (ms)")
    ax.set_ylabel("Mean per-utterance WER")
    ax.set_title("Streaming latency vs WER trade-off")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    path = out_dir / "phaseA_latency_wer_tradeoff.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_lambda_curve(dev_wer_by_lambda: dict, best_lambda: float,
                      out_dir: str | Path, dataset: str = "") -> Path:
    """Phase B: dev-set WER as a function of the fusion weight lambda."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lams = sorted(float(k) for k in dev_wer_by_lambda)
    wers = [dev_wer_by_lambda[str(l)] if str(l) in dev_wer_by_lambda
            else dev_wer_by_lambda[l] for l in lams]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(lams, wers, marker="o")
    ax.axvline(best_lambda, color="red", linestyle="--",
               label=f"best lambda={best_lambda}")
    ax.set_xlabel("LM fusion weight (lambda)")
    ax.set_ylabel("Dev-set WER")
    ax.set_title(f"Phase B lambda tuning{' - ' + dataset if dataset else ''}")
    ax.legend(); ax.grid(True, alpha=0.3)
    path = out_dir / "phaseB_lambda_curve.png"
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_baseline_vs_adapted(strat: list, out_dir: str | Path,
                             dataset: str = "") -> Path:
    """Phase B: baseline vs adapted WER per accent sub-group."""
    out_dir = Path(out_dir)
    rows = [s for s in strat if s.get("wer_baseline") is not None]
    labels = [s["accent"] for s in rows]
    base = [s["wer_baseline"] for s in rows]
    adapt = [s["wer_adapted"] for s in rows]
    import numpy as np
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), 4.5))
    ax.bar(x - 0.2, base, 0.4, label="baseline")
    ax.bar(x + 0.2, adapt, 0.4, label="adapted")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("WER"); ax.set_title(f"Baseline vs adapted by accent{' - ' + dataset if dataset else ''}")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    path = out_dir / "phaseB_by_accent.png"
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_error_mix(df: pd.DataFrame, out_dir: str | Path) -> Path:
    """Stacked S/D/I counts per condition (summed over utterances), per dataset."""
    out_dir = Path(out_dir)
    agg = (df.groupby(["dataset", "condition"])
             .agg(S=("substitutions", "sum"), D=("deletions", "sum"),
                  I=("insertions", "sum"))
             .reset_index())
    datasets = agg["dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.5),
                             squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        g = agg[agg["dataset"] == ds].set_index("condition")[["S", "D", "I"]]
        g.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title(f"{ds}: error type counts by condition")
        ax.set_ylabel("count")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    path = out_dir / "phaseA_error_mix.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
