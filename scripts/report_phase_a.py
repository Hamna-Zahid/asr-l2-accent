#!/usr/bin/env python
"""Phase A.4 — assemble the Phase A summary report (markdown + plots).

Reads the streaming-harness results table (and the error CSV if present) and
writes:
  results/<dataset>/phaseA_report.md
  results/<dataset>/phaseA_latency_wer_tradeoff.png
  results/<dataset>/phaseA_error_mix.png

All error-type proportions are labelled PRELIMINARY — they are pre-manual-
annotation (auto S/D/I only, not yet the accent/disfluency/VAD/hallucination
breakdown, which requires your manual pass on the error CSV).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asr_l2.config import load_config                       # noqa: E402
from asr_l2.report.plots import (plot_latency_wer_tradeoff,  # noqa: E402
                                 plot_error_mix)


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(
            f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])
            for c in cols) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    rdir = Path(args.results_dir) if args.results_dir else \
        Path(cfg["paths"]["results_root"]) / args.dataset

    table = rdir / "phaseA_streaming_results.parquet"
    if not table.exists():
        print(f"ERROR: results table not found: {table}", file=sys.stderr)
        return 2
    df = pd.read_parquet(table)

    # Condition-level summary: per-utterance mean WER/CER + corpus-style stats.
    summ = (df.groupby("condition")
              .agg(n=("utt_id", "count"),
                   mean_wer=("wer", "mean"), mean_cer=("cer", "mean"),
                   mean_rtf=("rtf", "mean"), mean_proc_s=("proc_time_s", "mean"),
                   alat_ms=("algorithmic_latency_ms", "max"))
              .reset_index()
              .sort_values("alat_ms", na_position="first"))

    p1 = plot_latency_wer_tradeoff(df, rdir)
    p2 = plot_error_mix(df, rdir)

    # Preliminary error-type proportions (auto S/D/I, summed).
    totals = df[["substitutions", "deletions", "insertions"]].sum()
    grand = float(totals.sum()) or 1.0
    prop_lines = [f"- **{k}**: {int(v)} ({v / grand:.1%})"
                  for k, v in totals.items()]

    # Optional: pull op breakdown from the error CSV if it exists.
    err_csv = rdir / "phaseA_error_annotation_offline.csv"
    err_note = ""
    if err_csv.exists():
        e = pd.read_csv(err_csv)
        if "why_category" in e.columns:
            annotated = e["why_category"].astype(str).str.strip().replace("nan", "")
            n_annotated = int((annotated != "").sum())
            err_note = (f"\nError CSV: `{err_csv.name}` — {len(e)} rows, "
                        f"{n_annotated} manually annotated with a why-category "
                        f"({n_annotated / len(e):.0%} done).\n")

    md = f"""# Phase A report — {args.dataset}

> **PRELIMINARY.** Error-type proportions below are automatic substitution/
> deletion/insertion counts only. The accent vs disfluency vs VAD-cutoff vs
> hallucination breakdown requires the manual annotation pass on the error CSV.

## Hardware / run config
- Model: `{cfg['asr']['model']}`  | compute: `{cfg['asr']['compute_type']}` | device: `{cfg['asr']['device']}`
- Chunk sizes (ms): {cfg['streaming']['chunk_sizes_ms']}  | left context: {cfg['streaming']['left_context_ms']} ms
- Utterances/condition cap: {cfg['eval']['max_utterances']}

## Per-condition WER / CER / latency
(`mean_wer`/`mean_cer` are means of per-utterance scores; `mean_rtf` is mean
real-time factor; `alat_ms` is the algorithmic latency = chunk size.)

{_md_table(summ.round(4))}

## Latency vs WER trade-off
![latency vs WER]({p1.name})

## Error-type mix (PRELIMINARY, auto S/D/I)
![error mix]({p2.name})

Totals across all conditions:
{chr(10).join(prop_lines)}
{err_note}
## What is automated vs. manual here
- **Automated:** all numbers and plots above (S/D/I counts, WER/CER, latency).
- **Manual (yours):** open the error CSV and fill `why_category` per error.
  Phase B's category-level deltas only become meaningful after that pass.
"""
    out = rdir / "phaseA_report.md"
    out.write_text(md, encoding="utf-8")
    print(">> DONE")
    print(f"   report: {out}")
    print(f"   plots:  {p1}\n           {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
