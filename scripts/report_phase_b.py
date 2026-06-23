#!/usr/bin/env python
"""Phase B.5 — assemble the Phase B summary report (markdown + plots).

Reads results/<dataset>/phaseB_summary.json (from rescore_nbest.py) and writes:
  results/<dataset>/phaseB_report.md
  results/<dataset>/phaseB_lambda_curve.png
  results/<dataset>/phaseB_by_accent.png

The category-level before/after breakdown (accent vs disfluency vs ...) needs
your manual annotation pass joined in; this report covers overall + accent-
stratified deltas, clearly flagged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asr_l2.config import load_config                          # noqa: E402
from asr_l2.report.plots import (plot_lambda_curve,            # noqa: E402
                                 plot_baseline_vs_adapted)


def _fmt(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) and x is not None else str(x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    rdir = Path(cfg["paths"]["results_root"]) / args.dataset
    summ_path = rdir / "phaseB_summary.json"
    if not summ_path.exists():
        print(f"ERROR: {summ_path} not found. Run rescore_nbest.py first.",
              file=sys.stderr)
        return 2
    s = json.loads(summ_path.read_text(encoding="utf-8"))

    p1 = plot_lambda_curve(s["dev_wer_by_lambda"], s["best_lambda"], rdir, args.dataset)
    p2 = plot_baseline_vs_adapted(s["stratified_by_accent"], rdir, args.dataset)

    base, adapt = s["test_wer_baseline"], s["test_wer_adapted"]
    delta = s["test_wer_delta"]
    rel = (delta / base * 100) if (delta is not None and base) else None
    verdict = ("**helps**" if (delta is not None and delta < 0) else
               "**no improvement**" if (delta is not None and delta >= 0) else "n/a")

    strat_rows = "\n".join(
        f"| {r['accent']} | {r['n']} | {_fmt(r['wer_baseline'])} | "
        f"{_fmt(r['wer_adapted'])} | {_fmt(r['delta'])} |"
        for r in s["stratified_by_accent"])

    md = f"""# Phase B report - {args.dataset}

Audio-free domain adaptation via n-best rescoring on the **frozen** Whisper
decoder (no audio fine-tuning). LM: `{s['lm']}`, n-best={s['nbest']}.

## Setup
- Eval clips: {s['n_eval']}  (dev {s['n_dev']} for lambda tuning / test {s['n_test']} for reporting)
- lambda grid: {s['lambda_grid']}  ->  **best lambda = {s['best_lambda']}** (chosen on dev)
- Hypotheses changed by rescoring on test: {s['n_changed']}

## Overall result (TEST split) - adaptation {verdict}
| | WER |
|---|---|
| baseline (Whisper top-1) | {_fmt(base)} |
| adapted (LM rescored) | {_fmt(adapt)} |
| absolute delta | {_fmt(delta)} |
| relative | {f'{rel:.1f}%' if rel is not None else 'n/a'} |

![lambda curve]({p1.name})

## Stratified by accent / L1 (honest sub-group check)
Negative delta = improvement; positive = hurt. Watch for adaptation that helps
some groups and hurts others.

| accent | n | WER baseline | WER adapted | delta |
|---|---|---|---|---|
{strat_rows}

![baseline vs adapted by accent]({p2.name})

## Pending: category-level delta
The before/after split by ERROR CATEGORY (accent_phoneme / disfluency / vad /
hallucination) requires your manual annotation pass (ANNOTATION.md) joined with
these results. Once the error CSV is annotated, that breakdown can be added -
it's the key question of *which kind* of error the LM fixes.
"""
    out = rdir / "phaseB_report.md"
    out.write_text(md, encoding="utf-8")
    print(">> DONE")
    print(f"   report: {out}")
    print(f"   plots:  {p1}  {p2}")
    print(f"   overall: baseline {_fmt(base)} -> adapted {_fmt(adapt)} "
          f"(delta {_fmt(delta)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
