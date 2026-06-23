#!/usr/bin/env python
"""Paired bootstrap significance test for the Phase B WER deltas.

Reviewers will ask whether a -15.5% relative WER change on ~99 clips is real.
We resample the test utterances with replacement (paired: the same resample is
scored for baseline and adapted), recompute corpus WER each time, and report the
delta distribution: mean, 95% CI, and a one-sided p-value (the fraction of
resamples in which adaptation did NOT help, i.e. delta >= 0).

Corpus WER per resample = sum(edits)/sum(ref_words), so resampling is fast.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asr_l2.scoring.metrics import score_pair      # noqa: E402


def _per_utt(rescore_csv: Path):
    """Return list of (ref_words, base_edits, adapt_edits, speaker) per utterance.

    Speaker is parsed from L2-ARCTIC utt_ids of the form ``l2arctic_<SPK>_...``;
    unknown formats get speaker=utt_id (so by-speaker == by-utterance there).
    """
    out = []
    for r in csv.DictReader(open(rescore_csv, encoding="utf-8")):
        b = score_pair(r["ref"], r["baseline_hyp"])
        a = score_pair(r["ref"], r["adapted_hyp"])
        n = b["n_ref_words"]
        if n == 0:
            continue
        be = b["substitutions"] + b["deletions"] + b["insertions"]
        ae = a["substitutions"] + a["deletions"] + a["insertions"]
        uid = r.get("utt_id", "")
        parts = uid.split("_")
        spk = parts[1] if (len(parts) >= 3 and parts[0] == "l2arctic") else uid
        out.append((n, be, ae, spk))
    return out


def _corpus_wer(sample):
    nref = sum(x[0] for x in sample)
    return (sum(x[1] for x in sample) / nref, sum(x[2] for x in sample) / nref) \
        if nref else (None, None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--by-speaker", action="store_true",
                    help="cluster bootstrap: resample SPEAKERS not utterances "
                         "(accounts for within-speaker correlation)")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    rdir = repo / "results" / args.dataset
    data = _per_utt(rdir / "phaseB_rescore.csv")
    n = len(data)
    base_wer, adapt_wer = _corpus_wer(data)
    obs_delta = adapt_wer - base_wer

    rng = random.Random(args.seed)
    deltas, base_s, adapt_s = [], [], []
    if args.by_speaker:
        # cluster bootstrap: group clips by speaker, resample speaker clusters
        from collections import defaultdict
        clusters = defaultdict(list)
        for row in data:
            clusters[row[3]].append(row)
        groups = list(clusters.values())
        ng = len(groups)
        for _ in range(args.n_boot):
            sample = []
            for _ in range(ng):
                sample.extend(groups[rng.randrange(ng)])
            b, a = _corpus_wer(sample)
            base_s.append(b); adapt_s.append(a); deltas.append(a - b)
    else:
        for _ in range(args.n_boot):
            sample = [data[rng.randrange(n)] for _ in range(n)]
            b, a = _corpus_wer(sample)
            base_s.append(b); adapt_s.append(a); deltas.append(a - b)
    deltas.sort()
    lo, hi = deltas[int(0.025 * args.n_boot)], deltas[int(0.975 * args.n_boot)]
    p_not_help = sum(1 for d in deltas if d >= 0) / args.n_boot

    def ci(v):
        v = sorted(v); return [round(v[int(0.025 * len(v))], 4), round(v[int(0.975 * len(v))], 4)]

    summary = {
        "dataset": args.dataset, "n_test": n, "n_boot": args.n_boot,
        "resample_unit": "speaker" if args.by_speaker else "utterance",
        "n_speakers": len({row[3] for row in data}),
        "wer_baseline": round(base_wer, 4), "wer_baseline_ci95": ci(base_s),
        "wer_adapted": round(adapt_wer, 4), "wer_adapted_ci95": ci(adapt_s),
        "delta_abs": round(obs_delta, 4), "delta_abs_ci95": [round(lo, 4), round(hi, 4)],
        "delta_rel_pct": round(100 * obs_delta / base_wer, 2) if base_wer else None,
        "p_value_one_sided": round(p_not_help, 4),
        "significant_at_0.05": bool(p_not_help < 0.05 and hi < 0),
    }
    fname = "phaseB_significance_byspeaker.json" if args.by_speaker \
        else "phaseB_significance.json"
    (rdir / fname).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(">> DONE", args.dataset)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
