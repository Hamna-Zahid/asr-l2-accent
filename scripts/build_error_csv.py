#!/usr/bin/env python
"""Phase A.2 — build the human-annotatable error CSV (offline condition).

Consumes the offline ASR word-timing sidecar written by the streaming harness
(`phaseA_offline_words.jsonl`) plus the dataset manifest, aligns ref vs hyp,
auto-tags every substitution / deletion / insertion with local context and an
approximate audio timestamp, and writes a CSV SORTED so the most frequent
error patterns surface first.

The output CSV has blank `why_category` and `annotator_notes` columns — THIS IS
THE FILE YOU ANNOTATE BY HAND. The auto pass never fills in *why* an error
occurred (accent vs disfluency vs VAD cutoff vs hallucination); see
`asr_l2.errors.tagging.WHY_CATEGORIES` for the controlled vocabulary.

Example:
  python scripts/build_error_csv.py --dataset svarah
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asr_l2.config import load_config                  # noqa: E402
from asr_l2.asr.engine import Word                     # noqa: E402
from asr_l2.io.manifest import read_manifest           # noqa: E402
from asr_l2.errors.tagging import tag_utterance, WHY_CATEGORIES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--condition", default="offline",
                    help="which condition's sidecar to tag (default: offline)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_root = Path(cfg["paths"]["data_root"])
    results_dir = Path(args.results_dir) if args.results_dir else \
        Path(cfg["paths"]["results_root"]) / args.dataset

    manifest_path = data_root / "processed" / args.dataset / "manifest.jsonl"
    sidecar = results_dir / f"phaseA_{args.condition}_words.jsonl"
    if not sidecar.exists():
        print(f"ERROR: word-timing sidecar not found: {sidecar}\n"
              f"       Run scripts/run_streaming_harness.py first.", file=sys.stderr)
        return 2

    meta = {u.utt_id: u for u in read_manifest(manifest_path)}

    # Load sidecar: utt_id -> (hyp_text, [Word, ...])
    hyp_by_utt: dict[str, tuple[str, list[Word]]] = {}
    with open(sidecar, "r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            words = [Word(w["w"], w["start"], w["end"], w.get("prob", 0.0))
                     for w in d.get("words", [])]
            hyp_by_utt[d["utt_id"]] = (d.get("hyp_text", ""), words)

    rows = []
    for utt_id, (hyp_text, words) in hyp_by_utt.items():
        u = meta.get(utt_id)
        if u is None:
            continue
        for er in tag_utterance(utt_id, args.dataset, args.condition,
                                u.text, hyp_text, words,
                                speaker=u.speaker, accent=u.accent):
            rows.append(asdict(er))

    if not rows:
        print("WARNING: no errors found (or no matching utterances).", file=sys.stderr)

    df = pd.DataFrame(rows)
    # Frequency of each (op, ref_word -> hyp_word) pattern, to sort by.
    if not df.empty:
        pair = df.apply(lambda r: (r["op"], r["ref_word"], r["hyp_word"]), axis=1)
        freq = Counter(pair)
        df["pattern_freq"] = pair.map(freq)
        df = df.sort_values(["pattern_freq", "op", "ref_word"],
                            ascending=[False, True, True]).reset_index(drop=True)
        # Drop the dict-typed 'extra' column for a clean spreadsheet.
        if "extra" in df.columns:
            df = df.drop(columns=["extra"])

    out_dir = results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"phaseA_error_annotation_{args.condition}.csv"
    df.to_csv(out_csv, index=False)

    # A tiny legend file so the annotator knows the allowed why_category values.
    (out_dir / "phaseA_why_categories.txt").write_text(
        "Allowed why_category values (fill in the error CSV by hand):\n  - "
        + "\n  - ".join(WHY_CATEGORIES) + "\n", encoding="utf-8")

    n_err = len(df)
    print(">> DONE")
    print(f"   error CSV: {out_csv}  ({n_err} error rows)")
    print(f"   legend:    {out_dir / 'phaseA_why_categories.txt'}")
    if n_err:
        by_op = df["op"].value_counts().to_dict()
        print(f"   breakdown by op (PRELIMINARY, pre-manual): {by_op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
