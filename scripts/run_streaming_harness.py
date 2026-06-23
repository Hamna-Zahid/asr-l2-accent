#!/usr/bin/env python
"""Phase A.1 — streaming-mismatch harness (CLI).

Runs offline + chunked-streaming ASR over a prepared dataset manifest and writes
a per-utterance results table (Parquet + CSV) with WER/CER/latency for every
condition. A stranger can reproduce the whole table with one command.

Example:
  python scripts/run_streaming_harness.py --dataset svarah
  python scripts/run_streaming_harness.py --dataset svarah --max-utterances 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asr_l2.config import load_config                       # noqa: E402
from asr_l2.asr.engine import AsrEngine                     # noqa: E402
from asr_l2.io.manifest import read_manifest                # noqa: E402
from asr_l2.scoring.harness import build_conditions, run_one  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="e.g. svarah | l2arctic")
    ap.add_argument("--config", default=None)
    ap.add_argument("--manifest", default=None,
                    help="override manifest path (default: data/processed/<ds>/manifest.jsonl)")
    ap.add_argument("--max-utterances", type=int, default=None,
                    help="override eval.max_utterances (None = config value)")
    ap.add_argument("--min-duration", type=float, default=None,
                    help="drop utterances shorter than this (s); overrides config")
    ap.add_argument("--model", default=None, help="override asr.model")
    ap.add_argument("--chunk-sizes", default=None,
                    help="comma-separated chunk sizes ms, e.g. 320,640,1280 "
                         "(override config; cheap on GPU)")
    ap.add_argument("--out", default=None, help="override results dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_root = Path(cfg["paths"]["data_root"])
    results_root = Path(cfg["paths"]["results_root"])

    manifest = Path(args.manifest) if args.manifest else \
        data_root / "processed" / args.dataset / "manifest.jsonl"
    if not manifest.exists():
        print(f"ERROR: manifest not found: {manifest}\n"
              f"       Run the dataset download/prep script first.", file=sys.stderr)
        return 2

    max_utts = args.max_utterances
    if max_utts is None:
        max_utts = cfg["eval"]["max_utterances"]

    min_dur = args.min_duration
    if min_dur is None:
        min_dur = cfg["eval"].get("min_duration_s", 0.0)

    utts = list(read_manifest(manifest))
    n_before = len(utts)
    if min_dur and min_dur > 0:
        utts = [u for u in utts if u.duration_s >= min_dur]
        print(f">> min-duration filter {min_dur}s: kept {len(utts)}/{n_before}")
    if max_utts is not None:
        utts = utts[:max_utts]
    if not utts:
        print("ERROR: no utterances to process.", file=sys.stderr)
        return 2

    chunk_sizes = cfg["streaming"]["chunk_sizes_ms"]
    if args.chunk_sizes:
        chunk_sizes = [int(c) for c in args.chunk_sizes.split(",")]
    conds = build_conditions(chunk_sizes, cfg["streaming"]["left_context_ms"])
    model = args.model or cfg["asr"]["model"]
    print(f">> dataset={args.dataset} utts={len(utts)} model={model} "
          f"conditions={[c.name for c in conds]}")

    import json
    out_dir = Path(args.out) if args.out else results_root / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Checkpointing -----------------------------------------------------
    # Every completed (utt_id, condition) row is appended to this JSONL as it
    # finishes, so an interrupted run (e.g. the laptop sleeping) resumes instead
    # of restarting. Delete this file to force a clean re-run.
    ckpt_path = out_dir / "phaseA_checkpoint.jsonl"
    rows: list[dict] = []
    done_keys: set[tuple] = set()
    if ckpt_path.exists():
        with open(ckpt_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                rows.append(r)
                done_keys.add((r["utt_id"], r["condition"]))
        print(f">> resuming: {len(done_keys)} rows already done in {ckpt_path.name}")

    engine = None  # lazy: don't load the model if everything is already done
    t0 = time.perf_counter()
    total = len(utts) * len(conds)
    done = len(done_keys)
    ckpt_fh = open(ckpt_path, "a", encoding="utf-8")
    for cond in conds:
        for utt in utts:
            if (utt.utt_id, cond.name) in done_keys:
                continue
            if engine is None:
                engine = AsrEngine(
                    model=model, device=cfg["asr"]["device"],
                    compute_type=cfg["asr"]["compute_type"],
                    cpu_threads=cfg["asr"]["cpu_threads"],
                    language=cfg["asr"]["language"])
                print(f">> engine ready on device={engine.device}")
            row = run_one(engine, utt, cond, cfg["asr"]["beam_size"])
            rows.append(row)
            ckpt_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            ckpt_fh.flush()  # durable after every row
            done += 1
            el = time.perf_counter() - t0
            n_new = done - len(done_keys)
            print(f"   {done}/{total}  [{cond.name} {utt.utt_id}]  "
                  f"({el:.0f}s this session, "
                  f"{el / max(n_new, 1):.1f}s/row)", flush=True)
    ckpt_fh.close()

    # Persist per-condition word-timing sidecars (used by build_error_csv.py so
    # it never re-decodes), then drop the heavy column from the results table.
    # Write one word-timing sidecar per condition, keyed by condition name.
    by_cond: dict[str, list] = {}
    for r in rows:
        by_cond.setdefault(r["condition"], []).append(r)
    for cond_name, crows in by_cond.items():
        with open(out_dir / f"phaseA_{cond_name}_words.jsonl", "w",
                  encoding="utf-8") as fh:
            for r in crows:
                fh.write(json.dumps({"utt_id": r["utt_id"],
                                     "hyp_text": r["hyp_text"],
                                     "words": r["_hyp_words"]},
                                    ensure_ascii=False) + "\n")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_hyp_words"}
                       for r in rows])
    pq = out_dir / "phaseA_streaming_results.parquet"
    csv = out_dir / "phaseA_streaming_results.csv"
    df.to_parquet(pq, index=False)
    df.to_csv(csv, index=False)

    # Condition-level summary (corpus micro-WER would need re-aggregation; here we
    # report the mean of per-utterance WER plus mean latency for a quick read).
    summ = (df.groupby("condition")
              .agg(n=("utt_id", "count"),
                   mean_wer=("wer", "mean"),
                   mean_cer=("cer", "mean"),
                   mean_rtf=("rtf", "mean"),
                   mean_proc_s=("proc_time_s", "mean"))
              .reset_index())
    summ.to_csv(out_dir / "phaseA_condition_summary.csv", index=False)

    print(">> DONE")
    print(f"   per-utterance: {pq}")
    print(f"   per-utterance: {csv}")
    print(f"   summary:       {out_dir / 'phaseA_condition_summary.csv'}")
    print(summ.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
