#!/usr/bin/env python
"""Decode scored n-best for an eval set with a chosen ASR family, and lay it out
so the existing Phase B scripts (rescore_nbest / oracle_bound / lm_ladder) run on
it unchanged.

Writes:
  results/<out>/phaseB_nbest.jsonl      (utt_id + hyps=[[text, acoustic_score]])
  data/processed/<out>/manifest.jsonl   (copy of the source manifest)

so that, e.g.:
  python scripts/decode_nbest.py --dataset l2arctic_24spk --engine wav2vec2 \
         --out l2arctic_24spk_w2v2
  python scripts/rescore_nbest.py --dataset l2arctic_24spk_w2v2 --lm neural:gpt2
  python scripts/oracle_bound.py  --dataset l2arctic_24spk_w2v2
  python scripts/lm_ladder.py     --dataset l2arctic_24spk_w2v2

GPU-friendly (wav2vec2 in fp16); resumable (skips clips already in the cache).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asr_l2.config import load_config                       # noqa: E402
from asr_l2.io.manifest import read_manifest                 # noqa: E402
from asr_l2.io.audio import load_audio                       # noqa: E402


def make_engine(engine: str, model: str | None, device: str):
    if engine == "wav2vec2":
        from asr_l2.asr.wav2vec2_nbest import Wav2Vec2NBestEngine, DEFAULT_MODEL
        return Wav2Vec2NBestEngine(model=model or DEFAULT_MODEL,
                                   device=None if device == "auto" else device)
    if engine == "whisper":
        from asr_l2.asr.nbest import NBestEngine
        return NBestEngine(model=model or "small.en", device=device)
    raise ValueError(f"unknown engine: {engine}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="source dataset (for the manifest)")
    ap.add_argument("--engine", choices=["wav2vec2", "whisper"], default="wav2vec2")
    ap.add_argument("--out", required=True, help="output dataset name (results + data dir)")
    ap.add_argument("--model", default=None, help="override model id")
    ap.add_argument("--nbest", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-utterances", type=int, default=None)
    ap.add_argument("--min-duration", type=float, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_root = Path(cfg["paths"]["data_root"])
    res_root = Path(cfg["paths"]["results_root"])
    src_manifest = data_root / "processed" / args.dataset / "manifest.jsonl"

    # lay out the output dataset: copy the manifest so downstream --dataset <out> works
    out_data = data_root / "processed" / args.out
    out_data.mkdir(parents=True, exist_ok=True)
    if not (out_data / "manifest.jsonl").exists():
        shutil.copy2(src_manifest, out_data / "manifest.jsonl")
    out_res = res_root / args.out
    out_res.mkdir(parents=True, exist_ok=True)
    cache_path = out_res / "phaseB_nbest.jsonl"

    # resumable: load any clips already decoded
    done = set()
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as fh:
            done = {json.loads(l)["utt_id"] for l in fh}

    utts = list(read_manifest(src_manifest))
    min_dur = args.min_duration if args.min_duration is not None else cfg["eval"].get("min_duration_s", 0.0)
    if min_dur:
        utts = [u for u in utts if u.duration_s >= min_dur]
    if args.max_utterances:
        utts = utts[:args.max_utterances]
    todo = [u for u in utts if u.utt_id not in done]
    print(f">> {args.engine}: {len(todo)} to decode ({len(done)} cached) -> {cache_path}")
    if not todo:
        print(">> nothing to do"); return 0

    eng = make_engine(args.engine, args.model, args.device)
    print(f">> engine {eng.name} on {eng.device}")
    fh = open(cache_path, "a", encoding="utf-8")
    for i, u in enumerate(todo):
        hyps = eng.nbest(load_audio(u.wav_path), n=args.nbest)
        rec = [[h.text, h.acoustic_score] for h in hyps]
        fh.write(json.dumps({"utt_id": u.utt_id, "hyps": rec}) + "\n")
        fh.flush()
        if (i + 1) % 25 == 0:
            print(f"   {i+1}/{len(todo)}", flush=True)
    fh.close()
    print(">> done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
