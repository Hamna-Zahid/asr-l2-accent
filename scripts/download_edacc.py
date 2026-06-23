#!/usr/bin/env python
"""Download & prepare EdAcc (Edinburgh International Accents of English) -- a
second SPONTANEOUS accented-English corpus, used to test whether the
"spontaneous speech defeats audio-free rescoring" finding (seen on Svarah)
replicates independently.

EdAcc is public (CC-BY-SA) on Hugging Face: edinburghcstr/edacc -- no token
needed. It is 40 h of dyadic conversations across many L1 accents.

Converts to the repo's unified format (like download_svarah.py):
  data/processed/edacc/wav/<utt_id>.wav   (16 kHz mono)
  data/processed/edacc/manifest.jsonl
Column names are auto-detected from the actual HF schema (printed for transparency).

Example:
  python scripts/download_edacc.py --split validation --max-utterances 200
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asr_l2.config import load_config, load_dotenv          # noqa: E402
from asr_l2.io.audio import TARGET_SR, save_wav             # noqa: E402
from asr_l2.io.manifest import Utterance, write_manifest    # noqa: E402

load_dotenv()
HF_ID = "edinburghcstr/edacc"
TEXT_KEYS = ["text", "transcript", "transcription", "sentence", "normalized_text"]
ACCENT_KEYS = ["l1", "first_language", "native_language", "accent", "mother_tongue", "language"]
SPEAKER_KEYS = ["speaker", "speaker_id", "spk_id", "spkid"]


def _first(keys, available):
    return next((k for k in keys if k in available), None)


def _decode_audio(field):
    data, path = field.get("bytes"), field.get("path")
    try:
        arr, sr = sf.read(io.BytesIO(data) if data is not None else path,
                          dtype="float32", always_2d=False)
    except Exception:
        try:
            arr, sr = librosa.load(path or io.BytesIO(data), sr=None, mono=True)
        except Exception:
            return None
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != TARGET_SR:
        arr = librosa.resample(arr.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    return arr.astype(np.float32)


def _heavy(v):
    return isinstance(v, (bytes, bytearray)) or (isinstance(v, (list, dict)) and len(v) > 32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--hf-id", default=HF_ID)
    ap.add_argument("--split", default="validation", help="EdAcc has validation/test")
    ap.add_argument("--max-utterances", type=int, default=200)
    ap.add_argument("--min-seconds", type=float, default=1.0,
                    help="drop clips shorter than this (sub-1s segments)")
    ap.add_argument("--max-seconds", type=float, default=30.0,
                    help="drop very long conversation segments (>30s)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out) if args.out else Path(cfg["paths"]["data_root"]) / "processed" / "edacc"
    wav_dir = out_dir / "wav"; wav_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset, Audio
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr); return 2

    print(f">> loading {args.hf_id} split={args.split}")
    ds = load_dataset(args.hf_id, split=args.split)
    ds_arrow = ds.with_format("arrow")
    available = list(ds.features.keys())
    print(">> schema:", available)
    text_key = _first(TEXT_KEYS, available)
    accent_key = _first(ACCENT_KEYS, available)
    speaker_key = _first(SPEAKER_KEYS, available)
    audio_key = next((k for k, v in ds.features.items() if isinstance(v, Audio)), None) \
        or _first(["audio", "audio_filepath"], available)
    if text_key is None or audio_key is None:
        print(f"ERROR: couldn't map text/audio in {available}", file=sys.stderr); return 3
    print(f">> mapped text='{text_key}' accent='{accent_key}' speaker='{speaker_key}' audio='{audio_key}'")

    total = len(ds)
    idx = list(range(total))
    rng = random.Random(cfg.get("seed", 1234)); rng.shuffle(idx)

    utts, dur_tot, kept, skipped = [], 0.0, 0, 0
    for i in idx:
        if kept >= args.max_utterances:
            break
        ex = ds_arrow[i].to_pylist()[0]
        text = (ex.get(text_key) or "").strip()
        if not text:
            skipped += 1; continue
        arr = _decode_audio(ex[audio_key])
        if arr is None:
            skipped += 1; continue
        dur = len(arr) / TARGET_SR
        if dur < args.min_seconds or dur > args.max_seconds:
            skipped += 1; continue
        uid = f"edacc_{i:06d}"
        wp = wav_dir / f"{uid}.wav"; save_wav(wp, arr, TARGET_SR)
        dur_tot += dur; kept += 1
        utts.append(Utterance(utt_id=uid, dataset="edacc", wav_path=str(wp), text=text,
                              duration_s=round(dur, 3),
                              speaker=str(ex.get(speaker_key)) if speaker_key else None,
                              accent=str(ex.get(accent_key)) if accent_key else None,
                              extra={k: ex[k] for k in available
                                     if k not in {audio_key, text_key} and not _heavy(ex[k])}))
        if kept % 25 == 0:
            print(f"   kept {kept}/{args.max_utterances}")

    written = write_manifest(out_dir / "manifest.jsonl", utts)
    report = {"dataset": "edacc", "hf_id": args.hf_id, "split": args.split,
              "utterances_written": written, "skipped": skipped,
              "total_duration_hours": round(dur_tot / 3600, 3), "schema_seen": available,
              "column_mapping": {"text": text_key, "accent": accent_key, "speaker": speaker_key}}
    (out_dir / "integrity.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    print(">> DONE"); print(json.dumps(report, indent=2))
    if written == 0:
        print("WARNING: 0 written -- check the schema mapping above.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
