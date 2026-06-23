# Datasets, licenses, and how to obtain them

This repository contains **no audio data**. It ships only code, download
scripts, and derived metrics/plots. Every dataset below must be fetched from
its official source by the provided script, which converts it into the
repo's unified internal format (16 kHz mono WAV + JSONL manifest) under
`data/` (which is git-ignored).

> ⚠️ **Before redistributing anything**, re-read each dataset's current license
> at its official source. License terms change; the summaries below are a
> starting point, not legal advice. Where a license is marked "VERIFY", we were
> not able to assert the exact terms with certainty and you should confirm.

---

## 1. Svarah (AI4Bharat) — PRIMARY evaluation set

- **What:** ~9.6 hours of transcribed Indian-accented English read/spontaneous
  speech across many L1 backgrounds.
- **Role here:** primary evaluation set for **both** Phase A and Phase B.
- **Source:** AI4Bharat (GitHub / Hugging Face). See `scripts/download_svarah.py`
  for the exact URL used.
- **License:** Open, released by AI4Bharat. **VERIFY** the exact license
  (commonly CC-BY-4.0) at the source before redistributing derived audio.
- **Redistribution in this repo:** none. Audio is downloaded at runtime;
  only WER/CER/derived metrics are committed under `results/`.
- **Access is GATED:** you must accept the agreement on the HF page and supply
  a token. Put `HF_TOKEN=hf_...` in a repo-root `.env` (git-ignored) or pass
  `--token`. See `scripts/download_svarah.py`.
- **Observed schema (HF `test` split, 6656 utts):** columns `audio_filepath`
  (the Audio-typed column), `text`, `duration`, `gender`, `age-group`,
  `primary_language` (used as the accent/L1 label), `native_place_state/
  district`, `highest_qualification`, `job_category`, `occupation_domain`.
  **No speaker_id** is exposed in the parquet (per-speaker analysis would need
  the separate `meta_speaker_stats.csv` from the GitHub repo). ~19% of clips
  are <1s single-word command/digit prompts ("Up", "Stop", "Four") — use
  `eval.min_duration_s` to exclude them if desired.
- **Windows note:** `datasets>=5` decodes audio via `torchcodec`, which is
  unreliable on Windows; our script reads rows in Arrow format and decodes the
  raw bytes with `soundfile` instead. No torchcodec needed.

## 2. L2-Arctic — error-taxonomy reference (Phase A.3)

- **What:** 24 non-native English speakers (6 L1s), with **manual phoneme-level
  mispronunciation annotations** — the ground truth we cross-reference against
  ASR substitution errors.
- **Role here:** Phase A.3 phoneme cross-reference; text-only transcripts may
  feed the Phase B LM corpus (see caveat below).
- **Source:** Texas A&M SPL — https://psi.engr.tamu.edu/l2-arctic-corpus/
  (requires accepting a license / registration; not anonymously downloadable).
- **License:** Free for **academic / non-commercial research**; redistribution
  of the audio is restricted. **VERIFY** current terms at the source.
- **Redistribution in this repo:** none. The download script expects you to
  obtain the archive after accepting the license, then points it at your local
  copy.
- ⚠️ **Phase B text caveat:** extracting transcripts as LM training text is a
  derivative use. We keep any such derived text **out of the public repo** and
  out of `results/` unless you confirm the license permits redistributing
  derived text. The corpus-builder script can use it **locally** regardless.

## 3. Speak & Improve Corpus 2025 (Cambridge / ELiT) — OPTIONAL secondary

- **What:** ~315 hours of CEFR-graded L2 English speaking-test responses.
- **Role here:** OPTIONAL, secondary — CEFR-stratified analysis only, if
  time/compute allow. Disabled by default in `config/default.yaml`.
- **Source:** https://www.speakandimprove.com/ / official corpus release
  (requires registration and license acceptance).
- **License:** **Non-commercial use only.** **VERIFY** at the source.
- ⚠️ **DO NOT bundle raw audio from this corpus in any public repo.** The
  download script fetches it at runtime from the official source after you
  accept the license; nothing from it is committed here.

---

## Public-redistribution risk flags (read before releasing your repo)

| Item | Safe to commit publicly? | Notes |
|---|---|---|
| Pipeline code, scripts | ✅ Yes | MIT (see `LICENSE`) |
| WER/CER/latency tables (`results/`) | ✅ Likely | Derived metrics, not audio. Confirm no dataset forbids publishing derived stats. |
| Plots / reports | ✅ Likely | Same as above. |
| Auto-generated error CSV (with ref/hyp **text snippets**) | ⚠️ Check | Contains short transcript excerpts. Fine for Svarah if CC-BY w/ attribution; **avoid** committing L2-Arctic / Speak & Improve text excerpts unless their license allows. |
| Any `.wav` / raw audio | ❌ Never | Git-ignored by default. |
| LM training text corpus | ⚠️ Check | May contain licensed transcript text — keep local unless cleared. |

**If in doubt, keep the artifact local and commit only aggregate numbers.**
