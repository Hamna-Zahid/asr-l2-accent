# Manual error annotation methodology (Phase A.2)

The pipeline auto-detects **what/where** each ASR error is (substitution /
deletion / insertion, with context and an approximate timestamp). It does **not**
guess **why**. Assigning the *why* is the manual step that turns the raw error
log into the paper's error taxonomy. This file documents the process so the
methodology is reproducible even though the filled-in annotations are private.

## The artifact

`results/<dataset>/phaseA_error_annotation_offline.csv`, produced by
`scripts/build_error_csv.py`. One row per error, sorted so the most frequent
error *patterns* appear first (annotate high-impact patterns earliest).

### Columns (auto-filled)
| column | meaning |
|---|---|
| `utt_id`, `dataset`, `condition` | which utterance / decoding condition |
| `op` | `substitute` \| `delete` \| `insert` |
| `ref_word` | reference word (empty for insertions) |
| `hyp_word` | ASR word (empty for deletions) |
| `ref_context` | reference words around the error, target in `[brackets]` |
| `approx_time_s` | approximate audio time of the error (to go listen) |
| `speaker`, `accent` | metadata if available |
| `pattern_freq` | how often this exact (op, ref→hyp) pattern occurs |

### Columns (YOU fill in)
| column | how to fill |
|---|---|
| `why_category` | one value from the controlled vocabulary below |
| `annotator_notes` | free text: anything notable (optional) |

## Controlled vocabulary for `why_category`

(Also written to `results/<dataset>/phaseA_why_categories.txt` at build time.)

- **accent_phoneme** — error driven by an accent/L2 phoneme realization
  (e.g. /v/–/w/, /θ/→/t/). On L2-Arctic, cross-check against the A.3 join.
- **disfluency** — caused by a filler, repetition, false start, or
  self-correction in the *speech* (not an ASR fault per se).
- **vad_cutoff** — a word lost/garbled at a VAD or streaming-chunk boundary.
  Most relevant in the streaming conditions.
- **hallucination** — ASR produced text with no acoustic basis (classic on
  silence/noise; insertions are prime suspects).
- **reference_error** — the ground-truth transcript itself is wrong; the ASR
  was actually right. Important to separate out — don't penalize the model.
- **normalization** — spurious error created by text normalization
  (e.g. "okay" vs "ok", numbers) rather than a real recognition mistake.
- **other** — none of the above; explain in `annotator_notes`.

## Suggested workflow

1. Open the CSV in a spreadsheet. Work top-down (most frequent patterns first).
2. For each row, read `ref_context`; if unsure, open the utterance WAV
   (`data/processed/<dataset>/wav/<utt_id>.wav`) and listen near `approx_time_s`.
3. Pick one `why_category`. For repeated identical patterns you can fill the
   first and copy down — but spot-check that context truly matches.
4. For L2-Arctic, open `results/l2arctic/phaseA3_phoneme_join.csv` alongside:
   `phoneme_classification = coincides_mispronunciation` is strong evidence for
   `accent_phoneme`; `asr_error_on_correct_speech` argues against it.
5. Save as a NEW file (e.g. `..._offline.annotated.csv`) so re-running the
   builder never overwrites your work.

## Stopping rule / coverage

You don't need to annotate every row. A defensible approach: annotate all rows
until cumulative `pattern_freq` covers ~80% of total errors, plus a random
sample of the long tail. Record what fraction you annotated — the Phase A
report prints this automatically once the annotated file is present.

## What this enables

Phase B compares before/after adaptation **per error category**. That breakdown
is only meaningful once `why_category` is filled — otherwise we can only report
raw S/D/I counts, not whether adaptation helps accent errors vs disfluencies.
