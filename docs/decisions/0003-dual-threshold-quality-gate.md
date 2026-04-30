# ADR 0003 — Dual-threshold quality gate before classification

- Status: Accepted

## Context

Whisper + Gemini sometimes produce structurally-valid transcripts that are
nonetheless garbage (loops of phonetic gibberish, clipped audio, near-silence).
Pushing those into classification and sentiment yields confident-but-wrong
labels that pollute downstream analytics.

## Decision

After refinement we gate at the `verify` node using **two** scores:

1. `confidence_score` (Whisper, ≥ `0.9`) — proxy for transcription quality.
2. `refinement_score` (Gemini, ≥ `0.5`) — proxy for transcript coherence.

If either score is below threshold, the run is routed straight to
`save_result` with `status = MANUAL_REVIEW`. Classification and sentiment are
skipped entirely.

## Consequences

- **Pro:** keeps the analytics dataset clean. Manual review queue receives
  exactly the calls that need human attention.
- **Pro:** Gemini's `score` field doubles as a "did anything meaningful happen
  here?" signal — useful when Whisper produces structurally-valid but
  meaningless output.
- **Con:** when Gemini API is unreachable, `refinement_score = 0.0` and *every*
  call routes to `MANUAL_REVIEW`. This is intentional — we'd rather queue for
  review than emit unrefined data.
- **Con:** the confidence score is `exp(mean token log-prob)` from a single
  `model.generate()` pass on the first 30s of audio (see
  `_compute_confidence` in `transcription.py`). Long calls are scored from a
  representative sample, not end-to-end. The threshold (`0.9`) was originally
  calibrated to a length-based heuristic and may want re-tuning now that the
  signal is real model log-prob.

## Tuning

Both thresholds are exposed as `PipelineSettings.confidence_threshold` and
`PipelineSettings.refinement_threshold`. Tune in `src/config/config.py` (or
override programmatically) — no code changes required.
