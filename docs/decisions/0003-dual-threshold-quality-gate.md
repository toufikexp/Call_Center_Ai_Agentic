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
- **Con:** the confidence score is a length-based heuristic
  (`recalculate_confidence`), not a true model log-prob. Threshold value
  (`0.9`) is calibrated to that heuristic and would need re-tuning if the
  scorer changed.

## Tuning

Both thresholds are exposed as `PipelineSettings.confidence_threshold` and
`PipelineSettings.refinement_threshold`. Tune in `src/config/config.py` (or
override programmatically) — no code changes required.
