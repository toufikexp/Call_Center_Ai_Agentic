# Runbook — Tuning quality thresholds

Two thresholds determine whether a call proceeds to classification or is
diverted to manual review:

| Setting                                | Default | What it gates on                                  |
|----------------------------------------|---------|---------------------------------------------------|
| `PipelineSettings.confidence_threshold` | `0.90`  | Whisper `exp(mean token log-prob)`                |
| `PipelineSettings.refinement_threshold` | `0.50`  | Gemini refinement quality score                   |

Both live in `src/config/config.py`.

## When to retune

- Too many calls land in `MANUAL_REVIEW` → thresholds may be too strict.
- Manual review queue is empty but classifications look unreliable →
  thresholds may be too loose.
- After swapping the Whisper model (the confidence "score" is a heuristic of
  transcript length; different models / audio quality shift the distribution).
- After changing the refinement prompt or model.

## Diagnostic — look at the score distribution

Run the pipeline on a representative batch (say 50–100 calls), then:

```bash
python - <<'PY'
import json, glob
rows = [json.load(open(p)) for p in glob.glob("data/results/*.json")]
def quantiles(xs):
    xs = sorted(xs)
    return {q: xs[int(len(xs)*q)] for q in [0.1, 0.25, 0.5, 0.75, 0.9]}
print("confidence_score quantiles:", quantiles([r["confidence_score"] for r in rows]))
print("refinement_score quantiles:", quantiles([r["refinement_score"] for r in rows]))
PY
```

Look at the median and p25:

- If most "good" calls are clustered near 0.95 confidence and only the broken
  ones drop below 0.85, `0.9` is reasonable.
- If a noticeable mass of perfectly fine refined transcripts has scores in
  `[0.4, 0.5)`, lowering `refinement_threshold` to `0.4` may unblock them.

## Procedure

1. Edit defaults in `src/config/config.py`:
   ```python
   pipeline=PipelineSettings(
       confidence_threshold=0.85,   # was 0.90
       refinement_threshold=0.40,   # was 0.50
   ),
   ```
   Or override programmatically without code changes:
   ```python
   from src.pipeline import CallAnalysisPipeline
   from src.core.config import get_settings

   s = get_settings()
   s.pipeline.confidence_threshold = 0.85
   pipeline = CallAnalysisPipeline(s)
   ```
2. Re-run the diagnostic batch.
3. Spot-check 10–20 newly-passing calls — do they look like ones a human
   would also accept?

## Notes

- The two thresholds are independent: a call must pass *both* to proceed.
- `confidence_score` is `exp(mean token log-prob)` from
  `_compute_confidence` in `src/services/transcription.py`. The score is
  computed from the **first 30 seconds** of the audio (the Whisper window
  size), as a representative sample for long-form input.
- `refinement_score` comes directly from the Gemini model. If you change the
  refinement prompt, the score distribution can shift abruptly.
