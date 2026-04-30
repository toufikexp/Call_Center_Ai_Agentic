# Runbook — Configuration & environment

How to configure the pipeline for a new machine or deployment.

## What the pipeline reads at startup

`Settings.create_default()` (in `src/config/config.py`) is called on first
`get_settings()`. It pulls from, in order:

1. Process environment.
2. `.env` at the project root (loaded by `load_env_file()` if present —
   only sets variables that aren't already in the environment).
3. Built-in defaults defined in the Pydantic models.

## Required configuration

| Concern                              | Where to set                                                  |
|--------------------------------------|---------------------------------------------------------------|
| Path to Whisper model                | `src/config/config.py` → `WhisperSettings.model_path`         |
| Gemini API key (refinement)          | `.env` → `GEMINI_API_KEY`                                     |
| vLLM endpoint (classification + sentiment) | `.env` → `VLLM_BASE_URL`, `VLLM_MODEL_NAME`             |

## Optional configuration

| Variable                       | Default                       | Used by                         |
|--------------------------------|-------------------------------|---------------------------------|
| `VLLM_API_KEY`                 | `none`                        | Classification, Sentiment       |
| `VLLM_TEMPERATURE`             | `0.1`                         | Classification, Sentiment       |
| `CLASSIFICATION_SCHEMA_PATH`   | `src/config/classification_schema.json` | Classification        |

Pipeline-level constants (`confidence_threshold`, `refinement_threshold`,
`audio_sample_rate`, `output_dir`, `logs_dir`) are not env-driven; edit
`PipelineSettings` defaults in `src/config/config.py` if you need to change
them globally, or override on the `Settings` instance programmatically.

## First-run checklist

```bash
# 1. .env
cp .env.example .env
# edit .env and fill in GEMINI_API_KEY at minimum

# 2. Whisper checkpoint exists?
python -c "import os; from src.config.config import get_settings; \
           p = get_settings().whisper.model_path; \
           print(p, '->', 'OK' if os.path.exists(p) else 'MISSING')"

# 3. vLLM reachable?
curl -s "$VLLM_BASE_URL/models" | head

# 4. Smoke test
python main.py data/audio_files/<one-file>.wav
```

If all four pass, you're good.

## Things that surprise people

- **Editing `.env` does not auto-reload a running process.** The settings are
  cached in module state (`_settings`) on first call. Restart the process.
- **A missing `GEMINI_API_KEY` is not an error.** Refinement is skipped
  (logged at `INFO`), `refinement_score = 0.0`, and the call lands in
  `MANUAL_REVIEW`. To make Gemini failures fatal, you'd need to change
  `RefinementService` — out of scope here.
- **`VLLM_BASE_URL` must include `/v1`.** It's appended directly by the
  `openai` client; no path manipulation is performed.
- **The classifier silently falls back to defaults if the schema JSON is
  malformed.** Validate JSON after editing (see
  `docs/runbooks/taxonomy-changes.md`).
