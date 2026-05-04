# CLAUDE.md

Guidance for Claude Code (and other AI assistants) working in this repository.

## Project at a glance

**Call Center AI Agentic Pipeline** — an on-premise LangGraph pipeline that turns
call center audio (Algerian Darija / Arabic) into structured analytics:
preprocess (channel split + VAD) → transcribe (Whisper + LoRA) → refine →
quality gate → subject classification → customer satisfaction.

- Entry point: `main.py`
- Orchestration: `src/pipeline/orchestrator.py` (LangGraph `StateGraph`)
- Service layer: `src/services/` (one file per stage)
- Configuration: `src/config/config.py` (Pydantic) + `src/config/classification_schema.json`
- State / result models: `src/core/state.py`
- Base service abstraction: `src/core/base.py`

## How to run

```bash
# 1. Create venv and install deps (CUDA build of torch by default)
python -m venv cc_agentic_env
source cc_agentic_env/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env       # then edit GEMINI_API_KEY, VLLM_BASE_URL, etc.

# 3. Run the pipeline
python main.py path/to/audio.mp3
```

Outputs land in `data/results/` as JSON; chunked audio (for files >30s) lands
in `data/chunks/`.

## External dependencies the pipeline expects at runtime

| Component | Purpose | Default |
|---|---|---|
| Silero VAD weights | Preprocessing (channel split + segmentation) | downloaded once via `torch.hub`; cache via `SILERO_CACHE_DIR` for air-gapped runs |
| Whisper base model | Transcription base | `WHISPER_BASE_MODEL_ID` (default `openai/whisper-large-v3`) |
| LoRA adapter (optional) | Fine-tune on top of base | `WHISPER_ADAPTER_PATH` — when unset, `WHISPER_MODEL_PATH` is loaded as a full merged checkpoint |
| Gemini API | Refinement (cloud) | `gemini-2.0-flash-exp` via `google-genai` |
| vLLM server | Classification + sentiment | `http://localhost:8080/v1`, model `Qwen/Qwen3-4B` |
| PostgreSQL (optional) | Analytics + idempotency for batch | enabled via `STORAGE_ENABLE=1` + `DATABASE_URL=postgresql://...`; see `docs/runbooks/storage.md` |

If neither the base model nor the adapter is loadable, transcription fails at
`initialize()`. If the adapter's `base_model_name_or_path` doesn't match
`WHISPER_BASE_MODEL_ID`, the service logs a warning and aligns to the
adapter's declared base.

## Repo conventions worth knowing before editing

- **No agents/tools, only services.** Each stage is a `BaseService` subclass
  with `initialize()` (lazy) and `process()` (returns `ServiceResult`). Don't
  introduce a different abstraction.
- **Lazy initialization.** Services load models on first `process()` call via
  `ensure_initialized()`. Avoid loading models in `__init__`.
- **`ServiceResult` is the boundary.** Services must not raise out of
  `process()` for normal failures — wrap in `_execute_with_timing()` so the
  orchestrator can branch on `result.success`.
- **Pipeline routing is in one place.** All conditional edges live in
  `_route_decision` and `_check_error` in `orchestrator.py`. Add new branches
  there, not inside service code.
- **Config is a single Pydantic tree.** `Settings` (in `src/config/config.py`)
  is re-exported from `src/core/config.py` for backward compat. New settings
  go on the relevant sub-model (`WhisperSettings`, `VLLMSettings`, …).
- **Classification taxonomy lives in JSON**, not in Python — see
  `src/config/classification_schema.json`. Edit there, not in code.
- **`.env` is auto-loaded** from project root by `load_env_file()` in
  `config.py`. Don't add `python-dotenv`; the loader is intentional.
- **Logs use a unified format** with timestamps and `service.<name>` /
  `pipeline.orchestrator` logger names. Match the existing format when adding
  log lines.

## Things that look like bugs but aren't

- `CorrectionService` (`src/services/correction.py`) is not wired into the
  graph. The file is kept as a reference for a future re-enablement of
  in-process Qwen correction. Don't delete the file; just leave it.
- `confidence_score` is the **mean of per-segment** `exp(mean token log-prob)`
  across all VAD segments, computed via `output_scores=True` inside
  `TranscriptionService._batch_confidences`. The default threshold (`0.9`)
  was calibrated under the previous heuristic; expect to retune it after
  this migration — see `docs/runbooks/threshold-tuning.md`.
- Pipeline routes to `MANUAL_REVIEW` whenever Gemini refinement fails OR has
  no API key — that is by design (refinement score 0.0 < 0.5 threshold).
- Preprocessing emitting **zero segments** (silent / unintelligible audio)
  routes to `MANUAL_REVIEW`: empty transcript → empty refinement → score 0
  → manual review. Refinement is short-circuited in this case so we don't
  send empty input to Gemini.
- `WhisperSettings.adapter_path` empty is **not** an error: the service
  loads `WHISPER_MODEL_PATH` as a full merged checkpoint (Path A artifact).
  Either route works.

## Where to make common changes

| Change | File |
|---|---|
| Swap LoRA adapter (new fine-tune) | env var `WHISPER_ADAPTER_PATH` (no code change) |
| Swap Whisper base model | env var `WHISPER_BASE_MODEL_ID` (no code change) |
| Tune VAD thresholds / segment lengths | env (`VAD_*`) or `PreprocessingSettings` defaults |
| Adjust GPU batch size | env `WHISPER_BATCH_SIZE` or `WhisperSettings.batch_size` |
| Adjust confidence / refinement thresholds | `src/config/config.py` → `PipelineSettings` |
| Add / rename a category or sub-category | `src/config/classification_schema.json` |
| Tweak the refinement prompt | `GeminiSettings.refinement_prompt_template` in `src/config/config.py` |
| Tweak classification prompt | `_build_classification_prompt` in `src/services/classification.py` |
| Tweak sentiment prompt | `_build_sentiment_prompt` in `src/services/sentiment.py` |
| Change pipeline graph (nodes/edges) | `_build_graph` in `src/pipeline/orchestrator.py` |
| Add a new stage | New file in `src/services/` + new node in `_build_graph` |
| Output JSON shape | `CallAnalysisResult` in `src/core/state.py` |

## What not to do

- **Do not commit `.env`** or any audio file under `data/audio_files/` —
  `.gitignore` is configured for this; respect it.
- **Do not rewrite the service abstraction** to use LangChain agents/tools.
  Services-with-orchestrator is the chosen architecture.
- **Do not bypass `_execute_with_timing`** in services; the orchestrator
  relies on `ServiceResult.success` for routing.
- **Do not add `python-dotenv`** as a dep; `load_env_file()` is intentionally
  hand-rolled.
- **Do not move the Whisper model path** out of code into env without also
  updating the runbook — several Whisper checkpoints are referenced in
  comments and switching them is an explicit experiment workflow.

## Documentation map

- `README.md` — user-facing overview, setup, output schema
- `docs/architecture.md` — pipeline diagram, data flow, component boundaries
- `docs/conventions.md` — coding / logging / config conventions
- `docs/data-contracts.md` — input/output schemas at every stage
- `docs/decisions/` — architecture decision records (ADRs)
- `docs/runbooks/` — operational procedures (incidents, model swap, taxonomy
  changes)
