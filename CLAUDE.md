# CLAUDE.md

Guidance for Claude Code (and other AI assistants) working in this repository.

## Project at a glance

**Call Center AI Agentic Pipeline** — an on-premise LangGraph pipeline that turns
call center audio (Algerian Darija / Arabic) into structured analytics:
transcript → refined transcript → quality gate → subject classification →
customer satisfaction.

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

| Component | Purpose | Default endpoint / path |
|---|---|---|
| Whisper checkpoint | Transcription | hard-coded path in `src/config/config.py` (`WhisperSettings.model_path`) |
| Gemini API | Refinement (cloud) | `gemini-2.0-flash-exp` via `google-genai` |
| vLLM server | Classification + sentiment | `http://localhost:8080/v1`, model `Qwen/Qwen3-4B` |

If the Whisper model path doesn't exist on the current machine, transcription
will fail at `initialize()` — update `WhisperSettings.model_path` in
`src/config/config.py` before assuming the pipeline is broken.

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
- `confidence_score` is a real Whisper signal — `exp(mean token log-prob)`
  computed by `_compute_confidence` in `transcription.py`. For audio longer
  than `chunk_length_seconds` (30s) the score is computed on the first chunk
  only, as a representative sample.
- Pipeline routes to `MANUAL_REVIEW` whenever Gemini refinement fails OR has
  no API key — that is by design (refinement score 0.0 < 0.5 threshold).

## Where to make common changes

| Change | File |
|---|---|
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
