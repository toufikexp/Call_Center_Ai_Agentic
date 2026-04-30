# Conventions

Coding, configuration, logging, and repo-hygiene conventions used across the
project. New code should match these.

## Service pattern

Every processing stage is a class that inherits from `BaseService`
(`src/core/base.py`).

```python
from src.core.base import BaseService, ServiceResult

class MyService(BaseService):
    def __init__(self, settings):
        super().__init__("my_service")        # logger name = service.my_service
        self.settings = settings
        self._client = None                   # all heavy state lazy-loaded

    def initialize(self) -> None:
        if self._initialized:
            return
        # load model / configure client here
        self._initialized = True

    def process(self, *args, **kwargs) -> ServiceResult:
        def _run():
            self.ensure_initialized()
            # ... real work ...
            return {"some": "data"}
        return self._execute_with_timing(_run)
```

Rules:

- `__init__` must be cheap. Never call out to a model or an API.
- `initialize` is idempotent (`if self._initialized: return`).
- `process` always returns a `ServiceResult`. Use `_execute_with_timing` so
  exceptions become `ServiceResult(success=False, error=...)`.
- Services do not know about each other. They only know `Settings` and the
  inputs passed to `process`.

## Configuration

- All config lives in `src/config/config.py` as Pydantic models. The leaf
  models (`WhisperSettings`, `GeminiSettings`, `VLLMSettings`,
  `ClassificationSettings`, `PipelineSettings`) are composed into one root
  `Settings`.
- `src/core/config.py` is a re-export shim — kept so `from src.core.config
  import ...` keeps working. New imports may use either.
- `Settings.create_default()` is the single source of truth for defaults and
  for reading environment variables.
- Environment variables are read **only** in `create_default()`. Don't sprinkle
  `os.getenv` calls in services.
- The `.env` loader (`load_env_file`) is hand-rolled. Don't add
  `python-dotenv`.
- Taxonomy data (categories, sub-categories, descriptions) lives in
  `src/config/classification_schema.json`, not in Python.

## Logging

All loggers use this format:

```
[YYYY-MM-DD HH:MM:SS,mmm] [<logger>] [<LEVEL>] <message>
```

Logger names:

- `service.<name>` — created automatically by `BaseService`.
- `pipeline.orchestrator` — created in `CallAnalysisPipeline._create_logger`.
- `config` — used by the `.env` loader and `create_default()`.

Conventions:

- One result-summary block per service, framed by 60 `=` characters and a
  title line with an emoji icon (see `RefinementService`, `ClassificationService`,
  etc. for the reference style).
- Log thresholds and decisions in `_route_decision` and `_save_node` so that a
  log is enough to explain why a call landed in `MANUAL_REVIEW`.
- Don't add a third logger backend; stick with `logging.StreamHandler`.

## Errors and routing

- Inside services, raise `RuntimeError` for hard failures that the orchestrator
  must treat as pipeline errors (e.g. JSON parse failure, network down). The
  `_execute_with_timing` wrapper converts the exception into a failed
  `ServiceResult`.
- Inside the orchestrator, set `result.status = ProcessingStatus.ERROR` and
  populate `result.error_message` when a `ServiceResult` comes back with
  `success=False`. Subsequent nodes check `result.status == ERROR` and
  short-circuit.
- Use `ProcessingStatus.MANUAL_REVIEW` for *quality* gating (low scores), not
  for runtime errors.

## Files and folders

- `src/services/` — one file per service, lowercase, singular noun.
- `src/pipeline/` — orchestrators only (today, just one).
- `src/core/` — types, base classes, shared utilities. No business logic.
- `data/` — runtime data. Gitignored except `.gitkeep` markers.
- `models/` — local model checkpoints. Gitignored.
- `notebooks/` — exploratory only (none today). If added, never import from
  `src/`-into-notebook for a notebook that gets committed; instead, copy any
  reusable code into `src/`.

## Naming

- Classes: `PascalCase`, ending in `Service` for services
  (`TranscriptionService`).
- Functions and variables: `snake_case`.
- Pydantic settings models: `<Domain>Settings` (`WhisperSettings`,
  `PipelineSettings`).
- Pipeline node methods: `_<action>_node` and routing methods: `_route_*` /
  `_check_*`. Both are private (leading underscore).
- Status enum values: ALL_CAPS with underscores (`MANUAL_REVIEW`).

## Output schema

`data/results/` JSON shape is governed by `CallAnalysisResult` plus a few
fields injected in `_save_node`. Treat that file as a versioned contract — see
`docs/data-contracts.md`. Don't rename or remove existing fields without bumping
the contract.

## Dependencies

- Pinned versions live in `requirements.txt`. Loose `>=` is reserved for
  framework packages (`langgraph`, `langchain`, `openai`, `google-genai`).
- The PyTorch index URL at the top of `requirements.txt` pulls a CUDA build.
  CPU-only environments must override with the upstream PyPI index.

## Git hygiene

- Branch convention for AI-generated work: `claude/<short-topic>-<id>`.
- `.env`, `data/audio_files/*`, `data/results/*`, `data/logs/*`, and
  `models/*` are gitignored. `.gitkeep` files preserve folder structure.
- Don't commit virtualenvs (`cc_agentic_env/`) or IDE config (`.cursor/`,
  `.vscode/`).
