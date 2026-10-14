# Project Context — Call Center AI Agentic Pipeline

Technical handover for an AI coding session that will deploy this project into
a two-environment production: a **CPU VM** plus a **GPU "AI Fabric"**
(Kubernetes + JupyterHub).

Every technical claim was verified against the code at commit **`ae86396`**
(GitHub `main`, 2026-06-21 — nothing newer has been pushed) unless tagged
otherwise.

| Tag | Meaning |
|---|---|
| `[code file:line]` | Read directly from the code at `ae86396` |
| `[executed]` | Proven by running the real code path (sandbox; only ML libraries stubbed) |
| `[observed]` | Seen in a real environment in an earlier session — may have changed, **verify** |
| `[user]` | Stated by the project owner; not verifiable from code |
| `[unknown]` | Not known — must be asked |

**Source of truth: code > this document > `CLAUDE.md` / `docs/`.** The docs
are partly outdated (see §14). When anything disagrees, trust the code and
re-verify.

---

## 1. What the system does

One call recording in → structured analysis out.

```
audio file (stereo WAV: agent = ch0, client = ch1)
 → preprocess   channel split · resample 16 kHz · Silero VAD segments   CPU, in-process
 → transcribe   Whisper large-v3 + LoRA adapter, batched per segment    GPU or CPU, in-process
 → refine       Gemini (google-genai): cleanup + quality score 0–1      cloud, HTTPS
 → verify       quality gate on refinement score and confidence
 → classify     subject / sub-subject from a JSON taxonomy              vLLM, HTTP
 → sentiment    satisfaction 1–10 + label + reasoning                   vLLM, HTTP
 → save         JSON file + PostgreSQL rows (optional)
```

Domain: Ooredoo Algeria customer-care calls in Algerian Darija/Arabic with
French code-switching [code config.py:164-184, sentiment.py:46].

Per call it produces: a JSON file in `data/results/`, rows in Postgres
`calls` / `call_results`, log lines, and (batch mode) moves the audio file to
an archive folder.

---

## 2. What has been done

50 commits, 2025-12-31 → 2026-06-21 [code: `git log origin/main`].

| Period | Delivered |
|---|---|
| 2025-12 → 2026-01 | Initial pipeline; migration to the `google-genai` SDK; centralized configuration |
| 2026-04-30 | Channel split + Silero VAD; runtime PEFT/LoRA loading; per-segment batched transcription; real Whisper confidence |
| 2026-05-04/06 | Offline-safe `.env` bootstrap; Silero via pip package; PostgreSQL persistence + provisioning script; batch runner with archive and live batch metrics; automatic column migrations |
| 2026-06-04/08 | Docker: one multi-stage Dockerfile, one compose file, one Makefile |
| 2026-06-09/11 | FastAPI job server; CPU↔GPU compose profiles; dedicated vLLM compose; `job_id == call_id`; one `batch_runs` row per server job; uuid `call_id` in batch; env-tunable thresholds; thread-safe lazy init; graceful Ctrl-C; vLLM `content=None` guard + Qwen3 thinking disabled; `TZ` for logs; default `batch_runs.notes` |
| 2026-06-21 | Docs: test-commands cheat sheet |

**Not present in the repo** [code: `git ls-files`]: automated tests, CI
configuration, Kubernetes manifests / Helm charts, notebooks, any non-Docker
install script, any UI. The only "backend" in the repo is the FastAPI job
server (§6.3).

---

## 3. Repository map

| Path | Purpose |
|---|---|
| `main.py` | Single-file CLI; also hosts `_bootstrap_env()` reused by the other entry points |
| `src/pipeline/orchestrator.py` | `CallAnalysisPipeline`: LangGraph graph, routing, JSON save, DB recording, batch lifecycle |
| `src/services/preprocessing.py` | Audio load, channel split, resample, Silero VAD |
| `src/services/transcription.py` | Whisper + LoRA, batched generate, confidence, transcript formatting |
| `src/services/refinement.py` | Gemini cleanup + score |
| `src/services/classification.py` | Subject classification via vLLM |
| `src/services/sentiment.py` | Satisfaction / sentiment via vLLM |
| `src/services/correction.py` | In-process Qwen correction — **not wired into the graph** |
| `src/core/base.py` | `BaseService`, `ServiceResult`, thread-safe `ensure_initialized()` |
| `src/core/state.py` | `CallAnalysisResult`, `SegmentResult`, `ProcessingStatus`, `PipelineState` |
| `src/core/config.py` | Re-export shim for `src/config/config.py` |
| `src/config/config.py` | Pydantic settings tree + env loading |
| `src/config/classification_schema.json` | Taxonomy: categories, sub-categories, descriptions, defaults |
| `src/batch/{__main__,cli,runner}.py` | `python -m src.batch run` |
| `src/server/{__main__,app,jobs}.py` | `python -m src.server` (FastAPI + in-memory job store) |
| `src/storage/results_store.py` | PostgreSQL schema, migrations, writes |
| `src/utils/ids.py` | `make_call_id` (content hash) — **no longer used** |
| `Dockerfile`, `compose.yaml`, `compose.vllm.yaml`, `Makefile` | Packaging and operations |
| `requirements.txt`, `requirements/{base,torch-cpu,torch-gpu}.txt` | Dependencies |
| `scripts/setup_postgres.sh` | Role/database provisioning for an external PostgreSQL |
| `.env.example` | Configuration template (`.env` itself is git-ignored) |

---

## 4. Pipeline graph and final status

Graph [code orchestrator.py:104-164]:

```
preprocess ──error───────────────────────────────┐
    │ continue                                   │
transcribe → refine → verify ──manual_review────►├──► save_result → END
                        │ proceed                │
                     classify ──error───────────►┤
                        │ continue               │
                 analyze_sentiment ──(always)───►┘
```

Routing [code orchestrator.py:286-312]: `refinement_score < REFINEMENT_THRESHOLD`
→ manual review; else `confidence_score < CONFIDENCE_THRESHOLD` → manual
review; else proceed. Final status is set in `_save_node` [370-383].

**Final status per failing stage [executed]** — the real compiled graph, with
each stage's `process()` forced to fail:

| Failing stage | Final status | `error_message` |
|---|---|---|
| none (scores above thresholds) | `COMPLETE` | — |
| preprocessing | `ERROR` | set |
| **transcription** | **`MANUAL_REVIEW`** | set |
| **refinement** | **`MANUAL_REVIEW`** | set |
| classification | `ERROR` | set |
| sentiment | `ERROR` | set |

Why: `_verify_node` sets `status = IN_PROGRESS` unconditionally
[orchestrator.py:271], erasing an `ERROR` set by transcribe or refine. The
`ERROR` check at the top of `_route_decision` [291-293] is therefore never
reached for those stages.

Consequences:
- Monitoring must look at `error_message`, not only `status`.
- **Batch mode archives `MANUAL_REVIEW` files** into `manual_review/`
  [code runner.py:183-222]. A transcription outage (e.g. a remote GPU service
  being down) would move files out of the input folder instead of leaving them
  for retry.
- No retry loop exists; `PipelineSettings.max_retry_attempts` is unused.

Other routes to `MANUAL_REVIEW`: no `GEMINI_API_KEY` (refinement score 0); no
speech detected (empty transcript → refinement skipped → score 0)
[code orchestrator.py:186-191, 232-236].

---

## 5. Stages in detail

### 5.1 Preprocessing — `src/services/preprocessing.py`
- Load with `soundfile`; on failure fall back to `librosa.load` (audioread /
  ffmpeg), which handles malformed WAV headers and mp3 [189-217].
- Mono → one `unknown` channel. Two or more channels → `agent` =
  `PREPROCESSING_AGENT_CHANNEL` (0), `client` = `PREPROCESSING_CLIENT_CHANNEL`
  (1); extra channels ignored [153-187].
- Resample to 16 kHz float32 [166-169].
- VAD: the `silero-vad` pip package; weights ship inside the wheel, so no
  download and no network [48-78]. Inference serialized by `_vad_lock` [46, 238].
- Duration filter after padding (`VAD_PADDING_MS`): segments shorter than
  `VAD_MIN_SEGMENT_SECONDS` or longer than `VAD_MAX_SEGMENT_SECONDS` are
  **dropped, not split** [253-259]. A high minimum (e.g. 5.0 s) silently
  discards short conversational turns and yields truncated transcripts
  [observed in prod].
- Output: list of `{channel, start_ms, end_ms, audio: np.float32}` sorted by
  `start_ms`, held **in memory** in `PipelineState["segments"]` and never
  serialized [state.py:59-62].
- `PREPROCESSING_SAVE_SEGMENTS=1` dumps WAVs to `data/segments/<basename>/`
  (relative to the working directory) [270-286].
- `PREPROCESSING_ENABLE` is read but **never used** — preprocessing always runs.

### 5.2 Transcription — `src/services/transcription.py`
- Base: `WhisperForConditionalGeneration.from_pretrained(WHISPER_BASE_MODEL_ID)`
  (default `openai/whisper-large-v3`), then
  `PeftModel.from_pretrained(base, WHISPER_ADAPTER_PATH).merge_and_unload()`
  [81-94]. If the adapter's `adapter_config.json` names a different base, the
  service warns and uses the adapter's base [65-76].
- Without an adapter path it loads `WHISPER_MODEL_PATH` as a merged checkpoint,
  whose **default is a developer home path**
  `/home/zerrougt/UBC_NLP_Casablanca/whisper_large_gemini_fixed/checkpoint-250`
  [config.py:369-372].
- Processor (tokenizer + feature extractor) always comes from the base model
  id, so offline mode needs it in the Hugging Face cache [112-117].
- Device: `cuda` if `torch.cuda.is_available()` else `cpu`; fp16 on GPU, fp32
  on CPU [174-189]. `WHISPER_USE_8BIT=1` needs `bitsandbytes`, which is not in
  the requirements.
- Inference: batches of `WHISPER_BATCH_SIZE` segments; greedy decoding
  (`do_sample=False`, `num_beams=1`), `repetition_penalty=1.2`,
  `no_repeat_ngram_size=4`, `max_new_tokens=444`, language `ar`, task
  `transcribe` [273-296]. CUDA OOM → batch halved recursively [253-271].
- **`model.generate()` is serialized by `_gen_lock`** [49, 295]. Within one
  process, worker threads never run two Whisper inferences at once. More GPU
  throughput means more processes (or pods), not more threads.
- Confidence: per segment `exp(mean token log-prob)` excluding eos/pad; call
  confidence = mean over segments with confidence > 0 [318-353, 230-233].
- Transcript line format: `MM:SS [Agent]|[Subscriber]|[Speaker]: text`
  [368-388].
- Model load happens on the first call and is guarded by a lock
  [base.py:97-110].

### 5.3 Refinement — `src/services/refinement.py`
- `genai.Client(api_key=GEMINI_API_KEY).models.generate_content(model=GEMINI_MODEL_NAME, contents=prompt, config={'response_mime_type': 'application/json'})`
  [32, 65-69].
- **No temperature and no timeout are passed**; `GeminiSettings.timeout_seconds`
  is unused. The same transcript yields different refined text across runs
  [observed].
- Prompt: `GeminiSettings.refinement_prompt_template` [config.py:163-186].
  A refined text shorter than 30 % of the input is rejected (score 0) [81-86].
- Default model `gemini-2.0-flash` [config.py:159]; `.env.example` sets
  `gemini-2.5-flash`.
- The only cloud service the pipeline calls by design: HTTPS egress to
  Google's Generative Language API. (The Hugging Face Hub may also be
  contacted by `from_pretrained` unless `HF_HUB_OFFLINE` /
  `TRANSFORMERS_OFFLINE` are set.)

### 5.4 Classification — `src/services/classification.py`
- OpenAI SDK client → `VLLM_BASE_URL`, `model=VLLM_MODEL_NAME`,
  `temperature=VLLM_TEMPERATURE` (default **0.1, i.e. sampling**),
  `max_tokens=VLLM_MAX_TOKENS`, and
  `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` when
  `VLLM_DISABLE_THINKING=1` [34-37, 169-182]. No client timeout is passed.
- Taxonomy (`classification_schema.json`, overridable with
  `CLASSIFICATION_SCHEMA_PATH`): 11 categories — Customer Service Process
  Related (1 sub-category), Customer Service Support Topic (16), Pricing (1),
  Network (4), Miscellaneous (2), Product Category (10), Channel (5), Mobile
  Device (6), Brand (2), Information (3), OTHER (0).
- Validation: unknown category → `OTHER`; unknown sub-category → the schema's
  `default_subcategory`, else the first allowed one [208-263].
- **Prompt conflicts with the schema**: rule 4 and the CRITICAL block
  [104, 125] restrict "Customer Service Support Topic" to support-process
  issues, while the schema describes it as complaints, credit loss, new
  offers and promos.
- Guard for reasoning-only responses (`content=None`) [187-195].

### 5.5 Sentiment — `src/services/sentiment.py`
- Same client and parameters as classification. Returns
  `satisfaction_score` 1–10 (out of range → 0.0), `sentiment_label`
  POSITIVE / NEUTRAL / NEGATIVE, `reasoning` [94-199].

### 5.6 Correction — not wired
`src/services/correction.py` loads a Qwen model in-process with transformers.
It is not in the graph; `Settings.qwen` (`Qwen/Qwen-7B-Chat`) only feeds it.

---

## 6. Run modes

### 6.1 Single file — `python main.py <audio_path>`
Loads `.env` before heavy imports; `call_id` = `call_<basename>_<uuid8>`;
exits 1 on `ERROR` [code main.py].

### 6.2 Batch — `python -m src.batch run [options]` (the Docker image's ENTRYPOINT)

| Option | Default | Effect |
|---|---|---|
| `--input-dir` | `$INPUT_DIR` or `data/audio_files` | Source directory |
| `--manifest` | — | File of paths, one per line (`#` = comment); mutually exclusive with `--input-dir` |
| `--pattern` | `*.wav` | Glob in the input dir (`.mp3` needs `--pattern '*.mp3'`) |
| `--workers` | 2 | Threads sharing one pipeline |
| `--limit` | — | First N files (sorted) |
| `--batch-name` | `batch-run-<timestamp>` | `batch_runs.notes` |
| `--archive-dir` | `$ARCHIVE_DIR` or the input dir | Parent of `processed_<YYYYMMDD>/` |
| `--no-archive` | off | Leave files in place |
| `--no-skip-completed` | off | (skip has no effect — see below) |
| `--dry-run` | off | List files, process nothing |

[code cli.py, runner.py]
- `call_id` = `uuid4().hex` per file [runner.py:250], so
  `is_already_processed()` never matches: **no resume** — re-running a folder
  reprocesses everything.
- Archive: `COMPLETE` → `…/processed_<date>/completed/`, `MANUAL_REVIEW` →
  `…/manual_review/`, anything else stays in the input dir [183-222].
- Log file: `data/logs/batch_<batch_id>.log`, relative to the working
  directory, not env-configurable [123-138].
- SIGINT / SIGTERM: stop submitting, finish in-flight calls, finish the batch
  [227-236, 365-376].
- Exit code 1 if any `ERROR` / `EXCEPTION` (unless stopped by a signal)
  [388-390].

### 6.3 HTTP server — `python -m src.server`
[code server/__main__.py, app.py, jobs.py]
- uvicorn, one process; `SERVER_HOST` (0.0.0.0), `SERVER_PORT` (8000),
  `SERVER_LOG_LEVEL`.
- `SERVER_WORKERS` job threads (default 1) share one pipeline; models stay warm
  after the first job.
- Endpoints:
  - `GET /health` → `{"status":"ok"}` — checks nothing.
  - `GET /ready` → `{"ready": true}` once the job store exists — does **not**
    check models, vLLM or DB.
  - `POST /jobs` (202): JSON `{"audio_path": …}` (must resolve inside the data
    root; traversal rejected) or multipart field `audio` (saved to
    `<data_root>/audio_files_uploads/<uuid>__<name>`, never cleaned up).
  - `GET /jobs/{id}`, `GET /jobs?limit=N`.
- `job_id == call_id`: `SELECT * FROM call_results WHERE call_id = '<job_id>'`.
- Each job writes its own `batch_runs` row (`notes = server-job:<id>`).
- Job registry is **in memory** (500 most recent); a restart loses queued jobs
  and history.
- **No authentication, no TLS, no upload size limit.** Compose publishes the
  port on `127.0.0.1` only [compose.yaml:104, 126].
- Data root: `$DATA_ROOT`, else `/app/data` if it exists, else the parent of
  `$INPUT_DIR` [app.py:190-205].

---

## 7. Configuration

### 7.1 Loading
- `.env` in the project root is read by two hand-rolled parsers
  (`main._bootstrap_env`, `config.load_env_file`). The first runs before
  `transformers` is imported so the HF offline flags take effect. Variables
  already in the environment are never overridden
  [main.py:17-47; config.py:17-84].
- **Neither parser strips inline comments**: `KEY=value   # note` yields the
  value `value   # note` [main.py:36-38; config.py:39-41]. `.env.example` has
  inline comments on `POSTGRES_PASSWORD`, `PREPROCESSING_SAVE_SEGMENTS`,
  `GEMINI_API_KEY` and `VLLM_MODEL`. In Docker the image contains no `.env`
  (values come from compose `env_file`), but **outside Docker — e.g. in a
  Jupyter pod — these parsers read `.env` directly**, so `GEMINI_API_KEY`
  would become the comment text. Keep `.env` free of inline comments
  everywhere.
- Settings: Pydantic tree `Settings`, singleton via `get_settings()`
  [config.py:266-516].

### 7.2 Variables read by Python code (34) [code, AST-extracted]

| Variable | Default | Effect | Read at |
|---|---|---|---|
| `GEMINI_API_KEY` | — | Enables refinement; empty → every call `MANUAL_REVIEW` | config.py:308 |
| `GEMINI_MODEL_NAME` | `gemini-2.0-flash` | Gemini model | config.py:461 |
| `VLLM_BASE_URL` | `http://localhost:8080/v1` | vLLM OpenAI endpoint | config.py:336 |
| `VLLM_MODEL_NAME` | `Qwen/Qwen3-4B` | Model name in requests — must equal vLLM's served name | config.py:337 |
| `VLLM_API_KEY` | `none` | Sent as the bearer key | config.py:338 |
| `VLLM_TEMPERATURE` | `0.1` | 0 = deterministic | config.py:339 |
| `VLLM_MAX_TOKENS` | `1024` | Completion cap | config.py:350 |
| `VLLM_DISABLE_THINKING` | `1` | Sends `enable_thinking=false` | config.py:354 |
| `CLASSIFICATION_SCHEMA_PATH` | bundled JSON | Taxonomy file | config.py:359 |
| `WHISPER_BASE_MODEL_ID` | `openai/whisper-large-v3` | Base model + processor | config.py:366 |
| `WHISPER_MODEL_PATH` | developer home path | Merged checkpoint (used only without adapter) | config.py:369 |
| `WHISPER_ADAPTER_PATH` | — | LoRA adapter directory | config.py:373 |
| `WHISPER_BATCH_SIZE` | `4` | Segments per `generate()` | config.py:376 |
| `WHISPER_USE_8BIT` | `0` | bitsandbytes 8-bit | config.py:380 |
| `PREPROCESSING_ENABLE` | `1` | **No effect** | config.py:430 |
| `VAD_THRESHOLD` | `0.5` | Silero speech threshold | config.py:431 |
| `VAD_MIN_SEGMENT_SECONDS` | `1.0` | Drop shorter segments | config.py:434 |
| `VAD_MAX_SEGMENT_SECONDS` | `30.0` | Drop longer segments | config.py:438 |
| `VAD_PADDING_MS` | `250` | Padding per side | config.py:442 |
| `PREPROCESSING_AGENT_CHANNEL` | `0` | Agent channel index | config.py:445 |
| `PREPROCESSING_CLIENT_CHANNEL` | `1` | Client channel index | config.py:449 |
| `PREPROCESSING_SAVE_SEGMENTS` | `0` | Debug segment dump | config.py:453 |
| `SILERO_CACHE_DIR` | — | Sets `torch.hub` dir — no effect today | config.py:457 |
| `CONFIDENCE_THRESHOLD` | `0.90` | Quality gate | config.py:487 |
| `REFINEMENT_THRESHOLD` | `0.5` | Quality gate | config.py:490 |
| `STORAGE_ENABLE` | `0` | Postgres on/off (compose sets `1`) | config.py:495 |
| `DATABASE_URL` | — | `postgresql://…` (compose builds it) | config.py:496 |
| `INPUT_DIR` | `data/audio_files` | Batch input; server data-root fallback | runner.py:74, app.py:204 |
| `ARCHIVE_DIR` | input dir | Archive parent | runner.py:83 |
| `DATA_ROOT` | `/app/data` if present | Server path sandbox | app.py:198 |
| `SERVER_HOST` | `0.0.0.0` | Bind address | server/__main__.py:31 |
| `SERVER_PORT` | `8000` | Bind port | server/__main__.py:32 |
| `SERVER_LOG_LEVEL` | `INFO` | Log level | server/__main__.py:24 |
| `SERVER_WORKERS` | `1` | Job threads | app.py:80 |

### 7.3 Used only by compose / Makefile
`IMAGE_CPU`, `IMAGE_GPU`, `APP_UID`, `APP_GID`, `DATA_DIR`, `ADAPTERS_DIR`,
`MODELS_DIR`, `HF_CACHE_DIR`, `POSTGRES_USER`, `POSTGRES_PASSWORD`,
`POSTGRES_DB`, `TZ`, `SERVER_HOST_PORT`, `VLLM_HOST_PORT`, `VLLM_MODEL`,
`VLLM_MAX_MODEL_LEN`, `VLLM_GPU_IMAGE`, `VLLM_GPU_MEM_UTIL`, `VLLM_CPU_IMAGE`,
`VLLM_CPU_KVCACHE_SPACE`, `VLLM_CPU_OMP_THREADS`; Makefile: `TORCH`, `IMAGE`,
`REGISTRY`, `TAG`, `ARGS`.

### 7.4 Read by libraries, not by project code
`HF_HOME` (`/hf` in the image), `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `TZ`,
proxy variables (`HTTPS_PROXY`, `NO_PROXY`) — prod needed `NO_PROXY` for
internal service names [observed]. `NUMBA_CACHE_DIR` (librosa → numba) was
needed when the container ran as a uid with no home directory in the image
[observed].

### 7.5 Settings that do nothing
`PREPROCESSING_ENABLE`; `PipelineSettings.max_retry_attempts`,
`audio_sample_rate`, `enable_verbose_logging`; `GeminiSettings.timeout_seconds`;
`VLLMSettings.timeout_seconds`; `WhisperSettings.chunk_length_seconds`,
`chunk_overlap_seconds`; `ClassificationSettings.other_category_threshold`;
`SILERO_CACHE_DIR`; `Settings.qwen` (only the unwired correction service).

### 7.6 Hard-coded (not configurable via env)
`data/results`, `data/logs`, `data/segments` (all relative to the working
directory — run from the project root, `/app` in Docker, or outputs land
elsewhere);
Whisper language `ar`, task, fp16, generation parameters; 16 kHz target rate;
classification and sentiment prompts (in the service files); refinement
prompt (config.py).

---

## 8. Data and storage

### 8.1 JSON result
`data/results/<audio_basename>_<suffix>_result.json`, where `<suffix>` is the
part of `call_id` after its last `_`, or its last 8 characters when it has no
`_` (batch and server ids) [orchestrator.py:386-405]. Contents: every
`CallAnalysisResult` field [state.py:32-50]
plus `audio_path` and `run_count`. Fields: `call_id`, `transcript`,
`refined_transcript`, `confidence_score`, `refinement_score`, `subject`,
`sub_subject`, `classification_confidence`, `satisfaction_score`,
`sentiment_label`, `sentiment_reasoning`, `segments[]`
(`channel, start_ms, end_ms, text, confidence`), `audio_duration_s`,
`channel_count`, `whisper_adapter_version`, `status`, `error_message`.

### 8.2 PostgreSQL — `src/storage/results_store.py`
- `psycopg` v3; a **new connection per operation**, no pool [179-184].
- Schema created at startup with `CREATE … IF NOT EXISTS`, plus idempotent
  `ADD COLUMN IF NOT EXISTS` migrations [42-122, 186-200]. Only the database
  and role must exist beforehand (the compose `db` service creates them from
  `POSTGRES_*`; `scripts/setup_postgres.sh` does it for an external server).
- Enabled when `STORAGE_ENABLE=1` and `DATABASE_URL` is set; compose sets both
  for the app/server services, with host `db:5432` [compose.yaml:30-35].
- If schema init fails at startup, storage is **disabled with a warning** and
  the pipeline continues JSON-only [orchestrator.py:43-51].
  `record_attempt()` never raises (logs only) [348-349]; `start_batch()` /
  `finish_batch()` do raise.

| Table | Columns |
|---|---|
| `calls` | `call_id` TEXT PK · `audio_path` TEXT NOT NULL · `duration_s` REAL · `channel_count` SMALLINT · `received_at` TIMESTAMPTZ default now() |
| `call_results` | `id` BIGSERIAL PK · `call_id` → calls · `batch_id` · `status` · `transcript` · `refined_transcript` · `confidence_score` · `refinement_score` · `subject` · `sub_subject` · `classification_confidence` · `satisfaction_score` · `sentiment_label` · `sentiment_reasoning` · `whisper_adapter_version` · `error_message` · `segments` JSONB · `started_at` · `finished_at` · `duration_s` |
| `batch_runs` | `batch_id` TEXT PK · `started_at` · `finished_at` · `file_count` · `success_count` · `error_count` · `review_count` (bumped live per call) · `notes` |

One `call_results` row per attempt; `calls` is upserted. Indexes on
`call_results(call_id, batch_id, status, subject, finished_at,
whisper_adapter_version)` and `calls(audio_path)`. Timestamps are UTC.

### 8.3 Files written at runtime
`data/results/`, `data/logs/`, the archive (`ARCHIVE_DIR`),
`data/audio_files_uploads/` (server), `data/segments/` (debug). In Docker all
of these live under the `/app/data` bind mount.

---

## 9. Packaging and deployment artifacts

### 9.1 Dockerfile
Multi-stage on `python:3.12-slim`. Builder: `build-essential`, `ffmpeg`,
`libsndfile1`, `libgomp1`; `pip install -r requirements/base.txt -r
requirements/torch-${TORCH_VARIANT}.txt`. Runtime: `ffmpeg`, `libsndfile1`,
`libgomp1`; user `app` (uid 1000); dirs `/app/data`, `/adapters`, `/hf`;
`HF_HOME=/hf`; copies only `src/` and `main.py`. ENTRYPOINT
`python -m src.batch run`, CMD `--workers 2`. No models and no `.env` in the
image.

### 9.2 `compose.yaml`

| Service | Profile | Image | Notes |
|---|---|---|---|
| `db` | (none — always) | `postgres:16` | Volume `pgdata`; port `127.0.0.1:5432`; healthcheck |
| `server` | `cpu` | `IMAGE_CPU` | `python -m src.server`; port `127.0.0.1:${SERVER_HOST_PORT:-8000}` |
| `server-gpu` | `gpu` | `IMAGE_GPU` | Same + 1 NVIDIA GPU |
| `app` | `cpu-batch` | `IMAGE_CPU` | Batch runner |
| `app-gpu` | `gpu-batch` | `IMAGE_GPU` | Batch runner + 1 NVIDIA GPU |

Shared by all pipeline services: `user: ${APP_UID:-1000}:${APP_GID:-1000}`,
`env_file: .env`, `host.docker.internal:host-gateway`, mounts
`DATA_DIR→/app/data`, `ADAPTERS_DIR→/adapters`, `HF_CACHE_DIR→/hf`, wait for
`db` healthy.

### 9.3 `compose.vllm.yaml`
Profiles `gpu` (`vllm-gpu`, image `VLLM_GPU_IMAGE`, default
`vllm/vllm-openai:latest`, adds `--gpu-memory-utilization`) and `cpu`
(`vllm-cpu`, image `VLLM_CPU_IMAGE`, default `vllm-cpu:local`; comment notes
`openeuler/vllm-cpu:latest` validated as vLLM 0.20.1). Args: `--model
$VLLM_MODEL`, **`--served-model-name "Qwen/Qwen3-4B"` hard-coded in both
profiles** [71, 113], `--reasoning-parser qwen3`, `--max-model-len`. `ipc:
host` ("vLLM needs a large /dev/shm") [47]. Mounts `MODELS_DIR→/models` and
the HF cache; port `127.0.0.1:${VLLM_HOST_PORT:-8080}`.

### 9.4 Makefile
`build` (`TORCH=cpu|gpu`), `push`, `up` (db only), `smoke`, `run ARGS=…`,
`shell`, `logs`, `psql`, `down`, `clean`, `serve`, `serve-down`, `serve-logs`,
`serve-shell`, `server-status`, `vllm-gpu`, `vllm-cpu`, `vllm-down`,
`vllm-logs`. `TORCH=gpu` switches profile, service and image for the build,
batch and server targets (`up` and the `vllm-*` targets ignore it).

**Bug: `make build TORCH=gpu` fails on `main`.** Line 21 maps `gpu` → `cu130`,
so the Dockerfile looks for `requirements/torch-cu130.txt`; only
`torch-gpu.txt` exists. [observed: fixed locally on dev by changing line 21
to `gpu`; not committed]

### 9.5 Requirements
`base.txt` pins the model stack (`transformers==4.57.3`, `peft==0.18.0`,
`huggingface-hub==0.36.0`, `accelerate==1.12.0`, `numpy==2.3.3`,
`pydantic==2.12.5`, `librosa==0.11.0`, `soundfile==0.13.1`, …) and leaves
`langgraph`, `openai`, `google-genai`, `silero-vad>=4.0`,
`psycopg[binary]>=3.1`, `fastapi`, `uvicorn`, `python-multipart` loose.
`torch-cpu.txt`: `torch==2.9.1` (CPU index). `torch-gpu.txt`:
`torch==2.9.1+cu130`, `torchvision==0.24.1+cu130`, `triton==3.5.1` (cu130
index) — **GPU hosts need a driver that supports CUDA 13.0**. Root
`requirements.txt` = base + torch-gpu. The Docker image uses Python 3.12.

---

## 10. Model artifacts to stage

| Artifact | Used by | Location in container | Source |
|---|---|---|---|
| `openai/whisper-large-v3` (weights + processor) | transcription | HF cache (`/hf`) | Hugging Face Hub |
| LoRA adapter directory (`adapter_config.json` + weights) | transcription | `/adapters/whisper_trained_LoRa_adaptator` | team artifact (`models/` on the host) |
| `Qwen/Qwen3-4B` | vLLM | `/models/Qwen3-4B` (`VLLM_MODEL`) or the HF id | Hugging Face Hub |
| Silero VAD | preprocessing | inside the `silero-vad` wheel | PyPI |

Offline: set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` and pre-populate
the caches; repo ids then resolve from the local cache.

---

## 11. Environments

### 11.1 Current

| | Dev | Prod (CPU) |
|---|---|---|
| Host | WSL [user] | One RHEL VM [user] |
| Privileges | — | Docker + Compose, Python packages; no sudo/root [user] |
| GPU | 24 GB NVIDIA GPU; `nvidia-smi` reports CUDA 13.1 [observed] | none [user] |
| Network | internet [user] | restricted; Docker Hub allowed [user]; corporate proxy, needs `NO_PROXY` for internal names [observed] |
| How code gets there | `git pull` + `make build` [observed] | image built on dev → `docker save` → copy → `docker load` [observed] |

Last-observed specifics — **verify before relying on any of them**:
- Prod project path `/app/Call_Center_Ai_Agentic`; host ports: server 8099,
  vLLM 8095.
- Dev had uncommitted edits: `Makefile:21` → `gpu`; port bindings without
  `127.0.0.1`; db published on 5435.
- Prod `.env` contained `NO_PROXY`, `NUMBA_CACHE_DIR=/tmp/numba`,
  `CONFIDENCE_THRESHOLD=0.85`, `VAD_MIN/MAX_SEGMENT_SECONDS=5.0/25.0` (which
  truncates transcripts — reverting to 1.0/30.0 was recommended), and a
  duplicated `APP_UID`/`APP_GID` pair.
- vLLM ran as a CPU container in both environments.
- `make down` does not stop services that belong to a profile; to apply a new
  image or `.env` to a long-lived container use
  `docker compose --profile <p> up -d --force-recreate <service>`.

### 11.2 Target [user]
- **VM (CPU)**: the processing/calculation part, PostgreSQL and the backend.
  Docker and any Python packages allowed; no sudo/root.
- **AI Fabric (GPU)**: only the AI parts that need a GPU. Kubernetes +
  JupyterHub; work runs in a Jupyter session, i.e. a pod with a predefined
  profile (resources).

---

## 12. What the code allows for the split

| Component | Coupling today | To run on AI Fabric |
|---|---|---|
| vLLM (classify + sentiment) | Already remote over HTTP (`VLLM_BASE_URL`) | Configuration only — if the pipeline host can reach the pod's endpoint, and `VLLM_MODEL_NAME` equals the served name |
| Whisper transcription | In-process; input is in-memory numpy segments from preprocessing | Either run the whole pipeline process there, or write new code (a transcription service + client). Raw segments are float32 at 16 kHz = 64 KB per second of segment audio |
| Preprocessing (Silero VAD) | In-process, CPU | Stays with whichever process runs transcription, unless split by new code |
| Refinement (Gemini) | Remote HTTPS | The process that runs it needs egress to Google |
| PostgreSQL | Remote via `DATABASE_URL` | Stays on the VM; the pipeline host needs network access to it |
| Files (input, results, logs, archive) | Local paths under `data/` | Must be on the host running the pipeline process |

Options this coupling allows (the choice is the owner's):

- **A — vLLM only on AI Fabric.** Configuration change only. Whisper stays on
  VM CPU, so transcription speed is unchanged.
- **B — whole pipeline process on AI Fabric; DB and files on the VM.** No code
  change in principle, but needs pod → VM database access, a way to exchange
  audio and results, Gemini egress from the pod, and a non-Docker Python
  environment in the pod.
- **C — new GPU transcription service on AI Fabric (plus vLLM); orchestration,
  DB and backend on the VM.** Matches "only GPU parts on AI Fabric", but
  requires new code and a network contract between VM and pod.

Facts that apply to every option:
- One Whisper `generate()` per process at a time (§5.2).
- A transcription failure becomes `MANUAL_REVIEW` and gets archived (§4) — a
  network hop to AI Fabric makes this matter more.
- vLLM needs a large `/dev/shm` (compose uses `ipc: host`).
- Outside Docker, the system libraries the image installs (`ffmpeg`,
  `libsndfile1`, `libgomp1`) must come from somewhere else, and `.env` must
  have no inline comments (§7.1).
- The only authentication the OpenAI client sends is
  `Authorization: Bearer $VLLM_API_KEY`; any other auth an AI Fabric endpoint
  requires would need a code change or a proxy.

---

## 13. Known issues (verified), most deployment-relevant first

1. `make build TORCH=gpu` broken on `main` (`Makefile:21`).
2. Transcription / refinement failures → `MANUAL_REVIEW` and archived [executed].
3. Server: no authentication or TLS; in-memory job registry; shallow `/ready`;
   uploads never cleaned up.
4. `.env` parsers keep inline comments (matters outside Docker).
5. Batch has no resume (uuid `call_id`).
6. `--served-model-name` hard-coded in `compose.vllm.yaml`.
7. Non-deterministic results: Gemini without temperature; vLLM temperature
   0.1 by default.
8. VAD drops — not splits — segments over the maximum; a high minimum
   truncates transcripts.
9. Classification prompt contradicts the schema for "Customer Service Support
   Topic".
10. `WHISPER_MODEL_PATH` defaults to a developer home path.
11. No timeouts on the Gemini or vLLM clients.
12. Dead settings (§7.5), notably `PREPROCESSING_ENABLE`.
13. Four call-audio files are tracked in git (`data/chunks/*.wav`) despite
    `.gitignore` — check whether they contain customer data.
14. One DB connection per operation (fine now; watch with many workers or
    processes).

---

## 14. Where `CLAUDE.md` / `docs/` disagree with the code

| Document says | Code does |
|---|---|
| Processed files move to `data/audio_files/processed_<YYYYMMDD>/` (CLAUDE.md:48) | Archive parent is `--archive-dir` → `$ARCHIVE_DIR` → input dir (runner.py:79-86); `.env.example` sets `/app/data/audio_files_archive` |
| Silero "downloaded once via `torch.hub`; cache via `SILERO_CACHE_DIR`" (CLAUDE.md:54) | `silero-vad` pip package with bundled weights; no download; `SILERO_CACHE_DIR` has no effect (preprocessing.py:48-78) |
| Gemini model `gemini-2.0-flash-exp` (CLAUDE.md:57) | Default `gemini-2.0-flash` (config.py:159); `.env.example` uses `gemini-2.5-flash` |
| Thresholds are changed in `PipelineSettings` (CLAUDE.md:118) | Also env-tunable: `CONFIDENCE_THRESHOLD`, `REFINEMENT_THRESHOLD` (config.py:486-493) |
| MANUAL_REVIEW when refinement fails or has no key (CLAUDE.md:100) | Also when **transcription** fails (§4) |

`docs/` (architecture, runbooks, ADRs) was not re-verified for this handover.
Treat it as background, not fact.

---

## 15. Open questions — ask, do not assume

**AI Fabric**
1. Can a JupyterHub pod expose an HTTP endpoint reachable from the VM
   (Service / Ingress / NodePort / JupyterHub proxy)? With what authentication?
2. Can a pod reach the VM (PostgreSQL port, file shares)?
3. Pod lifetime: idle culling, maximum session length, behavior on restart?
   Can a pod run an unattended long-lived server?
4. Can the profile use a custom container image, or only `pip`/`conda` inside
   the pod? Is Docker available inside?
5. GPU model and VRAM per profile; driver / CUDA version (the GPU pin needs
   CUDA 13.0).
6. Python version in the pod image.
7. Egress from pods: PyPI, `download.pytorch.org`, Hugging Face, Google
   (Gemini)? Internal mirrors?
8. Persistent storage: size and persistence of the home volume (models, HF
   cache, venv)?
9. Shared filesystem between the VM and AI Fabric?
10. Can the pod get a large `/dev/shm` (needed by vLLM)?

**VM**
11. Is it the existing prod VM or a new one? CPU / RAM / disk?
12. Can it open ports to AI Fabric (firewall)?
13. What does "backend" mean here — the existing FastAPI job server, or
    something outside this repository?

**Operations**
14. Batch, server, or both in the target production? How is a run triggered
    (manual, cron, an upstream system)?
15. Where do audio files arrive, and in what volume per day?
16. Is Gemini refinement kept in the target architecture?
17. Which Git repository is the deployment source — GitHub or an internal Git
    server? The VM and AI Fabric may only be able to reach one of them.

---

## 16. Ground-truth commands (run on each environment first)

```bash
# Repository state vs GitHub main
git log --oneline -1 && git status --short && git diff --stat
git fetch origin && git log --oneline HEAD..origin/main

# Effective configuration (never paste secrets: mask key/password values)
grep -vE '^\s*(#|$)' .env | sed -E 's/((KEY|PASSWORD|TOKEN)[A-Z_]*=).*/\1***/'

# Images and containers
docker images | grep -E 'cc-pipeline|vllm|postgres'
docker compose ps
docker compose -f compose.vllm.yaml ps

# Live DB schema vs §8.2
docker compose exec db psql -U cc_pipeline -d call_center -c '\d calls' -c '\d call_results' -c '\d batch_runs'
docker compose exec db psql -U cc_pipeline -d call_center -c \
  "SELECT status, COUNT(*) FROM call_results GROUP BY status;"

# GPU and Python (dev / AI Fabric pod)
nvidia-smi
python3 --version
```
