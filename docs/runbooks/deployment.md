# Runbook — Deployment

One Dockerfile, one `compose.yaml`, one `.env`, one `Makefile`. The same
artifact runs on dev (WSL) and prod (RHEL VM); the difference is the values
in `.env`.

## Mental model

```
                           build a CPU or GPU image
        cc-pipeline:cpu  ←─────────── make build (TORCH=cpu|gpu)
        cc-pipeline:gpu

                              same image, different env
                              ───────────────────────────►
  [dev WSL]            [prod RHEL VM]
  IMAGE=cc-pipeline:gpu (or cpu)         IMAGE=registry/.../cc-pipeline:cpu-<sha>
  DATA_DIR=./data                        DATA_DIR=/var/lib/cc/data
  ADAPTERS_DIR=./models                  ADAPTERS_DIR=/opt/cc/models/adapters
  HF_CACHE_DIR=~/.cache/huggingface      HF_CACHE_DIR=/opt/cc/models/huggingface
  HF_HUB_OFFLINE=0                       HF_HUB_OFFLINE=1
  GEMINI_API_KEY=...                     GEMINI_API_KEY=... (or empty if blocked)
```

## Quick reference

```
make help              # list targets
make build             # build cc-pipeline:cpu
make build TORCH=gpu   # build cc-pipeline:gpu (cu130)
make up                # start postgres
make smoke             # 3-file batch
make run ARGS="--workers 4 --batch-name nightly"
make psql              # open psql in db container
make shell             # bash shell in the app image
make logs              # tail compose logs
make down              # stop (keeps DB volume)
make clean             # stop + wipe DB volume
make push REGISTRY=registry.internal/team TAG=cpu-$(git rev-parse --short HEAD)

# Long-running HTTP API (alternative to one-shot batches)
make serve             # start db + server (detached). Listens on 127.0.0.1:8000
make serve-logs        # tail server logs
make serve-shell       # bash inside the server container
make serve-down        # stop the server (DB stays up)
make server-status     # health + recent jobs
```

## Long-running server (`make serve`)

When you want **models to stay loaded between runs** instead of paying the
~5–10 s reload on every batch, run the pipeline as a long-lived HTTP service.

- One process, one warm `CallAnalysisPipeline`. First job triggers lazy
  model load; subsequent jobs reuse it.
- Async API: `POST /jobs` returns a job id instantly; a worker thread runs
  the pipeline; `GET /jobs/{id}` polls.
- Per-job parallelism is `SERVER_WORKERS` (default 1). Keep it at 1 on CPU
  — Whisper-large-v3 saturates a single thread. Raise on GPU.

### Endpoints

| Method | Path | Notes |
|---|---|---|
| `GET`  | `/health` | Liveness — always returns 200 once the process is up. |
| `GET`  | `/ready`  | Readiness — `ready: true` once the worker pool is initialised. |
| `POST` | `/jobs`   | Submit a job. Two body shapes: JSON `{"audio_path": "..."}` OR `multipart/form-data` with field `audio`. Returns `202` with the job record. |
| `GET`  | `/jobs/{id}` | Fetch a job by id. |
| `GET`  | `/jobs?limit=N` | List the N most recent jobs (default 50). |

### Example

```bash
# Submit a path that already exists under the data mount
curl -X POST -H 'Content-Type: application/json' \
     -d '{"audio_path":"audio_files/399001002190006.wav"}' \
     http://127.0.0.1:8000/jobs

# Upload a file directly
curl -F "audio=@/tmp/sample.wav" http://127.0.0.1:8000/jobs

# Poll
curl http://127.0.0.1:8000/jobs/<job_id>
```

### Where results land (unchanged)

The server doesn't introduce a new output format. Each finished job
produces the same artifacts as the batch runner:

- `data/results/<basename>_<short_id>_result.json` — durable per-call JSON.
- A row in `call_results` (when storage is enabled), grouped under one
  `batch_id` per server lifetime (notes = `server-session-<timestamp>`).

The job record (`GET /jobs/{id}`) carries a small summary and the path to
the JSON file; query the DB or read the JSON for the full result.

### Configuration

| Env var | Default | Purpose |
|---|---|---|
| `SERVER_HOST_PORT` | `8000` | Host port the server is exposed on. |
| `SERVER_WORKERS`   | `1`    | Job worker threads. CPU: 1. GPU: 2–4. |
| `SERVER_LOG_LEVEL` | `info` | uvicorn / python logging level. |

The container always listens on `0.0.0.0:8000`; `SERVER_HOST_PORT` only
affects the host-side bind in compose. Default binding is `127.0.0.1`.


## Dev (WSL) workflow

```bash
cd ~/Call_Center_Ai_Agentic
git pull origin main

# 1. Build the image you'll use (CPU by default; GPU on a GPU box).
make build               # → cc-pipeline:cpu
# or:
make build TORCH=gpu     # → cc-pipeline:gpu

# 2. Env file
cp .env.example .env
# Edit .env:
#   - IMAGE                  match what you built
#   - GEMINI_API_KEY         paste your key
#   - POSTGRES_PASSWORD      anything for local
#   - WHISPER_ADAPTER_PATH   /adapters/<your-folder-under-./models>

# 3. Start postgres
make up

# 4. Smoke test
make smoke
```

That's it. No model staging — `./models` (your adapter) and
`~/.cache/huggingface` (your downloaded Whisper) are bind-mounted directly.

To check results:

```bash
make psql        # then run any SELECT against call_results / batch_runs
ls data/audio_files/processed_$(date +%Y%m%d)/completed/
```

To iterate on code:

```bash
# edit src/...
make build       # ≤30 s, deps layer cached
make smoke
```

## Production (RHEL 9, CPU-only)

### One-time

```bash
# Docker is installed and running. Then:
sudo mkdir -p /var/lib/cc/data/audio_files /opt/cc/models/adapters /opt/cc/models/huggingface
sudo chown -R 1000:1000 /var/lib/cc/data /opt/cc/models        # uid 1000 = appuser
```

### Stage models on the VM

The image expects two things under the bind-mount paths:

```
${HF_CACHE_DIR}/                                      # → /hf in container
└── hub/models--openai--whisper-large-v3/...          # HF cache layout

${ADAPTERS_DIR}/                                      # → /adapters in container
└── whisper_trained_LoRa_adaptator/                   # your LoRA adapter dir
    ├── adapter_config.json
    └── adapter_model.safetensors
```

Copy the Whisper base from your dev `~/.cache/huggingface` to the VM's
`/opt/cc/models/huggingface`. Use whatever your security policy allows
(rsync over SSH, internal share, tar over scp). Same for the adapter
directory. There is no required scripting; the layout is the contract.

### Build, push, pull

```bash
# On dev (internet available)
make build                                                # cc-pipeline:cpu
make push REGISTRY=registry.internal/team TAG=cpu-$(git rev-parse --short HEAD)

# On the prod VM
docker login registry.internal                            # one-time
docker pull registry.internal/team/cc-pipeline:cpu-<sha>
```

### Configure on the VM

```bash
cd /opt/cc/app                                            # wherever this repo is checked out
cp .env.example .env
```

Edit `.env`:

```bash
IMAGE=registry.internal/team/cc-pipeline:cpu-<sha>

DATA_DIR=/var/lib/cc/data
ADAPTERS_DIR=/opt/cc/models/adapters
HF_CACHE_DIR=/opt/cc/models/huggingface

HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1

WHISPER_ADAPTER_PATH=/adapters/whisper_trained_LoRa_adaptator

VLLM_BASE_URL=http://<vllm-host>:8080/v1                  # or http://localhost:8080/v1
GEMINI_API_KEY=                                            # empty if blocked
POSTGRES_PASSWORD=<strong-password>
```

### SELinux note

RHEL with SELinux requires Docker to relabel bind mounts. Edit
`compose.yaml` and append `:Z` to the three `app` volume mounts:

```yaml
    volumes:
      - ${DATA_DIR:-./data}:/app/data:Z
      - ${ADAPTERS_DIR:-./models}:/adapters:Z
      - ${HF_CACHE_DIR:-${HOME}/.cache/huggingface}:/hf:Z
```

Do this only on the SELinux host — adding `:Z` on a non-SELinux dev box
silently breaks mounts.

### Run

```bash
make up                                                   # start postgres
make run ARGS="--workers 3 --batch-name first-run"
```

First run creates the schema (`CREATE TABLE IF NOT EXISTS` + idempotent
`ALTER … ADD COLUMN IF NOT EXISTS`) and processes whatever audio sits in
`/var/lib/cc/data/audio_files`.

### Schedule the nightly batch

Two options. The simpler is a systemd timer that invokes `make run`:

```ini
# /etc/systemd/system/cc-nightly.service
[Service]
Type=oneshot
WorkingDirectory=/opt/cc/app
ExecStart=/usr/bin/make run ARGS=--workers\ 3\ --batch-name\ nightly

# /etc/systemd/system/cc-nightly.timer
[Timer]
OnCalendar=*-*-* 22:00:00
Persistent=true
[Install]
WantedBy=timers.target
```

```bash
sudo systemctl enable --now cc-nightly.timer
```

The batch is idempotent: a previous COMPLETE run won't be reprocessed
(content-hashed `call_id` + `is_already_processed` in the storage layer);
ERROR / unprocessed files stay in the input dir and are retried.

## Rollback

```bash
# Edit .env: IMAGE=registry.internal/team/cc-pipeline:cpu-<previous-sha>
docker pull "$(grep '^IMAGE=' .env | cut -d= -f2-)"
make down && make up
```

Models, data, and DB are untouched. Three minutes.

## Future GPU env

Classification, sentiment, and (eventually) refinement are all URL-driven.
When vLLM moves to a separate GPU env:

```bash
# In .env:
VLLM_BASE_URL=http://gpu-env.internal:8080/v1
```

…then restart the app. No image rebuild, no code change. Moving Whisper
transcription to the GPU env later would be a separate code change (a
remote-Whisper HTTP client) — out of scope for this deployment.

## Common issues

| Symptom | Cause / fix |
|---|---|
| `set POSTGRES_PASSWORD in .env` at start | `.env` missing or `POSTGRES_PASSWORD=` empty. Set it. |
| `Permission denied` on `/app/data` or `/adapters` | Host dirs not owned by uid 1000. `sudo chown -R 1000:1000 <path>`. On RHEL, also confirm `:Z` is on the volume specs. |
| `OSError ... huggingface.co` at startup with `HF_HUB_OFFLINE=1` | Cache incomplete. Verify `ls $HF_CACHE_DIR/hub/models--openai--whisper-large-v3/snapshots/*/`. Re-stage if needed. |
| `Adapter was trained on X but base is Y` warning | `WHISPER_BASE_MODEL_ID` doesn't match the adapter's base. Align them. |
| App can't reach the DB | `db` not healthy yet, or `POSTGRES_PASSWORD` mismatch between `.env` reads. `make up && make logs` and look at the db container. |
| Every call lands in `MANUAL_REVIEW` | Gemini unreachable or key empty (`refinement_score=0`). Expected if the prod VM can't reach Gemini; the local refinement LLM swap will fix it. |
| `host.docker.internal` not resolving | Linux Docker — the `extra_hosts: host.docker.internal:host-gateway` in `compose.yaml` is required and is already there. If it still fails, the host firewall is blocking the bridge gateway. |
| GPU image runs on a CPU box | The cu130 image will try to load CUDA libs and fail. Use the CPU image on CPU hosts. |

## Alternative path — host PostgreSQL

If you'd rather not run Postgres in a container (e.g., the VM has a managed
PG service), comment out the `db` service in `compose.yaml`, override
`DATABASE_URL` in `.env`, and provision the role + database once using
`scripts/setup_postgres.sh`. This is the alternative; the default path is
the container.
