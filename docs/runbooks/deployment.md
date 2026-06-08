# Runbook — Deployment (Docker)

Two flows:

- **Dev test on WSL** — verify the image locally against your existing models
  and HF cache before shipping. See "Dev test (WSL)" below.
- **Production on RHEL 9** — build → registry → prod VM with pre-staged
  offline models. The rest of this document.

---

## Dev test (WSL)

Test the exact image that will ship to prod, but against your existing local
files — no model staging, internet allowed, CPU inference (the image is CPU
torch; that's fine for verification).

```bash
# 1. Build the image (once; rebuilds are fast after code-only changes)
docker build -t cc-pipeline:local .

# 2. Dev env file
cp deploy/env.dev.example .env.dev
#    edit .env.dev → set GEMINI_API_KEY (and WHISPER_ADAPTER_PATH if your
#    adapter folder name differs from the default)

# 3. Start Postgres
docker compose -f compose.dev.yaml up -d db

# 4. Run a 3-file smoke batch
docker compose -f compose.dev.yaml run --rm app --limit 3 --batch-name dev-test

# 5. Inspect
docker compose -f compose.dev.yaml exec db \
    psql -U cc_pipeline -d call_center -c \
    "SELECT call_id, status, subject, audio_duration_s FROM call_results ORDER BY finished_at DESC LIMIT 5;"
ls data/audio_files/processed_$(date +%Y%m%d)/completed/ 2>/dev/null

# 6. Stop (keeps DB data; add -v to wipe)
docker compose -f compose.dev.yaml down
```

`compose.dev.yaml` mounts `./models`, `~/.cache/huggingface`, and `./data`
straight into the container, so it uses your already-downloaded Whisper and
your existing adapter. Iterate by editing `src/`, re-running
`docker build -t cc-pipeline:local .` (≈10–30 s, deps layer cached), and
re-running step 4.

---

## Production deployment (RHEL 9, CPU-only)

How to ship the pipeline to the production VM. The image is built on a
machine with internet (your dev box), pushed to the internal registry, and
pulled on the prod VM. Models are pre-staged separately and bind-mounted —
they are never baked into the image.

## Topology

```
[dev box: WSL Ubuntu, internet]
   build CPU image  ─push─►  [internal registry]  ─pull─►  [prod VM: RHEL 9, CPU]
   pack models      ─copy────────────────────────────────►  /opt/cc/models
                                                              docker compose up
                                                              ├── app  (this image)
                                                              └── db   (postgres:16)
```

What lives where:

| Thing | Location | In image? |
|---|---|---|
| Python deps + system libs (ffmpeg, libsndfile) | image | yes |
| Application code (`src/`, `main.py`) | image | yes |
| Silero VAD weights | image (baked at build) | yes |
| Whisper base model (~3 GB) | `/opt/cc/models/huggingface` on VM | no — bind-mounted |
| LoRA adapter (~35 MB) | `/opt/cc/models/adapters/<v>` on VM | no — bind-mounted |
| Audio in/out, logs, archive | `<CC_DATA_DIR>` on VM → `/app/data` | no — bind-mounted |
| PostgreSQL data | docker named volume `pgdata` | no — persistent volume |

## One-time prerequisites on the prod VM

- Docker is installed and active (`docker --version && systemctl is-active docker`).
- The internal registry is reachable and you can `docker login` to it
  (ask ops for hostname + credentials).
- A data directory exists, e.g. `/var/lib/cc/data`, with the input subdir:
  ```bash
  sudo mkdir -p /var/lib/cc/data/audio_files
  sudo chown -R 1000:1000 /var/lib/cc/data   # uid 1000 = appuser in the image
  ```
- A models directory exists: `sudo mkdir -p /opt/cc/models && sudo chown -R 1000:1000 /opt/cc/models`.

## Step 1 — Build + push (dev box)

```bash
docker login <registry>     # one-time

REGISTRY=<registry>/<team> TAG=$(git rev-parse --short HEAD) \
    ./scripts/build_and_push.sh
```

This builds the CPU image (uses `deploy/requirements.cpu.txt`), bakes in the
Silero VAD weights, tags it `:<sha>` and `:prod-latest`, and pushes both.
**Pin the `:<sha>` tag** in the prod compose env — `prod-latest` is for
convenience only and makes rollback ambiguous.

## Step 2 — Pre-stage models (dev box → prod VM)

On the dev box (internet available):

```bash
BASE_MODEL=openai/whisper-large-v3 \
ADAPTER_SRC=/path/to/whisper-large-v3-lora-algerian_v7 \
OUT=cc-models.tar.gz \
    ./scripts/stage_models.sh pack
```

Transfer `cc-models.tar.gz` to the prod VM via your approved channel, then on
the VM:

```bash
MODELS_DIR=/opt/cc/models TARBALL=cc-models.tar.gz \
    ./scripts/stage_models.sh unpack
```

Result on the VM:
```
/opt/cc/models/huggingface/   ← Whisper base (HF cache layout)
/opt/cc/models/adapters/<v>/  ← LoRA adapter
```

New adapter version later: pack just the adapter (or copy the folder
directly), drop it under `/opt/cc/models/adapters/`, update
`WHISPER_ADAPTER_PATH` in `deploy/.env.prod`, and restart the app container.
No image rebuild.

## Step 3 — Configure (prod VM)

```bash
cp deploy/env.prod.example deploy/.env.prod
# edit deploy/.env.prod:
#   POSTGRES_PASSWORD  → a real strong password
#   WHISPER_ADAPTER_PATH → matches the staged adapter folder
#   VLLM_BASE_URL      → where vLLM runs (localhost today; GPU env host later)
#   GEMINI_API_KEY     → leave empty if the VM can't reach Gemini
```

Export the deploy-time variables compose needs:

```bash
export CC_IMAGE=<registry>/<team>/cc-pipeline:<sha>
export CC_DATA_DIR=/var/lib/cc/data
export CC_MODELS_DIR=/opt/cc/models
# POSTGRES_* are read from deploy/.env.prod by the env_file directive,
# but compose also interpolates POSTGRES_PASSWORD for the app DATABASE_URL,
# so export it too:
export POSTGRES_PASSWORD=$(grep '^POSTGRES_PASSWORD=' deploy/.env.prod | cut -d= -f2-)
```

## Step 4 — Pull + start

```bash
docker pull "$CC_IMAGE"
docker compose --env-file deploy/.env.prod up -d db
# wait for db healthy, then a one-off batch (exits when done):
docker compose --env-file deploy/.env.prod run --rm app --workers 3 --batch-name "first-run"
```

The first app run creates the schema (`CREATE TABLE IF NOT EXISTS` + the
idempotent `ALTER` migrations) and processes whatever is in
`/var/lib/cc/data/audio_files`.

## Step 5 — Verify

```bash
# DB rows
docker compose --env-file deploy/.env.prod exec db \
    psql -U cc_pipeline -d call_center -c \
    "SELECT batch_id, file_count, success_count, error_count, review_count FROM batch_runs ORDER BY started_at DESC LIMIT 3;"

# Archived files moved out of input
ls /var/lib/cc/data/audio_files/processed_$(date +%Y%m%d)/completed/ | head

# Logs
docker compose --env-file deploy/.env.prod logs app | tail -40
```

You want to see `✅ ResultsStore ready (PostgreSQL)`, `✅ Silero VAD loaded`,
`Loading base Whisper`, and per-call `[done] COMPLETE` lines.

## Scheduling the nightly batch

Run the one-off `docker compose run --rm app ...` from cron or a systemd timer
on the VM. Example systemd timer (`/etc/systemd/system/cc-nightly.{service,timer}`):

```ini
# cc-nightly.service
[Service]
Type=oneshot
WorkingDirectory=/opt/cc/app          # where this repo's compose.yaml lives
Environment=CC_IMAGE=<registry>/<team>/cc-pipeline:<sha>
Environment=CC_DATA_DIR=/var/lib/cc/data
Environment=CC_MODELS_DIR=/opt/cc/models
ExecStart=/usr/bin/docker compose --env-file deploy/.env.prod run --rm app --workers 3 --batch-name nightly
```

```ini
# cc-nightly.timer
[Timer]
OnCalendar=*-*-* 22:00:00
Persistent=true
[Install]
WantedBy=timers.target
```

```bash
sudo systemctl enable --now cc-nightly.timer
```

The batch is idempotent: if a run is interrupted, the next run skips files
already marked COMPLETE (deterministic content-hashed `call_id` +
`is_already_processed`). ERROR / unprocessed files stay in the input dir and
are retried automatically.

## Rollback

```bash
export CC_IMAGE=<registry>/<team>/cc-pipeline:<previous-sha>
docker pull "$CC_IMAGE"
docker compose --env-file deploy/.env.prod up -d
```

Image is the only thing that changes; models, data, and DB are untouched.

## Future: moving GPU components to a separate env

Classification + sentiment (and later refinement) are URL-driven. When a GPU
env hosts vLLM, change one line in `deploy/.env.prod`:

```bash
VLLM_BASE_URL=http://gpu-env.internal:8080/v1
```

and restart the app container. No image rebuild, no code change. Whisper
transcription staying on the CPU VM is unchanged; moving it to the GPU env
would be a future code change (a remote-Whisper HTTP client), out of scope
for this deployment.

## SELinux note (RHEL)

The bind mounts in `compose.yaml` use the `:Z` suffix so Docker relabels them
with the right SELinux context. If you see `Permission denied` reading
`/app/data` or `/opt/cc/models` despite correct Unix ownership, that's
SELinux — confirm the `:Z` is present, or check `ausearch -m avc -ts recent`.

## Common issues

| Symptom | Cause / fix |
|---|---|
| `OSError ... huggingface.co` at startup | Model not staged, or `HF_HOME` wrong. Re-run stage_models unpack; confirm `HF_HOME=/opt/cc/models/huggingface`. |
| `Adapter was trained on X but base is Y` warning | `WHISPER_BASE_MODEL_ID` doesn't match the adapter's base. Align it. |
| App can't reach DB | `db` not healthy yet, or `DATABASE_URL` host wrong. Compose sets it to the `db` service automatically; don't override in the env file. |
| `Permission denied` on mounts | Host dir not owned by uid 1000, or missing `:Z`. `chown -R 1000:1000` + keep `:Z`. |
| Every call MANUAL_REVIEW | Gemini unreachable (`refinement_score=0`). Expected if `GEMINI_API_KEY` empty; wire local refinement LLM when ready. |
