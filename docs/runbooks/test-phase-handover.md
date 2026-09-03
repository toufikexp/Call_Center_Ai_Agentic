# Test Phase — Team Handover & Reference

**Scope:** supporting the Customer Care team during functional testing of the
Call Center AI pipeline on the **production server (CPU mode)**.

**Audience:** our team (technical support during the test phase). Everything
here is copy-pasteable and assumes you are on the prod host as the app user.

```bash
cd /app/Call_Center_Ai_Agentic     # all commands below start here
```

---

## 1. What the system does

One audio call in → structured analysis out:

```
call audio → transcription (Darija) → AI cleanup → quality gate
           → call subject (category) → satisfaction score → database + JSON
```

Every processed call produces:

| Output | Where |
|---|---|
| JSON result file | `data/results/<audio-name>_<id>_result.json` |
| Database row | table `call_results` |
| Pipeline log | `data/logs/batch_<batch-id>.log` |
| The audio file itself | moved to `data/audio_files_archive/processed_<YYYYMMDD>/` |

---

## 2. Environment facts (production)

| Item | Value |
|---|---|
| Project path | `/app/Call_Center_Ai_Agentic` |
| Mode | **CPU** (no GPU on prod yet — pending IT roll-out) |
| API server port | `8099` |
| vLLM (local LLM) port | `8095` |
| Database | PostgreSQL in container, service name `db` |
| Input folder | `data/audio_files/` |
| Archive folder | `data/audio_files_archive/` |
| Results folder | `data/results/` |
| Logs folder | `data/logs/` |
| Config file | `.env` (never commit it) |

Internet access is restricted; the only external call is the refinement API.

---

## 3. Before every test session — health check

```bash
docker compose ps                                 # db + server: healthy?
curl -s localhost:8095/v1/models | head -c 80     # vLLM answering?
curl -s localhost:8099/health                     # API server answering?
ls data/audio_files/*.wav | wc -l                 # files ready?
df -h . | tail -1                                 # disk space left?
```

Green = `db` healthy, vLLM returns JSON, `/health` returns `{"status":"ok"}`.
If any fails, see **Section 8 — Troubleshooting** before running tests.

---

## 4. Running a test

### 4.1 Put the audio files in place

```bash
cp /path/to/customer-care-files/*.wav data/audio_files/
ls -lh data/audio_files/
```

**Accepted format:** `.wav` by default. `.mp3` works but needs an explicit
pattern (see 4.4). Stereo files are strongly preferred — the pipeline splits
channel 0 = agent, channel 1 = client. Mono works but speaker labels degrade.

### 4.2 One file (quick end-to-end check)

```bash
make run ARGS="--limit 1 --no-archive --batch-name smoke-test"
```

`--no-archive` keeps the file in the input folder so it can be re-run.

### 4.3 All files in the input folder

```bash
make run ARGS="--workers 2 --batch-name cc-test-$(date +%Y%m%d)"
```

Keep `--workers 2` on CPU — more workers just compete for the same cores.

### 4.4 Useful options

| Option | Purpose |
|---|---|
| `--limit N` | process only the first N files |
| `--batch-name NAME` | label the run (appears in `batch_runs.notes`) |
| `--no-archive` | leave files in the input folder after processing |
| `--dry-run` | list the files that would be processed, process nothing |
| `--pattern '*.mp3'` | process mp3 instead of wav |
| `--workers N` | parallel workers (2 on CPU) |
| `--manifest FILE` | process an explicit list of paths (one per line) |

### 4.5 Long runs — do not lose them to an SSH drop

```bash
nohup make run ARGS="--workers 2 --batch-name cc-test-$(date +%Y%m%d)" \
  > data/logs/console_$(date +%Y%m%d).log 2>&1 &

tail -f data/logs/batch_*.log        # follow progress
```

---

## 5. Reading the results

### 5.1 Live progress

```bash
tail -f data/logs/batch_*.log
grep -E 'BATCH SUMMARY|completed in|Subject:' data/logs/batch_*.log | tail -20
```

### 5.2 Database

```bash
# last 10 processed calls
docker compose exec db psql -U cc_pipeline -d call_center -c \
 "SELECT call_id, status, subject, sub_subject, satisfaction_score,
         sentiment_label, ROUND(duration_s) AS secs, finished_at
  FROM call_results ORDER BY finished_at DESC LIMIT 10;"

# summary of one test run
docker compose exec db psql -U cc_pipeline -d call_center -c \
 "SELECT status, COUNT(*) FROM call_results r
  JOIN batch_runs b USING (batch_id)
  WHERE b.notes = 'cc-test-20260713' GROUP BY status;"
```

### 5.3 JSON result file

```bash
ls -t data/results/ | head -5
cat data/results/<file>_result.json | python3 -m json.tool | head -40
```

Fields the Customer Care team cares about:

| Field | Meaning |
|---|---|
| `transcript` | raw transcription, speaker-tagged |
| `refined_transcript` | cleaned-up version (used for classification) |
| `subject` / `sub_subject` | call category and sub-category |
| `satisfaction_score` | 0–10 (0 = not analysed) |
| `sentiment_label` | POSITIVE / NEUTRAL / NEGATIVE |
| `sentiment_reasoning` | one-sentence justification |
| `confidence_score` | transcription confidence |
| `audio_duration_s`, `channel_count` | input properties |
| `status` | see below |

### 5.4 Status meanings

| Status | Meaning | Action |
|---|---|---|
| `COMPLETE` | fully processed, all stages succeeded | none — this is the good case |
| `MANUAL_REVIEW` | processed, but quality gate flagged it (low transcription confidence or low refinement quality) | **normal and expected** — a human should read it |
| `ERROR` | a stage failed | check the log, see Section 8 |

`MANUAL_REVIEW` is **not** a failure — it is the system saying "I am not
confident, do not trust this automatically." Explain this to Customer Care so
they do not report it as a bug.

---

## 6. Expected processing time (CPU)

Production runs on CPU, which is roughly **25× slower than GPU**. As a rough
guide, expect **several minutes per call**, dominated by transcription.

Practical consequences to communicate to Customer Care:

- A 20-file test batch is a **coffee-break-plus** operation, not instant.
- Do not report "it is stuck" before checking Section 8.1.
- Real-time processing is not possible in this configuration; it is a
  batch/after-the-fact analysis tool until GPUs are deployed.

---

## 7. Using the API instead of batch (optional)

The HTTP server is an alternative to batch for one-off calls. Models stay
warm, so the second and later requests skip model loading.

```bash
# health
curl -s localhost:8099/health
curl -s localhost:8099/ready

# submit a file already on the server
curl -s -X POST localhost:8099/jobs \
  -H 'Content-Type: application/json' \
  -d '{"audio_path": "audio_files/399001002190001.wav"}'

# submit by upload
curl -s -X POST localhost:8099/jobs -F "audio=@/path/to/file.wav"

# check a job (job_id comes from the submit response)
curl -s localhost:8099/jobs/<job_id> | python3 -m json.tool

# list recent jobs
curl -s "localhost:8099/jobs?limit=20" | python3 -m json.tool
```

`job_id` and `call_id` are the same value, so a job can be traced straight
into the `call_results` table.

---

## 8. Troubleshooting

### 8.1 "It is stuck"

Almost always it is not — CPU transcription is simply slow, and the
transcription stage prints nothing while it works.

```bash
docker stats --no-stream          # batch container burning CPU (>100%) = working
tail -f data/logs/batch_*.log     # last line tells you the current stage
```

Container at ~0% CPU for several minutes = genuinely stuck; escalate.

### 8.2 Transcript is far too short for the call length

Cause: the voice-activity settings are dropping speech segments. Check:

```bash
grep -E 'VAD_MIN_SEGMENT_SECONDS|VAD_MAX_SEGMENT_SECONDS' .env
```

Correct values are `1.0` and `30.0`. Anything higher (e.g. `5.0` / `25.0`)
silently discards short conversational turns and produces mutilated
transcripts. Fix the `.env`, then re-run — no rebuild needed.

### 8.3 vLLM not answering (classification / sentiment fail)

```bash
curl -s localhost:8095/v1/models
docker compose -f compose.vllm.yaml ps
docker compose -f compose.vllm.yaml logs --tail=50 vllm-cpu
```

Restart if needed:
```bash
docker compose -f compose.vllm.yaml --profile cpu up -d --force-recreate vllm-cpu
```

### 8.4 Everything lands in MANUAL_REVIEW

Usually the refinement API key is missing or expired — refinement fails,
its score is 0, and the quality gate routes every call to manual review.

```bash
grep GEMINI_API_KEY .env | cut -c1-25      # is it set?
grep -i 'refinement' data/logs/batch_*.log | tail -5
```

### 8.5 Permission denied writing logs/results

```bash
ls -ln data data/logs data/results
```
The container user must be able to write these folders. If not, escalate —
do not change ownership blindly.

### 8.6 Out of disk space

```bash
df -h .
du -sh data/*
```
Old segment dumps and archived audio are the usual culprits. `data/segments/`
should not exist in prod (`PREPROCESSING_SAVE_SEGMENTS=0`).

---

## 9. Stopping / restarting

```bash
# stop a running batch gracefully (finishes files in flight)
docker ps | grep app-run
docker stop <container-name>

# after editing .env: batch needs NOTHING — the next run picks it up.
# the API server must be recreated (restart is NOT enough):
docker compose --profile cpu up -d --force-recreate server
```

Never `docker kill` a running batch — it leaves the run un-finalised in the
database.

---

## 10. Known limitations — tell Customer Care up front

| Limitation | Explanation |
|---|---|
| Slow on CPU | ~25× slower than GPU; GPU deployment pending with IT |
| Same file can give slightly different results | The cleanup stage is a cloud AI model that is not perfectly deterministic; borderline calls can shift category between runs |
| Some categories overlap | "Customer Service Support Topic" vs "Customer Service Process Related" is the known weak boundary — taxonomy refinement with Customer Care is planned |
| Satisfaction is AI-estimated | It is a trend indicator, not a survey result |
| Model confidence values | The classifier reports high confidence on nearly everything — do not use it to judge correctness |
| Mono audio | Loses agent/client separation and degrades speaker labelling |

---

## 11. What NOT to do

- Do **not** edit code on the production server — prod deploys tested images only.
- Do **not** commit `.env` or any audio file to git.
- Do **not** change VAD or threshold values in `.env` without telling the team
  — results between test runs stop being comparable.
- Do **not** delete files from `data/audio_files_archive/` during the test
  phase; they are the evidence trail for the results.
- Do **not** `docker kill` a batch (see Section 9).

---

## 12. Escalation

Collect these before escalating:

```bash
docker compose ps
tail -100 data/logs/batch_<batch-id>.log
docker compose exec db psql -U cc_pipeline -d call_center -c \
 "SELECT call_id, status, error_message FROM call_results
  ORDER BY finished_at DESC LIMIT 5;"
```

Include: what was run (exact command), how many files, when, and the batch id.

---

## 13. Related documentation

| Document | Content |
|---|---|
| `README.md` | project overview and setup |
| `docs/architecture.md` | pipeline design and data flow |
| `docs/data-contracts.md` | input/output schema at every stage |
| `docs/runbooks/deployment.md` | deployment and redeploy procedure |
| `docs/runbooks/configuration.md` | every `.env` setting explained |
| `docs/runbooks/threshold-tuning.md` | tuning the quality gate |
| `docs/runbooks/taxonomy-changes.md` | changing categories / sub-categories |
| `docs/runbooks/incident-response.md` | production incident procedure |
