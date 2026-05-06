# Runbook — Batch processing

`python -m src.batch run` processes many audio files in one process,
keeping the model loaded once across all of them. The pipeline behaviour
per file is identical to `python main.py <file>` — same services, same
output JSON, same DB rows.

## When to use which

| Use | Entry |
|---|---|
| Single ad-hoc file (dev, debugging) | `python main.py <audio_path>` |
| Many files in one go | `python -m src.batch run …` |

The batch runner does not exist to add new pipeline behaviour; it only
amortises the model cold-start (~30–60 s per `python main.py`) across
many files and adds idempotent resume.

## Quick start

```bash
# Use $INPUT_DIR from .env (default: data/audio_files)
python -m src.batch run

# Override the input directory on the CLI
python -m src.batch run --input-dir /mnt/calls/inbox --pattern '*.wav'

# Explicit list of files
python -m src.batch run --manifest /path/to/list.txt
```

`list.txt`: one absolute audio path per line. Blank lines and lines
starting with `#` are ignored. When `--manifest` is given, `--input-dir`
and `$INPUT_DIR` are ignored.

## Input directory resolution

Precedence (first non-empty wins):

1. `--input-dir <path>` on the CLI
2. `$INPUT_DIR` from `.env`
3. Default: `data/audio_files`

`--manifest` short-circuits all three.

## Archive layout (default behaviour)

After each call the runner moves the audio file out of the input
directory based on the result status:

```
ARCHIVE_DIR/
└── processed_<YYYYMMDD>/
    ├── completed/         ← COMPLETE
    └── manual_review/     ← MANUAL_REVIEW
```

`ARCHIVE_DIR` resolves the same way as input: CLI `--archive-dir` →
`$ARCHIVE_DIR` → `--input-dir` (so the example layout
`data/audio_files/processed_…/…` works out of the box). Multiple batches
on the same day share the same `processed_<YYYYMMDD>/` folder.

What does **not** get moved:

- `ERROR`, `EXCEPTION`, `ID_ERROR` — left in the input directory so the
  next batch retries them automatically. The deterministic `call_id` +
  `is_already_processed` check ensures they don't re-process if they
  later succeed.
- `SKIPPED` (idempotency match) — file isn't in the input directory
  in the first place; it was moved by the batch that completed it.

To turn the archive off:

```bash
python -m src.batch run --no-archive
```

Files then stay in the input directory regardless of status.

## Useful flags

| Flag | Meaning |
|---|---|
| `--workers N` | Parallel worker threads. Default `2`. Increase carefully. |
| `--limit N` | Process only the first N files. Smoke-test friendly. |
| `--no-skip-completed` | Re-process files even if a previous attempt landed in COMPLETE. Default skips them. |
| `--batch-name "nightly-2026-05-04"` | Free-text label persisted to `batch_runs.notes`. |
| `--dry-run` | Resolve and print the input list, do nothing. |
| `--input-dir PATH` | Override `$INPUT_DIR`. |
| `--archive-dir PATH` | Override `$ARCHIVE_DIR`. Defaults to the resolved input dir. |
| `--no-archive` | Don't move processed files; leave them in place. |

## How idempotent resume works

For each file the runner computes a deterministic `call_id`:

```
call_id = call_<basename>_<sha256(file)[:12]>
```

Before processing, the runner asks the storage layer
(`ResultsStore.is_already_processed(call_id)`) whether a prior attempt
already landed in `COMPLETE`. If yes, the file is skipped with a log
line `[skip] already COMPLETE: <call_id>`.

This means:
- Re-running the same batch is safe — completed files are skipped, the
  rest are processed.
- A crashed batch can be restarted; it picks up where it left off.
- `MANUAL_REVIEW` and `ERROR` runs are **not** considered done; they
  will be retried on the next run. To re-attempt a `COMPLETE` file
  (e.g. with a new prompt), pass `--no-skip-completed`.

If `STORAGE_ENABLE=0`, the skip check is a no-op. Every file is
processed; the JSON output under `data/results/` is the only durable
artifact.

## Logging

Per-batch log file at `data/logs/batch_<batch_id>.log` collects the
output of the orchestrator + every service for the run. Stdout still
shows the same content.

The file is created when `PipelineSettings.enable_file_logging` is True
(the default). Set it to False in code if you need stdout only.

## Inspect a batch

With `STORAGE_ENABLE=1`:

```bash
psql -h 127.0.0.1 -p 5432 -U cc_pipeline -d call_center

-- Recent batches
SELECT batch_id, started_at, finished_at, file_count,
       success_count, error_count, review_count, notes
FROM batch_runs
ORDER BY started_at DESC LIMIT 10;

-- Calls in a specific batch
SELECT call_id, status, confidence_score, refinement_score,
       subject, satisfaction_score, finished_at
FROM call_results
WHERE batch_id = '<batch_id>'
ORDER BY finished_at;

-- Failures only
SELECT call_id, status, error_message
FROM call_results
WHERE batch_id = '<batch_id>' AND status IN ('ERROR', 'MANUAL_REVIEW');
```

## Stopping a batch cleanly

`SIGTERM` (or Ctrl-C) sets a stop event. The runner stops submitting new
files; in-flight calls finish; `finish_batch()` runs; the process exits
cleanly. Already-completed files are durable in JSON + DB; the missing
files become re-tries on the next run.

If the process is killed hard (`kill -9`), in-flight calls leave no row
behind — they will be re-tried on the next run via the idempotency
check (since they never reached `COMPLETE`).

## Concurrency notes

The runner uses a thread pool with **one shared `CallAnalysisPipeline`
instance** (model loaded once). One shared model saves RAM compared to
a multi-process pool that would duplicate the ~3 GB Whisper checkpoint
per worker.

What overlaps across threads:

- Audio I/O (soundfile / librosa)
- Feature extraction
- Token decoding
- Network I/O to Gemini and vLLM
- Postgres writes

What is **serialised** by per-service locks (concurrent inference on a
shared torch model is not safe — Silero's LSTM hidden state and
Whisper's KV cache are mutated during a forward pass and crash under
concurrent calls):

- Silero VAD `_get_speech_timestamps` call
- Whisper `model.generate` call

Net throughput from `--workers > 1` is "API stages of call A run while
inference for call B runs", not full parallel inference. On a CPU-only
box where transcription dominates, the gain is modest. Tune accordingly:

| Host | Suggested `--workers` |
|---|---|
| CPU-only, 4–8 cores | 1–2 |
| CPU-only, 16+ cores | 2–4 |
| GPU box, single GPU | 2–4 (GPU work serialises on the CUDA queue regardless) |

Going too high on a CPU box just thrashes — Python threads contend on
the GIL during the orchestrator's sync code, and Whisper is the bottleneck.

## What this runner deliberately does NOT do

- No timeouts or retries on Gemini/vLLM. A slow API can stall a worker;
  a transient 5xx still becomes an `ERROR` on the call.
- No cross-call GPU batching. Each call's segments are batched within
  the call (`WHISPER_BATCH_SIZE`); no inter-call batching.
- No metrics export, no healthcheck command.
- No prompt or threshold changes.

These are separable improvements, not part of the batch entry point.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Every file is "skipped" but you wanted them processed | A previous batch already marked them COMPLETE in the DB. Use `--no-skip-completed` or wipe rows for those `call_id`s. |
| `is_already_processed lookup failed` warning followed by reprocess | Database is unreachable; the runner intentionally proceeds rather than block on a flaky DB. Fix the DB and resume. |
| Batch summary shows `EXCEPTION` count > 0 | An uncaught error in `pipeline.run`. Check the per-batch log for the traceback. |
| `enable_file_logging` is True but no batch log file | Check `PipelineSettings.logs_dir` exists and is writable. |
| `Archive move failed for X.wav: …` | Cross-filesystem move (input on one mount, archive on another) — `os.replace` requires same FS. Either keep them on the same mount or use `--no-archive` and move externally. |
| File reappears in input directory after a successful run | Status was `ERROR`/`EXCEPTION` (left in place by design for retry), not `COMPLETE`. Check the batch log. |
