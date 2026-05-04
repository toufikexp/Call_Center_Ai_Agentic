# ADR 0007 — PostgreSQL persistence layer for results

- Status: Accepted

## Context

At small scale (tens of calls per session), the per-call JSON files under
`data/results/` are sufficient as the durable artifact. Operational needs
that emerge as soon as throughput grows (idempotent resume after a crash,
analytics, batch lifecycle, multi-worker concurrency, comparison across
LoRA adapter versions) require a queryable store, not flat files.

The expected nightly batch is ~10 K calls. At that volume, scanning
thousands of JSON files for "all MANUAL_REVIEW from yesterday" or
"distribution of subjects this week" stops being viable.

## Decision

Add an opt-in PostgreSQL persistence layer (`src/storage/results_store.py`)
that records every pipeline run in three tables:

- `calls` — input registry, one row per audio file.
- `call_results` — one row per pipeline attempt (multiple per call when a
  file is reprocessed).
- `batch_runs` — high-level lifecycle for a batch context.

The store is enabled via `STORAGE_ENABLE=1` and `DATABASE_URL=postgresql://...`.
When disabled, the orchestrator does not even import psycopg, and the
pipeline behaves exactly as before.

JSON output under `data/results/` is **not removed** when the DB is
enabled. Both are written. JSON remains the per-call source of truth that
survives DB outages, schema migrations, and accidental drops.

## Consequences

- **Pro:** SQL analytics for free — subject distribution, latency
  percentiles, error rate, model-version comparison are one-line queries.
- **Pro:** True idempotency. A batch runner can call
  `is_already_processed(call_id)` to skip files that completed in a
  previous run, even if the in-process state was lost.
- **Pro:** Multi-worker safety. PostgreSQL MVCC handles concurrent writers
  without explicit coordination.
- **Pro:** Schema is portable. Adding indexes or columns is mechanical.
- **Con:** New runtime dependency (`psycopg[binary]`) and operational
  surface (PostgreSQL service, credentials, backups).
- **Con:** Schema migrations are manual today. Acceptable while the
  schema is small; revisit if it grows enough to need Alembic.
- **Con:** When the DB is unreachable, the pipeline still runs (failures
  inside the store are logged, not raised) but data is missed. Operators
  must monitor for `Failed to record attempt` log lines and replay from
  JSON if needed.

## Alternatives considered

- **SQLite.** Zero ops, single file. Rejected because the deployment
  target is multi-worker batch processing where MVCC and concurrent
  writers matter. SQLite would have been a fine first step but the
  upgrade path was foreseeable; better to start on Postgres.
- **DuckDB / file-scanning analytics.** Solves the read story but not
  idempotency or multi-worker writes. Postgres handles both.
- **Skip the DB, lean on JSON.** Works at <100 calls/run; does not scale.
- **Object storage (S3/MinIO) + Athena.** Overbuilt for an on-prem call
  center pipeline. Postgres is the simplest tool that fits.

## Related runbooks

- `docs/runbooks/storage.md` — setup, schema, useful queries.
