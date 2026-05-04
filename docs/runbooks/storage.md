# Runbook — PostgreSQL persistence

When enabled, the pipeline records every call attempt to PostgreSQL in
addition to the per-call JSON file under `data/results/`. JSON remains the
source of truth; the DB exists for analytics, idempotency, and batch
lifecycle tracking.

## Enable / disable

The store is **opt-in**. Two env vars (in `.env`):

```bash
STORAGE_ENABLE=1
DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/call_center
```

When `STORAGE_ENABLE` is unset or `0`, the orchestrator does not touch
PostgreSQL. JSON output under `data/results/` is unaffected either way.

## One-time setup

1. Install the driver (already pinned in `requirements.txt`):

   ```bash
   pip install -r requirements.txt
   ```

2. Make sure a PostgreSQL server is reachable. Local install (Linux):

   ```bash
   sudo apt-get install -y postgresql
   sudo systemctl start postgresql
   ```

3. Create a role and a database:

   ```bash
   sudo -u postgres psql <<'SQL'
   CREATE USER cc_pipeline WITH PASSWORD 'change_me';
   CREATE DATABASE call_center OWNER cc_pipeline;
   GRANT ALL PRIVILEGES ON DATABASE call_center TO cc_pipeline;
   SQL
   ```

4. Set `DATABASE_URL` in `.env`:

   ```bash
   DATABASE_URL=postgresql://cc_pipeline:change_me@localhost:5432/call_center
   STORAGE_ENABLE=1
   ```

5. Run the pipeline once. Tables are created automatically the first time
   `ResultsStore` initializes (`CREATE TABLE IF NOT EXISTS`). You should see
   `✅ ResultsStore ready (PostgreSQL)` in the logs.

## Schema

Three tables. `calls` is the input registry, `call_results` is one row per
attempt, `batch_runs` is the batch lifecycle.

```
calls
├── call_id        TEXT PRIMARY KEY
├── audio_path     TEXT
├── duration_s     REAL
├── channel_count  SMALLINT
└── received_at    TIMESTAMPTZ

call_results
├── id                          BIGSERIAL PRIMARY KEY
├── call_id                     TEXT REFERENCES calls
├── batch_id                    TEXT       (nullable; non-batch runs leave NULL)
├── status                      TEXT       (COMPLETE / MANUAL_REVIEW / ERROR)
├── transcript                  TEXT
├── refined_transcript          TEXT
├── confidence_score            REAL
├── refinement_score            REAL
├── subject                     TEXT
├── sub_subject                 TEXT
├── classification_confidence   REAL
├── satisfaction_score          REAL
├── sentiment_label             TEXT
├── sentiment_reasoning         TEXT
├── whisper_adapter_version     TEXT
├── error_message               TEXT
├── segments                    JSONB
├── started_at                  TIMESTAMPTZ
├── finished_at                 TIMESTAMPTZ
└── duration_s                  REAL

batch_runs
├── batch_id        TEXT PRIMARY KEY
├── started_at      TIMESTAMPTZ
├── finished_at     TIMESTAMPTZ
├── file_count      INTEGER
├── success_count   INTEGER
├── error_count     INTEGER
├── review_count    INTEGER
└── notes           TEXT
```

Indexes: `call_results.call_id`, `batch_id`, `status`, `subject`,
`finished_at`, `whisper_adapter_version`. `calls.audio_path`.

## Useful queries

### Overall throughput by day

```sql
SELECT date_trunc('day', finished_at) AS day,
       count(*) AS calls,
       count(*) FILTER (WHERE status = 'COMPLETE')      AS complete,
       count(*) FILTER (WHERE status = 'MANUAL_REVIEW') AS manual,
       count(*) FILTER (WHERE status = 'ERROR')         AS errors
FROM call_results
GROUP BY 1
ORDER BY 1 DESC;
```

### Subject distribution

```sql
SELECT subject, sub_subject, count(*) AS n,
       avg(satisfaction_score) AS avg_sat
FROM call_results
WHERE status = 'COMPLETE'
GROUP BY 1, 2
ORDER BY n DESC;
```

### Compare adapter versions

```sql
SELECT whisper_adapter_version,
       count(*) AS calls,
       avg(confidence_score)  AS avg_conf,
       avg(refinement_score)  AS avg_refine,
       avg(duration_s)        AS avg_seconds
FROM call_results
GROUP BY 1
ORDER BY calls DESC;
```

### Per-batch summary

```sql
SELECT batch_id, started_at, finished_at,
       file_count, success_count, error_count, review_count
FROM batch_runs
ORDER BY started_at DESC
LIMIT 10;
```

### Find calls flagged for manual review with low refinement

```sql
SELECT call_id, refinement_score, confidence_score, finished_at
FROM call_results
WHERE status = 'MANUAL_REVIEW'
  AND refinement_score < 0.3
  AND finished_at > now() - interval '24 hours'
ORDER BY finished_at DESC;
```

### Per-segment text for one call

`segments` is JSONB. Each row has `channel`, `start_ms`, `end_ms`, `text`,
`confidence`.

```sql
SELECT s->>'channel'    AS channel,
       (s->>'start_ms')::int AS start_ms,
       s->>'text'       AS text,
       (s->>'confidence')::real AS confidence
FROM call_results, jsonb_array_elements(segments) s
WHERE call_id = 'call_399001002190030_abcdef12'
ORDER BY (s->>'start_ms')::int;
```

## Operational notes

- **Failures are non-fatal.** If PostgreSQL is unreachable mid-run, the
  store logs the error and the pipeline finishes anyway. JSON file output
  always succeeds.
- **Schema migrations** are not handled by code. To add a column later,
  write the `ALTER TABLE` by hand (or adopt Alembic when the schema gets
  more complex).
- **Vacuum / analyze** PostgreSQL handles automatically. Nothing to do at
  10 K rows/day. Consider a dedicated `VACUUM ANALYZE` schedule above
  ~10 M rows.
- **Backups.** The DB is queryable state, not the source of truth — JSON
  files are. `pg_dump call_center > snapshot.sql` weekly is enough.
- **Multi-worker safety.** `record_attempt` opens its own connection per
  call and runs the upsert + insert in one transaction. Many workers can
  point at the same database without a lock.

## Switching back to JSON-only

Set `STORAGE_ENABLE=0` in `.env` (or remove it). No schema cleanup needed
on the DB side; existing rows stay.
