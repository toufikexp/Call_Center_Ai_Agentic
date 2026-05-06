"""
PostgreSQL persistence layer for pipeline results.

Backend: `psycopg` (v3) over a PostgreSQL server. All tables and indexes are
created on first connection if missing (`CREATE ... IF NOT EXISTS`), so a
fresh database needs only `CREATE DATABASE call_center;` before the
pipeline runs — no migration tool required for the initial schema.

Idempotency
-----------
- `record_attempt` upserts the `calls` row by `call_id` and inserts a new
  row in `call_results` for every attempt. Re-running the same audio file
  therefore creates a *new* result row but does not duplicate the call row.
- `is_already_processed(call_id)` returns True if a previous attempt landed
  in COMPLETE (configurable). Use this from a batch runner to skip files
  that already finished successfully.

Opt-in
------
The store is opt-in via `StorageSettings.enable`. When disabled, the
orchestrator does not construct a ResultsStore at all and JSON output
under `data/results/` is unaffected — JSON remains the per-call durable
artifact regardless of DB state.

Concurrency
-----------
PostgreSQL handles concurrent writers natively via MVCC. Each
`record_attempt` call opens its own connection and runs the upsert + insert
inside a single transaction (`with psycopg.connect(...)` commits on
success, rolls back on exception). For a multi-worker batch runner you can
safely call this from many threads or processes pointing at the same
database.
"""
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Optional, Sequence

from src.core.state import CallAnalysisResult


SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    call_id        TEXT PRIMARY KEY,
    audio_path     TEXT NOT NULL,
    duration_s     REAL,
    channel_count  SMALLINT,
    received_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_calls_audio_path ON calls(audio_path);

CREATE TABLE IF NOT EXISTS call_results (
    id                          BIGSERIAL PRIMARY KEY,
    call_id                     TEXT NOT NULL REFERENCES calls(call_id),
    batch_id                    TEXT,
    status                      TEXT NOT NULL,
    transcript                  TEXT,
    refined_transcript          TEXT,
    confidence_score            REAL,
    refinement_score            REAL,
    subject                     TEXT,
    sub_subject                 TEXT,
    classification_confidence   REAL,
    satisfaction_score          REAL,
    sentiment_label             TEXT,
    sentiment_reasoning         TEXT,
    whisper_adapter_version     TEXT,
    error_message               TEXT,
    segments                    JSONB,
    started_at                  TIMESTAMPTZ NOT NULL,
    finished_at                 TIMESTAMPTZ NOT NULL,
    duration_s                  REAL
);

CREATE INDEX IF NOT EXISTS idx_results_call_id        ON call_results(call_id);
CREATE INDEX IF NOT EXISTS idx_results_batch_id       ON call_results(batch_id);
CREATE INDEX IF NOT EXISTS idx_results_status         ON call_results(status);
CREATE INDEX IF NOT EXISTS idx_results_subject        ON call_results(subject);
CREATE INDEX IF NOT EXISTS idx_results_finished_at    ON call_results(finished_at);
CREATE INDEX IF NOT EXISTS idx_results_adapter        ON call_results(whisper_adapter_version);

CREATE TABLE IF NOT EXISTS batch_runs (
    batch_id        TEXT PRIMARY KEY,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    file_count      INTEGER,
    success_count   INTEGER NOT NULL DEFAULT 0,
    error_count     INTEGER NOT NULL DEFAULT 0,
    review_count    INTEGER NOT NULL DEFAULT 0,
    notes           TEXT
);
"""


# Idempotent in-place migrations.
#
# CREATE TABLE IF NOT EXISTS does nothing when the table already exists, so
# columns that were added to the schema *after* a database was first
# deployed never appear there. The block below uses
# ADD COLUMN IF NOT EXISTS (PostgreSQL ≥ 9.6) to backfill any column that's
# missing on an older instance. New databases run these statements as
# no-ops; older databases get the missing columns added without a separate
# migration tool.
#
# Convention: whenever you add a column to one of the CREATE TABLE blocks
# above, also append the matching ALTER here. Keep types identical to the
# CREATE statement. Only NULLABLE columns (or columns with a DEFAULT) are
# safe to add on a populated table — never add NOT NULL without DEFAULT
# this way; that requires a real migration.
MIGRATIONS = """
ALTER TABLE calls         ADD COLUMN IF NOT EXISTS duration_s    REAL;
ALTER TABLE calls         ADD COLUMN IF NOT EXISTS channel_count SMALLINT;

ALTER TABLE call_results  ADD COLUMN IF NOT EXISTS classification_confidence REAL;
ALTER TABLE call_results  ADD COLUMN IF NOT EXISTS sentiment_label           TEXT;
ALTER TABLE call_results  ADD COLUMN IF NOT EXISTS sentiment_reasoning       TEXT;
ALTER TABLE call_results  ADD COLUMN IF NOT EXISTS whisper_adapter_version   TEXT;
ALTER TABLE call_results  ADD COLUMN IF NOT EXISTS segments                  JSONB;

ALTER TABLE batch_runs    ADD COLUMN IF NOT EXISTS notes TEXT;
"""


def _make_logger() -> logging.Logger:
    logger = logging.getLogger("storage.results")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class ResultsStore:
    """PostgreSQL-backed persistence for call analysis results."""

    def __init__(self, database_url: str):
        self._logger = _make_logger()
        self._validate_url(database_url)
        self._database_url = database_url

        # Lazy import: psycopg is only required when storage is enabled.
        try:
            import psycopg
            from psycopg.types.json import Jsonb
        except ImportError as e:
            raise RuntimeError(
                "psycopg (v3) is required for the ResultsStore. "
                "Install with: pip install 'psycopg[binary]>=3.1'"
            ) from e

        self._psycopg = psycopg
        self._Jsonb = Jsonb
        self._initialize_schema()

    # ------------------------------------------------------------------
    # URL handling
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_url(url: str) -> None:
        if not url:
            raise ValueError(
                "DATABASE_URL is required when storage is enabled. "
                "Example: postgresql://user:pass@localhost:5432/call_center"
            )
        if not url.startswith(("postgresql://", "postgres://")):
            raise ValueError(
                f"Unsupported database URL scheme: {url!r}. "
                "Expected 'postgresql://...' or 'postgres://...'."
            )

    # ------------------------------------------------------------------
    # Connection / schema
    # ------------------------------------------------------------------
    @contextmanager
    def _connect(self) -> Iterator["psycopg.Connection"]:  # noqa: F821
        """Yield a psycopg connection. Auto-commits on success, rolls back on
        exception (the `with psycopg.connect(...)` semantics)."""
        with self._psycopg.connect(self._database_url) as conn:
            yield conn

    def _initialize_schema(self) -> None:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    # Create tables and indexes if missing (no-op on
                    # already-deployed databases).
                    cur.execute(SCHEMA)
                    # Backfill any columns added after the database was
                    # first created. Idempotent on up-to-date schemas.
                    cur.execute(MIGRATIONS)
            self._logger.info("✅ ResultsStore ready (PostgreSQL)")
        except Exception as e:
            # Re-raise — without a working schema the store is unusable.
            self._logger.error(f"Failed to initialize schema: {e}")
            raise

    # ------------------------------------------------------------------
    # Batch lifecycle
    # ------------------------------------------------------------------
    def start_batch(
        self,
        batch_id: str,
        file_count: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> str:
        """Insert a `batch_runs` row. Returns the batch_id for chaining."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO batch_runs (batch_id, file_count, notes)
                           VALUES (%s, %s, %s)""",
                    (batch_id, file_count, notes),
                )
        return batch_id

    def finish_batch(self, batch_id: str) -> None:
        """Stamp finished_at on `batch_runs`.

        Counters (success_count / error_count / review_count) are kept
        live by `record_attempt` inside the same transaction that
        inserts each `call_results` row, so this method only marks the
        batch as done and does not need to re-aggregate.
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE batch_runs
                           SET finished_at = now()
                         WHERE batch_id = %s""",
                    (batch_id,),
                )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record_attempt(
        self,
        call_id: str,
        audio_path: str,
        result: CallAnalysisResult,
        batch_id: Optional[str] = None,
        started_at: Optional[datetime] = None,
        finished_at: Optional[datetime] = None,
    ) -> None:
        """Upsert `calls` and insert a new `call_results` row.

        DB issues are logged but not raised — the pipeline must keep going
        even if PostgreSQL is unreachable. The JSON output under
        `data/results/` always contains the same data and is the source of
        truth for replay.
        """
        started = started_at or datetime.now(timezone.utc)
        finished = finished_at or datetime.now(timezone.utc)
        duration = (finished - started).total_seconds()

        segments_json = self._Jsonb([s.model_dump() for s in result.segments])

        # Source-the-truth values for calls.duration_s / channel_count.
        # When the audio failed to load (e.g. corrupt file), result fields
        # are 0; treat those as NULL so analytics queries don't average
        # over fake zeros.
        audio_duration_s = result.audio_duration_s if result.audio_duration_s > 0 else None
        channel_count = result.channel_count if result.channel_count > 0 else None

        # Map status to the matching batch_runs counter column. A status
        # that doesn't correspond to a counter (e.g. PENDING, IN_PROGRESS)
        # leaves all three columns alone.
        counter_column = {
            "COMPLETE": "success_count",
            "ERROR": "error_count",
            "MANUAL_REVIEW": "review_count",
        }.get(result.status.value)

        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    # Upsert calls — pull duration / channel_count forward
                    # so old rows get backfilled when re-processed.
                    cur.execute(
                        """INSERT INTO calls (call_id, audio_path, duration_s, channel_count)
                               VALUES (%s, %s, %s, %s)
                           ON CONFLICT (call_id) DO UPDATE
                               SET audio_path    = EXCLUDED.audio_path,
                                   duration_s    = COALESCE(EXCLUDED.duration_s, calls.duration_s),
                                   channel_count = COALESCE(EXCLUDED.channel_count, calls.channel_count)""",
                        (call_id, audio_path, audio_duration_s, channel_count),
                    )

                    # Insert the per-attempt row.
                    cur.execute(
                        """INSERT INTO call_results (
                                call_id, batch_id, status,
                                transcript, refined_transcript,
                                confidence_score, refinement_score,
                                subject, sub_subject, classification_confidence,
                                satisfaction_score, sentiment_label, sentiment_reasoning,
                                whisper_adapter_version, error_message,
                                segments, started_at, finished_at, duration_s
                           ) VALUES (
                                %s, %s, %s,
                                %s, %s,
                                %s, %s,
                                %s, %s, %s,
                                %s, %s, %s,
                                %s, %s,
                                %s, %s, %s, %s
                           )""",
                        (
                            call_id,
                            batch_id,
                            result.status.value,
                            result.transcript,
                            result.refined_transcript,
                            result.confidence_score,
                            result.refinement_score,
                            result.subject,
                            result.sub_subject,
                            result.classification_confidence,
                            result.satisfaction_score,
                            result.sentiment_label,
                            result.sentiment_reasoning,
                            result.whisper_adapter_version,
                            result.error_message,
                            segments_json,
                            started,
                            finished,
                            duration,
                        ),
                    )

                    # Live batch metrics: bump the matching counter in the
                    # same transaction so the batch_runs row reflects
                    # progress as workers finish, not only at end-of-batch.
                    # PostgreSQL row-level locking serialises concurrent
                    # workers updating the same batch_id correctly.
                    if batch_id and counter_column:
                        cur.execute(
                            f"""UPDATE batch_runs
                                    SET {counter_column} = {counter_column} + 1
                                  WHERE batch_id = %s""",
                            (batch_id,),
                        )
        except Exception as e:
            self._logger.error(f"Failed to record attempt for {call_id}: {e}")

    # ------------------------------------------------------------------
    # Idempotency
    # ------------------------------------------------------------------
    def is_already_processed(
        self,
        call_id: str,
        batch_id: Optional[str] = None,
        statuses: Sequence[str] = ("COMPLETE",),
    ) -> bool:
        """True if `call_id` already has a result with one of the given statuses.

        Default: returns True only if a previous attempt completed cleanly.
        MANUAL_REVIEW and ERROR runs are considered re-tryable.
        """
        if not statuses:
            return False
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    if batch_id:
                        cur.execute(
                            """SELECT 1 FROM call_results
                                WHERE call_id = %s
                                  AND batch_id = %s
                                  AND status = ANY(%s)
                                LIMIT 1""",
                            (call_id, batch_id, list(statuses)),
                        )
                    else:
                        cur.execute(
                            """SELECT 1 FROM call_results
                                WHERE call_id = %s
                                  AND status = ANY(%s)
                                LIMIT 1""",
                            (call_id, list(statuses)),
                        )
                    return cur.fetchone() is not None
        except Exception as e:
            self._logger.warning(f"is_already_processed lookup failed: {e}")
            return False
