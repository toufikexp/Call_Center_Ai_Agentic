"""In-process job store + worker pool for the HTTP server.

Design notes
------------
- ONE long-lived `CallAnalysisPipeline` is shared across all jobs. Models
  load lazily on the first job and stay resident — that's the whole point
  of running the server vs. the one-shot batch container.
- A `ThreadPoolExecutor` runs jobs. Default `max_workers=1` because
  Whisper-large-v3 on CPU is the bottleneck and the transcription service's
  internal `_gen_lock` already serialises generate() calls. Raising it on
  GPU is fine — set `SERVER_WORKERS` in `.env`.
- The pipeline already records to the JSON output dir AND to Postgres (when
  storage is enabled), so this module deliberately does NOT duplicate that.
  Job records here are *metadata*; the durable artifacts are the JSON file
  under `data/results/` and the `call_results` row.
- Each job runs as its own batch (file_count=1): _run() calls start_batch
  before the pipeline and finish_batch after, so every `batch_runs` row gets
  a `finished_at` when the job completes — matching the one-shot batch
  runner. The batch_id is passed explicitly into `run(batch_id=...)` so
  concurrent workers don't race on the pipeline's shared `_current_batch_id`.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config.config import get_settings
from src.pipeline import CallAnalysisPipeline


logger = logging.getLogger("server.jobs")


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    MANUAL_REVIEW = "manual_review"
    ERROR = "error"


@dataclass
class JobRecord:
    """Metadata for one submitted job. Result data lives on disk / in DB.

    `job_id` is passed straight through to the pipeline as the `call_id`, so
    `call_id == job_id` once the job starts running. To find a job in the
    database: `SELECT * FROM call_results WHERE call_id = '<job_id>';`.
    """
    job_id: str
    audio_path: str
    status: JobStatus = JobStatus.QUEUED
    submitted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    # Equals job_id (set from the pipeline result once it completes).
    call_id: Optional[str] = None
    # Small summary lifted from the pipeline result for cheap polling.
    summary: Optional[Dict[str, Any]] = None
    # Path to the JSON result file under data/results/, when produced.
    result_path: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


class JobStore:
    """Thread-safe job registry + worker pool.

    One instance per server process. Owns the shared pipeline and the
    executor. Construction is cheap; the pipeline lazy-loads on the first
    `submit()`.
    """

    def __init__(self, workers: int = 1, max_history: int = 500):
        self._lock = threading.Lock()
        self._jobs: Dict[str, JobRecord] = {}
        # Order of submission, for /jobs listing without scanning the dict
        # each time.
        self._order: List[str] = []
        self._max_history = max(50, int(max_history))

        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(workers)),
            thread_name_prefix="job-worker",
        )

        self._settings = get_settings()
        self._pipeline = CallAnalysisPipeline(self._settings)
        # Each job runs as its OWN batch (file_count=1), opened and finished
        # inside _run(). This mirrors the one-shot batch runner — every
        # batch_runs row gets a finished_at when its job completes — instead
        # of one ever-open server-lifetime batch that was only stamped at
        # shutdown.
        logger.info(
            "Job store ready (workers=%d, output_dir=%s)",
            workers,
            self._settings.pipeline.output_dir,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def shutdown(self, wait: bool = True) -> None:
        logger.info("Shutting down job store (wait=%s)", wait)
        # No server-lifetime batch to finish — batches are per-job now.
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------
    def submit(self, audio_path: str) -> JobRecord:
        job_id = uuid.uuid4().hex
        record = JobRecord(job_id=job_id, audio_path=audio_path)
        with self._lock:
            self._jobs[job_id] = record
            self._order.append(job_id)
            self._evict_if_full()
        self._executor.submit(self._run, job_id)
        return record

    def _evict_if_full(self) -> None:
        """Drop oldest finished jobs to bound memory. Holds `self._lock`."""
        overflow = len(self._order) - self._max_history
        if overflow <= 0:
            return
        # Prefer to evict finished jobs first. Walk from oldest forward.
        kept: List[str] = []
        dropped = 0
        terminal = {JobStatus.COMPLETE, JobStatus.MANUAL_REVIEW, JobStatus.ERROR}
        for jid in self._order:
            if dropped < overflow:
                rec = self._jobs.get(jid)
                if rec is not None and rec.status in terminal:
                    self._jobs.pop(jid, None)
                    dropped += 1
                    continue
            kept.append(jid)
        self._order = kept

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            rec = self._jobs.get(job_id)
            # Return a copy so callers can't mutate live state.
            return JobRecord(**asdict(rec)) if rec else None

    def list(self, limit: int = 50) -> List[JobRecord]:
        with self._lock:
            ids = self._order[-max(1, limit):][::-1]
            return [JobRecord(**asdict(self._jobs[i])) for i in ids if i in self._jobs]

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _run(self, job_id: str) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return
            rec.status = JobStatus.RUNNING
            rec.started_at = datetime.now(timezone.utc).isoformat()
            audio_path = rec.audio_path

        t0 = time.time()
        # One batch per job (file_count=1), finished in the `finally` below so
        # batch_runs.finished_at is always stamped — same lifecycle as the
        # one-shot batch runner. The batch_id is passed explicitly into run()
        # so concurrent workers don't race on the shared _current_batch_id.
        batch_id = self._pipeline.start_batch(file_count=1, notes=f"server-job:{job_id}")
        try:
            # Pass the API job_id straight through as the pipeline call_id so
            # the value the client polls on IS the primary key of the durable
            # row in `calls` / `call_results`. One id, traceable end to end —
            # no separate job_id↔call_id mapping to keep.
            state = self._pipeline.run(audio_path, call_id=job_id, batch_id=batch_id)
            result = state["result"]
            status_val = result.status.value
            mapped = _STATUS_MAP.get(status_val, JobStatus.ERROR)

            summary = {
                "call_id": result.call_id,
                "status": status_val,
                "confidence": float(result.confidence_score or 0.0),
                "refinement_score": float(result.refinement_score or 0.0),
                "subject": result.subject,
                "sub_subject": result.sub_subject,
                "satisfaction_score": float(result.satisfaction_score or 0.0),
                "transcript_length": len(result.transcript or ""),
            }
            result_path = _result_json_path(self._settings, audio_path, result.call_id)

            with self._lock:
                rec2 = self._jobs.get(job_id)
                if rec2 is None:
                    return
                rec2.status = mapped
                rec2.finished_at = datetime.now(timezone.utc).isoformat()
                rec2.call_id = result.call_id
                rec2.summary = summary
                rec2.result_path = result_path
                if status_val == "ERROR":
                    rec2.error = result.error_message or "pipeline reported ERROR"

            logger.info(
                "Job %s finished in %.2fs → %s (call_id=%s)",
                job_id, time.time() - t0, status_val, result.call_id,
            )
        except FileNotFoundError as e:
            self._mark_error(job_id, f"audio not found: {e}")
        except Exception as e:  # pragma: no cover — defensive
            logger.exception("Job %s failed: %s", job_id, e)
            self._mark_error(job_id, f"{type(e).__name__}: {e}")
        finally:
            # Stamp batch_runs.finished_at for this job's batch, always.
            try:
                self._pipeline.finish_batch(batch_id)
            except Exception as e:  # pragma: no cover — defensive
                logger.warning("finish_batch failed for job %s: %s", job_id, e)

    def _mark_error(self, job_id: str, message: str) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return
            rec.status = JobStatus.ERROR
            rec.finished_at = datetime.now(timezone.utc).isoformat()
            rec.error = message


_STATUS_MAP = {
    "COMPLETE": JobStatus.COMPLETE,
    "MANUAL_REVIEW": JobStatus.MANUAL_REVIEW,
    "ERROR": JobStatus.ERROR,
    "PENDING": JobStatus.ERROR,
    "IN_PROGRESS": JobStatus.ERROR,
}


def _result_json_path(settings, audio_path: str, call_id: str) -> Optional[str]:
    """Reconstruct the path that orchestrator._save_node writes to.

    Returns the relative path if it exists, None otherwise.
    """
    if not call_id:
        return None
    audio_basename = Path(audio_path).stem
    unique = call_id.split("_")[-1] if "_" in call_id else call_id[-8:]
    path = Path(settings.pipeline.output_dir) / f"{audio_basename}_{unique}_result.json"
    return str(path) if path.exists() else str(path)
