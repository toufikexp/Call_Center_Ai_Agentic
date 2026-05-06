"""Batch runner.

Runs the existing `CallAnalysisPipeline` against a list of audio files
in one process, with a thread pool of workers. Reuses the orchestrator's
batch lifecycle (`start_batch` / `finish_batch`) and the storage
layer's idempotency check (`is_already_processed`). Per-batch logging
goes to `data/logs/batch_<batch_id>.log` when `enable_file_logging` is
True (default).

What this is NOT:
- Not a per-stage queue / cross-call GPU batching system.
- Not a place to add timeouts, retries, or backoff (those would change
  service code, which is out of scope for this introduction).
- Not a multi-process runner. Threads on a single shared
  `CallAnalysisPipeline` keep the model loaded once and avoid the RAM
  cost of duplicating Whisper across processes.
"""
import logging
import os
import shutil
import signal
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config.config import get_settings, set_settings
from src.pipeline import CallAnalysisPipeline
from src.utils.ids import make_call_id


_LOGGERS_TO_CAPTURE = (
    "pipeline.orchestrator",
    "service.preprocessing",
    "service.transcription",
    "service.refinement",
    "service.classification",
    "service.sentiment",
    "storage.results",
    "batch.runner",
    "config",
)

_DEFAULT_INPUT_DIR = "data/audio_files"


class BatchRunner:
    """Process many audio files using a single shared pipeline instance."""

    def __init__(
        self,
        input_dir: Optional[str] = None,
        manifest: Optional[str] = None,
        pattern: str = "*.wav",
        workers: int = 2,
        limit: Optional[int] = None,
        skip_completed: bool = True,
        batch_name: Optional[str] = None,
        dry_run: bool = False,
        archive_dir: Optional[str] = None,
        archive_enabled: bool = True,
    ):
        # Resolve input source: --manifest wins over --input-dir; otherwise
        # CLI arg → $INPUT_DIR → default.
        if manifest:
            self.manifest = manifest
            self.input_dir = None
        else:
            self.manifest = None
            self.input_dir = (
                input_dir
                or os.getenv("INPUT_DIR")
                or _DEFAULT_INPUT_DIR
            )

        # Archive parent (where processed_<YYYYMMDD>/ is created):
        # CLI arg → $ARCHIVE_DIR → fall back to the resolved input dir
        # (matches the "data/audio_files/processed_…" example layout).
        self.archive_dir = (
            archive_dir
            or os.getenv("ARCHIVE_DIR")
            or self.input_dir
            or _DEFAULT_INPUT_DIR
        )
        self.archive_enabled = archive_enabled

        self.pattern = pattern
        self.workers = max(1, workers)
        self.limit = limit
        self.skip_completed = skip_completed
        self.batch_name = batch_name
        self.dry_run = dry_run

        self._stop_event = threading.Event()
        self._logger = self._make_logger()

        # Computed once when run() starts so all moves go to the same
        # date-stamped folder for this batch.
        self._completed_dir: Optional[Path] = None
        self._review_dir: Optional[Path] = None
        # Serialise file moves so concurrent workers don't race on
        # mkdir / replace for the same target dir.
        self._move_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    @staticmethod
    def _make_logger() -> logging.Logger:
        logger = logging.getLogger("batch.runner")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _setup_batch_log_file(
        self, batch_id: str, settings
    ) -> Optional[logging.FileHandler]:
        if not settings.pipeline.enable_file_logging:
            return None
        logs_dir = Path(settings.pipeline.logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"batch_{batch_id}.log"
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
        )
        for name in _LOGGERS_TO_CAPTURE:
            logging.getLogger(name).addHandler(handler)
        self._logger.info(f"📁 Per-batch log: {log_path}")
        return handler

    @staticmethod
    def _teardown_batch_log_file(handler: Optional[logging.FileHandler]) -> None:
        if handler is None:
            return
        for name in _LOGGERS_TO_CAPTURE:
            logging.getLogger(name).removeHandler(handler)
        handler.close()

    # ------------------------------------------------------------------
    # Input resolution
    # ------------------------------------------------------------------
    def _resolve_inputs(self) -> List[str]:
        if self.manifest:
            with open(self.manifest, "r", encoding="utf-8") as f:
                paths = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.lstrip().startswith("#")
                ]
        else:
            base = Path(self.input_dir)
            if not base.is_dir():
                raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
            paths = sorted(str(p) for p in base.glob(self.pattern) if p.is_file())

        if self.limit is not None:
            paths = paths[: self.limit]
        return paths

    # ------------------------------------------------------------------
    # Archive (move processed files out of the input dir)
    # ------------------------------------------------------------------
    def _build_archive_paths(self) -> Tuple[Path, Path]:
        """Compute completed/ and manual_review/ paths for this batch.

        Folder name is `processed_<YYYYMMDD>` so multiple batches in one
        day share one archive. Sub-folders are created lazily on first
        move.
        """
        date_str = datetime.now().strftime("%Y%m%d")
        base = Path(self.archive_dir) / f"processed_{date_str}"
        return (base / "completed", base / "manual_review")

    def _archive_one(self, audio_path: str, status: str) -> None:
        """Move a single processed file to the right archive folder.

        Policy:
          COMPLETE        → completed/
          MANUAL_REVIEW   → manual_review/
          (anything else) → leave in place (ERROR / EXCEPTION / ID_ERROR /
                            SKIPPED retry on the next batch run)
        """
        if not self.archive_enabled:
            return
        if status == "COMPLETE":
            target_dir = self._completed_dir
        elif status == "MANUAL_REVIEW":
            target_dir = self._review_dir
        else:
            return

        if target_dir is None:
            return  # archive disabled for this run

        src = Path(audio_path)
        if not src.exists():
            # Already moved (e.g. by a previous run) or never existed.
            return

        try:
            with self._move_lock:
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / src.name
                # os.replace is atomic on the same filesystem and
                # silently overwrites if the target exists.
                os.replace(str(src), str(target))
            self._logger.info(
                f"📦 archived {src.name} → {target_dir.parent.name}/{target_dir.name}/"
            )
        except OSError as e:
            self._logger.warning(
                f"Archive move failed for {src.name}: {e}; file left in place"
            )

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------
    def _install_signal_handlers(self) -> None:
        def _handler(signum, frame):
            self._logger.warning(
                f"Signal {signum} received — finishing in-flight calls and stopping"
            )
            self._stop_event.set()

        signal.signal(signal.SIGTERM, _handler)
        # SIGINT is delivered by Ctrl-C; keep the handler so cleanup happens.
        signal.signal(signal.SIGINT, _handler)

    # ------------------------------------------------------------------
    # Per-call work
    # ------------------------------------------------------------------
    def _process_one(
        self, pipeline: CallAnalysisPipeline, audio_path: str
    ) -> Tuple[str, str]:
        """Process one file. Always returns (call_id, status); never raises."""
        try:
            call_id = make_call_id(audio_path)
        except Exception as e:
            self._logger.error(
                f"Failed to compute call_id for {audio_path}: {e}"
            )
            return ("?", "ID_ERROR")

        # Idempotency: only when storage is enabled
        store = getattr(pipeline, "_results_store", None)
        if self.skip_completed and store is not None:
            try:
                if store.is_already_processed(call_id):
                    self._logger.info(f"[skip] already COMPLETE: {call_id}")
                    return (call_id, "SKIPPED")
            except Exception as e:
                self._logger.warning(
                    f"is_already_processed lookup failed for {call_id}: {e}; will reprocess"
                )

        try:
            state = pipeline.run(audio_path, call_id=call_id)
            status = state["result"].status.value
            self._logger.info(f"[done] {status}: {call_id}")
            return (call_id, status)
        except Exception as e:
            self._logger.exception(
                f"[fail] uncaught exception for {audio_path}: {e}"
            )
            return (call_id, "EXCEPTION")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> int:
        settings = get_settings()
        set_settings(settings)

        try:
            paths = self._resolve_inputs()
        except FileNotFoundError as e:
            self._logger.error(str(e))
            return 1

        self._logger.info(f"📦 Resolved {len(paths)} input file(s)")
        if not paths:
            self._logger.warning("No inputs to process; exiting.")
            return 0

        if self.dry_run:
            self._logger.info("(dry-run) files that would be processed:")
            for p in paths:
                print(p)
            return 0

        self._install_signal_handlers()

        # One pipeline instance, shared across worker threads.
        # Construction loads no models (lazy init); the first call per
        # service does the heavy lifting and the result is cached.
        pipeline = CallAnalysisPipeline(settings)

        batch_id = pipeline.start_batch(
            file_count=len(paths), notes=self.batch_name
        )
        if batch_id is None:
            # Storage disabled — synthesize a local id for the log file only.
            batch_id = uuid.uuid4().hex[:12]
            self._logger.info(
                f"📦 Batch {batch_id} (storage disabled — DB tracking off)"
            )

        # Compute archive paths once so all moves in this batch land in the
        # same date-stamped folder, even if the run spans midnight.
        if self.archive_enabled:
            self._completed_dir, self._review_dir = self._build_archive_paths()
            self._logger.info(
                f"📁 Archive: COMPLETE → {self._completed_dir}, "
                f"MANUAL_REVIEW → {self._review_dir}"
            )
        else:
            self._logger.info("📁 Archive: disabled (--no-archive)")

        log_handler = self._setup_batch_log_file(batch_id, settings)

        counts: Dict[str, int] = {}

        try:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futures = {}
                for p in paths:
                    if self._stop_event.is_set():
                        self._logger.warning(
                            "Stop requested — not submitting remaining files"
                        )
                        break
                    futures[ex.submit(self._process_one, pipeline, p)] = p

                for fut in as_completed(futures):
                    audio_path = futures[fut]
                    call_id, status = fut.result()
                    counts[status] = counts.get(status, 0) + 1
                    # Move the file out of the input dir if applicable.
                    self._archive_one(audio_path, status)
                    if self._stop_event.is_set():
                        # Cancel anything that hasn't started yet; running
                        # calls finish naturally.
                        for other in futures:
                            if not other.done():
                                other.cancel()
        finally:
            try:
                pipeline.finish_batch()
            except Exception as e:
                self._logger.error(f"finish_batch failed: {e}")
            self._teardown_batch_log_file(log_handler)

        self._logger.info("=" * 60)
        self._logger.info(f"📋 BATCH SUMMARY: {batch_id}")
        self._logger.info("=" * 60)
        if not counts:
            self._logger.info("  (no files processed)")
        else:
            for k in sorted(counts):
                self._logger.info(f"  {k}: {counts[k]}")
        self._logger.info("=" * 60)

        # Exit code reflects whether anything failed (excluding skips).
        failures = counts.get("ERROR", 0) + counts.get("EXCEPTION", 0) + counts.get("ID_ERROR", 0)
        return 1 if failures and not self._stop_event.is_set() else 0
