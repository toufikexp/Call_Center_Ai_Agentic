"""FastAPI app for the long-lived pipeline server.

Endpoints
---------
GET  /health        Liveness — fixed response.
GET  /ready         Readiness — pipeline constructed and worker pool up.
POST /jobs          Submit a job. Accepts EITHER:
                      - JSON: {"audio_path": "<path under /app/data>"}
                      - multipart/form-data with a file field named "audio".
                    Returns 202 with the job record.
GET  /jobs/{id}     Get a job record.
GET  /jobs          List recent jobs (newest first, ?limit=N).

Notes
-----
- Audio paths are constrained to the data root (`PipelineSettings.audio_dir`
  or /app/data) to avoid arbitrary filesystem reads.
- Uploads land under `<data_root>/audio_files_uploads/<job_id>__<filename>`.
- Job state lives in memory only; the durable artifacts are the JSON file
  under data/results/ and the row in Postgres. A restart loses pending
  queued jobs that were never picked up — re-submit them.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config.config import get_settings
from src.server.jobs import JobRecord, JobStatus, JobStore


logger = logging.getLogger("server.app")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class SubmitJobRequest(BaseModel):
    audio_path: str = Field(
        ...,
        description="Path under the data root (e.g. '/app/data/audio_files/foo.wav' "
                    "or 'audio_files/foo.wav').",
    )


class JobResponse(BaseModel):
    job_id: str
    status: str
    audio_path: str
    submitted_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    call_id: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None
    result_path: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def from_record(cls, rec: JobRecord) -> "JobResponse":
        d = rec.to_dict()
        return cls(**d)


class JobListResponse(BaseModel):
    jobs: List[JobResponse]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    settings = get_settings()
    workers = int(os.getenv("SERVER_WORKERS", "1"))
    data_root = _resolve_data_root(settings)
    upload_dir = data_root / "audio_files_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title="Call Center AI Agentic Pipeline",
        version="1.0",
        description="Long-running job API around the LangGraph pipeline.",
    )

    # Lazy-built so multiple uvicorn workers (if anyone ever sets >1) each
    # have their own. Default is one worker / one process.
    store: Dict[str, JobStore] = {}

    @app.on_event("startup")
    def _startup() -> None:
        logger.info("Starting job store with %d worker(s)", workers)
        store["jobs"] = JobStore(workers=workers)
        logger.info("Data root: %s", data_root)
        logger.info("Upload dir: %s", upload_dir)

    @app.on_event("shutdown")
    def _shutdown() -> None:
        js = store.get("jobs")
        if js is not None:
            js.shutdown(wait=True)

    # ----- Health / readiness ---------------------------------------------
    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> Dict[str, Any]:
        js = store.get("jobs")
        return {"ready": js is not None, "workers": workers}

    # ----- Job submission -------------------------------------------------
    @app.post("/jobs", status_code=202, response_model=JobResponse)
    async def submit_job(request: Request) -> JobResponse:
        """Submit a job.

        Use EITHER a JSON body with `audio_path` OR a multipart upload with
        a file field named `audio`. The two cannot be mixed in one endpoint
        signature (a `File()` param forces FastAPI to treat the whole body
        as multipart), so we branch on Content-Type from the raw request.
        """
        js = store.get("jobs")
        if js is None:
            raise HTTPException(503, "Job store not ready")

        content_type = (request.headers.get("content-type") or "").lower()

        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            audio = form.get("audio")
            if not isinstance(audio, UploadFile):
                raise HTTPException(
                    400, "multipart request must include a file field named 'audio'."
                )
            audio_path = _save_upload(audio, upload_dir)
        elif content_type.startswith("application/json"):
            try:
                payload = await request.json()
            except Exception:
                raise HTTPException(400, "Body is not valid JSON.")
            audio_path_raw = (payload or {}).get("audio_path")
            if not audio_path_raw:
                raise HTTPException(400, "JSON body must include 'audio_path'.")
            audio_path = _resolve_input_path(str(audio_path_raw), data_root)
        else:
            raise HTTPException(
                400,
                "Send JSON (Content-Type: application/json, body {\"audio_path\": ...}) "
                "or a multipart upload (field 'audio').",
            )

        if not Path(audio_path).exists():
            raise HTTPException(404, f"Audio file not found: {audio_path}")

        rec = js.submit(audio_path)
        logger.info("Submitted job %s → %s", rec.job_id, audio_path)
        return JobResponse.from_record(rec)

    # ----- Job lookup -----------------------------------------------------
    @app.get("/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str) -> JobResponse:
        js = store.get("jobs")
        if js is None:
            raise HTTPException(503, "Job store not ready")
        rec = js.get(job_id)
        if rec is None:
            raise HTTPException(404, f"Unknown job: {job_id}")
        return JobResponse.from_record(rec)

    @app.get("/jobs", response_model=JobListResponse)
    def list_jobs(limit: int = 50) -> JobListResponse:
        js = store.get("jobs")
        if js is None:
            raise HTTPException(503, "Job store not ready")
        recs = js.list(limit=limit)
        return JobListResponse(jobs=[JobResponse.from_record(r) for r in recs])

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_data_root(settings) -> Path:
    """The directory all input paths must live inside.

    Inside the container compose mounts `${DATA_DIR}` at /app/data, so the
    root is /app/data by default. Outside the container we honour
    `$DATA_ROOT` if set, otherwise fall back to the parent of the configured
    input dir.
    """
    explicit = os.getenv("DATA_ROOT")
    if explicit:
        return Path(explicit).resolve()
    if Path("/app/data").exists():
        return Path("/app/data").resolve()
    # Outside Docker: parent of $INPUT_DIR (default data/audio_files).
    input_dir = os.getenv("INPUT_DIR", "data/audio_files")
    return Path(input_dir).resolve().parent


def _resolve_input_path(audio_path: str, data_root: Path) -> str:
    """Map a user-supplied path to an absolute path under `data_root`.

    Rejects path traversal that would escape the data root.
    """
    raw = Path(audio_path)
    candidate = raw if raw.is_absolute() else (data_root / raw)
    try:
        candidate = candidate.resolve(strict=False)
    except Exception:
        raise HTTPException(400, f"Invalid audio_path: {audio_path}")

    try:
        candidate.relative_to(data_root)
    except ValueError:
        raise HTTPException(
            400,
            f"audio_path must be inside the data root ({data_root}); got {candidate}",
        )
    return str(candidate)


def _save_upload(audio: UploadFile, upload_dir: Path) -> str:
    """Persist an uploaded file to disk; return the absolute path."""
    import uuid as _uuid
    safe_name = Path(audio.filename or "upload.wav").name  # strip dirs
    dest = upload_dir / f"{_uuid.uuid4().hex}__{safe_name}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(audio.file, f)
    return str(dest)
