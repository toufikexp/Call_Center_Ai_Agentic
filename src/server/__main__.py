"""Entry point: `python -m src.server`.

Reads SERVER_HOST / SERVER_PORT from the environment and launches the
FastAPI app under uvicorn. One uvicorn worker by default because the
pipeline state (warm models, the shared CallAnalysisPipeline) must not be
duplicated across processes. Parallelism inside the server comes from the
job worker pool (`SERVER_WORKERS`), not from uvicorn workers.
"""
from __future__ import annotations

import logging
import os

import uvicorn

# Load .env BEFORE importing the app — same reason main.py does it: any HF
# offline / cache vars must be set before transformers imports run.
from main import _bootstrap_env  # type: ignore  # noqa: E402

_bootstrap_env()


def _setup_root_logging() -> None:
    level = os.getenv("SERVER_LOG_LEVEL", "INFO").upper()
    fmt = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=fmt)


def main() -> None:
    _setup_root_logging()
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))
    # One uvicorn worker. Job parallelism is configured via SERVER_WORKERS
    # and lives inside the app process.
    uvicorn.run(
        "src.server.app:create_app",
        host=host,
        port=port,
        factory=True,
        workers=1,
        log_level=os.getenv("SERVER_LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
