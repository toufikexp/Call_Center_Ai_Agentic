"""Long-lived HTTP server entry point.

An alternative to `python -m src.batch run`. Holds a single warm
`CallAnalysisPipeline` instance, accepts jobs over HTTP, processes them on
a worker thread, and lets clients poll for results. The pipeline, services,
and orchestrator are reused unchanged.

Run inside the container with:
    python -m src.server
"""
from src.server.app import create_app

__all__ = ["create_app"]
