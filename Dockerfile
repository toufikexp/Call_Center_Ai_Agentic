# syntax=docker/dockerfile:1.7
#
# One Dockerfile, two variants. Build via the Makefile:
#   make build              # → cc-pipeline:cpu (TORCH_VARIANT=cpu)
#   make build TORCH=gpu    # → cc-pipeline:gpu (TORCH_VARIANT=cu130)
#
# Models are NOT in the image. The runtime stage mounts them at /adapters
# and /hf via compose. Offline behaviour is controlled by HF_HUB_OFFLINE
# and TRANSFORMERS_OFFLINE in `.env`, not by anything baked here.
#
# Multi-stage layout:
#   builder  — has build-essential + pip cache, installs deps, bakes Silero
#   runtime  — only runtime libs; copies site-packages and Silero from builder

ARG PYTHON_VERSION=3.12-slim
ARG TORCH_VARIANT=cpu

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libsndfile1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Re-declare ARG (multi-stage builds reset args at each FROM)
ARG TORCH_VARIANT

COPY requirements/ /app/requirements/

# Install base deps + the torch variant matching this build.
# pip resolves --extra-index-url from the variant file's header.
RUN pip install --upgrade pip \
    && pip install \
        -r /app/requirements/base.txt \
        -r /app/requirements/torch-${TORCH_VARIANT}.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf

# Runtime system libs only — no compilers, no headers.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /adapters /hf \
    && chown -R app:app /app /hf

# Bring installed packages + binaries from builder. Silero VAD weights ship
# inside the `silero-vad` PyPI package itself (under site-packages), so they
# come along automatically with the line above — no separate cache copy.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Application code last — biggest layer to change between builds, so it
# stays on top to keep the deps layer cached.
COPY --chown=app:app src/ /app/src/
COPY --chown=app:app main.py /app/main.py

USER app

# Default to the batch runner. Compose overrides via `command:` for the
# `app` service; users override per-invocation with `make run ARGS="..."`.
ENTRYPOINT ["python", "-m", "src.batch", "run"]
CMD ["--workers", "2"]
