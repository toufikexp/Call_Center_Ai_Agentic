# Production image for the Call Center AI Agentic Pipeline (CPU-only).
#
# Build (dev box with internet):
#   docker build -t <registry>/cc-pipeline:<tag> .
# Push:
#   docker push <registry>/cc-pipeline:<tag>
# Pull + run (prod RHEL 9 VM):
#   docker pull <registry>/cc-pipeline:<tag>
#   docker compose up   (see compose.yaml)
#
# Models are NOT baked into the image — they are bind-mounted at runtime
# (see compose.yaml / deployment runbook). This keeps the image ~1.5 GB and
# lets the LoRA adapter change without an image rebuild.

FROM python:3.12-slim

# --- System libraries the pipeline needs at runtime ---
#   ffmpeg     : audioread/librosa fallback for malformed WAV + format support
#   libsndfile1: soundfile backend
#   libgomp1   : OpenMP runtime for torch/numpy on CPU
# Pinned-clean apt install, then cache cleanup to keep the layer small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Offline by default in prod — no HF Hub / torch.hub network calls at runtime.
    # Override to 0 only if you intentionally want online behaviour.
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# --- Python deps (CPU torch stack) ---
# Copy only the requirements first so this layer caches across code changes.
COPY deploy/requirements.cpu.txt /app/deploy/requirements.cpu.txt
RUN pip install --upgrade pip \
    && pip install -r /app/deploy/requirements.cpu.txt

# --- Bake the Silero VAD weights into the image ---
# The silero-vad pip package downloads its ONNX/JIT weights on first
# load_silero_vad() call. Trigger it now (build host has internet) so the
# prod container never reaches the network for VAD.
RUN python -c "from silero_vad import load_silero_vad; load_silero_vad()"

# --- Application code ---
COPY src/ /app/src/
COPY main.py /app/main.py

# Non-root user; data + model dirs are bind-mounted at runtime.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data /opt/cc/models \
    && chown -R appuser:appuser /app /opt/cc
USER appuser

# Default command processes a batch from the configured input dir. Override
# in compose / docker run for one-off single-file processing or healthchecks.
ENTRYPOINT ["python", "-m", "src.batch", "run"]
CMD ["--workers", "3"]
