#!/usr/bin/env bash
# Pre-stage the Whisper base model + LoRA adapter so the prod VM never needs
# to reach Hugging Face Hub at runtime.
#
# Two phases:
#   (A) On the dev box (internet): download the base model into a local HF
#       cache dir and pack it + the adapter into a tarball.
#   (B) On the prod VM (no internet): unpack the tarball into the models dir
#       that compose.yaml bind-mounts at /opt/cc/models.
#
# Usage — phase A (dev box):
#   BASE_MODEL=openai/whisper-large-v3 \
#   ADAPTER_SRC=/path/to/whisper-large-v3-lora-algerian_v7 \
#   OUT=cc-models.tar.gz \
#       ./scripts/stage_models.sh pack
#
# Usage — phase B (prod VM):
#   MODELS_DIR=/opt/cc/models \
#   TARBALL=cc-models.tar.gz \
#       ./scripts/stage_models.sh unpack

set -euo pipefail

CMD="${1:-}"

pack() {
    BASE_MODEL="${BASE_MODEL:-openai/whisper-large-v3}"
    ADAPTER_SRC="${ADAPTER_SRC:?Set ADAPTER_SRC to the local LoRA adapter directory}"
    OUT="${OUT:-cc-models.tar.gz}"

    STAGE="$(mktemp -d)"
    HF_DIR="${STAGE}/huggingface"
    ADAPTER_DIR="${STAGE}/adapters/$(basename "${ADAPTER_SRC}")"
    mkdir -p "${HF_DIR}" "${ADAPTER_DIR}"

    echo "→ Downloading base model '${BASE_MODEL}' into HF cache"
    HF_HOME="${HF_DIR}" python - <<PY
from transformers import WhisperForConditionalGeneration, WhisperProcessor
WhisperForConditionalGeneration.from_pretrained("${BASE_MODEL}")
WhisperProcessor.from_pretrained("${BASE_MODEL}")
print("base model cached")
PY

    echo "→ Copying adapter from ${ADAPTER_SRC}"
    cp -r "${ADAPTER_SRC}/." "${ADAPTER_DIR}/"

    echo "→ Packing ${OUT}"
    tar -C "${STAGE}" -czf "${OUT}" huggingface adapters
    rm -rf "${STAGE}"

    cat <<EOM

✅ Packed ${OUT}
Transfer it to the prod VM (registry-adjacent share / scp / approved channel),
then run:
   MODELS_DIR=/opt/cc/models TARBALL=${OUT} ./scripts/stage_models.sh unpack
EOM
}

unpack() {
    MODELS_DIR="${MODELS_DIR:-/opt/cc/models}"
    TARBALL="${TARBALL:?Set TARBALL to the transferred cc-models.tar.gz}"

    mkdir -p "${MODELS_DIR}"
    echo "→ Unpacking ${TARBALL} into ${MODELS_DIR}"
    tar -C "${MODELS_DIR}" -xzf "${TARBALL}"

    cat <<EOM

✅ Models staged under ${MODELS_DIR}:
   ${MODELS_DIR}/huggingface/   (Whisper base, HF cache layout)
   ${MODELS_DIR}/adapters/      (LoRA adapter(s))

Ensure deploy/.env.prod has:
   HF_HOME=/opt/cc/models/huggingface
   WHISPER_ADAPTER_PATH=/opt/cc/models/adapters/<adapter-folder>
EOM
}

case "${CMD}" in
    pack)   pack ;;
    unpack) unpack ;;
    *)
        echo "Usage: $0 {pack|unpack}" >&2
        echo "  pack   — on the dev box (downloads base model, builds tarball)" >&2
        echo "  unpack — on the prod VM (extracts into MODELS_DIR)" >&2
        exit 1
        ;;
esac
