#!/usr/bin/env bash
# Build the CPU production image on the dev box and push it to the registry.
#
# Run on the machine WITH internet (your WSL dev box), not the prod VM.
#
# Usage:
#   REGISTRY=registry.internal.example.com/team \
#   TAG=$(git rev-parse --short HEAD) \
#       ./scripts/build_and_push.sh
#
# Defaults: TAG is the short git sha; IMAGE_NAME is cc-pipeline.

set -euo pipefail

REGISTRY="${REGISTRY:?Set REGISTRY to your registry path, e.g. registry.internal/team}"
IMAGE_NAME="${IMAGE_NAME:-cc-pipeline}"
TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo latest)}"

FULL="${REGISTRY}/${IMAGE_NAME}:${TAG}"

echo "→ Building ${FULL} (CPU image)"
docker build -t "${FULL}" .

# Also tag a moving 'prod-latest' for convenience; pin the sha tag in prod units.
PROD_LATEST="${REGISTRY}/${IMAGE_NAME}:prod-latest"
docker tag "${FULL}" "${PROD_LATEST}"

echo "→ Pushing ${FULL}"
docker push "${FULL}"
echo "→ Pushing ${PROD_LATEST}"
docker push "${PROD_LATEST}"

cat <<EOM

✅ Pushed:
   ${FULL}
   ${PROD_LATEST}

On the prod VM:
   docker pull ${FULL}
   CC_IMAGE=${FULL} docker compose --env-file deploy/.env.prod up -d

Pin the sha tag (${TAG}) in prod — avoid relying on prod-latest for rollback.
EOM
