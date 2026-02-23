#!/usr/bin/env bash
# One-shot CI-style check: build image (if needed), install in container, run single-GPU smoke test, exit with its code.
# No interactive shell; use for PR verification or CI. Requires one GPU.
# Inside the container this runs scripts/single_gpu_smoke.sh via docker_install_and_build.sh --smoke.

set -e

DISTCA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${DISTCA_IMAGE:-distca-pytorch:24.12}"

echo "DistCA root: $DISTCA_ROOT"
echo "Docker image: $IMAGE_NAME"

if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
  echo "Building image $IMAGE_NAME (NGC base ~4GB, may take a few minutes)..."
  docker build -t "$IMAGE_NAME" "$DISTCA_ROOT"
fi

echo "Running container: install + single-GPU smoke test (non-interactive)..."
echo "Using --shm-size=2g for NCCL/pretrain."
docker run --gpus all --rm --shm-size=2g \
  -v "$DISTCA_ROOT:/workspace/DistCA" \
  -e DISTCA_ROOT=/workspace/DistCA \
  "$IMAGE_NAME" \
  /workspace/DistCA/scripts/docker_install_and_build.sh --smoke
