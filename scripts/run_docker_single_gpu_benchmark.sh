#!/usr/bin/env bash
# One-shot: build image (if needed), install in container, run single-GPU benchmark, exit with its code.
# Requires one GPU. For an interactive shell after install, run scripts/run_docker_benchmark.sh, then inside the container run:
#   bash /workspace/DistCA/scripts/single_gpu_benchmark.sh

set -e

DISTCA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${DISTCA_IMAGE:-distca-pytorch:24.12}"

echo "DistCA root: $DISTCA_ROOT"
echo "Docker image: $IMAGE_NAME"

if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
  echo "Building image $IMAGE_NAME (NGC base ~4GB, may take a few minutes)..."
  docker build -t "$IMAGE_NAME" "$DISTCA_ROOT"
fi

echo "Running container: install + single-GPU benchmark (non-interactive)..."
docker run --gpus all --rm --shm-size=2g \
  -v "$DISTCA_ROOT:/workspace/DistCA" \
  -e DISTCA_ROOT=/workspace/DistCA \
  "$IMAGE_NAME" \
  /workspace/DistCA/scripts/docker_install_and_build.sh --benchmark
