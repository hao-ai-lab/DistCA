#!/usr/bin/env bash
# One-shot: build DistCA Docker image (if needed), run container with repo mounted,
# install distca + Megatron-LM + build csrc, then drop into a shell where you can run
# pretrain_llama.py or scripts/single_gpu_smoke.sh / scripts/single_gpu_benchmark.sh.

set -e

DISTCA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${DISTCA_IMAGE:-distca-pytorch:24.12}"

echo "DistCA root: $DISTCA_ROOT"
echo "Docker image: $IMAGE_NAME"

if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
  echo "Building image $IMAGE_NAME (NGC base ~4GB, may take a few minutes)..."
  docker build -t "$IMAGE_NAME" "$DISTCA_ROOT"
fi

echo "Running container (mount $DISTCA_ROOT -> /workspace/DistCA), running install + build then bash..."
echo "Using --shm-size=2g for NCCL/pretrain."
docker run --gpus all -it --rm --shm-size=2g \
  -v "$DISTCA_ROOT:/workspace/DistCA" \
  "$IMAGE_NAME" \
  /workspace/DistCA/scripts/docker_install_and_build.sh
