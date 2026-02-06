#!/usr/bin/env bash
# Single-GPU smoke test: run minimal pretrain_llama.py (1 GPU, 1 layer, 1 batch).
# Run inside the container after docker_install_and_build.sh, or from the host via:
#   docker run ... /workspace/DistCA/scripts/docker_install_and_build.sh --smoke
# Host one-shot: ./scripts/run_docker_single_gpu_smoke.sh

set -e

DISTCA_ROOT="${DISTCA_ROOT:-/workspace/DistCA}"
cd "$DISTCA_ROOT"

# NVSHMEM env (required for libas_comm.so and pretrain)
export NVSHMEM_PREFIX=$(python -c "
import nvidia.nvshmem
import os
p = getattr(nvidia.nvshmem, '__path__', None)
if p:
    print(os.path.normpath(p[0]))
elif getattr(nvidia.nvshmem, '__file__', None):
    print(os.path.dirname(nvidia.nvshmem.__file__))
else:
    import sys
    for p in sys.path:
        d = os.path.join(p, 'nvidia', 'nvshmem')
        if os.path.isdir(d) and os.path.exists(os.path.join(d, 'lib')):
            print(os.path.normpath(d))
            break
    else:
        raise SystemExit('Could not find NVSHMEM_PREFIX')
")
export LD_LIBRARY_PATH="${NVSHMEM_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Avoid NVSHMEM buffer size error for minimal 1-GPU config
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB="${EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB:-0.1}"

# Minimal single-GPU pretrain: 1 node, 1 GPU, 1 layer, 1 batch, 1024 tokens
# Model is downloaded from Hugging Face on first run; set HF_HOME or DISTCA_SMOKE_MODEL to override.
MODEL_PATH="${DISTCA_SMOKE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
OUTPUT_DIR="${DISTCA_SMOKE_OUTPUT:-./logs/single_gpu_smoke}"

echo "=== Single-GPU smoke test (pretrain_llama.py) ==="
echo "  MODEL_PATH=$MODEL_PATH"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  NVSHMEM_PREFIX=$NVSHMEM_PREFIX"
echo "  EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=$EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB"

torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 \
  pretrain_llama.py \
  --num-tokens 1024 --num-batches 1 --num-nodes 1 --num-gpus-per-node 1 \
  --cp-size 1 --tp-size 1 --pp-size 1 --num-microbatch 1 --max-sample-id 1 \
  --num-layers 1 --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR"

echo "=== Single-GPU smoke test PASSED ==="
