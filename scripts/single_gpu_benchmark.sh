#!/usr/bin/env bash
# Single-GPU benchmark: run pretrain_llama.py with multiple steps and report throughput (tokens/s).
# Run inside the container after docker_install_and_build.sh. Same env as single_gpu_smoke.sh.
# Host one-shot: ./scripts/run_docker_single_gpu_benchmark.sh

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
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB="${EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB:-0.1}"

# Benchmark config: steps = max_sample_id, each step has num_tokens * num_batches tokens per rank
NUM_TOKENS="${DISTCA_BENCH_NUM_TOKENS:-1024}"
NUM_BATCHES="${DISTCA_BENCH_NUM_BATCHES:-1}"
MAX_SAMPLE_ID="${DISTCA_BENCH_MAX_SAMPLE_ID:-5}"
MODEL_PATH="${DISTCA_SMOKE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
OUTPUT_DIR="${DISTCA_SMOKE_OUTPUT:-./logs/single_gpu_benchmark}"

echo "=== Single-GPU benchmark (pretrain_llama.py) ==="
echo "  MODEL_PATH=$MODEL_PATH"
echo "  num_tokens=$NUM_TOKENS num_batches=$NUM_BATCHES max_sample_id=$MAX_SAMPLE_ID (steps=$MAX_SAMPLE_ID)"
echo "  OUTPUT_DIR=$OUTPUT_DIR"

start_sec=$(date +%s)

torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 \
  pretrain_llama.py \
  --num-tokens "$NUM_TOKENS" --num-batches "$NUM_BATCHES" --num-nodes 1 --num-gpus-per-node 1 \
  --cp-size 1 --tp-size 1 --pp-size 1 --num-microbatch 1 --max-sample-id "$MAX_SAMPLE_ID" \
  --num-layers 1 --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --val-every-n-steps 0

end_sec=$(date +%s)
duration_sec=$(( end_sec - start_sec ))
total_tokens=$(( NUM_TOKENS * NUM_BATCHES * MAX_SAMPLE_ID ))
if [ "$duration_sec" -gt 0 ]; then
  tokens_per_sec=$(( total_tokens / duration_sec ))
  echo ""
  echo "=== Single-GPU benchmark result ==="
  echo "  steps=$MAX_SAMPLE_ID  total_tokens=$total_tokens  wall_time=${duration_sec}s  throughput=${tokens_per_sec} tokens/s"
else
  echo "  wall_time < 1s (report total_tokens=$total_tokens)"
fi
