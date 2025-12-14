#!/bin/bash

# Get the current directory of this script
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pre-requisites:
# 1. Enable the conda environment "jd-d2"
# 2. Set JOBID and HEAD_NODE_IP environment variables
# 3. Call this file: bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks3/_251103_correctness/run_megatron_layer.sh

# Environment variables
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO__DISABLE_CHECK=1

# Job configuration
# export JOBID=${JOBID:-1026987}
# export HEAD_NODE_IP=${HEAD_NODE_IP:-fs-mbz-gpu-180}
# export TP_SIZE=${TP_SIZE:-8}
export JOBID=${JOBID:-1026987}
export HEAD_NODE_IP=${HEAD_NODE_IP:-fs-mbz-gpu-180}
export TP_SIZE=${TP_SIZE:-1}
NNODES=${NNODES:-2}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

if [ $NNODES -ge 2 ]; then
    # export NVSHMEM_DEBUG=DEBUG 
    export NVSHMEM_IB_ENABLE_IBGDA=true
fi
unset NVSHMEM_DEBUG


# Test configuration
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))  # 16 for 2 nodes with 8 GPUs each
MASTER_PORT=29500

# Test parameters
SEED=${SEED:-42}
NUM_TOKENS=${NUM_TOKENS:-8192}
NUM_DOCS=${NUM_DOCS:-3}
MAX_CP_DEGREE=${MAX_CP_DEGREE:-2}
HIDDEN_SIZE=${HIDDEN_SIZE:-1024}  # Increased to support more heads with TP=8
NUM_HEADS=${NUM_HEADS:-16}  # Increased proportionally (16 heads / 8 TP = 2 heads per TP rank)
NUM_QUERY_HEADS=${NUM_QUERY_HEADS:-16}


echo "====================================="
echo "Running test_megatron_layer_correctness_multiiter.py"
echo "====================================="
echo "JOBID: $JOBID"
echo "HEAD_NODE_IP: $HEAD_NODE_IP"
echo "NNODES: $NNODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "TP_SIZE: $TP_SIZE"
echo "====================================="


# Build the torchrun command
TORCHRUN_CMD="torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=$JOBID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$HEAD_NODE_IP:$MASTER_PORT \
    /mnt/weka/home/yonghao.zhuang/jd/d2/tests/test_megatron_layer_correctness_multiiter.py \
    --world-size $WORLD_SIZE \
    --tp-size $TP_SIZE \
    --seed $SEED \
    --num-tokens $NUM_TOKENS \
    --num-docs $NUM_DOCS \
    --max-cp-degree $MAX_CP_DEGREE \
    --hidden-size $HIDDEN_SIZE \
    --num-heads $NUM_HEADS \
    --num-query-heads $NUM_QUERY_HEADS"


if [ -n "${TORCHRUN_EXTRA_ARGS}" ]; then
    TORCHRUN_CMD="$TORCHRUN_CMD ${TORCHRUN_EXTRA_ARGS}"
fi


# Run with srun
cd /mnt/weka/home/yonghao.zhuang/jd/d2/tests

srun -N $NNODES -G $((NNODES * NPROC_PER_NODE)) -w $HEAD_NODE_IP --jobid $JOBID \
    bash -c "$TORCHRUN_CMD"

echo "====================================="
echo "Test completed"
echo "====================================="
