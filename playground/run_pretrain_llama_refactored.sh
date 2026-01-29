#!/bin/bash
#
# Run script for pretrain_llama_refactored.py
#
# Usage: ./run_pretrain_llama_refactored.sh
#

set -e

# ================================
# Configuration
# ================================

# Parallelism (adjust based on your hardware)
export TP=1                 # Tensor Parallel
export PP=1                 # Pipeline Parallel
export CP=1                 # Context Parallel
export DP=2                 # Data Parallel (computed as world_size / (TP * PP * CP))

# Hardware
NUM_NODES=1
NUM_GPUS_PER_NODE=2
export WORLD_SIZE=$((NUM_NODES * NUM_GPUS_PER_NODE))

# Batch configuration
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=$DP
export SEQ_LENGTH=4096

# Model configuration
MODEL_NAME="meta-llama/Llama-3.1-8B"
NUM_LAYERS_OVERRIDE=2       # Use fewer layers for faster testing

# Training configuration
TRAIN_ITERS=10
NUM_TOKENS=1024
NUM_BATCHES=1
NUM_MICROBATCH=$PP          # Typically set to PP size
NUM_SEQS=3

# Dataset configuration (synthetic for now)
SAMPLE_NAME="wlbllm"        # Options: wlbllm, prolong, bookcorpus, wikitext, openwebtext, c4
UP_SAMPLE_FACTOR=4
ELONGATE_FACTOR=1
FILTER_THRESHOLD=65536
FILTER_RATIO=0.50

# For real datasets, uncomment and set:
# DATA_PATH="/path/to/your/dataset"
# TOKENIZER_TYPE="HuggingFaceTokenizer"

# DistCA configuration
USE_PLANNER="--use-planner"
NVSHMEM_BUFFER_SIZE_GB=1.0
# QUIT_IF_MAYBE_OOM="--quit-if-maybe-oom"

# Mixed precision
USE_BF16="--use-bf16"       # Use bfloat16 (otherwise fp16)

# Logging
SEED=42

# ================================
# Sanity Checks
# ================================
echo "======================================"
echo "DistCA LLaMA Pre-training (Refactored)"
echo "======================================"
echo ""
echo "Configuration:"
echo "  TP=${TP}, PP=${PP}, CP=${CP}, DP=${DP}"
echo "  World Size: ${WORLD_SIZE}"
echo "  Micro Batch Size: ${MICRO_BATCH_SIZE}"
echo "  Global Batch Size: ${GLOBAL_BATCH_SIZE}"
echo "  Sequence Length: ${SEQ_LENGTH}"
echo "  Model: ${MODEL_NAME}"
echo "  Layers: ${NUM_LAYERS_OVERRIDE}"
echo "  Training Iters: ${TRAIN_ITERS}"
echo "  Sample Name: ${SAMPLE_NAME}"
echo ""

# Check that TP * PP * CP * DP == WORLD_SIZE
EXPECTED_WORLD_SIZE=$((TP * PP * CP * DP))
if [ $EXPECTED_WORLD_SIZE -ne $WORLD_SIZE ]; then
    echo "ERROR: TP * PP * CP * DP != WORLD_SIZE"
    echo "  TP * PP * CP * DP = ${TP} * ${PP} * ${CP} * ${DP} = ${EXPECTED_WORLD_SIZE}"
    echo "  WORLD_SIZE = ${WORLD_SIZE}"
    exit 1
fi

# ================================
# Launch Training
# ================================

# Build the command
CMD="python playground/pretrain_llama_refactored.py \
    --model-name ${MODEL_NAME} \
    --num-layers-override ${NUM_LAYERS_OVERRIDE} \
    --train-iters ${TRAIN_ITERS} \
    --seed ${SEED} \
    --tp ${TP} \
    --pp ${PP} \
    --cp ${CP} \
    --dp ${DP} \
    --seq-length ${SEQ_LENGTH} \
    --num-tokens ${NUM_TOKENS} \
    --num-batches ${NUM_BATCHES} \
    --num-microbatch ${NUM_MICROBATCH} \
    --num-seqs ${NUM_SEQS} \
    --sample-name ${SAMPLE_NAME} \
    --up-sample-factor ${UP_SAMPLE_FACTOR} \
    --elongate-factor ${ELONGATE_FACTOR} \
    --filter-threshold ${FILTER_THRESHOLD} \
    --filter-ratio ${FILTER_RATIO} \
    --nvshmem-buffer-size-gb ${NVSHMEM_BUFFER_SIZE_GB} \
    ${USE_PLANNER} \
    ${USE_BF16}"

# Add optional arguments
if [ ! -z "${DATA_PATH}" ]; then
    CMD="${CMD} --data-path ${DATA_PATH}"
fi

if [ ! -z "${TOKENIZER_TYPE}" ]; then
    CMD="${CMD} --tokenizer-type ${TOKENIZER_TYPE}"
fi

if [ ! -z "${MAX_TOTAL_TOKENS}" ]; then
    CMD="${CMD} --max-total-tokens ${MAX_TOTAL_TOKENS}"
fi

if [ ! -z "${QUIT_IF_MAYBE_OOM}" ]; then
    CMD="${CMD} ${QUIT_IF_MAYBE_OOM}"
fi

# Launch with torchrun
echo "Launching training with torchrun..."
echo ""

torchrun \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    --nnodes=${NUM_NODES} \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    ${CMD}

echo ""
echo "Training complete!"
