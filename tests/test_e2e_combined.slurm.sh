#!/bin/bash

#SBATCH --job-name=e2e
#SBATCH --partition=main
#SBATCH --nodes=4
#SBATCH --output=logs/slurm.%j.stdout
#SBATCH --error=logs/slurm.%j.stderr
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --exclusive
#SBATCH --time=720:00:00
#SBATCH --qos=iq
#SBATCH --exclude=fs-mbz-gpu-286,fs-mbz-gpu-302,fs-mbz-gpu-476,fs-mbz-gpu-597,fs-mbz-gpu-684,fs-mbz-gpu-697,fs-mbz-gpu-868,fs-mbz-gpu-877


# ===================================
# D2 E2E Combined Test - Slurm Script
# ===================================
#
# USAGE:
#   sbatch test_e2e_combined.slurm.sh                      # Run with default parameters
#   PP_SIZE=4 BATCH_SIZE=2 sbatch test_e2e_combined.slurm.sh  # Override specific parameters
#
# CHANGING NODE/GPU CONFIGURATION:
# - To change the number of nodes:
#   * Edit the "#SBATCH --nodes=4" line at the top of this script
#   * Or use: sbatch --nodes=8 test_e2e_combined.slurm.sh
#
# - To change the number of GPUs per node:
#   * Edit the "#SBATCH --gres=gpu:8" line at the top of this script
#   * Or use: sbatch --gres=gpu:4 test_e2e_combined.slurm.sh
#   * The script automatically detects GPU count from Slurm environment variables
#
# PARAMETER CONFIGURATION:
# - All experiment parameters can be set as environment variables before submission
# - Parameters will use default values if not explicitly set
# - Key parameters to consider setting:
#   * PP_SIZE: Pipeline parallelism size (default: 2)
#   * TP_SIZE: Tensor parallelism size (default: GPUS_PER_NODE)
#   * BATCH_SIZE: Batch size for training (default: 1)
#   * NUM_TOKENS: Number of tokens to process (default: 262144)
#   * NUM_LAYERS: Number of model layers (default: 4)
#   * MODE: Experiment mode (default: baseline)
#   * MODEL_PATH: Path to the model (default: deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
#
# EXAMPLES:
#   PP_SIZE=4 TP_SIZE=2 MODE=dynamic sbatch test_e2e_combined.slurm.sh
#   BATCH_SIZE=2 NUM_TOKENS=131072 sbatch test_e2e_combined.slurm.sh
#   sbatch --nodes=2 test_e2e_combined.slurm.sh
#



# ---- Los Angeles timestamp + output dir ----
TS=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)

# Get the current directory of the script
cd $HOME/jd/d2/tests

# TOOD: Fix this hardcode output dir.
OUTPUT_DIR="$HOME/jd/d2/tests/logs/$TS.job$SLURM_JOB_NAME-${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# Redirect the output of this script to the output directory
# exec > $OUTPUT_DIR/slurm.stdout 2> $OUTPUT_DIR/slurm.stderr
exec > $OUTPUT_DIR/slurm.stdout 2>&1



set -x

# ---------------------------
# Env and Sanity Check
# ---------------------------

conda activate jd-d2

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name --format=csv

env > $OUTPUT_DIR/slurm.env

# ---------------------------
# Environment variables
# ---------------------------
export NVSHMEM_IB_ENABLE_IBGDA=true
# export NVSHMEM_DEBUG=DEBUG
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 
# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NVSHMEM_IBGDA_NUM_DCI=8
# export NVSHMEM_IBGDA_NUM_DCT=8
# export NVSHMEM_IBGDA_NUM_RC_PER_PE=4

# Comment out for a clean log.
# export CUDA_LAUNCH_BLOCKING=1 
# export D2_DEBUG_PRINT=0
# export D2_FA2A_DISABLE_SEND_RECV=0
# export WLBLLM_DISABLE_LSE=1
# export WLBLLM_SYNC_TIME_FLASH_ATTN=1

# ---------------------------
# Setup paths
# ---------------------------
export CUDA_DIR=/usr/local/cuda
export NCCL_HOME=/usr
export NCCL_LIB=/usr/lib/x86_64-linux-gnu
export NVSHMEM_DIR=/mnt/weka/home/yonghao.zhuang/opt/nvshmem
export NVSHMEM_PREFIX=/mnt/weka/home/yonghao.zhuang/opt/nvshmem
export OPENMPI_DIR=/mnt/weka/home/yonghao.zhuang/opt/openmpi

export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${CUDA_DIR}/lib64:${OPENMPI_DIR}/lib:${NCCL_LIB}/:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:${OPENMPI_DIR}/bin:${CUDA_DIR}/bin:$PATH"

# ---------------------------
# Setup experiment variables
# ---------------------------


DRY_RUN=${DRY_RUN:-0}

ENABLE_NSYS=${ENABLE_NSYS:-0}
WLBLLM_SYNC_TIME_AG=${WLBLLM_SYNC_TIME_AG:-0}
# How many time should each iteration repeat
EXPERIMENT_REPEAT_TIMES=${EXPERIMENT_REPEAT_TIMES:-3}
EXPERIMENT_WARMUP_TIMES=${EXPERIMENT_REPEAT_TIMES:-5}
EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=${EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB:--1}
EXPERIMENT_SHOULD_FORCE_EXIT=${EXPERIMENT_SHOULD_FORCE_EXIT:-0}
EXPERIMENT_EMIT_BACKWARD_NVTX=${EXPERIMENT_EMIT_BACKWARD_NVTX:-0}

export NVTE_NVTX_ENABLED=1
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


# ---------------------------
# Setup distributed args
# ---------------------------
echo $(hostname)
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
export head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port=29500

RZV_BACKEND=c10d
RZV_ENDPOINT=$head_node_ip:$port

# Get GPU count from Slurm's environment variables
# SLURM_GPUS_PER_NODE is set by Slurm when using --gpus-per-node or --gres=gpu:N
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
  GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
else
  GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
fi
NUM_NODES=$SLURM_NNODES
WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))

# ---------------------------
# D2 Experiment Important Vars
# ---------------------------
# These variables can be set as environment variables before submitting the job
# Example: PP_SIZE=4 BATCH_SIZE=2 sbatch test_e2e_combined.slurm.sh

# Model configuration
MODEL_PATH=${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}
NUM_LAYERS=${NUM_LAYERS:-4}

# Parallelism settings
TP_SIZE=${TP_SIZE:-$GPUS_PER_NODE}   # Tensor Parallelism size, defaults to GPUs per node
PP_SIZE=${PP_SIZE:-1}                # Pipeline Parallelism size
CP_SIZE=${CP_SIZE:-1}                # Only useful in WLBLLM (D2 will have DPCP anyways)

# Experiment settings
MODE=${MODE:-d2}               # Experiment mode (baseline, dynamic, etc.)
BATCH_SIZE=${BATCH_SIZE:-1}          # Batch size for training
NUM_TOKENS=${NUM_TOKENS:-65536}     # Number of tokens to process
MAX_SAMPLE_ID=${MAX_SAMPLE_ID:-3}   # Maximum sample ID

# Dataset settings
UP_SAMPLE_FACTOR=${UP_SAMPLE_FACTOR:-4}
ELONGATE_FACTOR=${ELONGATE_FACTOR:-1}
FILTER_THRESHOLD=${FILTER_THRESHOLD:-65536}
FILTER_RATIO=${FILTER_RATIO:-0.50}
SHOULD_ADD_DEBUG_CASES=${SHOULD_ADD_DEBUG_CASES:-0}


# Build the common stem *format* (anything that needs per-node must be computed on the node)
COMMON_STEM="nnodes${NNODES}.bs${BATCH_SIZE}.maxid${MAX_SAMPLE_ID}.tp${TP_SIZE}.pp${PP_SIZE}.cp${CP_SIZE}.t${NUM_TOKENS}.elong${ELONGATE_FACTOR}.up${UP_SAMPLE_FACTOR}.ft${FILTER_THRESHOLD}.fr${FILTER_RATIO}"

# Define missing variables with defaults
NNODES=${NNODES:-$NUM_NODES}
NPROC_PER_NODE=${NPROC_PER_NODE:-$GPUS_PER_NODE}
RZV_ID=${RZV_ID:-$RANDOM}
REPLAN_ITER=${REPLAN_ITER:-0}
SHOULD_PROFILE_MEMORY=${SHOULD_PROFILE_MEMORY:-0}

TORCHRUN_CMD=(
  --nnodes=${NNODES} \
  --nproc_per_node=${NPROC_PER_NODE} \
  --rdzv_backend=${RZV_BACKEND} \
  --rdzv_endpoint=${RZV_ENDPOINT} \
  --rdzv_id=${RZV_ID} \
  --max_restarts=0 \
  test_e2e_combined.py \
    --model-path ${MODEL_PATH} \
    --mode ${MODE} \
    --replan-iter ${REPLAN_ITER} \
    --batch-size ${BATCH_SIZE} \
    --num-nodes ${NNODES} \
    --num-gpus-per-node ${NPROC_PER_NODE} \
    --num-layers ${NUM_LAYERS} \
    --max-sample-id ${MAX_SAMPLE_ID} \
    --tp-size ${TP_SIZE} \
    --cp-degree ${CP_SIZE} \
    --up-sample-factor ${UP_SAMPLE_FACTOR} \
    --num-tokens ${NUM_TOKENS} \
    --elongate-factor ${ELONGATE_FACTOR} \
    --filter-threshold ${FILTER_THRESHOLD} \
    --filter-ratio ${FILTER_RATIO}
)

if [ ${SHOULD_ADD_DEBUG_CASES} -eq 1 ]; then
    TORCHRUN_CMD+=(--should-add-debug-cases)
fi

if [ ${EXPERIMENT_SHOULD_FORCE_EXIT} -eq 1 ]; then
    TORCHRUN_CMD+=(--force-exit)
fi

if [ ${SHOULD_PROFILE_MEMORY} -eq 1 ]; then
    TORCHRUN_CMD+=(--should-profile-memory ${PROFILE_MEMORY_PATH} )
fi

# Serialize TORCHRUN_CMD array so we can pass it through bash -lc cleanly
TORCHRUN_STR=$(printf " %q" "${TORCHRUN_CMD[@]}")

# ---- Per-node logs + per-node NSYS outputs ----
# %N and %j are expanded by Slurm *only* in --output/--error.
# Inside the bash -lc block we compute HOST and build node-specific file names for NSYS/other artifacts.

SRUN_BASE=(
  srun
  -N ${NNODES}
  --ntasks-per-node=1
  --gpus-per-task=${GPUS_PER_NODE}     # <= crucial
  --gpu-bind=per_task:${GPUS_PER_NODE} # <= bind GPUs to the task
  --kill-on-bad-exit=1
)


if [ ${ENABLE_NSYS} -eq 1 ]; then
  "${SRUN_BASE[@]}" \ \
    --output="${OUTPUT_DIR}/${TS}.${MODE}.%N.%j.out" \
    --error="${OUTPUT_DIR}/${TS}.${MODE}.%N.%j.out" \
      nsys profile --show-output=true --force-overwrite=true \
        -o "${OUTPUT_DIR}/%h.nsys-rep" --sample=none -t cuda,nvtx \
        torchrun $TORCHRUN_STR
else
#   srun \
#     --output="${OUTPUT_DIR}/%N.%j.out" \
#     --error="${OUTPUT_DIR}/%N.%j.out" \
#     torchrun $TORCHRUN_STR

    "${SRUN_BASE[@]}" \
        --output="${OUTPUT_DIR}/%N.%j.out" \
        --error="${OUTPUT_DIR}/%N.%j.out" \
        bash -lc '
            set -x
            hostname
            nvidia-smi
            python -c "import torch; print(torch.cuda.is_available()); torch.cuda.set_device(0); print(torch.cuda.get_device_name()); x = torch.ones(1).cuda(); print(x)"
            exec torchrun '"$TORCHRUN_STR"'
        '
fi

set +x

# set -euox pipefail