#!/bin/bash

#SBATCH --job-name=distca-debug
#SBATCH --output=logs/slurm/stdout.%j.log
#SBATCH --error=logs/slurm/stderr.%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00



export TP=1
export PP=1
export CP=1
export DP=1
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=2
export SEQ_LENGTH=4096
export NUM_GPUS=$((TP * PP * CP * DP))
NPROC_PER_NODE=$((NUM_GPUS < 8 ? NUM_GPUS : 8))
NNODES=$((NUM_GPUS / NPROC_PER_NODE))
echo TP=$TP, PP=$PP, CP=$CP, DP=$DP, NUM_GPUS=$NUM_GPUS, NPROC_PER_NODE=$NPROC_PER_NODE, NNODES=$NNODES


source .env.sh

# ================================
# Setup SLURM environment variables
# - JOBID: SLURM job ID
# - HEAD_NODE_IP: IP of head node
# ================================
if [ -n "$SLURM_JOB_ID" ] || [ -n "$SLURM_NODELIST" ] || [ -n "$SLURM_NNODES" ]; then
    is_in_slurm_env=1
else
    is_in_slurm_env=0
fi

if [ "$is_in_slurm_env" -eq 1 ]; then
    # Use SLURM variables when present
    JOBID=$SLURM_JOB_ID
    # Set HEAD_NODE_IP from SLURM_NODELIST if not already set
    echo SLURM_NODELIST=$SLURM_NODELIST
    SCONTROL_NODES=$(scontrol show hostnames "$SLURM_NODELIST")
    HEAD_NODE_IP=$(echo "$SCONTROL_NODES" | head -n 1)
    export JOBID
    export HEAD_NODE_IP
else
    # Not in SLURM: rely on whatever was set in .env.sh or environment
    JOBID="${JOBID}"
    
    # If JOBID is provided, query SLURM to get the node list
    if [ -n "$JOBID" ]; then
        echo "Querying SLURM for job $JOBID node list..."
        # Get the node list from scontrol show job
        NODELIST=$(scontrol show job "$JOBID" | grep -oP '^[\s]+NodeList=\K[^\s]+' || echo "")
        
        if [ -n "$NODELIST" ]; then
            echo "Found node list: $NODELIST"
            # Convert node list to hostnames and get the first one
            SCONTROL_NODES=$(scontrol show hostnames "$NODELIST")
            HEAD_NODE_IP=$(echo "$SCONTROL_NODES" | head -n 1)
            echo "Using head node: $HEAD_NODE_IP"
        else
            echo "Warning: Could not retrieve node list for job $JOBID"
            exit 1
        fi
    else
        HEAD_NODE_IP="${HEAD_NODE_IP}"
    fi
fi

echo JOBID=$JOBID
echo HEAD_NODE_IP=$HEAD_NODE_IP

# =====================================
# NCCL debug flags for troubleshooting
# =====================================
# export NCCL_DEBUG=INFO
unset NCCL_DEBUG
# export NCCL_DEBUG_SUBSYS=ALL
unset NCCL_DEBUG_SUBSYS
export NCCL_IB_DISABLE=0
export NVSHMEM_IB_ENABLE_IBGDA=true
export CUDA_DEVICE_MAX_CONNECTIONS=1
# unset CUDA_DEVICE_MAX_CONNECTIONS
export TORCH_NCCL_CONNECT_TIMEOUT=60000 # 60s
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 
export NVTE_NVTX_ENABLED=1
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0 

export OMP_NUM_THREADS=16

export TORCH_EXTENSIONS_DIR=/tmp/$USER/torch_extensions
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"

export PYTHONPYCACHEPREFIX=/tmp/$USER/pycache
mkdir -p "$PYTHONPYCACHEPREFIX"

# ==============================================================
# Logging directory setup
# - DISTCA_LOG_TIMESTAMP: Timestamp of the log
# - DISTCA_LOG_BASE_DIR: Base directory of the logs
# - DISTCA_LOG_ROOT_DIR: Root directory of the logs
# - DISTCA_LOG_LATEST_LINK: Link to the latest log directory
# ==============================================================
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DISTCA_LOG_TIMESTAMP=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)
export DISTCA_LOG_BASE_DIR="${CURDIR}/logs"
export DISTCA_LOG_ROOT_DIR="${DISTCA_LOG_BASE_DIR}/${DISTCA_LOG_TIMESTAMP}"
export DISTCA_LOG_LATEST_LINK="${CURDIR}/logs-latest"

# Create directories upfront so nsys can write to them
mkdir -p "${DISTCA_LOG_ROOT_DIR}/nsys"
mkdir -p "${DISTCA_LOG_ROOT_DIR}/rank_logs"
mkdir -p "${DISTCA_LOG_ROOT_DIR}/checkpoints"
mkdir -p "${DISTCA_LOG_ROOT_DIR}/tensorboard"
mkdir -p "${DISTCA_LOG_ROOT_DIR}/data_cache"

# Update the symlink to point to latest log directory
ln -sfn "${DISTCA_LOG_ROOT_DIR}" "${DISTCA_LOG_LATEST_LINK}"


# ==============================================================
# Run the test using SLURM
# ==============================================================

# TORCHRUN_DISTARGS=(
#     --nnodes=${NNODES}
#     --nproc_per_node=${NPROC_PER_NODE}
#     --rdzv_backend=c10d
#     --rdzv_endpoint=${HEAD_NODE_IP}:29800
#     --rdzv_id=0000
#     --max_restarts=0
# )

if [ -n "$ENABLE_NSYS" ]; then
set -x
    # Run with nsys profiling
    srun -w ${HEAD_NODE_IP} -N ${NNODES} --gres=gpu:${NPROC_PER_NODE} --jobid=${JOBID} \
    nsys profile --trace=cuda,nvtx --force-overwrite=true -o "${DISTCA_LOG_ROOT_DIR}/nsys/nsys-rep.%h.nsys-rep" --sample=none --capture-range=cudaProfilerApi --capture-range-end=stop \
    torchrun --nnodes=${NNODES} --nproc_per_node=${NPROC_PER_NODE} --rdzv_backend=c10d --rdzv_endpoint=${HEAD_NODE_IP}:29800 --rdzv_id=0000 --max_restarts=0 \
        simple_4d.py
set +x
else
set -x
    # Run without nsys profiling
    srun -w ${HEAD_NODE_IP} -N ${NNODES} --gres=gpu:${NPROC_PER_NODE} --jobid=${JOBID} \
    torchrun --nnodes=${NNODES} --nproc_per_node=${NPROC_PER_NODE} --rdzv_backend=c10d --rdzv_endpoint=${HEAD_NODE_IP}:29800 --rdzv_id=0000 --max_restarts=0 \
        simple_4d.py
set +x
fi
