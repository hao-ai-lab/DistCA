#! /bin/bash

#SBATCH --job-name=distca-debug
#SBATCH --nodes=4
#SBATCH --output=logs/slurm/stdout.%j.log
#SBATCH --error=logs/slurm/stderr.%j.log
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --exclusive
#SBATCH --time=01:00:00


NNODES=1
NPROC_PER_NODE=1
JOBID=1390200
HEAD_NODE_IP=fs-mbz-gpu-044

source .env.sh

# NCCL debug flags for troubleshooting
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

# --------------------------------
# Logging directory setup
# Generate timestamp once and share between nsys and Python
# --------------------------------
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

set -x
# nsys profile --trace=cuda,nvtx --force-overwrite=true -o "${DISTCA_LOG_ROOT_DIR}/nsys/nsys-rep.%h.nsys-rep" --sample=none --capture-range=cudaProfilerApi --capture-range-end=stop \

srun -w ${HEAD_NODE_IP} -N ${NNODES} --gres=gpu:${NPROC_PER_NODE} --jobid=${JOBID} \
torchrun --nnodes=${NNODES} --nproc_per_node=${NPROC_PER_NODE} --rdzv_backend=c10d --rdzv_endpoint=${HEAD_NODE_IP}:29800 --rdzv_id=0000 --max_restarts=0 \
    test_megatron_init.py 

set +x