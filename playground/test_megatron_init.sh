#! /bin/bash

#SBATCH --job-name=test-megatron-init
#SBATCH --nodes=2
#SBATCH --output=logs/slurm/stdout.%j.log
#SBATCH --error=logs/slurm/stderr.%j.log
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --exclusive
#SBATCH --time=01:00:00


NNODES=1
NPROC_PER_NODE=8
JOBID=1343050
HEAD_NODE_IP=fs-mbz-gpu-942:29500

export WANDB_API_KEY="02575b6c73e438f9885daa7cf691a45939d26a71"


export OMP_NUM_THREADS=4

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
export DISTCA_LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
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
srun -N ${NNODES} --gres=gpu:${NPROC_PER_NODE} --jobid=${JOBID} \
nsys profile --trace=cuda,nvtx --force-overwrite=true -o "${DISTCA_LOG_ROOT_DIR}/nsys/nsys-rep.%h.nsys-rep" --sample=none --capture-range=cudaProfilerApi --capture-range-end=stop \
torchrun --nproc_per_node=${NPROC_PER_NODE} --nnodes=${NNODES} --rdzv_backend=c10d --rdzv_endpoint=${HEAD_NODE_IP} --rdzv_id=0000 --max_restarts=0 \
    test_megatron_init.py 
set +x