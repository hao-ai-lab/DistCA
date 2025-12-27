#! /bin/bash

#SBATCH --job-name=distca-llama
#SBATCH --nodes=1
#SBATCH --output=logs/slurm/stdout.%j.log
#SBATCH --error=logs/slurm/stderr.%j.log
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --exclusive
#SBATCH --time=01:00:00

JOBID=${JOBID:-$SLURM_JOB_ID}
NNODES=${SLURM_NNODES:-1}
# TODO: Get the head node IP from the job ID.
# HEAD_NODE_IP=$(scontrol show hostnames $(scontrol show job $JOBID | awk -F= '/NodeList=fs/ {print $2}') | head -n 1)
# echo HEAD_NODE_IP=$HEAD_NODE_IP

# export CUDA_DEVICE_MAX_CONNECTIONS=1
unset CUDA_DEVICE_MAX_CONNECTIONS


# ------------------------------------------------------
# Input Variables (what you will set outside)
# ------------------------------------------------------

# Model configuration
# MODEL_PATH=${MODEL_PATH:-codellama/CodeLlama-34b-hf}
MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
MODEL_PATH_normalized=$(echo $MODEL_PATH | sed 's/\//_/g')
NUM_LAYERS=4
# NUM_LAYERS=1

# Parallelism settings
TP_SIZE=8
PP_SIZE=1                # Pipeline Parallelism size
CP_SIZE=1                # Only useful in WLBLLM (D2 will have DPCP anyways)
NUM_MICROBATCH=1

# Experiment settings
MODE=distca               # Experiment mode (baseline, dynamic, etc.)
BATCH_SIZE=1          # Batch size for training
NUM_TOKENS=16384     # Number of tokens to process
MAX_SAMPLE_ID=15      # Maximum sample ID
VAL_EVERY_N_STEPS=0  # Validate every N training steps
CKPT_EVERY_N_STEPS=0 # Checkpoint every N training steps (0=disable)
# SAMPLE_EXPR=${SAMPLE_EXPR:-""}   # Sample expression
SAMPLE_NAME=wlbllm
CHANGE_LONG_DOC_RATIO=0.0

# Dataset sampling settings
UP_SAMPLE_FACTOR=4
ELONGATE_FACTOR=1
FILTER_THRESHOLD=65536
FILTER_RATIO=0.50
SHOULD_ADD_DEBUG_CASES=0


# EXPERIMENT_ENABLE_CUDA_GRAPHS=1
export EXPERIMENT_ENABLE_CUDA_GRAPHS=0
export EXPERIMENT_DISABLE_ROPE=0
export D2_LOG_TENSOR_SHAPES=1
export D2_DEBUG_PRINT=0

# ------------------------------------------------------
# Setup loggings and artifact directories
# ------------------------------------------------------
TS=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)
SHORT_TS=$(TZ=America/Los_Angeles date +%d_%H%M%S)
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_LATEST="${CURDIR}/logs-latest"
OUTPUT_DIR_PREFIX="${CURDIR}/logs"
OUTPUT_DIR_SUFFIX=${OUTPUT_DIR_SUFFIX:-"$SHORT_TS.${MODE}-n${NNODES}-t${NUM_TOKENS}-b${BATCH_SIZE}-mb${NUM_MICROBATCH}-cp${CP_SIZE}tp${TP_SIZE}pp${PP_SIZE}-${MODEL_PATH_normalized}-L${NUM_LAYERS}-${SAMPLE_NAME}_${CHANGE_LONG_DOC_RATIO}"}
OUTPUT_DIR_SUFFIX_ADDON=${OUTPUT_DIR_SUFFIX_ADDON:-""}
OUTPUT_DIR="$OUTPUT_DIR_PREFIX/$OUTPUT_DIR_SUFFIX$OUTPUT_DIR_SUFFIX_ADDON"
mkdir -p "$OUTPUT_DIR"

if [ -d $OUTPUT_LATEST ]; then
    rm $OUTPUT_LATEST
fi
ln -s $OUTPUT_DIR $OUTPUT_LATEST

# Redirect the output of this script to the output directory
exec > >(tee "$OUTPUT_DIR/slurm.stdout") 2>&1

export LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
export CKPT_DIR="${CKPT_DIR:-${OUTPUT_DIR}/ckpts}"
mkdir -p "$CKPT_DIR"

PROFILE_MEMORY_PATH=${PROFILE_MEMORY_PATH:"${OUTPUT_DIR}/"}
# ---------------------------
# Environment variables
# ---------------------------
# export NVSHMEM_BOOTSTRAP=mpi
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 # otherwise (if set to 0) attention kernels may get horrible performance
export NVSHMEM_IB_ENABLE_IBGDA=true
export ENABLE_NSYS=0
export EXPERIMENT_LOG_MEMORY_USAGE=0
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2
export EXPERIMENT_ADD_SELECTIVE_CKPT=1
export EXPERIMENT_ENABLE_CUDA_GRAPHS=0


# ---------------------------
# Setup Logging
# ---------------------------
# Source the `env.sh` file from project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$PROJECT_ROOT/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/env.sh"
else
  echo -e "\033[31mError: missing $PROJECT_ROOT/env.sh (computed from script location).\nPlease copy $PROJECT_ROOT/env.template.sh to $PROJECT_ROOT/env.sh and customize it.\033[0m" >&2
  exit 1
fi


# ---------------------------
# Setup distributed args
# ---------------------------
RZV_BACKEND=c10d
RZV_ENDPOINT=$HEAD_NODE_IP:29500
echo RZV_ENDPOINT=$RZV_ENDPOINT

# Get GPU count from Slurm's environment variables
# SLURM_GPUS_PER_NODE is set by Slurm when using --gpus-per-node or --gres=gpu:N
GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE}}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))


# Build the common stem *format* (anything that needs per-node must be computed on the node)
COMMON_STEM="mode${MODE}.nnodes${NNODES}.bs${BATCH_SIZE}.maxid${MAX_SAMPLE_ID}.tp${TP_SIZE}.pp${PP_SIZE}.cp${CP_SIZE}.t${NUM_TOKENS}"

touch ${OUTPUT_DIR}/desc.${COMMON_STEM} # just a description of this experiment, in its file name

# Define missing variables with defaults
TOTAL_GPUS=$((TP_SIZE * PP_SIZE * CP_SIZE))
echo TOTAL_GPUS=$TOTAL_GPUS
NPROC_PER_NODE=$(( GPUS_PER_NODE < TOTAL_GPUS ? GPUS_PER_NODE : TOTAL_GPUS ))
NPROC_PER_NODE=$(( NPROC_PER_NODE < 8 ? NPROC_PER_NODE : 8 ))
echo NPROC_PER_NODE=$NPROC_PER_NODE
RZV_ID=${RZV_ID:-$head_node_ip-${TS}}
REPLAN_ITER=${REPLAN_ITER:-0}
SHOULD_PROFILE_MEMORY=${SHOULD_PROFILE_MEMORY:-0}
EXPERIMENT_SHOULD_LOG_MEMORY_DURING_WARMUP=${EXPERIMENT_SHOULD_LOG_MEMORY_DURING_WARMUP:-0}

# ------------------------------------------------------
# Construct command
# ------------------------------------------------------

# -vvvv # Touching the --cpu-per-task and --cpu-bind variables may cause srun to hang.
SRUN_BASE=(
  srun --kill-on-bad-exit=1 
  -N ${NNODES}
  -G ${WORLD_SIZE}
  --ntasks-per-node=1
  # --gpus-per-task=${GPUS_PER_NODE}     # <= crucial
  --cpus-per-task=128
  --cpu-bind=cores
  # --gpu-bind=closest
  # --kill-on-bad-exit=1
  -w "$HEAD_NODE_IP"
  --mem=0 # inherit the memory from the salloc
)

if [ ${JOBID} -ne 0 ]; then
  SRUN_BASE+=(--jobid=${JOBID})
fi


TORCHRUN_CMD=(
  --nnodes=${NNODES}
  --nproc_per_node=${NPROC_PER_NODE}
  --rdzv_backend=${RZV_BACKEND}
  --rdzv_endpoint=${RZV_ENDPOINT}
  --rdzv_id=${RZV_ID}
  --max_restarts=0
  --no-python bash ./utils/bind_and_exec.sh 
    python pretrain_llama.py
    --num-tokens ${NUM_TOKENS}
    --num-batches ${BATCH_SIZE}
    --num-nodes ${NNODES}
    --num-gpus-per-node ${NPROC_PER_NODE}
    --cp-size ${CP_SIZE}
    --tp-size ${TP_SIZE}
    --pp-size ${PP_SIZE}
    --num-microbatch ${NUM_MICROBATCH}
    --val-every-n-steps ${VAL_EVERY_N_STEPS}
    --ckpt-every-n-steps ${CKPT_EVERY_N_STEPS}
    --use-planner
    
    --model-path ${MODEL_PATH}
    --num-layers ${NUM_LAYERS}

    # TODO: 
    --max-sample-id ${MAX_SAMPLE_ID}
    --sample-name ${SAMPLE_NAME}
    --change-long-doc-ratio ${CHANGE_LONG_DOC_RATIO}

    --up-sample-factor ${UP_SAMPLE_FACTOR}
    --elongate-factor ${ELONGATE_FACTOR}
    --filter-threshold ${FILTER_THRESHOLD}
    --filter-ratio ${FILTER_RATIO}
    --output-dir ${OUTPUT_DIR}
)

if [ ${SHOULD_ADD_DEBUG_CASES} -eq 1 ]; then
    TORCHRUN_CMD+=(--should-add-debug-cases)
fi

# Serialize TORCHRUN_CMD array so we can pass it through bash -lc cleanly
TORCHRUN_STR=$(printf " %q" "${TORCHRUN_CMD[@]}")

# ---- Per-node logs + per-node NSYS outputs ----
# %N and %j are expanded by Slurm *only* in --output/--error.
# Inside the bash -lc block we compute HOST and build node-specific file names for NSYS/other artifacts.

# ---------------------------
# Log environment variables (for debugging)
# ---------------------------
env > $OUTPUT_DIR/slurm.env

# Define helper function to echo and tee to file
echo_and_tee() {
    local file="$1"
    shift
    echo "$@" | tee -a "$file"
}

# ---------------------------
# Run the experiment
# ---------------------------

echo "Start running sbatch at $(TZ='America/Los_Angeles' date)"

NSYS_PATH="${OUTPUT_DIR}/nsys-reps"
mkdir -p ${NSYS_PATH}

nsys_str=""
if [ ${ENABLE_NSYS} -eq 1 ]; then
  nsys_str="nsys profile --show-output=true --force-overwrite=true -o ${NSYS_PATH}/%h.nsys-rep -t cuda,nvtx"
fi

start_time=$(TZ='America/Los_Angeles' date +%s)

set -x
"${SRUN_BASE[@]}" \
    bash -lc '
        '"$nsys_str"' torchrun '"$TORCHRUN_STR"'
    '
set +x

end_time=$(TZ='America/Los_Angeles' date +%s)
elapsed_time=$((end_time - start_time))
echo "Finished running sbatch at $(TZ='America/Los_Angeles' date)."
echo_and_tee "$EXP_README" "- Elapsed time: $elapsed_time seconds"


# Check if the experiment finished successfully
if [ ! -f ${OUTPUT_DIR}/benchmark.json ]; then
    echo "Experiment failed. The benchmark.json file does not exist."
else 
    echo "Experiment success. See the $OUTPUT_DIR/benchmark.json file."
    echo_and_tee "$EXP_README" "Experiment success. See the $OUTPUT_DIR/benchmark.json file."
fi


# Check if OOM happened by checking all log files.
if grep -H -C 20 'OutOfMemoryError' "$LOG_DIR"/logs/*.log > /dev/null 2>&1; then
    grep -H -C 20 'OutOfMemoryError' "$LOG_DIR"/logs/*.log > "$OUTPUT_DIR/exit_status.oom.txt"
fi
