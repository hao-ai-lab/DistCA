#! /bin/bash

#SBATCH --job-name=d2pp-e2e
#SBATCH --nodes=4
#SBATCH --output=logs/slurm/stdout.%j.log
#SBATCH --error=logs/slurm/stderr.%j.log
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --partition=lowprio 
#SBATCH --qos=lowprio
#SBATCH --requeue 

# ------------------------------------------------------
# Input Variables (what you will set outside)
# ------------------------------------------------------

# Model configuration
# MODEL_PATH=${MODEL_PATH:-codellama/CodeLlama-34b-hf}
MODEL_PATH=${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}
NUM_LAYERS=${NUM_LAYERS:-8}

# Parallelism settings
TP_SIZE=${TP_SIZE:-$GPUS_PER_NODE}   # Tensor Parallelism size, defaults to GPUs per node
TP_SIZE=${TP_SIZE:-8}
PP_SIZE=${PP_SIZE:-1}                # Pipeline Parallelism size
CP_SIZE=${CP_SIZE:-1}                # Only useful in WLBLLM (D2 will have DPCP anyways)
NUM_MICROBATCH=${NUM_MICROBATCH:-${PP_SIZE}}            # Number of microbatches per pipeline stage, has to be >= PP_SIZE - 1

# Experiment settings
MODE=${MODE:-d2}               # Experiment mode (baseline, dynamic, etc.)
BATCH_SIZE=${BATCH_SIZE:-1}          # Batch size for training
# NUM_TOKENS=${NUM_TOKENS:-131072}     # Number of tokens to process
NUM_TOKENS=${NUM_TOKENS:-16384}     # Number of tokens to process
MAX_SAMPLE_ID=${MAX_SAMPLE_ID:-3}   # Maximum sample ID
# SAMPLE_EXPR=${SAMPLE_EXPR:-""}   # Sample expression

# Dataset sampling settings
UP_SAMPLE_FACTOR=${UP_SAMPLE_FACTOR:-4}
ELONGATE_FACTOR=${ELONGATE_FACTOR:-1}
FILTER_THRESHOLD=${FILTER_THRESHOLD:-65536}
FILTER_RATIO=${FILTER_RATIO:-0.50}
SHOULD_ADD_DEBUG_CASES=${SHOULD_ADD_DEBUG_CASES:-0}
PROFILE_MEMORY_PATH=${PROFILE_MEMORY_PATH:"${OUTPUT_DIR}/"}
SAMPLE_NAME=${SAMPLE_NAME:-"wlbllm"}
CHANGE_LONG_DOC_RATIO=${CHANGE_LONG_DOC_RATIO:-0.0}

JOBID=${JOBID:-${SLURM_JOB_ID}}
if [ -z "$JOBID" ]; then
  echo -e "\033[31mJOBID is not set. Must set JOBID environment variable.\033[0m"
  exit 1
fi

# Check if job is still running
if ! squeue -j "$JOBID" &>/dev/null; then
  echo -e "\033[31mError: Job $JOBID is no longer running in SLURM queue\033[0m"
  exit 1
fi

NNODES=${NNODES:-$SLURM_NNODES}
if [ -z "$NNODES" ]; then
    NNODES=$(squeue -j $JOBID -h -o %D)
fi
echo -e "\033[33mRecognized JOBID=$JOBID, NNODES=$NNODES\033[0m"
sleep 1


# ------------------------------------------------------
# Setup loggings and artifact directories
# ------------------------------------------------------
TS=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)
SHORT_TS=$(TZ=America/Los_Angeles date +%d_%H%M%S)

# TOOD: Fix this hardcode output dir.
OUTPUT_DIR_PREFIX=${OUTPUT_DIR_PREFIX:-"$HOME/jd/d2/tests/logs"}
# OUTPUT_DIR_SUFFIX=${OUTPUT_DIR_SUFFIX:-"$TS.job$SLURM_JOB_NAME-${JOBID}.${MODE}-cp${CP_SIZE}tp${TP_SIZE}pp${PP_SIZE}-n${NNODES}-b${BATCH_SIZE}-t${NUM_TOKENS}-mb${NUM_MICROBATCH}"}
OUTPUT_DIR_SUFFIX=${OUTPUT_DIR_SUFFIX:-"$SHORT_TS.${MODE}-cp${CP_SIZE}tp${TP_SIZE}pp${PP_SIZE}-n${NNODES}-b${BATCH_SIZE}-t${NUM_TOKENS}-mb${NUM_MICROBATCH}"}
OUTPUT_DIR_SUFFIX_ADDON=${OUTPUT_DIR_SUFFIX_ADDON:-""}
OUTPUT_DIR="$OUTPUT_DIR_PREFIX/$OUTPUT_DIR_SUFFIX$OUTPUT_DIR_SUFFIX_ADDON"
mkdir -p "$OUTPUT_DIR"

# Redirect the output of this script to the output directory
# exec > $OUTPUT_DIR/slurm.stdout 2> $OUTPUT_DIR/slurm.stderr
# exec > $OUTPUT_DIR/slurm.stdout 2>&1
exec > >(tee "$OUTPUT_DIR/slurm.stdout") 2>&1

export LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"


# ---------------------------
# Environment variables
# ---------------------------
# export NVSHMEM_BOOTSTRAP=mpi
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 
export NVSHMEM_IB_ENABLE_IBGDA=true
# export TORCH_CPP_LOG_LEVEL=INFO 
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_DIST_INIT_RETRY_TIMEOUT=15
export TORCH_DIST_INIT_RETRY_TIMEOUT=30
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_SHOW_CPP_STACKTRACES=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,COLL
# export NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_NCCL_BLOCKING_WAIT=1
# export NCCL_IB_TIMEOUT=14
# export NCCL_NET_GDR_LEVEL=3
# export NVSHMEM_DEBUG=INFO        # or DEBUG/TRACE for deeper
# export NVSHMEM_LOG_LEVEL=INFO
# export NVSHMEM_DEBUG=DEBUG
# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NVSHMEM_IBGDA_NUM_DCI=8
# export NVSHMEM_IBGDA_NUM_DCT=8
# export NVSHMEM_IBGDA_NUM_RC_PER_PE=4
# export NCCL_ASYNC_ERROR_HANDLING=1
# Comment out for a clean log.
# export CUDA_LAUNCH_BLOCKING=1 
# export D2_DEBUG_PRINT=0
# export D2_FA2A_DISABLE_SEND_RECV=0
# export WLBLLM_DISABLE_LSE=1
# export WLBLLM_SYNC_TIME_FLASH_ATTN=1

# ---------------------------
# Setup paths
# ---------------------------
export CUDA_DIR=/mnt/sharefs/software/DeepEP/cuda-12-6
export NCCL_HOME=/usr
export NCCL_LIB=/usr/lib/x86_64-linux-gnu
export NVSHMEM_DIR=/mnt/weka/home/yonghao.zhuang/opt/nvshmem
export NVSHMEM_PREFIX=/mnt/weka/home/yonghao.zhuang/opt/nvshmem
export OPENMPI_DIR=/mnt/weka/home/yonghao.zhuang/opt/openmpi

export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${CUDA_DIR}/lib64:${OPENMPI_DIR}/lib:${NCCL_LIB}/:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:${OPENMPI_DIR}/bin:${CUDA_DIR}/bin:$PATH"

export NVTE_NVTX_ENABLED=1
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0 

# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


# ---------------------------
# Setup experiment variables
# ---------------------------

DRY_RUN=${DRY_RUN:-0}

ENABLE_NSYS=${ENABLE_NSYS:-0}
EXPERIMENT_REPEAT_TIMES=${EXPERIMENT_REPEAT_TIMES:-3}
EXPERIMENT_WARMUP_TIMES=${EXPERIMENT_WARMUP_TIMES:-5}
EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=${EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB:--1}


# ---------------------------
# Setup distributed args
# ---------------------------

nodes=( $(scontrol show hostnames $(scontrol show job $JOBID | awk -F= '/NodeList=fs/ {print $2}') ) )
echo nodes "${nodes[@]}"
# export head_node=${nodes[0]}
export head_node_ip=${HEAD_NODE_IP:-$head_node}
export head_node=${head_node:-$head_node_ip}
# port=$(shuf -i 29000-30000 -n 1)
port=29500
echo head_node_ip=$head_node_ip port=$port
# RZV_ID=$(shuf -i 9000-300000 -n 1)
RZV_BACKEND=c10d
RZV_ENDPOINT=$head_node_ip:$port
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
NPROC_PER_NODE=${NPROC_PER_NODE:-$GPUS_PER_NODE}
RZV_ID=${RZV_ID:-$head_node_ip-${TS}}
REPLAN_ITER=${REPLAN_ITER:-0}
SHOULD_PROFILE_MEMORY=${SHOULD_PROFILE_MEMORY:-0}


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
  -w "$head_node"
  --mem=0 # inherit the memory from the salloc
)

# -vv


if [ ${JOBID} -ne 0 ]; then
  IS_LOCAL_RUN=${IS_LOCAL_RUN:-0}
  if [ ${IS_LOCAL_RUN} -eq 0 ]; then
    SRUN_BASE+=(--jobid=${JOBID})
  fi
fi


TORCHRUN_CMD=(
  --nnodes=${NNODES}
  --nproc_per_node=${NPROC_PER_NODE}
  --rdzv_backend=${RZV_BACKEND}
  --rdzv_endpoint=${RZV_ENDPOINT}
  --rdzv_id=${RZV_ID}
  --max_restarts=0
  --no-python bash ./bind_and_exec.sh 
    python test_megatron_e2e_pipeline_with_cp.py
    --num-tokens ${NUM_TOKENS}
    --num-batches ${BATCH_SIZE}
    --num-nodes ${NNODES}
    --num-gpus-per-node ${NPROC_PER_NODE}
    --cp-size ${CP_SIZE}
    --tp-size ${TP_SIZE}
    --pp-size ${PP_SIZE}
    --num-microbatch ${NUM_MICROBATCH}
    --use-planner
    
    --model-path ${MODEL_PATH}
    --num-layers ${NUM_LAYERS}

    --max-sample-id ${MAX_SAMPLE_ID}
    --sample-name ${SAMPLE_NAME}
    --change-long-doc-ratio ${CHANGE_LONG_DOC_RATIO}

    --up-sample-factor ${UP_SAMPLE_FACTOR}
    --elongate-factor ${ELONGATE_FACTOR}
    --filter-threshold ${FILTER_THRESHOLD}
    --filter-ratio ${FILTER_RATIO}
    --output-dir ${OUTPUT_DIR}
)

# if [ ${USE_PLANNER} -eq 1 ]; then
#     TORCHRUN_CMD+=(--use-planner)
# fi


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

EXP_README="$OUTPUT_DIR/README.md"
echo_and_tee "$EXP_README" "Running experiment with the following parameters:"

echo_and_tee "$EXP_README" "## Model settings"
echo_and_tee "$EXP_README" "- MODE: $MODE"
echo_and_tee "$EXP_README" "- MODEL_PATH: $MODEL_PATH"
echo_and_tee "$EXP_README" "- NNODES: $NNODES"
echo_and_tee "$EXP_README" "- NUM_LAYERS: $NUM_LAYERS"
echo_and_tee "$EXP_README" "- TP_SIZE: $TP_SIZE"
echo_and_tee "$EXP_README" "- PP_SIZE: $PP_SIZE"
echo_and_tee "$EXP_README" "- CP_SIZE: $CP_SIZE"
echo_and_tee "$EXP_README" "- NUM_MICROBATCH: $NUM_MICROBATCH"
echo_and_tee "$EXP_README" "- BATCH_SIZE: $BATCH_SIZE"
echo_and_tee "$EXP_README" "- NUM_TOKENS: $NUM_TOKENS"
echo_and_tee "$EXP_README" "- MAX_SAMPLE_ID: $MAX_SAMPLE_ID"
echo_and_tee "$EXP_README" "- UP_SAMPLE_FACTOR: $UP_SAMPLE_FACTOR"
echo_and_tee "$EXP_README" "- ELONGATE_FACTOR: $ELONGATE_FACTOR"
echo_and_tee "$EXP_README" "- FILTER_THRESHOLD: $FILTER_THRESHOLD"
echo_and_tee "$EXP_README" "- FILTER_RATIO: $FILTER_RATIO"
echo_and_tee "$EXP_README" "- SHOULD_ADD_DEBUG_CASES: $SHOULD_ADD_DEBUG_CASES"
echo_and_tee "$EXP_README" "- SAMPLE_START_IDX: $SAMPLE_START_IDX"

echo_and_tee "$EXP_README" "## Experiment Flags"
echo_and_tee "$EXP_README" "- ENABLE_NSYS: $ENABLE_NSYS"
echo_and_tee "$EXP_README" "- WLBLLM_SYNC_TIME_AG: $WLBLLM_SYNC_TIME_AG"
echo_and_tee "$EXP_README" "- EXPERIMENT_REPEAT_TIMES: $EXPERIMENT_REPEAT_TIMES"
echo_and_tee "$EXP_README" "- EXPERIMENT_WARMUP_TIMES: $EXPERIMENT_WARMUP_TIMES"
echo_and_tee "$EXP_README" "- EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB: $EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB"
echo_and_tee "$EXP_README" "- EXPERIMENT_SHOULD_FORCE_EXIT: $EXPERIMENT_SHOULD_FORCE_EXIT"
echo_and_tee "$EXP_README" "- EXPERIMENT_EMIT_BACKWARD_NVTX: $EXPERIMENT_EMIT_BACKWARD_NVTX"
echo_and_tee "$EXP_README" "- EXPERIMENT_WARMUP_TIMEOUT_SEC: $EXPERIMENT_WARMUP_TIMEOUT_SEC"
echo_and_tee "$EXP_README" "- D2_SHOULD_REPLAN: $D2_SHOULD_REPLAN"
echo_and_tee "$EXP_README" "- SHOULD_PROFILE_MEMORY: $SHOULD_PROFILE_MEMORY"
echo_and_tee "$EXP_README" "- SHOULD_ADD_DEBUG_CASES: $SHOULD_ADD_DEBUG_CASES"
echo_and_tee "$EXP_README" "- EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0: $EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0"
echo_and_tee "$EXP_README" "- EXPERIMENT_LOG_MEMORY_USAGE: $EXPERIMENT_LOG_MEMORY_USAGE"
echo_and_tee "$EXP_README" "- EXPERIMENT_ADD_SELECTIVE_CKPT: $EXPERIMENT_ADD_SELECTIVE_CKPT"
echo_and_tee "$EXP_README" "- EXPERIMENT_SHOULD_RESEND_QKV: $EXPERIMENT_SHOULD_RESEND_QKV"
echo_and_tee "$EXP_README" "- D2_SKIP_FLOAT_CONVERSION: $D2_SKIP_FLOAT_CONVERSION"
echo_and_tee "$EXP_README" "- EXPERIMENT_D2_FLASH_ATTN_SKIP_GET_BACKEND: $EXPERIMENT_D2_FLASH_ATTN_SKIP_GET_BACKEND"
echo_and_tee "$EXP_README" "- EXPERIMENT_SKIP_OPTIMIZER_STEP: $EXPERIMENT_SKIP_OPTIMIZER_STEP"
echo_and_tee "$EXP_README" "- EXPERIMENT_OVERLAP_PARAM_GATHER_WITH_OPTIMIZER_STEP: $EXPERIMENT_OVERLAP_PARAM_GATHER_WITH_OPTIMIZER_STEP"
echo_and_tee "$EXP_README" "- EXPERIMENT_D2_BALANCE_PING_PONG: $EXPERIMENT_D2_BALANCE_PING_PONG"
echo_and_tee "$EXP_README" "- EXPERIMENT_SHOULD_DUMP_TRACEBACK: $EXPERIMENT_SHOULD_DUMP_TRACEBACK"
echo_and_tee "$EXP_README" "- EXPERIMENT_TORCH_DIST_TIMEOUT: $EXPERIMENT_TORCH_DIST_TIMEOUT"
echo_and_tee "$EXP_README" "- EXPERIMENT_ENABLE_BENCHMARK_SAVING: $EXPERIMENT_ENABLE_BENCHMARK_SAVING"


echo_and_tee "$EXP_README" "## Other Variables"
echo_and_tee "$EXP_README" "- TS: $TS"
echo_and_tee "$EXP_README" "- JOBID: $JOBID"
echo_and_tee "$EXP_README" "- OUTPUT_DIR: $OUTPUT_DIR"


# ---------------------------
# Run the experiment
# ---------------------------

DRY_RUN=${DRY_RUN:-0}

if [ ${DRY_RUN} -eq 1 ]; then
  SRUN_BASE=(echo ${SRUN_BASE[@]})
fi

echo "Start running sbatch at $(TZ='America/Los_Angeles' date)"

NSYS_PATH="${OUTPUT_DIR}/nsys-reps"
mkdir -p ${NSYS_PATH}



# nsys_str="nsys profile --show-output=true --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true -o ${NSYS_PATH}/%h.nsys-rep --sample=none -t cuda,nvtx"
# WARNING: CUDA backtraces will not be collected because CPU sampling is disabled.
# WARNING: CPU IP/backtrace sampling not supported, disabling.
# Try the 'nsys status --environment' command to learn more.
nsys_str=""
if [ ${ENABLE_NSYS} -eq 1 ]; then
  nsys_str="nsys profile --show-output=true --force-overwrite=true -o ${NSYS_PATH}/%h.nsys-rep -t cuda,nvtx"
  # nsys_str="nsys profile --show-output=true --force-overwrite=true -o ${NSYS_PATH}/%h.nsys-rep -t cuda,nvtx,osrt --cudabacktrace=true --backtrace=auto "
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
fi


echo '\a'
echo '\a'
echo '\a'
