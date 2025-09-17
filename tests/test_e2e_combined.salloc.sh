#! /bin/bash

#SBATCH --job-name=d2-e2e
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
MODEL_PATH=${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}
NUM_LAYERS=${NUM_LAYERS:-4}

# Parallelism settings
TP_SIZE=${TP_SIZE:-$GPUS_PER_NODE}   # Tensor Parallelism size, defaults to GPUs per node
TP_SIZE=${TP_SIZE:-8}
PP_SIZE=${PP_SIZE:-1}                # Pipeline Parallelism size
CP_SIZE=${CP_SIZE:-1}                # Only useful in WLBLLM (D2 will have DPCP anyways)

# Experiment settings
MODE=${MODE:-d2}               # Experiment mode (baseline, dynamic, etc.)
BATCH_SIZE=${BATCH_SIZE:-1}          # Batch size for training
NUM_TOKENS=${NUM_TOKENS:-65536}     # Number of tokens to process
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

# cmd="MODE=$MODE MODEL_PATH=$MODEL_PATH BATCH_SIZE=$BATCH_SIZE NUM_TOKENS=$NUM_TOKENS MAX_SAMPLE_ID=$MAX_SAMPLE_ID TP_SIZE=$TP_SIZE CP_SIZE=$CP_SIZE NUM_LAYERS=$NUM_LAYERS sbatch --nodes $NNODES test_e2e_combined.slurm.sh"


# ------------------------------------------------------
# Setup loggings and artifact directories
# ------------------------------------------------------
TS=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)

# Get the current directory of the script
cd $HOME/jd/d2/tests

# TOOD: Fix this hardcode output dir.
OUTPUT_DIR_PREFIX=${OUTPUT_DIR_PREFIX:-"$HOME/jd/d2/tests/logs"}
OUTPUT_DIR_SUFFIX=${OUTPUT_DIR_SUFFIX:-"$TS.job$SLURM_JOB_NAME-${JOBID}.${MODE}-cp${CP_SIZE}-n${NNODES}-b${BATCH_SIZE}-t${NUM_TOKENS}"}
OUTPUT_DIR_SUFFIX_ADDON=${OUTPUT_DIR_SUFFIX_ADDON:-""}
OUTPUT_DIR="$OUTPUT_DIR_PREFIX/$OUTPUT_DIR_SUFFIX$OUTPUT_DIR_SUFFIX_ADDON"
mkdir -p "$OUTPUT_DIR"

# Redirect the output of this script to the output directory
# exec > $OUTPUT_DIR/slurm.stdout 2> $OUTPUT_DIR/slurm.stderr
# exec > $OUTPUT_DIR/slurm.stdout 2>&1
exec > >(tee "$OUTPUT_DIR/slurm.stdout") 2>&1

export LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"




# ------------------------------------------------------

# ---------------------------
# Env and Sanity Check
# ---------------------------

# conda activate jd-d2

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name --format=csv



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
EXPERIMENT_WARMUP_TIMEOUT_SEC=${EXPERIMENT_WARMUP_TIMEOUT_SEC:-240}
EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0=${EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0:-0}
D2_SHOULD_REPLAN=${D2_SHOULD_REPLAN:-1}
D2_SKIP_FLOAT_CONVERSION=${D2_SKIP_FLOAT_CONVERSION:-1}
SHOULD_ADD_DEBUG_CASES=${SHOULD_ADD_DEBUG_CASES:-1}
# EXPERIMENT_LOG_MEMORY_USAGE: 
#   Log memory usage (using torch.cuda.memory_summary and pynvml) to file and console.
#   May have overhead, so only enable it when needed.
EXPERIMENT_LOG_MEMORY_USAGE=${EXPERIMENT_LOG_MEMORY_USAGE:-0}
EXPERIMENT_ADD_SELECTIVE_CKPT=${EXPERIMENT_ADD_SELECTIVE_CKPT:-0}
# EXPERIMENT_SHOULD_RESEND_QKV:
#   Flag to resend QKV to the WLBLLM.
#   If set to 1, the QKV will be resend to the WLBLLM after the first forward pass.
#   This is useful when the WLBLLM is not able to process the QKV in the first forward pass.
#   However, this may have overhead, so only enable it when needed.
EXPERIMENT_SHOULD_RESEND_QKV=${EXPERIMENT_SHOULD_RESEND_QKV:-0}
EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0=${EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0:-0}
EXPERIMENT_D2_FLASH_ATTN_SKIP_GET_BACKEND=${EXPERIMENT_D2_FLASH_ATTN_SKIP_GET_BACKEND:-1}
EXPERIMENT_SKIP_OPTIMIZER_STEP=${EXPERIMENT_SKIP_OPTIMIZER_STEP:-0}
EXPERIMENT_OVERLAP_PARAM_GATHER_WITH_OPTIMIZER_STEP=${EXPERIMENT_OVERLAP_PARAM_GATHER_WITH_OPTIMIZER_STEP:-0}
EXPERIMENT_SHOULD_DUMP_TRACEBACK=${EXPERIMENT_SHOULD_DUMP_TRACEBACK:-0}
EXPERIMENT_TORCH_DIST_TIMEOUT=${EXPERIMENT_TORCH_DIST_TIMEOUT:--1}
EXPERIMENT_ENABLE_BENCHMARK_SAVING=${EXPERIMENT_ENABLE_BENCHMARK_SAVING:-1}
EXPERIMENT_D2_BALANCE_PING_PONG=${EXPERIMENT_D2_BALANCE_PING_PONG:-0}
SAMPLE_START_IDX=${SAMPLE_START_IDX:-0}

export NVTE_NVTX_ENABLED=1
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


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
if [ -z "$head_node" ]; then
    echo -e "\033[31mERROR: head_node is empty. Please set HEAD_NODE_IP environment variable.\033[0m"
    exit 1
fi


# Get GPU count from Slurm's environment variables
# SLURM_GPUS_PER_NODE is set by Slurm when using --gpus-per-node or --gres=gpu:N
GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE}}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))


# Build the common stem *format* (anything that needs per-node must be computed on the node)
COMMON_STEM="mode${MODE}.nnodes${NNODES}.bs${BATCH_SIZE}.maxid${MAX_SAMPLE_ID}.tp${TP_SIZE}.pp${PP_SIZE}.cp${CP_SIZE}.t${NUM_TOKENS}.elong${ELONGATE_FACTOR}.up${UP_SAMPLE_FACTOR}.ft${FILTER_THRESHOLD}.fr${FILTER_RATIO}"

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
  # -vv
  --ntasks-per-node=1
  # --gpus-per-task=${GPUS_PER_NODE}     # <= crucial
  --cpus-per-task=128
  --cpu-bind=cores
  # --gpu-bind=closest
  # --kill-on-bad-exit=1
  -w "$head_node"
  --mem=0 # inherit the memory from the salloc
)

# SRUN_INCLUDE_NODES
# if not "", then add `-w` with this env var
echo "SRUN_INCLUDE_NODES=${SRUN_INCLUDE_NODES}"
if [ -n "${SRUN_INCLUDE_NODES}" ]; then
  SRUN_BASE+=(-w "${SRUN_INCLUDE_NODES}")
fi
echo srun command: 
echo "${SRUN_BASE[@]}"
# exit 0


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
  --no-python bash ./bind_and_exec.sh python test_e2e_combined.py
    --model-path ${MODEL_PATH}
    --mode ${MODE}
    --replan-iter ${REPLAN_ITER}
    --batch-size ${BATCH_SIZE}
    --num-nodes ${NNODES}
    --num-gpus-per-node ${NPROC_PER_NODE}
    --num-layers ${NUM_LAYERS}
    --max-sample-id ${MAX_SAMPLE_ID}
    --tp-size ${TP_SIZE}
    --cp-degree ${CP_SIZE}
    --up-sample-factor ${UP_SAMPLE_FACTOR}
    --num-tokens ${NUM_TOKENS}
    --elongate-factor ${ELONGATE_FACTOR}
    --filter-threshold ${FILTER_THRESHOLD}
    --filter-ratio ${FILTER_RATIO}
    --output-dir ${OUTPUT_DIR}
    --sample-name ${SAMPLE_NAME}
    --change-long-doc-ratio ${CHANGE_LONG_DOC_RATIO}
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

if [ ${EXPERIMENT_SHOULD_RESEND_QKV} -eq 1 ]; then
    TORCHRUN_CMD+=(--should-resend-qkv)
fi
if [ ${SAMPLE_START_IDX} -ne "" ]; then
    TORCHRUN_CMD+=(--sample-start-idx ${SAMPLE_START_IDX})
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
# Generate equivalent command



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

nsys_str=""
if [ ${ENABLE_NSYS} -eq 1 ]; then
  nsys_str="nsys profile --show-output=true --force-overwrite=true -o ${NSYS_PATH}/%h.nsys-rep --sample=none -t cuda,nvtx"
fi

start_time=$(TZ='America/Los_Angeles' date +%s)

# export OMP_NUM_THREADS=1

# Add some sanity checks here to ensure torchrun is functioning correctly, or no connectivity issue arises.
# srun -N 32 -G 256 --jobid=$JOBID torchrun --nnodes=32 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=fs-mbz-gpu-033:29500 --rdzv_id=59090 --no-python hostname

# Test 1
# "${SRUN_BASE[@]}" hostname

# Test 2
set -x
# "${SRUN_BASE[@]}" torchrun --nnodes=${NNODES} \
#   --nproc_per_node=${NPROC_PER_NODE} \
#   --rdzv_backend=${RZV_BACKEND} \
#   --rdzv_endpoint=${RZV_ENDPOINT} \
#   --rdzv_id=${RZV_ID} \
#   --max_restarts=0 \
#   --no-python hostname


# This checks torchrun is working correctly, and allreduce is working correctly.
PRECHECK_TORCHRUN_SIMPLE_ALL_REDUCE=${PRECHECK_TORCHRUN_SIMPLE_ALL_REDUCE:-0}
if [ ${PRECHECK_TORCHRUN_SIMPLE_ALL_REDUCE} -eq 1 ]; then
  "${SRUN_BASE[@]}" --output="${LOG_DIR}/%N.%j.%s.out" \
    --error="${LOG_DIR}/%N.%j.%s.out" \
    torchrun --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --rdzv_backend=${RZV_BACKEND} \
    --rdzv_endpoint=${RZV_ENDPOINT} \
    --rdzv_id=${RZV_ID} \
    --max_restarts=0 \
    --no-python python simple_torch.py
fi

# Test 3
# "${SRUN_BASE[@]}" --output="${LOG_DIR}/%N.%j.%s.out" \
#   --error="${LOG_DIR}/%N.%j.%s.out" \
#   torchrun --nnodes=${NNODES} \
#   --nproc_per_node=${NPROC_PER_NODE} \
#   --rdzv_backend=${RZV_BACKEND} \
#   --rdzv_endpoint=${RZV_ENDPOINT} \
#   --rdzv_id=${RZV_ID} \
#   --max_restarts=0 \
#   --no-python python -c "
# import torch
# import torch.distributed as dist
# import os
# local_rank = int(os.environ.get('LOCAL_RANK'))
# print(f'CUDA available: {torch.cuda.is_available()}')
# torch.cuda.set_device(f'cuda:{local_rank}')
# print(f'Using device: {torch.cuda.get_device_name()}')

# dist.init_process_group('nccl')
# rank = dist.get_rank()
# world_size = dist.get_world_size()
# print(f'Rank {rank}/{world_size} initialized')

# x = torch.ones(1).cuda()
# dist.all_reduce(x)
# print(f'Rank {rank}: After all_reduce x = {x} (should be {world_size})')
# "


# exit 1


# --output="${LOG_DIR}/%N.%j.%s.out" \
# --error="${LOG_DIR}/%N.%j.%s.out" \
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
    echo "🔴 Experiment failed. The benchmark.json file does not exist. See $OUTPUT_DIR/slurm.stdout and the logs for more details."
else 
    echo "🟢 Experiment success. See the $OUTPUT_DIR/benchmark.json file."
fi


echo '\a'
echo '\a'
echo '\a'
