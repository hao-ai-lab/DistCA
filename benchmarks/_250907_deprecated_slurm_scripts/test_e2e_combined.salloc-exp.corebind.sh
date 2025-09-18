
# ------------------------------------------------------
# Handle salloc+srun env variables
# ------------------------------------------------------

echo $SLURM_JOB_NAME $SLURM_JOB_ID


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

# Dataset sampling settings
UP_SAMPLE_FACTOR=${UP_SAMPLE_FACTOR:-4}
ELONGATE_FACTOR=${ELONGATE_FACTOR:-1}
FILTER_THRESHOLD=${FILTER_THRESHOLD:-65536}
FILTER_RATIO=${FILTER_RATIO:-0.50}
SHOULD_ADD_DEBUG_CASES=${SHOULD_ADD_DEBUG_CASES:-0}
PROFILE_MEMORY_PATH=${PROFILE_MEMORY_PATH:"${OUTPUT_DIR}/"}

NNODES=${NNODES:-$SLURM_NNODES}

cmd="MODE=$MODE MODEL_PATH=$MODEL_PATH BATCH_SIZE=$BATCH_SIZE NUM_TOKENS=$NUM_TOKENS MAX_SAMPLE_ID=$MAX_SAMPLE_ID TP_SIZE=$TP_SIZE CP_SIZE=$CP_SIZE NUM_LAYERS=$NUM_LAYERS sbatch --nodes $NNODES test_e2e_combined.slurm.sh"


# ------------------------------------------------------
# Setup loggings and artifact directories
# ------------------------------------------------------
TS=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)

# Get the current directory of the script
cd $HOME/jd/d2/tests

# TOOD: Fix this hardcode output dir.
OUTPUT_DIR_PREFIX=${OUTPUT_DIR_PREFIX:-"$HOME/jd/d2/tests/logs"}
OUTPUT_DIR_SUFFIX=${OUTPUT_DIR_SUFFIX:-"$TS.job$SLURM_JOB_NAME-${SLURM_JOB_ID}.${MODE}-cp${CP_SIZE}-n${NNODES}-b${BATCH_SIZE}-t${NUM_TOKENS}"}
OUTPUT_DIR="$OUTPUT_DIR_PREFIX/$OUTPUT_DIR_SUFFIX"
mkdir -p "$OUTPUT_DIR"

# Redirect the output of this script to the output directory
# exec > $OUTPUT_DIR/slurm.stdout 2> $OUTPUT_DIR/slurm.stderr
# exec > $OUTPUT_DIR/slurm.stdout 2>&1
exec > >(tee "$OUTPUT_DIR/slurm.stdout") 2>&1

export LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

set -x


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
# export NVSHMEM_DEBUG=INFO        # or DEBUG/TRACE for deeper
# export NVSHMEM_LOG_LEVEL=INFO
# export NVSHMEM_DEBUG=DEBUG
# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NVSHMEM_IBGDA_NUM_DCI=8
# export NVSHMEM_IBGDA_NUM_DCT=8
# export NVSHMEM_IBGDA_NUM_RC_PER_PE=4

# TODO: Debug
# echo "Debugging: unsetting NVSHMEM_IB_ENABLE_IBGDA, NVSHMEM_IBGDA_NUM_DCI, NVSHMEM_IBGDA_NUM_DCT, NVSHMEM_IBGDA_NUM_RC_PER_PE to avoid the nvshmem init failure"
# unset NVSHMEM_IB_ENABLE_IBGDA 
# unset NVSHMEM_IBGDA_NUM_DCI 
# unset NVSHMEM_IBGDA_NUM_DCT 
# unset NVSHMEM_IBGDA_NUM_RC_PER_PE

# Comment out for a clean log.
# export CUDA_LAUNCH_BLOCKING=1 
# export D2_DEBUG_PRINT=0
# export D2_FA2A_DISABLE_SEND_RECV=0
# export WLBLLM_DISABLE_LSE=1
# export WLBLLM_SYNC_TIME_FLASH_ATTN=1

# ---------------------------
# Setup paths
# ---------------------------
# export CUDA_DIR=/usr/local/cuda
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
SHOULD_ADD_DEBUG_CASES=${SHOULD_ADD_DEBUG_CASES:-1}
EXPERIMENT_LOG_MEMORY_USAGE=${EXPERIMENT_LOG_MEMORY_USAGE:-0}
EXPERIMENT_ADD_SELECTIVE_CKPT=${EXPERIMENT_ADD_SELECTIVE_CKPT:-0}
EXPERIMENT_SHOULD_RESEND_QKV=${EXPERIMENT_SHOULD_RESEND_QKV:-0}

export NVTE_NVTX_ENABLED=1
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


# ---------------------------
# Setup distributed args
# ---------------------------
# echo $(hostname)
# echo SLURM_JOB_NODELIST $SLURM_JOB_NODELIST

# # nodes=$SLURM_JOB_NODELIST # 
# nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
# echo nodes $nodes
# exit 0
# nodes_array=($nodes)
# export head_node=${nodes[0]}
# head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# echo "$SLURM_STEP_NODELIST"        # compact nodelist for THIS step
# echo "$SLURM_STEP_NUM_NODES"       # number of nodes in this step
# scontrol show hostnames "$SLURM_STEP_NODELIST"  # expand to one hostname per line
# exit 0

# nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
JOBID=${JOBID:-SLURM_JOB_ID}
if [ -z "$JOBID" ]; then
  echo "JOBID is not set"
  exit 1
fi

nodes=( $(scontrol show hostnames $(scontrol show job $JOBID | awk -F= '/NodeList=fs/ {print $2}') ) )
echo nodes "${nodes[@]}"
export head_node=${nodes[0]}
# export head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
export head_node_ip=$head_node
port=29500
echo head_node_ip=$head_node_ip port=$port

RZV_BACKEND=c10d
RZV_ENDPOINT=$head_node_ip:$port
echo RZV_ENDPOINT=$RZV_ENDPOINT

# Get GPU count from Slurm's environment variables
# SLURM_GPUS_PER_NODE is set by Slurm when using --gpus-per-node or --gres=gpu:N
GPUS_PER_NODE=8
# if [ -n "$SLURM_GPUS_ON_NODE" ]; then
#   GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
# else
#   GPUS_PER_NODE=8
# fi
NUM_NODES=${NUM_NODES:-$NNODES}
WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))


# Build the common stem *format* (anything that needs per-node must be computed on the node)
COMMON_STEM="mode${MODE}.nnodes${NNODES}.bs${BATCH_SIZE}.maxid${MAX_SAMPLE_ID}.tp${TP_SIZE}.pp${PP_SIZE}.cp${CP_SIZE}.t${NUM_TOKENS}.elong${ELONGATE_FACTOR}.up${UP_SAMPLE_FACTOR}.ft${FILTER_THRESHOLD}.fr${FILTER_RATIO}"

touch ${OUTPUT_DIR}/desc.${COMMON_STEM} # just a description of this experiment, in its file name

# Define missing variables with defaults
NNODES=${NNODES:-$NUM_NODES}
NPROC_PER_NODE=${NPROC_PER_NODE:-$GPUS_PER_NODE}
RZV_ID=${RZV_ID:-$head_node_ip}
REPLAN_ITER=${REPLAN_ITER:-0}
SHOULD_PROFILE_MEMORY=${SHOULD_PROFILE_MEMORY:-0}


# bash ./redirect_log.sh 
TORCHRUN_CMD=(
  --nnodes=${NNODES} \
  --nproc_per_node=${NPROC_PER_NODE} \
  --rdzv_backend=${RZV_BACKEND} \
  --rdzv_endpoint=${RZV_ENDPOINT} \
  --rdzv_id=${RZV_ID} \
  --max_restarts=0 \
  --no-python \
  python test_e2e_combined.py \
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
    --filter-ratio ${FILTER_RATIO} \
    --output-dir ${OUTPUT_DIR}
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

# Serialize TORCHRUN_CMD array so we can pass it through bash -lc cleanly
TORCHRUN_STR=$(printf " %q" "${TORCHRUN_CMD[@]}")

# ---- Per-node logs + per-node NSYS outputs ----
# %N and %j are expanded by Slurm *only* in --output/--error.
# Inside the bash -lc block we compute HOST and build node-specific file names for NSYS/other artifacts.

SRUN_BASE=(
  srun
  -N ${NNODES}
  -G ${WORLD_SIZE}
  # --ntasks-per-node=1
  # --gpus-per-task=8
  # --cpus-per-task=64
  # --cpu-bind=cores
  # --gpu-bind=closest
  # --kill-on-bad-exit=1
  -w "$head_node"
  --mem=0 # inherit the memory from the salloc
)

#   --jobid=${JOBID:-SLURM_JOB_ID}
if [ ${JOBID} -ne 0 ]; then
  SRUN_BASE+=(--jobid=${JOBID})
fi


# ---------------------------
# Log environment variables (for debugging)
# ---------------------------
env > $OUTPUT_DIR/slurm.env

EXP_README="$OUTPUT_DIR/README.md"
echo "Running experiment with the following parameters:" > $EXP_README
echo "## Model settings" >> $EXP_README
echo "- MODE: $MODE" >> $EXP_README
echo "- MODEL_PATH: $MODEL_PATH" >> $EXP_README
echo "- NNODES: $NNODES" >> $EXP_README
echo "- NUM_LAYERS: $NUM_LAYERS" >> $EXP_README
echo "- TP_SIZE: $TP_SIZE" >> $EXP_README
echo "- PP_SIZE: $PP_SIZE" >> $EXP_README
echo "- CP_SIZE: $CP_SIZE" >> $EXP_README
echo "- BATCH_SIZE: $BATCH_SIZE" >> $EXP_README
echo "- NUM_TOKENS: $NUM_TOKENS" >> $EXP_README
echo "- MAX_SAMPLE_ID: $MAX_SAMPLE_ID" >> $EXP_README
echo "- UP_SAMPLE_FACTOR: $UP_SAMPLE_FACTOR" >> $EXP_README
echo "- ELONGATE_FACTOR: $ELONGATE_FACTOR" >> $EXP_README
echo "- FILTER_THRESHOLD: $FILTER_THRESHOLD" >> $EXP_README
echo "- FILTER_RATIO: $FILTER_RATIO" >> $EXP_README
echo "- SHOULD_ADD_DEBUG_CASES: $SHOULD_ADD_DEBUG_CASES" >> $EXP_README
echo "## Experiment Flags" >> $EXP_README
echo "- EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB: $EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB" >> $EXP_README
echo "- EXPERIMENT_LOG_MEMORY_USAGE: $EXPERIMENT_LOG_MEMORY_USAGE" >> $EXP_README
echo "- EXPERIMENT_SHOULD_FORCE_EXIT: $EXPERIMENT_SHOULD_FORCE_EXIT" >> $EXP_README
echo "- EXPERIMENT_EMIT_BACKWARD_NVTX: $EXPERIMENT_EMIT_BACKWARD_NVTX" >> $EXP_README
echo "- EXPERIMENT_WARMUP_TIMEOUT_SEC: $EXPERIMENT_WARMUP_TIMEOUT_SEC" >> $EXP_README
echo "- D2_SHOULD_REPLAN: $D2_SHOULD_REPLAN" >> $EXP_README
echo "- EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0: $EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0" >> $EXP_README
echo "- EXPERIMENT_ADD_SELECTIVE_CKPT: $EXPERIMENT_ADD_SELECTIVE_CKPT" >> $EXP_README
echo "- EXPERIMENT_SHOULD_RESEND_QKV: $EXPERIMENT_SHOULD_RESEND_QKV" >> $EXP_README

# Generate equivalent command
echo "" >> $EXP_README
echo "- Command: $cmd" >> $EXP_README


# ---------------------------
# Run the experiment
# ---------------------------

DRY_RUN=${DRY_RUN:-0}

if [ ${DRY_RUN} -eq 1 ]; then
  SRUN_BASE=(
    echo
    ${SRUN_BASE[@]}
  )
    
fi

echo "Start running sbatch at $(TZ='America/Los_Angeles' date)"

NSYS_PATH="${OUTPUT_DIR}/nsys-reps"
mkdir -p ${NSYS_PATH}

nsys_str=""
if [ ${ENABLE_NSYS} -eq 1 ]; then
  nsys_str="nsys profile --show-output=true --force-overwrite=true -o ${NSYS_PATH}/%h.nsys-rep --sample=none -t cuda,nvtx"
fi

unset OMP_NUM_THREADS
start_time=$(TZ='America/Los_Angeles' date +%s)

"${SRUN_BASE[@]}" \
    --output="${LOG_DIR}/%N.%j.%s.out" \
    --error="${LOG_DIR}/%N.%j.%s.out" \
    bash -lc '
        set -x
        # export OMP_NUM_THREADS=1
        hostname
        nvidia-smi topo -p2p w     # Check nvidia-smi topology
        
        env # to check the nvshmem environment variables
        '"$nsys_str"' torchrun '"$TORCHRUN_STR"'
    '

end_time=$(TZ='America/Los_Angeles' date +%s)
elapsed_time=$((end_time - start_time))
echo "Finished running sbatch at $(TZ='America/Los_Angeles' date). Does not guarantee that the experiment finished successfully. Please check if the benchmark.json file exists."
echo "Elapsed time: $elapsed_time seconds"
echo "Elapsed time: $elapsed_time seconds" >> $OUTPUT_DIR/README.md


# Check if the experiment finished successfully
if [ ! -f ${OUTPUT_DIR}/benchmark.json ]; then
    echo "Experiment failed. The benchmark.json file does not exist."
fi


set +x

echo '\a'
echo '\a'
echo '\a'
# set -euox pipefail