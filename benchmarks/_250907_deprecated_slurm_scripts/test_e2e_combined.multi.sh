
# Usage:
# bash test_e2e_combined.multi.sh localhost:29500 1 2 2 baseline unique_rdbzid
# bash test_e2e_combined.multi.sh localhost:29500 2 8 8 baseline unique_rdbzid_baseline
#

# export CUDA_LAUNCH_BLOCKING=1 
D2_DEBUG_PRINT=${D2_DEBUG_PRINT:-0}
# export WLBLLM_DISABLE_LSE=1
# export WLBLLM_SYNC_TIME_FLASH_ATTN=1

WLBLLM_SYNC_TIME_AG=${WLBLLM_SYNC_TIME_AG:-0}
# How many time should each iteration repeat
EXPERIMENT_REPEAT_TIMES=${EXPERIMENT_REPEAT_TIMES:-3}
EXPERIMENT_WARMUP_TIMES=${EXPERIMENT_REPEAT_TIMES:-5}
EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=${EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB:--1}
EXPERIMENT_SHOULD_FORCE_EXIT=${EXPERIMENT_SHOULD_FORCE_EXIT:-0}
EXPERIMENT_EMIT_BACKWARD_NVTX=${EXPERIMENT_EMIT_BACKWARD_NVTX:-0}

export NVTE_NVTX_ENABLED=1
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0 
export NVSHMEM_IB_ENABLE_IBGDA=true 
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

ENABLE_NSYS=${ENABLE_NSYS:-0}

# ---------------------------
RZV_BACKEND=c10d
RZV_ENDPOINT=$1
NNODES=${2:-1}
NPROC_PER_NODE=${3:-8}
TP_SIZE=${4:-8}
MODE=${5:-baseline}
RZV_ID=$6

# MODE=d2

if [ -z "${RZV_ENDPOINT}" ]; then
    echo "Usage: bash $0 <rzv_endpoint> <n_nodes> <n_gpus_per_node> <tp_size> <mode> <rzv_id>"
    echo "   - rzv_endpoint: the endpoint of the master node. For example, <node_address:29400>"
    echo "   - n_nodes: the number of nodes to use. (default: 1)"
    echo "   - n_gpus_per_node: the number of GPUs per node. (default: 8)"
    echo "   - tp_size: the tensor parallel size. (default: 8)"
    echo "   - mode: the mode to use. (default: baseline; choices: baseline, d2, wlbllm)"
    echo "   - rzv_id: the rzv id. (default: megatron_d2_unique_id)"
    exit 1
fi


D2_FA2A_DISABLE_SEND_RECV=${D2_FA2A_DISABLE_SEND_RECV:-0}

CP_DEGREE=${CP_DEGREE:-1}
# Tweek the mode between baseline vs d2.
REPLAN_ITER=${REPLAN_ITER:-10}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_TOKENS=${NUM_TOKENS:-131072}
NUM_LAYERS=${NUM_LAYERS:-4}
UP_SAMPLE_FACTOR=${UP_SAMPLE_FACTOR:-4}
ELONGATE_FACTOR=${ELONGATE_FACTOR:-1}
FILTER_THRESHOLD=${FILTER_THRESHOLD:-65536}
FILTER_RATIO=${FILTER_RATIO:-0.50}
MAX_SAMPLE_ID=${MAX_SAMPLE_ID:-20}
MODEL_PATH=${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}
SHOULD_ADD_DEBUG_CASES=${SHOULD_ADD_DEBUG_CASES:-0}

THIS_HOST=$(hostname)

CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OUTPUT_DIR=${DEFAULT_OUTPUT_DIR:-${CURDIR}/../test-logs}
OUTPUT_DIR=${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}
OUTPUT_DIR=$(realpath ${OUTPUT_DIR})

SHOULD_PROFILE_MEMORY=${SHOULD_PROFILE_MEMORY:-0}

now="$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M%S)_PST"
mkdir -p ${OUTPUT_DIR}
output_file_stem=${OUTPUT_DIR}/${now}.${MODE}.${THIS_HOST}.nnodes${NNODES}.bs${BATCH_SIZE}.maxid${MAX_SAMPLE_ID}.tp${TP_SIZE}.cp${CP_DEGREE}.t${NUM_TOKENS}.elong${ELONGATE_FACTOR}.up${UP_SAMPLE_FACTOR}.ft${FILTER_THRESHOLD}.fr${FILTER_RATIO}
NSYS_PROFILE_PATH=${output_file_stem}.nsys-rep
LOG_PATH=${output_file_stem}.log
PROFILE_MEMORY_PATH=${output_file_stem}.mem.json


function echo_and_tee() {
    echo "$@" | tee -a ${LOG_PATH}
}

echo_and_tee "Running with the following parameters:" 
echo_and_tee "  MODE=${MODE}"
echo_and_tee "  BATCH_SIZE=${BATCH_SIZE}"
echo_and_tee "  TP_SIZE=${TP_SIZE}"
echo_and_tee "  CP_DEGREE=${CP_DEGREE}"
echo_and_tee "  REPLAN_ITER=${REPLAN_ITER}"
echo_and_tee "  NUM_TOKENS=${NUM_TOKENS}"
echo_and_tee "  NUM_LAYERS=${NUM_LAYERS}"
echo_and_tee "  UP_SAMPLE_FACTOR=${UP_SAMPLE_FACTOR}"
echo_and_tee "  ELONGATE_FACTOR=${ELONGATE_FACTOR}"
echo_and_tee "  FILTER_THRESHOLD=${FILTER_THRESHOLD}"
echo_and_tee "  FILTER_RATIO=${FILTER_RATIO}"
echo_and_tee "  MAX_SAMPLE_ID=${MAX_SAMPLE_ID}"
echo_and_tee "  MODEL_PATH=${MODEL_PATH}"
echo_and_tee "  NNODES=${NNODES}"
echo_and_tee "  NPROC_PER_NODE=${NPROC_PER_NODE}"
echo_and_tee "  ENABLE_NSYS=${ENABLE_NSYS}"
echo_and_tee "  NSYS_PROFILE_PATH=${NSYS_PROFILE_PATH}"
echo_and_tee "  SHOULD_ADD_DEBUG_CASES=${SHOULD_ADD_DEBUG_CASES}"
echo_and_tee "  WLBLLM_SYNC_TIME_AG=${WLBLLM_SYNC_TIME_AG}"
echo_and_tee "  EXPERIMENT_REPEAT_TIMES=${EXPERIMENT_REPEAT_TIMES}"
echo_and_tee "  EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=${EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB}"
echo_and_tee "  EXPERIMENT_WARMUP_TIMES=${EXPERIMENT_WARMUP_TIMES}"
echo_and_tee "  EXPERIMENT_EMIT_BACKWARD_NVTX=${EXPERIMENT_EMIT_BACKWARD_NVTX}"
echo_and_tee "  OUTPUT_DIR=${OUTPUT_DIR}"
echo_and_tee "  LOG_PATH=${LOG_PATH}"

# Prepare the common torchrun command and arguments
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
    --cp-degree ${CP_DEGREE} \
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

if [ ${ENABLE_NSYS} -eq 1 ]; then
    echo_and_tee nsys profile \
      --show-output=true \
      --force-overwrite=true \
      -o ${NSYS_PROFILE_PATH} \
      --sample=none \
      -t cuda,nvtx \
    torchrun "${TORCHRUN_CMD[@]}"

    nsys profile \
      --show-output=true \
      --force-overwrite=true \
      -o ${NSYS_PROFILE_PATH} \
      --sample=none \
      -t cuda,nvtx \
    torchrun "${TORCHRUN_CMD[@]}" | tee -a ${LOG_PATH} 2>&1
else
    echo_and_tee torchrun "${TORCHRUN_CMD[@]}"


    torchrun "${TORCHRUN_CMD[@]}" | tee -a ${LOG_PATH} 2>&1
fi
