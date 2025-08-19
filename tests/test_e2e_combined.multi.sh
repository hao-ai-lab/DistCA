
# Usage:
# bash test_e2e_combined.multi.sh localhost:29500 1 2 2 baseline unique_rdbzid
# bash test_e2e_combined.multi.sh localhost:29500 2 8 8 baseline unique_rdbzid_baseline
#

# export CUDA_LAUNCH_BLOCKING=1 
export D2_DEBUG_PRINT=1
# export WLBLLM_DISABLE_LSE=1
# export WLBLLM_SYNC_TIME_FLASH_ATTN=1

export NVTE_NVTX_ENABLED=1
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0 
export D2_FA2A_DISABLE_SEND_RECV=0 
export NVSHMEM_IB_ENABLE_IBGDA=true 
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

ENABLE_NSYS=0

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


echo
echo "Executing: bash $0 rzv_endpoint=$1 n_nodes=$2 n_gpus_per_node=$3 tp_size=$4 mode=$5 rzv_id=$6"
echo 
echo "bash $0 $1 $2 $3 $4 $5 $6"
echo



# Tweek the mode between baseline vs d2.
REPLAN_ITER=10
# NUM_TOKENS=2048
# NUM_TOKENS=32768
# NUM_TOKENS=65536
NUM_TOKENS=131072
# NUM_TOKENS=174080
NUM_LAYERS=4
# NUM_LAYERS=32
# NUM_LAYERS=4
UP_SAMPLE_FACTOR=4
# UP_SAMPLE_FACTOR=32
ELONGATE_FACTOR=1
FILTER_THRESHOLD=65536
FILTER_RATIO=0.50
# MAX_SAMPLE_ID=20
MAX_SAMPLE_ID=20
# MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Llama-8B

THIS_HOST=$(hostname)

OUTPUT_DIR=nsys-profile
now=$(date +%Y%m%d_%H%M%S)
mkdir -p ${OUTPUT_DIR}
NSYS_PROFILE_PATH=${OUTPUT_DIR}/${now}.${MODE}${REPLAN_ITER}.${THIS_HOST}.t${NUM_TOKENS}.elong${ELONGATE_FACTOR}.up${UP_SAMPLE_FACTOR}.ft${FILTER_THRESHOLD}.fr${FILTER_RATIO}.nsys-rep
LOG_PATH=${OUTPUT_DIR}/${now}.${MODE}${REPLAN_ITER}.${THIS_HOST}.t${NUM_TOKENS}.elong${ELONGATE_FACTOR}.up${UP_SAMPLE_FACTOR}.ft${FILTER_THRESHOLD}.fr${FILTER_RATIO}.log


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
    --num-nodes ${NNODES} \
    --num-gpus-per-node ${NPROC_PER_NODE} \
    --tp-size ${TP_SIZE} \
    --num-layers ${NUM_LAYERS} \
    --max-sample-id ${MAX_SAMPLE_ID} \
    --up-sample-factor ${UP_SAMPLE_FACTOR} \
    --num-tokens ${NUM_TOKENS} \
    --elongate-factor ${ELONGATE_FACTOR} \
    --filter-threshold ${FILTER_THRESHOLD} \
    --filter-ratio ${FILTER_RATIO} 
    # --should-add-debug-cases
)

set -x
if [ ${ENABLE_NSYS} -eq 1 ]; then
    nsys profile \
      --show-output=true \
      --force-overwrite=true \
      -o ${NSYS_PROFILE_PATH} \
      --sample=none \
      -t cuda,nvtx \
    torchrun "${TORCHRUN_CMD[@]}" | tee ${LOG_PATH} 2>&1
else
    torchrun "${TORCHRUN_CMD[@]}" --force-exit | tee ${LOG_PATH} 2>&1
fi
set +x