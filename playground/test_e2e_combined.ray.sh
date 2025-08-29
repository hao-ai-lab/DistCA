

env_vars=(
    "NVSHMEM_IB_ENABLE_IBGDA=true"
    "D2_FA2A_DISABLE_SEND_RECV=0"
    "D2_DEBUG_PRINT=1"
    "D2_SYNC_TIME_TRANSFORMER_LAYER=0"
    "WLBLLM_DISABLE_LSE=1"
    "WLBLLM_SYNC_TIME_FLASH_ATTN=1"
    "NVTE_NVTX_ENABLED=1"
    "NSYS_NVTX_PROFILER_REGISTER_ONLY=0"
    "WLBLLM_SYNC_TIME_PERDOC_ATTN=0"
    "WLBLLM_SYNC_TIME_FLASH_ATTN=0"
)

run_vars=(
    "ENABLE_NSYS=1"
    "SHOULD_ADD_DEBUG_CASES=0"
    "CP_DEGREE=1" 
    "MAX_SAMPLE_ID=3"
    "REPLAN_ITER=10"
    "NUM_TOKENS=8192"
    "NUM_LAYERS=4"
    "UP_SAMPLE_FACTOR=4"
    "ELONGATE_FACTOR=1"
    "FILTER_THRESHOLD=65536"
    "FILTER_RATIO=0.50"
    "MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)

vars_str=$(echo "${env_vars[@]}" | tr '\n' ' ')
run_vars_str=$(echo "${run_vars[@]}" | tr '\n' ' ')


NNODES=4
NPROC_PER_NODE=8
TP_SIZE=8
MODE=d2
RZV_ID=wlb-8-$(hostname)-$(date +%Y%m%d_%H%M%S)



torchrun_cmd=$(echo "${TORCHRUN_CMD[@]}" | tr '\n' ' ')

set -x
python -m d2.ray_launch --num-tasks ${NNODES} --cmd "${vars_str} ${run_vars_str} bash test_e2e_combined.multi.sh $(hostname):29500 ${NNODES} ${NPROC_PER_NODE} ${TP_SIZE} ${MODE} ${RZV_ID}"
set +x