export NVTE_NVTX_ENABLED=1
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0 
export D2_FA2A_DISABLE_SEND_RECV=0 
export NVSHMEM_IB_ENABLE_IBGDA=true 
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

NNODES=4
NPROC_PER_NODE=8
RZV_BACKEND=c10d
RZV_ENDPOINT=fs-mbz-gpu-022:29400
RZV_ID=megatron_d2_unique_id_6
MODE=baseline
# MODE=d2
REPLAN_ITER=10
# NUM_TOKENS=65536
NUM_TOKENS=131072
# NUM_TOKENS=174080
NUM_LAYERS=4
# NUM_LAYERS=32
# NUM_LAYERS=4
# UP_SAMPLE_FACTOR=4
UP_SAMPLE_FACTOR=32
ELONGATE_FACTOR=1
FILTER_THRESHOLD=65536
FILTER_RATIO=0.10
MAX_SAMPLE_ID=20
# MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Llama-8B

THIS_HOST=$(hostname)
mkdir -p nsys-profile-15
NSYS_PROFILE_PATH=nsys-profile-15/${MODE}${REPLAN_ITER}.${THIS_HOST}.t${NUM_TOKENS}.elong${ELONGATE_FACTOR}.up${UP_SAMPLE_FACTOR}.ft${FILTER_THRESHOLD}.fr${FILTER_RATIO}.nsys-rep



# nsys profile \
#   --show-output=true \
#   --force-overwrite=true \
#   -o ${NSYS_PROFILE_PATH} \
#   --sample=none \
#   -t cuda,nvtx \
torchrun \
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
    --tp-size ${NPROC_PER_NODE} \
    --num-layers ${NUM_LAYERS} \
    --max-sample-id ${MAX_SAMPLE_ID} \
    --up-sample-factor ${UP_SAMPLE_FACTOR} \
    --num-tokens ${NUM_TOKENS} \
    --elongate-factor ${ELONGATE_FACTOR} \
    --filter-threshold ${FILTER_THRESHOLD} \
    --filter-ratio ${FILTER_RATIO} \
    --force-exit

