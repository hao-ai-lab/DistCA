# 
# Usage
#   SHOULD_PROFILE_MEMORY=1 BATCH_SIZE=1 NUM_LAYERS=32 NUM_TOKENS=262144 ELONGATE_FACTOR=4 MAX_SAMPLE_ID=2 \
#   bash test_e2e_combined.onesrun.sh

# Parameters
echo "=== Run with parameters ==="
echo "BATCH_SIZE: $BATCH_SIZE"
echo "NUM_LAYERS: $NUM_LAYERS"
echo "NUM_TOKENS: $NUM_TOKENS"
echo "ELONGATE_FACTOR: $ELONGATE_FACTOR"
echo "MAX_SAMPLE_ID: $MAX_SAMPLE_ID"
echo "SHOULD_PROFILE_MEMORY: $SHOULD_PROFILE_MEMORY"

# 
set -x
stem="bs${BATCH_SIZE}_nt${NUM_TOKENS}_ef${ELONGATE_FACTOR}"
TS=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)_PST
OUTPUT_DIR=./logs/${TS}_${stem}
LOG_DIR=${OUTPUT_DIR}/logs
NSYS_DIR=${OUTPUT_DIR}/nsys-profiles
mkdir -p $LOG_DIR

# exec > $OUTPUT_DIR/slurm.stdout 2>&1 # redirect only
exec > >(tee -a "$OUTPUT_DIR/slurm.stdout") 2>&1 # tee, so you can see it in console

export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 
export NVSHMEM_IB_ENABLE_IBGDA=true

export CUDA_DIR=/mnt/sharefs/software/DeepEP/cuda-12-6
export NCCL_HOME=/usr
export NCCL_LIB=/usr/lib/x86_64-linux-gnu
export NVSHMEM_DIR=/mnt/weka/home/yonghao.zhuang/opt/nvshmem
export NVSHMEM_PREFIX=/mnt/weka/home/yonghao.zhuang/opt/nvshmem
export OPENMPI_DIR=/mnt/weka/home/yonghao.zhuang/opt/openmpi

export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${CUDA_DIR}/lib64:${OPENMPI_DIR}/lib:${NCCL_LIB}/:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:${OPENMPI_DIR}/bin:${CUDA_DIR}/bin:$PATH"


mkdir -p $LOG_DIR

# OUTPUT_DIR

EXTRA_ARGS=""
if [ ${SHOULD_PROFILE_MEMORY} -eq 1 ]; then
    EXTRA_ARGS="--should-profile-memory true"
fi

if [ ${SHOULD_RESEND_QKV} -eq 1 ]; then
    EXTRA_ARGS="--should-resend-qkv"
fi

# HOST_IDS="fs-mbz-gpu-[004,036,041,064,124,137-138,143-144,153,184,209,217,268,272,279,294,311,341,369,402,441,444,460,481,488,646,649,743,753,770,880]"

# HOST_IDS_4="fs-mbz-gpu-[004,036,041,064,124,137-138,143-144,153,184,209,217,268,272,279,294,311,341,369,402,441,444,460,481,488,646,649,743,753,770,880]"

# fs-mbz-gpu-[017,056,067,072,076,087,092,109,114,139,142,145,158,163,230,266,399,449,476,615,678,700,709,716,737,755,824,891,914,946,964,967]

# start_time=$(date +%s)
# srun --time=00:10:00 -N 4 -G 32 --ntasks-per-node=1 -w "$HOST_IDS_4" \
#     --output=${LOG_DIR}/%N.%j.out \
#     --error=${LOG_DIR}/%N.%j.out \
#     bash -lc "
#         set -x
#         exec torchrun --nnodes=4  --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=fs-mbz-gpu-004:29500 --rdzv_id=fs-mbz-gpu-004 --max_restarts=0 test_e2e_combined.py --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B --mode d2 --replan-iter 0 --batch-size $BATCH_SIZE --num-nodes 4 --num-gpus-per-node 8 --num-layers $NUM_LAYERS --max-sample-id $MAX_SAMPLE_ID --tp-size 8 --cp-degree 1 --up-sample-factor 4 --num-tokens $NUM_TOKENS --elongate-factor $ELONGATE_FACTOR --filter-threshold 65536 --filter-ratio 0.50 --output-dir ${OUTPUT_DIR} --should-add-debug-cases $EXTRA_ARGS
#     "

# start_time=$(date +%s)
# srun --time=00:10:00 -N 32 -G 256 --ntasks-per-node=1 --jobid=679144 \
#     --output=${LOG_DIR}/%N.%j.out \
#     --error=${LOG_DIR}/%N.%j.out \
#     bash -lc "
#         set -x
#         exec torchrun --nnodes=32 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=fs-mbz-gpu-017:29500 --rdzv_id=fs-mbz-gpu-017 --max_restarts=0 test_e2e_combined.py --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B --mode d2 --replan-iter 0 --batch-size $BATCH_SIZE --num-nodes 32 --num-gpus-per-node 8 --num-layers $NUM_LAYERS --max-sample-id $MAX_SAMPLE_ID --tp-size 8 --cp-degree 1 --up-sample-factor 4 --num-tokens $NUM_TOKENS --elongate-factor $ELONGATE_FACTOR --filter-threshold 65536 --filter-ratio 0.50 --output-dir ${OUTPUT_DIR} --should-add-debug-cases $MEMORY_PROFILE_ARG
#     "


# nnodes=4

# ngpus=32

nnodes=2
ngpus=16
head_node=fs-mbz-gpu-012
start_time=$(date +%s)
# NSYS_PROFILE_PATH=""
# nsys profile --show-output=true -o ${NSYS_DIR}/%N.%j.nsys-rep
#   --sample=none -t cuda,nvtx 
srun --time=00:10:00 -N $nnodes -G $ngpus --ntasks-per-node=1 --jobid=679854 -w "$head_node" \
    --output=${LOG_DIR}/%N.%j.out \
    --error=${LOG_DIR}/%N.%j.out \
    bash -lc "
        set -x
        exec torchrun --nnodes=$nnodes --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=$head_node:29500 --rdzv_id=fs-mbz-gpu-017-1 --max_restarts=0 test_e2e_combined.py --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B --mode d2 --replan-iter 0 --batch-size $BATCH_SIZE --num-nodes $nnodes --num-gpus-per-node 8 --num-layers $NUM_LAYERS --max-sample-id $MAX_SAMPLE_ID --tp-size 8 --cp-degree 1 --up-sample-factor 4 --num-tokens $NUM_TOKENS --elongate-factor $ELONGATE_FACTOR --filter-threshold 65536 --filter-ratio 0.50 --output-dir ${OUTPUT_DIR} --should-add-debug-cases $EXTRA_ARGS
    "


end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"
set +x