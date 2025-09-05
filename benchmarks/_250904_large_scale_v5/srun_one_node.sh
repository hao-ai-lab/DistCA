# 
# Usage
#   SHOULD_PROFILE_MEMORY=1 BATCH_SIZE=1 NUM_LAYERS=32 NUM_TOKENS=262144 ELONGATE_FACTOR=4 MAX_SAMPLE_ID=2 \
#   bash test_e2e_combined.onesrun.sh


nnodes=1
ngpus=8
tp_size=2
# head_node=fs-mbz-gpu-098
# jobid=681144


# 
set -x
stem="n${nnodes}_bs${BATCH_SIZE}_nt${NUM_TOKENS}_ef${ELONGATE_FACTOR}"
TS=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)_PST
OUTPUT_DIR=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250904_large_scale_v5/logs/${TS}_${stem}
LOG_DIR=${OUTPUT_DIR}/logs
mkdir -p $LOG_DIR

# Parameters
README_FILE=$OUTPUT_DIR/README.txt
echo "=== Run with parameters ===" >> $README_FILE
echo "NNODES: $nnodes" >> $README_FILE
echo "BATCH_SIZE: $BATCH_SIZE" >> $README_FILE
echo "NUM_LAYERS: $NUM_LAYERS" >> $README_FILE
echo "NUM_TOKENS: $NUM_TOKENS" >> $README_FILE
echo "ELONGATE_FACTOR: $ELONGATE_FACTOR" >> $README_FILE
echo "MAX_SAMPLE_ID: $MAX_SAMPLE_ID" >> $README_FILE
echo "SHOULD_PROFILE_MEMORY: $SHOULD_PROFILE_MEMORY" >> $README_FILE


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

EXTRA_ARGS=""
if [ ${SHOULD_PROFILE_MEMORY} -eq 1 ]; then
    EXTRA_ARGS="--should-profile-memory true"
fi

if [ ${SHOULD_RESEND_QKV} -eq 1 ]; then
    EXTRA_ARGS="--should-resend-qkv"
fi

nsys_str=""
if [ ${SHOULD_NSYS_PROFILE} -eq 1 ]; then
    nsys_str="nsys profile --output=${OUTPUT_DIR}/nsys_report.%h.%p --force-overwrite=true --trace=cuda,nvtx,osrt"
fi



start_time=$(date +%s)
srun -N $nnodes -G $ngpus --ntasks-per-node=1 --jobid=$jobid -w "$head_node" \
    --output=${LOG_DIR}/%N.%j.out \
    --error=${LOG_DIR}/%N.%j.out \
    bash -lc "
    env
    set -x
    exec $nsys_str torchrun --nnodes=$nnodes --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=$head_node:29500 --rdzv_id=d2-n$nnodes --max_restarts=0 test_e2e_combined.py --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B --mode d2 --replan-iter 0 --batch-size $BATCH_SIZE --num-nodes $nnodes --num-gpus-per-node 8 --num-layers $NUM_LAYERS --max-sample-id $MAX_SAMPLE_ID --tp-size $tp_size --cp-degree 1 --up-sample-factor 4 --num-tokens $NUM_TOKENS --elongate-factor $ELONGATE_FACTOR --filter-threshold 65536 --filter-ratio 0.50 --output-dir ${OUTPUT_DIR} $EXTRA_ARGS
    "


end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"
set +x