set -x
stem="bs${BATCH_SIZE}_nt${NUM_TOKENS}_ef${ELONGATE_FACTOR}"
TS=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)_PST
OUTPUT_DIR=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/logs/${TS}_${stem}
LOG_DIR=${OUTPUT_DIR}/logs
mkdir -p $LOG_DIR

exec > $OUTPUT_DIR/slurm.stdout 2>&1

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

start_time=$(date +%s)
srun --time=00:10:00 -N 1 -G 8 --ntasks-per-node=1 \
    --output=${LOG_DIR}/%N.%j.out \
    --error=${LOG_DIR}/%N.%j.out \
    bash -lc "
        set -x
        exec torchrun --nnodes=1 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=fs-mbz-gpu-004:29500 --rdzv_id=fs-mbz-gpu-004 --max_restarts=0 test_e2e_combined.py --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B --mode d2 --replan-iter 0 --batch-size $BATCH_SIZE --num-nodes 1 --num-gpus-per-node 8 --num-layers $NUM_LAYERS --max-sample-id $MAX_SAMPLE_ID --tp-size 2 --cp-degree 1 --up-sample-factor 4 --num-tokens $NUM_TOKENS --elongate-factor $ELONGATE_FACTOR --filter-threshold 65536 --filter-ratio 0.50 --output-dir ${OUTPUT_DIR} --should-add-debug-cases
    "

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"
set +x