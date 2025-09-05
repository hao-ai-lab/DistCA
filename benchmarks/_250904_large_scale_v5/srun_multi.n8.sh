set -x


export EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0=0 
export SHOULD_RESEND_QKV=1 
export EXPERIMENT_WARMUP_TIMEOUT_SEC=300 
export EXPERIMENT_WARMUP_TIMES=3 
export EXPERIMENT_REPEAT_TIMES=3 
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=6 
export EXPERIMENT_LOG_MEMORY_USAGE=0 
export NUM_LAYERS=32
export MAX_SAMPLE_ID=50


export nnodes=8
export ngpus=64
export head_node=fs-mbz-gpu-084
export jobid=681776

for NNODES in 8; do
# finished 131072
    for NUM_TOKENS in 1048576 524288 262144 ; do
        # define BATCH_SIZE
        
        if [ $NUM_TOKENS -eq 131072 ]; then
            BATCH_SIZE=$((2 * NNODES / 8))
        elif [ $NUM_TOKENS -eq 262144 ]; then
            BATCH_SIZE=$((1 * NNODES / 8))
        elif [ $NUM_TOKENS -eq 524288 ]; then
            BATCH_SIZE=$((1 * NNODES / 8))
        elif [ $NUM_TOKENS -eq 1048576 ]; then
            BATCH_SIZE=$((1 * NNODES / 8))
        else
            BATCH_SIZE=1
        fi

        ELONGATE_FACTOR=$((NUM_TOKENS / 65536))
        NGPUS=$((NNODES * 8))
        NNODES=$NNODES NGPUS=$NGPUS BATCH_SIZE=$BATCH_SIZE NUM_TOKENS=$NUM_TOKENS ELONGATE_FACTOR=$ELONGATE_FACTOR  bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250904_large_scale_v5/srun_one_case.sh
    done
done

set +x
