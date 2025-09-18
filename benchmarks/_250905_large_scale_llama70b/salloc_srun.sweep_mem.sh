

export ENABLE_NSYS=0
export EXPERIMENT_LOG_MEMORY_USAGE=0
export NNODES=${NNODES:-32}
export TP_SIZE=8
# export JOBID=

if [ -z "${JOBID}" ]; then
    echo "Error: JOBID is not set. This script must be run within a SLURM allocation."
    exit 1
fi
export JOBID=${JOBID:-${SLURM_JOB_ID}}

# export MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# export MODEL_PATH=codellama/CodeLlama-34b-hf
export MODEL_PATH=codellama/CodeLlama-34b-hf
export NUM_LAYERS=4

export OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250905_large_scale_llama70b/sweep
export MAX_SAMPLE_ID=1
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2
export TP_SIZE=8
export ENABLE_NSYS=1
export EXPERIMENT_LOG_MEMORY_USAGE=1
export EXPERIMENT_REPEAT_TIMES=1
export EXPERIMENT_WARMUP_TIMES=0
export SHOULD_ADD_DEBUG_CASES=0
echo "Sweeping parameters for testing:"
echo "- NNODES: $NNODES"
echo "- JOBID: $JOBID"
echo "- MODEL_PATH: $MODEL_PATH"
echo "- NUM_LAYERS: $NUM_LAYERS"
echo "- MAX_SAMPLE_ID: $MAX_SAMPLE_ID"
echo "- ENABLE_NSYS: $ENABLE_NSYS"
echo "- EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB: $EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB"
echo "- EXPERIMENT_LOG_MEMORY_USAGE: $EXPERIMENT_LOG_MEMORY_USAGE"
echo "- EXPERIMENT_REPEAT_TIMES: $EXPERIMENT_REPEAT_TIMES"
echo "- EXPERIMENT_WARMUP_TIMES: $EXPERIMENT_WARMUP_TIMES"
echo "Final chance to check your variables...."
DRY_RUN=${DRY_RUN:-0}
if [ $DRY_RUN -eq 0 ]; then
    sleep 5
fi



# Run experiments with different configurations
for config in \
    "0 0 $((NNODES / 8)) 131072 2" \
    "1 1 $((NNODES / 8)) 262144 4" \
    "1 1 $((NNODES / 8)) 524288 8"; do
    
    read -r selective_ckpt resend_qkv batch_size num_tokens elongate_factor <<< "$config"
    
    export EXPERIMENT_ADD_SELECTIVE_CKPT=$selective_ckpt
    export EXPERIMENT_SHOULD_RESEND_QKV=$resend_qkv
    export BATCH_SIZE=$batch_size
    export NUM_TOKENS=$num_tokens
    export ELONGATE_FACTOR=$elongate_factor

    # Run d2 mode
    export MODE=d2
    echo -e "\033[32mRunning d2 with NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR\033[0m"
    if [ $DRY_RUN -eq 0 ]; then
        bash test_e2e_combined.salloc.sh
    fi
    
    # Run wlbllm mode with different CP sizes
    for CP_SIZE in 32 16 8 4 2 1; do
        if [ $CP_SIZE -gt $NNODES ]; then
            continue
        fi
        DP_SIZE=$((NNODES / CP_SIZE))
        if [ $DP_SIZE -gt $(($BATCH_SIZE * 2)) ]; then
            continue
        fi
        echo -e "\033[32mRunning wlbllm with CP_SIZE=$CP_SIZE, DP_SIZE=$DP_SIZE, NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR\033[0m"
        export MODE=wlbllm CP_SIZE=$CP_SIZE
        if [ $DRY_RUN -eq 0 ]; then
            bash test_e2e_combined.salloc.sh
        fi
    done
    

    
done



# salloc -N 8 -G 64 bash salloc_srun.sh