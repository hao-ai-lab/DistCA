# 
# Usage
#   export JOBID=<JOBID>
#   export NNODES=<NNODES>
#   bash check_corebind.sh
# 
# 
#
JOBID=${JOBID:-${SLURM_JOB_ID}}
if [ -z "$JOBID" ]; then
  echo -e "\033[31mJOBID is not set. Must set JOBID environment variable.\033[0m"
  exit 1
fi
NNODES=${NNODES:-$SLURM_NNODES}
if [ -z "$NNODES" ]; then
    NNODES=$(squeue -j $JOBID -h -o %D)
fi
echo -e "\033[33mRecognized JOBID=$JOBID, NNODES=$NNODES\033[0m"

sleep 1


# export OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_2509907_large_scale_v6/logs.v1-core-bind
export OUTPUT_DIR_PREFIX=~/jd/d2/tests/logs
export MAX_SAMPLE_ID=3
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2
export TP_SIZE=8
export ENABLE_NSYS=1
export EXPERIMENT_LOG_MEMORY_USAGE=1
export EXPERIMENT_REPEAT_TIMES=1
export EXPERIMENT_WARMUP_TIMES=0
export SHOULD_ADD_DEBUG_CASES=0

DRY_RUN=${DRY_RUN:-0}

# export MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# export MODEL_PATH=codellama/CodeLlama-34b-hf
export MODEL_PATH=codellama/CodeLlama-34b-hf
export NUM_LAYERS=4

# Run experiments with different configurations
for config in \
    "0 0 $((NNODES / 8)) 131072 2" ; do
    
    read -r selective_ckpt resend_qkv batch_size num_tokens elongate_factor <<< "$config"

    if [ $batch_size -le 0 ]; then
        batch_size=1
    fi
    
    export EXPERIMENT_ADD_SELECTIVE_CKPT=$selective_ckpt
    export EXPERIMENT_SHOULD_RESEND_QKV=$resend_qkv
    export BATCH_SIZE=$batch_size
    export NUM_TOKENS=$num_tokens
    export ELONGATE_FACTOR=$elongate_factor

    # Run d2 mode and check corebinding
    export MODE=d2
    echo "Running d2 with NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR"
    if [ $DRY_RUN -eq 0 ]; then
        bash test_e2e_combined.salloc.sh
    fi

    break
    # Run wlbllm mode with different CP sizes and check corebinding does not fail.
    for CP_SIZE in 32 16 8 4 2 1; do
        if [ $CP_SIZE -gt $NNODES ]; then
            continue
        fi
        DP_SIZE=$((NNODES / CP_SIZE))
        if [ $DP_SIZE -gt $(($BATCH_SIZE * 2)) ]; then
            continue
        fi
        echo "Running wlbllm with CP_SIZE=$CP_SIZE, DP_SIZE=$DP_SIZE, NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR"
        export MODE=wlbllm CP_SIZE=$CP_SIZE
        if [ $DRY_RUN -eq 0 ]; then
            bash test_e2e_combined.salloc.sh
        fi
        break
    done
    

done

echo -e "\033[32mCorebinding check completed.\033[0m"


# salloc -N 8 -G 64 bash check_corebind.sh