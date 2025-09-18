# 
# Usage
#   export JOBID=<JOBID>
#   export NNODES=<NNODES>
#   bash salloc_srun.sh
# 
# set -e

export NNODES=${NNODES:-8}
export TP_SIZE=8
# export JOBID=

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


TS=$(TZ=America/Los_Angeles date +%m%d_%H%M%S)_PST
export OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250915_basic_exps/item_02_wlb_memimbalance
export MAX_SAMPLE_ID=5
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2
export TP_SIZE=8
export ENABLE_NSYS=0
# export EXPERIMENT_LOG_MEMORY_USAGE=1
export EXPERIMENT_LOG_MEMORY_USAGE=1
export EXPERIMENT_REPEAT_TIMES=1
export EXPERIMENT_WARMUP_TIMES=1
export EXPERIMENT_D2_FLASH_ATTN_SKIP_GET_BACKEND=1 # default 1
export SHOULD_ADD_DEBUG_CASES=0
export EXPERIMENT_SKIP_OPTIMIZER_STEP=0
export EXPERIMENT_FA2A_BARRIER=0
export EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0=0
# export EXPERIMENT_FA2A_BARRIER=1
# export EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0=0 # default 0


DRY_RUN=${DRY_RUN:-0}

export MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# export MODEL_PATH=codellama/CodeLlama-34b-hf
# export MODEL_PATH=codellama/CodeLlama-34b-hf
export NUM_LAYERS=32


# ------------------------------------
# Check and skip success runs
# ------------------------------------

# Run one d2 + one wlbllm-cpMax to justify the result.
# selective_ckpt resend_qkv batch_size num_tokens elongate_factor
for config in \
    "0 0 1 65536 1" \
    "0 0 1 131072 2" \
    "0 0 1 262144 4" \
    ; do


    read -r selective_ckpt resend_qkv batch_size num_tokens elongate_factor <<< "$config"
    
    export EXPERIMENT_ADD_SELECTIVE_CKPT=$selective_ckpt
    export EXPERIMENT_SHOULD_RESEND_QKV=$resend_qkv
    export BATCH_SIZE=$batch_size
    export NUM_TOKENS=$num_tokens
    export ELONGATE_FACTOR=$elongate_factor

    # Run d2 mode with all on
    export MODE=d2
    export EXPERIMENT_D2_BALANCE_PING_PONG=0
    export OUTPUT_DIR_SUFFIX_ADDON="-normal"
    echo "🟡 Running d2 with NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR"
    if [ $DRY_RUN -eq 0 ]; then
        # bash test_e2e_combined.salloc.sh
        echo "🟡 Finished running d2 with NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR. Not guaranteed to be successful."
        echo "\a"
    fi

    
    # Run wlbllm mode with different CP sizes
    counter=0
    # for CP_SIZE in 32 16 8 4 2 1; do
    for CP_SIZE in 4 2 1; do
        if [ $CP_SIZE -gt $NNODES ]; then
            continue
        fi
        DP_SIZE=$((NNODES / CP_SIZE))
        if [ $DP_SIZE -gt $(($BATCH_SIZE * 2)) ]; then
            continue
        fi

        export MODE=wlbllm CP_SIZE=$CP_SIZE
        export OUTPUT_DIR_SUFFIX_ADDON=""
        echo "🟡 Running wlbllm with CP_SIZE=$CP_SIZE, DP_SIZE=$DP_SIZE, NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR"
        if [ $DRY_RUN -eq 0 ]; then
            bash test_e2e_combined.salloc.sh
            echo "🟡 Finished running wlbllm with CP_SIZE=$CP_SIZE, DP_SIZE=$DP_SIZE, NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR. Not guaranteed to be successful."
            echo "\a"
        fi
        # break # <- breaking after the max permissible cp size is run
    done

done


# Run all wlbllm-cp except max 
for config in \
    "1 1 4 131072 2" \
    "1 1 4 262144 4" \
    "1 1 2 524288 8" \
    "1 1 1 131072 2" \
    "1 1 1 262144 4" \
    "1 1 1 524288 8" \
    "1 1 4 524288 8" \
    ; do


    read -r selective_ckpt resend_qkv batch_size num_tokens elongate_factor <<< "$config"
    
    export EXPERIMENT_ADD_SELECTIVE_CKPT=$selective_ckpt
    export EXPERIMENT_SHOULD_RESEND_QKV=$resend_qkv
    export BATCH_SIZE=$batch_size
    export NUM_TOKENS=$num_tokens
    export ELONGATE_FACTOR=$elongate_factor
    
    # Run wlbllm mode with different CP sizes
    # counter=0
    # for CP_SIZE in 32 16 8 4 2 1; do
    for CP_SIZE in 4 2 1; do
        if [ $CP_SIZE -gt $NNODES ]; then
            continue
        fi
        DP_SIZE=$((NNODES / CP_SIZE))
        if [ $DP_SIZE -gt $(($BATCH_SIZE * 2)) ]; then
            continue
        fi

        counter=$((counter + 1))
        if [ $counter -le 1 ]; then
            continue # <- skipping the first one case as it has been run before
        fi

        echo "🟡 Running wlbllm with CP_SIZE=$CP_SIZE, DP_SIZE=$DP_SIZE, NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR"
        export MODE=wlbllm CP_SIZE=$CP_SIZE
        export OUTPUT_DIR_SUFFIX_ADDON=""
        if [ $DRY_RUN -eq 0 ]; then
            bash test_e2e_combined.salloc.sh
            echo "🟡 Finished running wlbllm with CP_SIZE=$CP_SIZE, DP_SIZE=$DP_SIZE, NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR. Not guaranteed to be successful."
            echo "\a"
        fi
    done


    # # Run d2 mode with balance ping pong
    # export MODE=d2
    # export EXPERIMENT_D2_BALANCE_PING_PONG=1
    # export OUTPUT_DIR_SUFFIX_ADDON="-pingpong_balance"
    # echo "🟡 Running d2 with NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR"
    # if [ $DRY_RUN -eq 0 ]; then
    #     bash test_e2e_combined.salloc.sh
    #     echo "🟡 Finished running d2 with NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR. Not guaranteed to be successful."
    #     echo "\a"
    # fi


    

done


set +e