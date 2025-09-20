

# ----------------------
# Set these flags yourself
# ----------------------
export OUTPUT_DIR_PREFIX="/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250918_test_refactor/logs.v1-test-refactor"

export NNODES=${NNODES:-32}
export TP_SIZE=8

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

export TP_SIZE=8
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2
export EXPERIMENT_LOG_MEMORY_USAGE=0
export EXPERIMENT_WARMUP_TIMES=2
export EXPERIMENT_REPEAT_TIMES=1
export EXPERIMENT_D2_FLASH_ATTN_SKIP_GET_BACKEND=1 # default 1
export EXPERIMENT_SKIP_OPTIMIZER_STEP=1
export EXPERIMENT_FA2A_BARRIER=0
export EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0=0
export EXPERIMENT_D2_BALANCE_PING_PONG=0
# export EXPERIMENT_FA2A_BARRIER=1
# export EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0=0 # default 0


DRY_RUN=${DRY_RUN:-0}


# ------------------------------------
# Check and skip success runs
# ------------------------------------
# Run one d2 + one wlbllm-cpMax to justify the result.
for sample_config in \
"wlbllm 0.0" \
; do

# "astronomer/Llama-3-70B-Special-Tokens-Adjusted 170000 80" \
# "codellama/CodeLlama-34b-hf 131072 48" \
for model_config in \
"deepseek-ai/DeepSeek-R1-Distill-Llama-8B 64000 32" \
; do


export ENABLE_NSYS=1
export MAX_SAMPLE_ID=3
# export SHOULD_ADD_DEBUG_CASES=1 # usually this case will find all the bugs

#    s r b  tok   e N
for config in \
    "1 1 8 131072 2 16" \
    "1 1 16 131072 2 32" \
    ; do


    read -r sample_name change_long_doc_ratio <<< "$sample_config"
    read -r model_path attn_linear_breakpoint num_layers <<< "$model_config"
    read -r selective_ckpt resend_qkv batch_size num_tokens elongate_factor nnodes <<< "$config"
    
    export EXPERIMENT_ADD_SELECTIVE_CKPT=$selective_ckpt
    export EXPERIMENT_SHOULD_RESEND_QKV=$resend_qkv
    export BATCH_SIZE=$batch_size
    export NUM_TOKENS=$num_tokens
    export ELONGATE_FACTOR=$elongate_factor
    export MODEL_PATH=$model_path
    export NNODES=$nnodes
    export SAMPLE_NAME=$sample_name
    export CHANGE_LONG_DOC_RATIO=$change_long_doc_ratio
    export ATTN_LINEAR_BREAKPOINT=$attn_linear_breakpoint
    export NUM_LAYERS=$num_layers
    
    # Run d2 with signal only
    export MODE=d2
    export MIN_TOLERANCE_FACTOR=0.05
    echo "🟡 Running d2-signal with NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR, MIN_TOLERANCE_FACTOR=$MIN_TOLERANCE_FACTOR"
    if [ $DRY_RUN -eq 0 ]; then
        bash test_e2e_combined.salloc.sh
        echo "🟡 Finished running d2-signal with NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR, MIN_TOLERANCE_FACTOR=$MIN_TOLERANCE_FACTOR. Not guaranteed to be successful."
        echo "\a"
    fi
        
done
done
done


set +e