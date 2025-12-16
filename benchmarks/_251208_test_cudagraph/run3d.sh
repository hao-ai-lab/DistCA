
export TP_SIZE=${TP_SIZE:-8}
JOBID=${JOBID:-${SLURM_JOB_ID}}


TS=$(TZ=America/Los_Angeles date +%m%d_%H%M%S)_PST
export OUTPUT_DIR_PREFIX="/mnt/weka/home/hao.zhang/jd/d2/benchmarks/_251208_test_cudagraph/logs.v1"
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2
export EXPERIMENT_LOG_MEMORY_USAGE=0
export EXPERIMENT_REPEAT_TIMES=1
export EXPERIMENT_WARMUP_TIMES=0
export EXPERIMENT_D2_FLASH_ATTN_SKIP_GET_BACKEND=1 # default 1
export SHOULD_ADD_DEBUG_CASES=0
export EXPERIMENT_SKIP_OPTIMIZER_STEP=1
export EXPERIMENT_FA2A_BARRIER=0
export EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0=0
# export EXPERIMENT_FA2A_BARRIER=1
# export EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0=0 # default 0


DRY_RUN=${DRY_RUN:-0}

# export MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# export MODEL_PATH=codellama/CodeLlama-34b-hf
# export MODEL_PATH=codellama/CodeLlama-34b-hf

export EXPERIMENT_D2_BALANCE_PING_PONG=0

# ------------------------------------
# Check and skip success runs
# ------------------------------------


export EXPERIMENT_FA2A_SPLIT_SENDRECV=1

export ENABLE_NSYS=1
export MAX_SAMPLE_ID=1
export SHOULD_ADD_DEBUG_CASES=1

# Run one d2 + one wlbllm-cpMax to justify the result.
for sample_config in \
"wlbllm 0.0" \
; do

# "astronomer/Llama-3-70B-Special-Tokens-Adjusted 170000 80" \
# "codellama/CodeLlama-34b-hf 131072 24" \
for model_config in \
"deepseek-ai/DeepSeek-R1-Distill-Llama-8B 64000 3" \
; do
    
    
    # "1 1 1  65536 1 8" \

#    s r b   tok  e n
for config in \
    "1 1 2 32768 2 1" \
    ; do

    read -r selective_ckpt resend_qkv batch_size num_tokens elongate_factor nnodes <<< "$config"
    read -r sample_name change_long_doc_ratio <<< "$sample_config"
    read -r model_path attn_linear_breakpoint num_layers <<< "$model_config"
    
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
    
    CP_SIZE=$NNODES
    DP_SIZE=$((NNODES / CP_SIZE))
    export MODE=d2 CP_SIZE=$CP_SIZE
    export OUTPUT_DIR_SUFFIX_ADDON=""
    echo "🟡 Running d2 with CP_SIZE=$CP_SIZE, DP_SIZE=$DP_SIZE, NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR"
    bash test_e2e_combined.salloc.sh
    echo "🟡 Finished running d2 with CP_SIZE=$CP_SIZE, DP_SIZE=$DP_SIZE, NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR. Not guaranteed to be successful."

done
done
done


set +e