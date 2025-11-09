export NNODES=${NNODES:-1}
export TP_SIZE=1

TS=$(TZ=America/Los_Angeles date +%m%d_%H%M%S)_PST

export EXPERIMENT_LOG_MEMORY_USAGE=0
export EXPERIMENT_REPEAT_TIMES=1
export EXPERIMENT_WARMUP_TIMES=1
export EXPERIMENT_D2_FLASH_ATTN_SKIP_GET_BACKEND=1 # default 1
export SHOULD_ADD_DEBUG_CASES=0
export EXPERIMENT_SKIP_OPTIMIZER_STEP=1
export EXPERIMENT_FA2A_BARRIER=0
export EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0=0
# export EXPERIMENT_FA2A_BARRIER=1
# export EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0=0 # default 0

# torch: avoid recording streams 
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1

# Control how many GPUs per node we should use.
export GPUS_PER_NODE=8
# Control if we should use srun.
export EXPERIMENT_NO_SRUN=0
export EXPERIMENT_USE_PYTORCH_A2A=0
export EXPERIMENT_LOAD_FROM_PLAN_AHEAD=0
export EXPERIMENT_ILP_TIME_LIMIT=5
export D2_SHOULD_USE_SAME_STREAM_FOR_COMM_AND_COMPUTE=0
export EXPERIMENT_SET_SEQUENCE_PARALLEL=1

DRY_RUN=${DRY_RUN:-0}

# ------------------------------------
# Check and skip success runs
# ------------------------------------

export ENABLE_NSYS=0
export MAX_SAMPLE_ID=3

export OUTPUT_DIR_PREFIX="/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks3/_251103_correctnes/logs.v1"

# "deepseek-ai/DeepSeek-R1-Distill-Llama-8B 64000 32" \
# "astronomer/Llama-3-70B-Special-Tokens-Adjusted 170000 80" \
for model_config in \
"deepseek-ai/DeepSeek-R1-Distill-Llama-8B 64000 32" \
; do

configs=(
# N = 1
#    s r b   tok  e  N  mode   cp   tp  sample_name  change_long_doc_ratio
    "1 1 1 32768  1  1  d2     1   8   prolong      0.3"
# N = 8
#      s r b    tok  e  N  mode   cp   tp  sample_name  change_long_doc_ratio
    # "1 1 4 131072  2  8  d2     8   8   prolong      0.3"
    # "1 1 2 262144  4  8  d2     8   8   prolong      0.3"
    # "1 1 1 524288  8  8  d2     8   8   prolong      0.3"
    
# N = 16
#      s r b    tok  e  N  mode   cp   tp  sample_name  change_long_doc_ratio
    # "1 1 8 131072  2 16  d2     16   8  prolong      0.3"
    # "1 1 4 262144  4 16  d2     16   8  prolong      0.3"
    # "1 1 2 524288  8 16  d2     16   8  prolong      0.3"
    
# N = 32
#      s r b    tok  e  N  mode   cp   tp  sample_name  change_long_doc_ratio
    # "1 1 16 131072  2 32  d2     32   8  prolong      0.3"
    # "1 1 8 262144  4 32  d2     32   8   prolong      0.3"
    # "1 1 4 524288  8 32  d2     32   8   prolong      0.3"

)


# export EXPERIMENT_D2_BALANCE_PING_PONG=1
export EXPERIMENT_PROFILE_RUN=0
export WLBLLM_ENABLE_SHUFFLE=0


for config in "${configs[@]}"; do
    read -r selective_ckpt resend_qkv batch_size num_tokens elongate_factor nnodes mode cp_size tp_size sample_name change_long_doc_ratio <<< "$config"
    read -r model_path attn_linear_breakpoint num_layers <<< "$model_config"
    
    export EXPERIMENT_ADD_SELECTIVE_CKPT=$selective_ckpt
    export EXPERIMENT_SHOULD_RESEND_QKV=$resend_qkv
    export BATCH_SIZE=$batch_size
    export NUM_TOKENS=$num_tokens
    export ELONGATE_FACTOR=$elongate_factor
    export MODEL_PATH=$model_path
    export MODEL_PATH_NORMALIZED=$(echo $model_path | sed 's/\//_/g')
    export NNODES=$nnodes
    export SAMPLE_NAME=$sample_name
    export CHANGE_LONG_DOC_RATIO=$change_long_doc_ratio
    export ATTN_LINEAR_BREAKPOINT=$attn_linear_breakpoint
    export NUM_LAYERS=$num_layers
    export CP_SIZE=$cp_size
    export TP_SIZE=$tp_size

    export OUTPUT_DIR_SUFFIX_ADDON="-${mode}-nsys-${sample_name}"
    
    if [ "$mode" != "d2" ]; then
        echo "Error: mode must be d2, got: $mode"
        exit 1
    fi
    
    # Run d2 mode with all on
    export MODE=d2
    # export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=$buffer_size
    echo "🟡 Running d2 with NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR"
    if [ $DRY_RUN -eq 0 ]; then
        bash test_correctness_dpcp_e2e.salloc.sh
        echo "🟡 Finished running d2 with NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR. Not guaranteed to be successful."
        echo "\a"
    fi



done
done


set +e