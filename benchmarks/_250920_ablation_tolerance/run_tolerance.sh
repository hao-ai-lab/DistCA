# 
# Usage
#   export JOBID=<JOBID>
#   export NNODES=<NNODES>
#   bash salloc_srun.sh
# 
# set -e



export NNODES=${NNODES:-32}
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
export OUTPUT_DIR_PREFIX="/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250920_ablation_tolerance/logs.v2-tolerance"


export TP_SIZE=8
export ENABLE_NSYS=1
# export EXPERIMENT_LOG_MEMORY_USAGE=1
export EXPERIMENT_LOG_MEMORY_USAGE=0
export EXPERIMENT_REPEAT_TIMES=1
export EXPERIMENT_WARMUP_TIMES=1
export EXPERIMENT_D2_FLASH_ATTN_SKIP_GET_BACKEND=1 # default 1

export EXPERIMENT_SKIP_OPTIMIZER_STEP=1
export EXPERIMENT_FA2A_BARRIER=0
export EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0=0
# export EXPERIMENT_FA2A_BARRIER=1
# export EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0=0 # default 0


# Get current directory of this script
export LOGFILE="${OUTPUT_DIR_PREFIX}/script.log"
exec &> >(tee -a "$LOGFILE")


DRY_RUN=${DRY_RUN:-0}

# export MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# export MODEL_PATH=codellama/CodeLlama-34b-hf

export EXPERIMENT_D2_BALANCE_PING_PONG=0
# ------------------------------------
# Check and skip success runs
# ------------------------------------
# Distribution Name
# CHANGE_LONG_DOC_RATIO  
# ATTN_LINEAR_BREAKPOINT

export ENABLE_NSYS=1
export SAMPLE_START_IDX=0
export MAX_SAMPLE_ID=10
export SHOULD_ADD_DEBUG_CASES=0 # Control the sequence imbalance.
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=8
# Run one d2 + one wlbllm-cpMax to justify the result.
for sample_config in \
"wlbllm 0.0" \
; do

# "astronomer/Llama-3-70B-Special-Tokens-Adjusted 170000 80" \
# "codellama/CodeLlama-34b-hf 131072 48" \


#                                                   L  s r b   tok   e  n tolerance_factor
for config in \
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B 64000 32 1 1 8 131072  2  8  0.0" \
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B 64000 32 1 1 8 131072  2  16 0.0" \
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B 64000 32 1 1 8 131072  2  16 1.0" \
    "codellama/CodeLlama-34b-hf 131072              48 1 1 2 131072  2  8  0.0" \
    "codellama/CodeLlama-34b-hf 131072              48 1 1 4 131072  2  16 0.0" \
    "codellama/CodeLlama-34b-hf 131072              48 1 1 4 131072  2  16 1.0" \
    ; do

    read -r model_path attn_linear_breakpoint num_layers selective_ckpt resend_qkv batch_size num_tokens elongate_factor nnodes tolerance_factor <<< "$config"
    read -r sample_name change_long_doc_ratio <<< "$sample_config"
    
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

    # Run d2 with sweeping some tolerance factors
    export MODE=d2
    export MIN_TOLERANCE_FACTOR=$tolerance_factor
    export OUTPUT_DIR_SUFFIX_ADDON="-tol${tolerance_factor}"
    echo "🟡 Running d2 with NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR, MIN_TOLERANCE_FACTOR=$MIN_TOLERANCE_FACTOR"
    bash test_e2e_combined.salloc.sh
    echo "🟡 Finished running d2 with NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR, MIN_TOLERANCE_FACTOR=$MIN_TOLERANCE_FACTOR. Not guaranteed to be successful."
    echo "\a"


done
done
# done


set +e