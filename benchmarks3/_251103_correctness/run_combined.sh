# 
# Usage
#   export JOBID=<JOBID>
#   export NNODES=<NNODES>
#   bash salloc_srun.sh
# 
# set -e

export NNODES=${NNODES:-32}
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
export OUTPUT_DIR_PREFIX="/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks3/_251103_correctness/logs.v1"
export MAX_SAMPLE_ID=30
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1
export TP_SIZE=${TP_SIZE:-8}
export ENABLE_NSYS=0
# export EXPERIMENT_LOG_MEMORY_USAGE=1
export EXPERIMENT_LOG_MEMORY_USAGE=0
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

# export MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# export MODEL_PATH=codellama/CodeLlama-34b-hf
# export MODEL_PATH=codellama/CodeLlama-34b-hf

export EXPERIMENT_D2_BALANCE_PING_PONG=0

export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO__DISABLE_CHECK=1    


# export TENSOR_DUMP_DIR=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks3/_251103_correctness/logs.v1.tensors
# export TENSOR_DUMP_SUFFIX=d2


# ------------------------------------
# Check and skip success runs
# ------------------------------------
# Distribution Name
# CHANGE_LONG_DOC_RATIO  
# ATTN_LINEAR_BREAKPOINT

# Run one d2 + one wlbllm-cpMax to justify the result.
for sample_config in \
"wlbllm 0.0" \
; do

# "astronomer/Llama-3-70B-Special-Tokens-Adjusted 170000 80" \
# "codellama/CodeLlama-34b-hf 131072 48" \
# "deepseek-ai/DeepSeek-R1-Distill-Llama-8B 64000 32" \
# meta-llama/Llama-3.2-1B
# "deepseek-ai/DeepSeek-R1-Distill-Llama-8B 64000 16" \
for model_config in \
"meta-llama/Llama-3.2-1B 64000 16" \
; do


# "1 1 4 131072 2 32" \
#     "1 1 4 262144 4 32" \
#     "1 1 4 524288 8 32" \
#     "1 1 1 262144 4 16" \
#     "1 1 1 524288 8 16" \
#     "1 1 1 262144 4 32" \
#     "1 1 1 524288 8 32" \

# selective_ckpt resend_qkv batch_size num_tokens elongate_factor nnodes
for config in \
    "1 1 1 32768 2 2" \
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


    tolerance_factor=0.05
    export MODE=d2
    export MIN_TOLERANCE_FACTOR=$tolerance_factor
    export OUTPUT_DIR_SUFFIX_ADDON="-tol${tolerance_factor}"
    eid="d2-cp1-n${NNODES}-b${BATCH_SIZE}-t${NUM_TOKENS}-tol${tolerance_factor}"
    echo "🟡 Running d2 with TP_SIZE=$TP_SIZE, NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR, MIN_TOLERANCE_FACTOR=$MIN_TOLERANCE_FACTOR"
    bash training_3d.sh
    echo "🟡 Finished running d2 with NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR, MIN_TOLERANCE_FACTOR=$MIN_TOLERANCE_FACTOR. Not guaranteed to be successful."
    echo "\a"

done
done
done


set +e