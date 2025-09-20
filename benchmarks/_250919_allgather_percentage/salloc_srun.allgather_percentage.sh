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
export OUTPUT_DIR_PREFIX="/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250919_allgather_percentage/logs.v1-allgather_percentage"
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2
export TP_SIZE=8
# export EXPERIMENT_LOG_MEMORY_USAGE=1
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

# check the success_eids.txt file
# SUCCESS_EIDS=""
# if [ -f $eid_file ]; then
#     SUCCESS_EIDS=$(cat $eid_file)
#     echo "🟡 SUCCESS_EIDS=$SUCCESS_EIDS"
# else
#     echo "🟡 $eid_file file does not exist"
# fi

# Distribution Name
# CHANGE_LONG_DOC_RATIO  
# ATTN_LINEAR_BREAKPOINT

export EXPERIMENT_FA2A_SPLIT_SENDRECV=1

export ENABLE_NSYS=1
export MAX_SAMPLE_ID=1
export SHOULD_ADD_DEBUG_CASES=1

# Run one d2 + one wlbllm-cpMax to justify the result.
for sample_config in \
"wlbllm 0.0" \
; do

# "astronomer/Llama-3-70B-Special-Tokens-Adjusted 170000 80" \
# "codellama/CodeLlama-34b-hf 131072 48" \
for model_config in \
"deepseek-ai/DeepSeek-R1-Distill-Llama-8B 64000 4" \
; do

#    s r b   tok  e n
for config in \
    "1 1 8 131072 2 8" \
    "1 1 4 262144 4 8" \
    "1 1 2 524288 8 8" \
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
        counter=$((counter + 1))
        # if [ $counter -ge 3 ]; then
        #     break
        # fi
    done

done
done
done


set +e