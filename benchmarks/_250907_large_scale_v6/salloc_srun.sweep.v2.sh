# 
# Usage
#   export JOBID=<JOBID>
#   export NNODES=<NNODES>
#   bash salloc_srun.sh
# 

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
export OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250907_large_scale_v6/logs.v-sweep
export MAX_SAMPLE_ID=20
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2
export TP_SIZE=8
export ENABLE_NSYS=1
# export EXPERIMENT_LOG_MEMORY_USAGE=1
export EXPERIMENT_LOG_MEMORY_USAGE=0
export EXPERIMENT_REPEAT_TIMES=1
export EXPERIMENT_WARMUP_TIMES=1
export EXPERIMENT_D2_FLASH_ATTN_SKIP_GET_BACKEND=1 # default 1
export SHOULD_ADD_DEBUG_CASES=0
# export EXPERIMENT_FA2A_BARRIER=1
# export EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0=0 # default 0

DRY_RUN=${DRY_RUN:-0}

# export MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# export MODEL_PATH=codellama/CodeLlama-34b-hf
export MODEL_PATH=codellama/CodeLlama-34b-hf
export NUM_LAYERS=3


# ------------------------------------
# Check and skip success runs
# ------------------------------------

python query_success_runs.py --folder $OUTPUT_DIR_PREFIX

# check the success_eids.txt file
SUCCESS_EIDS=""
if [ -f success_eids.txt ]; then
    SUCCESS_EIDS=$(cat success_eids.txt)
    echo "🟡 SUCCESS_EIDS=$SUCCESS_EIDS"
else
    echo "🟡 success_eids.txt file does not exist"
fi


# ------------------------------------
# Run experiments
# ------------------------------------
# for config in \
#     "1 1 $((NNODES / 8)) 524288 8" \
#     "1 1 $((NNODES / 8)) 262144 4" \
#     "0 0 $((NNODES / 8)) 131072 2"; do

# for config in \
#     "1 1 1 524288 8" \
#     "1 1 1 262144 4" \
#     "0 0 1 131072 2" \
#     "1 1 2 524288 8" \
#     "1 1 2 262144 4" \
#     "0 0 2 131072 2" \
#     "1 1 4 524288 8" \
#     "1 1 4 262144 4" \
#     "0 0 4 131072 2" \
#     "1 1 1 1048576 8" \
#     ; do

for config in \
    "1 1 2 524288 8" \
    "1 1 4 262144 4" \
    "0 0 4 131072 2" \
    "1 1 1 524288 8" \
    "1 1 2 262144 4" \
    "0 0 2 131072 2" \
    "1 1 4 524288 8" \
    "1 1 4 1048576 16" \
    ; do


    read -r selective_ckpt resend_qkv batch_size num_tokens elongate_factor <<< "$config"
    
    export EXPERIMENT_ADD_SELECTIVE_CKPT=$selective_ckpt
    export EXPERIMENT_SHOULD_RESEND_QKV=$resend_qkv
    export BATCH_SIZE=$batch_size
    export NUM_TOKENS=$num_tokens
    export ELONGATE_FACTOR=$elongate_factor

    # Run d2 mode
    export MODE=d2
    eid="d2-cp1-n${NNODES}-b${BATCH_SIZE}-t${NUM_TOKENS}"
    if [[ "$SUCCESS_EIDS" =~ "$eid" ]]; then
        echo "🟢 Skip: $eid"
        continue
    fi

    echo "🟡 Running d2 with NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR"
    if [ $DRY_RUN -eq 0 ]; then
        bash test_e2e_combined.salloc.sh
        echo "🟡 Finished running d2 with NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR. Not guaranteed to be successful."
    fi
    
    # Run wlbllm mode with different CP sizes

    # for CP_SIZE in 16 ; do
    for CP_SIZE in 32 16 8 4 2 1; do
        if [ $CP_SIZE -gt $NNODES ]; then
            continue
        fi
        DP_SIZE=$((NNODES / CP_SIZE))
        if [ $DP_SIZE -gt $(($BATCH_SIZE * 2)) ]; then
            continue
        fi
        eid="wlbllm-cp${CP_SIZE}-n${NNODES}-b${BATCH_SIZE}-t${NUM_TOKENS}"
        if [[ "$SUCCESS_EIDS" =~ "$eid" ]]; then
            echo "🟢 Skip: $eid"
            continue
        fi

        echo "🟡 Running wlbllm with CP_SIZE=$CP_SIZE, DP_SIZE=$DP_SIZE, NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR"
        export MODE=wlbllm CP_SIZE=$CP_SIZE
        if [ $DRY_RUN -eq 0 ]; then
            bash test_e2e_combined.salloc.sh
            echo "🟡 Finished running wlbllm with CP_SIZE=$CP_SIZE, DP_SIZE=$DP_SIZE, NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR. Not guaranteed to be successful."
        fi
    done
    
done



# salloc -N 8 -G 64 bash salloc_srun.sh