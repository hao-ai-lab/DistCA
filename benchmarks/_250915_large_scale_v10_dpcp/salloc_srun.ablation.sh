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
export OUTPUT_DIR_PREFIX="/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250915_large_scale_v10_dpcp/logs.v2-ablation"
export MAX_SAMPLE_ID=10
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2
export TP_SIZE=8
export ENABLE_NSYS=1
# export EXPERIMENT_LOG_MEMORY_USAGE=1
export EXPERIMENT_LOG_MEMORY_USAGE=0
export EXPERIMENT_REPEAT_TIMES=1
export EXPERIMENT_WARMUP_TIMES=2
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
export MODEL_PATH=codellama/CodeLlama-34b-hf

export EXPERIMENT_D2_BALANCE_PING_PONG=0

# ------------------------------------
# Check and skip success runs
# ------------------------------------
eid_file=$OUTPUT_DIR_PREFIX/success_eids.txt
python /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250907_large_scale_v6/query_success_runs.py --folder $OUTPUT_DIR_PREFIX --output_file $eid_file

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

# Run one d2 + one wlbllm-cpMax to justify the result.
for sample_config in \
"wlbllm 0.0" \
; do

# "astronomer/Llama-3-70B-Special-Tokens-Adjusted 170000 80" \
# "codellama/CodeLlama-34b-hf 131072 48" \
for model_config in \
"deepseek-ai/DeepSeek-R1-Distill-Llama-8B 64000 32" \
; do


# "1 1 4 131072 2 32" \
#     "1 1 4 262144 4 32" \
#     "1 1 4 524288 8 32" \
#     "1 1 1 262144 4 16" \
#     "1 1 1 524288 8 16" \
#     "1 1 1 262144 4 32" \
#     "1 1 1 524288 8 32" \

for config in \
    "1 1 1 131072 2 2" \
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
    
#    1. Signal Communication for non-PP (EST: 1hr for fig, 1hr for text?) (no need for extra coding. Experiment est: 1hr)
#       1. Only run one model, only one or two #nodes to speedup experiment
#    2. Non-PingPong (use one stream for compute and comm) (EST: 1hr for fig, 1hr for text?) (no need for extra coding. Experiment est: 1hr)
#       1. Same as above.
#    3. Modifying scheduler's tolerance factor for non-PP (EST: 1hr for fig, 1hr for text?) (no need for extra coding. Experiment est: 



    
    # Run d2 with signal only
    export MODE=d2
    export MIN_TOLERANCE_FACTOR=0.05
    export OUTPUT_DIR_SUFFIX_ADDON="-signal"
    export EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0=1 # <- set to 1 to enable signal-only communication
    echo "🟡 Running d2-signal with NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR, MIN_TOLERANCE_FACTOR=$MIN_TOLERANCE_FACTOR"
    if [ $DRY_RUN -eq 0 ]; then
        bash test_e2e_combined.salloc.sh
        echo "🟡 Finished running d2-signal with NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR, MIN_TOLERANCE_FACTOR=$MIN_TOLERANCE_FACTOR. Not guaranteed to be successful."
        echo "\a"
    fi
        

    # # Run d2 with no pingpong
    # export MODE=d2
    # export MIN_TOLERANCE_FACTOR=0.05
    # export OUTPUT_DIR_SUFFIX_ADDON="-no-pingpong"
    # echo "🟡 Running d2-signal with NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR, MIN_TOLERANCE_FACTOR=$MIN_TOLERANCE_FACTOR"
    # if [ $DRY_RUN -eq 0 ]; then
    #     bash test_e2e_combined.salloc.sh
    #     echo "🟡 Finished running d2-signal with NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR, MIN_TOLERANCE_FACTOR=$MIN_TOLERANCE_FACTOR. Not guaranteed to be successful."
    #     echo "\a"
    # fi

    # Run d2 with sweeping some tolerance factors
    for tolerance_factor in 0.05 0.2 0.5 0.8 1.0; do
        export MODE=d2
        export MIN_TOLERANCE_FACTOR=$tolerance_factor
        export OUTPUT_DIR_SUFFIX_ADDON="-tol${tolerance_factor}"
        eid="d2-cp1-n${NNODES}-b${BATCH_SIZE}-t${NUM_TOKENS}-tol${tolerance_factor}"
        echo "🟡 Running d2 with NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR, MIN_TOLERANCE_FACTOR=$MIN_TOLERANCE_FACTOR"
        if [ $DRY_RUN -eq 0 ]; then
            bash test_e2e_combined.salloc.sh
            echo "🟡 Finished running d2 with NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR, MIN_TOLERANCE_FACTOR=$MIN_TOLERANCE_FACTOR. Not guaranteed to be successful."
            echo "\a"
        fi
    done


done
done
done


set +e