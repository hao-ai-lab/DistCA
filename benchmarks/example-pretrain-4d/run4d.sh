set -e


# Create output directory
TS=$(TZ=America/Los_Angeles date +%m%d_%H%M%S)_PST
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUTPUT_DIR_PREFIX="${CURDIR}/logs"
mkdir -p "${OUTPUT_DIR_PREFIX}"

# -----------------------------
# Debugging Flags
# -----------------------------
export ENABLE_NSYS=0
export EXPERIMENT_LOG_MEMORY_USAGE=0
export EXPERIMENT_SKIP_FA2A_OP=0 # (DebugOnly: Ensure not stuck at fast a2a op)
export EXPERIMENT_SKIP_OPTIMIZER_STEP=1
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2
export SHOULD_ADD_DEBUG_CASES=0

export MAX_SAMPLE_ID=5
export EXPERIMENT_0TH_SAMPLE_WARMUP_TIMES=1
export EXPERIMENT_WARMUP_TIMES=0
export EXPERIMENT_REPEAT_TIMES=1

export EXPERIMENT_FA2A_SPLIT_SENDRECV=1
export EXPERIMENT_SHOULD_LOG_MEMORY_DURING_WARMUP=1
export EXPERIMENT_ADD_SELECTIVE_CKPT=1


model_config="deepseek-ai/DeepSeek-R1-Distill-Llama-8B 64000 8"


# Example configs running for formal benchmarking
# param_configs_cases=(
# # 8B 128k
# #    n bs mb t       mode  cp  pp  tp  comment 
#     "16 1 4 131072    d2   8  2  8  'd2 subopt-mem'  " 
#     "16 1 4 131072    d2   4  4  8  'd2 subopt-mem'  " 
#     "16 1 4 131072 wlbllm  4  4  8  'wlbllm subopt-mem'  "
#     "16 1 4 131072 wlbllm  8  2  8  'wlbllm subopt-mem'  "  
#     "16 2 2 131072 wlbllm  4  2  8  'wlbllm subopt-mem'  "  

# )
# for config in "${param_configs_cases[@]}"; do
#     cases+=("$model_config $config")
# done

# Example configs running for debugging
param_configs_cases=(
# 8B 128k
#    n bs mb t     mode  cp  pp  tp  comment 
    # "1 1 4 65536     d2   1  2  4  'd2 subopt-mem'  " 
    "2 1 4 65536     d2   1  2  8  'd2 subopt-mem'  " 
    # "2 1 4 65536     d2   1  2  8  'd2 subopt-mem'  " 
    # "2 1 4 65536 wlbllm   1  2  8  'wlbllm subopt-mem'  "

)
for config in "${param_configs_cases[@]}"; do
    cases+=("$model_config $config")
done


dists=(
    # "prolong 0.3"
    "wlbllm 0.0"
)



set -e
# -------------------------------

wlbllm_cases=()
d2_cases=()
echo "🟡 All cases:"
printf "%-50s  %10s  %6s  %10s  %14s  %10s  %6s  %7s  %7s  %7s\n" \
    "model_path" "attn_linear_breakpoint" "num_layers" "nnodes" "batch_size" "microbatch_size" "num_tokens" "mode" "cp_size" "pp_size" "tp_size" "comment"
for case in "${cases[@]}"; do
    read -r model_path attn_linear_breakpoint num_layers nnodes batch_size microbatch_size num_tokens mode cp_size pp_size tp_size comment <<< "$case"
    if [ "$mode" == "wlbllm" ]; then
        wlbllm_cases+=("$case")
        printf "%-50s  %10s  %6s  %10s  %14s  %10s  %6s  %7s  %7s  %7s\n" \
            "$model_path" "$attn_linear_breakpoint" "$num_layers" "$nnodes" "$batch_size" "$microbatch_size" "$num_tokens" "$mode" "$cp_size" "$pp_size" "$tp_size" "$comment"
    fi
    if [ "$mode" == "d2" ]; then
        d2_cases+=("$case")
        printf "%-50s  %10s  %6s  %10s  %14s  %10s  %6s  %7s  %7s  %7s\n" \
            "$model_path" "$attn_linear_breakpoint" "$num_layers" "$nnodes" "$batch_size" "$microbatch_size" "$num_tokens" "$mode" "$cp_size" "$pp_size" "$tp_size" "$comment"
    fi
done
echo "🟡 Total wlbllm cases: ${#wlbllm_cases[@]}"
echo "🟡 Total d2 cases: ${#d2_cases[@]}"
echo ""


sleep 3


max_cases=100
echo "🏁 Start regression sweep. Only running $max_cases cases."
cases_index=0
    

for sample_config in "${dists[@]}"; do
# for config in "${wlbllm_cases[@]}"; do
for config in "${cases[@]}"; do
    read -r model_path attn_linear_breakpoint num_layers nnodes batch_size microbatch_size num_tokens mode cp_size pp_size tp_size comment <<< "$config"
    read -r sample_name change_long_doc_ratio <<< "$sample_config"
    
    export MODE=$mode
    export BATCH_SIZE=$batch_size
    export NNODES=$nnodes
    export NUM_TOKENS=$num_tokens
    export NUM_MICROBATCH=$microbatch_size
    export CP_SIZE=$cp_size
    export PP_SIZE=$pp_size
    export TP_SIZE=$tp_size
    export ATTN_LINEAR_BREAKPOINT=$attn_linear_breakpoint
    export NUM_LAYERS=$num_layers
    export MODEL_PATH=$model_path
    export MODEL_PATH_normalized=$(echo $model_path | sed 's/\//_/g')
    export SAMPLE_NAME=$sample_name
    export CHANGE_LONG_DOC_RATIO=$change_long_doc_ratio
    

    export ELONGATE_FACTOR=$(($NUM_TOKENS / 65536))
    if [ $ELONGATE_FACTOR -lt 1 ]; then
        ELONGATE_FACTOR=1
    fi
    
    # D2 normal
    echo "🟡 Running config: MODE=$MODE, NNODES=$NNODES, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, NUM_MICROBATCH=$NUM_MICROBATCH, CP_SIZE=$CP_SIZE, PP_SIZE=$PP_SIZE, TP_SIZE=$TP_SIZE, MODE=$MODE, MODEL_PATH=$MODEL_PATH, MODEL_PATH_normalized=$MODEL_PATH_normalized, NUM_LAYERS=$NUM_LAYERS, ATTN_LINEAR_BREAKPOINT=$ATTN_LINEAR_BREAKPOINT, COMMENT=$COMMENT"

    if [ "${DRY_RUN:-0}" == "1" ]; then
        # echo "🔵 DRY_RUN=1, skipping execution"
        continue
    fi

    export OUTPUT_DIR_SUFFIX_ADDON="-${MODEL_PATH_normalized}"
    
    
    if [ $MODE == "d2" ]; then
        bash pretrain_llama.sh
    fi

    if [ $MODE == "wlbllm" ]; then
        bash test_wlb_e2e.sh
    fi

    echo "🟡 Finished running config: MODE=$MODE, NNODES=$NNODES, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, NUM_MICROBATCH=$NUM_MICROBATCH, CP_SIZE=$CP_SIZE, PP_SIZE=$PP_SIZE, TP_SIZE=$TP_SIZE, MODE=$MODE, MODEL_PATH=$MODEL_PATH, MODEL_PATH_normalized=$MODEL_PATH_normalized, NUM_LAYERS=$NUM_LAYERS, ATTN_LINEAR_BREAKPOINT=$ATTN_LINEAR_BREAKPOINT, COMMENT=$COMMENT"
    

    cases_index=$((cases_index + 1))
    if [ $cases_index -gt $max_cases ]; then
        break
    fi

done
done
