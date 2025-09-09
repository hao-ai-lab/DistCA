set -e

# export JOBID=${JOBID:-710588}
export ENABLE_NSYS=1
export OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250909_pp_debug/logs

# export MODEL_PATH=codellama/CodeLlama-34b-hf
# export NUM_LAYERS=8
export MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
export NUM_LAYERS=8



cases=(
    # (n bs mb t mode cp pp tp)
    # 🟢 Passed
    # "2 1 2 32768 d2 1 2 8"
    
    # 🟡 Running
    "2 1 1 32768 d2 2 1 8"

    # 🔴 Failed
    # "2 2 2 32768 d2 1 2 8" # Stuck at send_backward_recv_backward: num batch vs num microbatch?
    
    
    # ⚪Ready
    # "2 2 2 32768 wlbllm 2 1 8" 
)


# -----------------------------
# Debugging Flags
# -----------------------------
export EXPERIMENT_PYTHON_DEBUG_TRACE_CALLS=1
export EXPERIMENT_LOG_MEMORY_USAGE=1


max_cases=1
echo "🏁 Start regression sweep. Only running $max_cases cases."
cases_index=0
for config in "${cases[@]}"; do


    read -r nnodes batch_size microbatch_size num_tokens mode cp_size pp_size tp_size <<< "$config"
    
    export MODE=$mode
    export BATCH_SIZE=$batch_size
    export NNODES=$nnodes
    export NUM_TOKENS=$num_tokens
    export NUM_MICROBATCH=$microbatch_size
    export CP_SIZE=$cp_size
    export PP_SIZE=$pp_size
    export TP_SIZE=$tp_size
    
    echo "🟡 Running config: NNODES=$NNODES, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, NUM_MICROBATCH=$NUM_MICROBATCH, CP_SIZE=$CP_SIZE, PP_SIZE=$PP_SIZE, TP_SIZE=$TP_SIZE, MODE=$MODE"
    
    cases_index=$((cases_index + 1))
    if [ $cases_index -gt $max_cases ]; then
        break
    fi
    sleep 2.5
    bash test_megatron_e2e_pipeline_combined.sh
done