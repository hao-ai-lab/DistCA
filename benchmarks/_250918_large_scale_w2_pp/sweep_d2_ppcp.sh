set -e

# export JOBID=${JOBID:-710588}
TS=$(TZ=America/Los_Angeles date +%m%d_%H%M%S)_PST
export OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250918_large_scale_w2_pp/logs.v2-sweep-pp-memory/${TS}

# -----------------------------
# TorchRun Final Logging Flag (for cleaner console)
# -----------------------------
# Quiet C++/glog + c10d
export TORCH_CPP_LOG_LEVEL=ERROR        # hides INFO/WARNING from libtorch/c10d
export GLOG_minloglevel=2               # 0=INFO,1=WARNING,2=ERROR,3=FATAL
export TORCH_DISTRIBUTED_DEBUG=OFF      # don't spam states
export NCCL_DEBUG=ERROR                 # only errors from NCCL
unset TORCH_SHOW_CPP_STACKTRACES        # disable those C++ frames in tracebacks

# -----------------------------
# Debugging Flags
# -----------------------------
export ENABLE_NSYS=1
# export EXPERIMENT_PYTHON_DEBUG_TRACE_CALLS=1
export EXPERIMENT_LOG_MEMORY_USAGE=1
export EXPERIMENT_SKIP_FA2A_OP=0 # (DebugOnly: Ensure not stuck at fast a2a op)
export EXPERIMENT_SKIP_OPTIMIZER_STEP=1
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2
# export NNODES=8
export MAX_SAMPLE_ID=3

set -e
export EXPERIMENT_FA2A_SPLIT_SENDRECV=1


model_configs=(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B 64000 32" \
    "codellama/CodeLlama-34b-hf 24" \
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B 40"
)

cases=(
    #n bs mb t       mode cp  pp  tp )
    "8  1 4 262144 wlbllm  2  4   8"
    "8  1 4 262144     d2  2  4   8"
    "8  2 2 262144 wlbllm  4  2   8"
    "8  2 2 262144     d2  4  2   8"

    "8  1 8 131072 wlbllm  2  4   8"
    "8  1 8 131072     d2  2  4   8"
    "8  2 4 131072 wlbllm  4  2   8"
    "8  2 4 131072     d2  4  2   8"

    "8  1 8 65536 wlbllm  2  4   8"
    "8  1 8 65536     d2  2  4   8"
    "8  2 4 65536 wlbllm  4  2   8"
    "8  2 4 65536     d2  4  2   8"
)

max_cases=10000
echo "🏁 Start regression sweep. Only running $max_cases cases."
cases_index=0

for model_config in "${model_configs[@]}"; do
    
for config in "${cases[@]}"; do
    
    read -r model_path num_layers <<< "$model_config"
    read -r nnodes batch_size microbatch_size num_tokens mode cp_size pp_size tp_size <<< "$config"
    
    export MODE=$mode
    export BATCH_SIZE=$batch_size
    export NNODES=$nnodes
    export NUM_TOKENS=$num_tokens
    export NUM_MICROBATCH=$microbatch_size
    export CP_SIZE=$cp_size
    export PP_SIZE=$pp_size
    export TP_SIZE=$tp_size
    export NUM_LAYERS=$num_layers
    export MODEL_PATH=$model_path
    export MODEL_PATH_normalized=$(echo $model_path | sed 's/\//_/g')

    export ELONGATE_FACTOR=$(($NUM_TOKENS / 65536))
    if [ $ELONGATE_FACTOR -lt 1 ]; then
        ELONGATE_FACTOR=1
    fi
    
    # D2 normal
    echo "🟡 Running config: MODE=$MODE, NNODES=$NNODES, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, NUM_MICROBATCH=$NUM_MICROBATCH, CP_SIZE=$CP_SIZE, PP_SIZE=$PP_SIZE, TP_SIZE=$TP_SIZE, MODE=$MODE, MODEL_PATH=$MODEL_PATH, MODEL_PATH_normalized=$MODEL_PATH_normalized, NUM_LAYERS=$NUM_LAYERS"

    if [ "${DRY_RUN:-0}" == "1" ]; then
        # echo "🔵 DRY_RUN=1, skipping execution"
        continue
    fi

    export OUTPUT_DIR_SUFFIX_ADDON="-${MODEL_PATH_normalized}"
    
    
    if [ $MODE == "d2" ]; then
        bash test_megatron_e2e_pipeline_with_cp.sh
    fi

    if [ $MODE == "wlbllm" ]; then
        bash test_wlb_e2e.sh
    fi

    echo "🟡 Finished running config: MODE=$MODE, NNODES=$NNODES, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, NUM_MICROBATCH=$NUM_MICROBATCH, CP_SIZE=$CP_SIZE, PP_SIZE=$PP_SIZE, TP_SIZE=$TP_SIZE, MODE=$MODE, MODEL_PATH=$MODEL_PATH, MODEL_PATH_normalized=$MODEL_PATH_normalized, NUM_LAYERS=$NUM_LAYERS"
    

    cases_index=$((cases_index + 1))
    if [ $cases_index -gt $max_cases ]; then
        break
    fi

done
done