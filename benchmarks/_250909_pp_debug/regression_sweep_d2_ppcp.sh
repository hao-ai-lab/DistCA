set -e

# export JOBID=${JOBID:-710588}
export OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250909_pp_debug/logs

# export NUM_LAYERS=8
# export MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# export MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Llama-70B
# export MODEL_PATH=codellama/CodeLlama-34b-hf
# export NUM_LAYERS=16
# 


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
# export EXPERIMENT_LOG_MEMORY_USAGE=1
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2
export EXPERIMENT_SKIP_FA2A_OP=0 # (DebugOnly: Ensure not stuck at fast a2a op)
export EXPERIMENT_SKIP_OPTIMIZER_STEP=1
# export NNODES=8
export MAX_SAMPLE_ID=1

export NUM_LAYERS=4

export MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Llama-70B


set -e

export NUM_LAYERS=6
cases=(
    # (n bs mb t mode   cp pp tp    model_path)
    
    # "1  1 1 16384 wlbllm 1 1 8  deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # "1  1 1 16384     d2 1 1 8  deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    # "1  1 2 16384 wlbllm 1 2 2  deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # "1  1 2 16384     d2 1 2 2  deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    # "1  1 4 131072 wlbllm 2 2 2  deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # "1  1 4 131072     d2 2 2 2  deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    "4  1 4 131072 wlbllm 2 2 8  deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    "4  1 4 131072     d2 2 2 8  deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

    # "1  1 4 16384 wlbllm 2 2 2  dphn/Dolphin3.0-Llama3.2-3B"
    # "1  1 4 16384     d2 2 2 2  dphn/Dolphin3.0-Llama3.2-3B"



    # "2  2 2 65536 wlbllm 2 1 8  deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    # "2  2 2 65536     d2 2 1 8  deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    # "8  2 2 65536     d2 4 2 8  deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    # "8  2 2 65536 wlbllm 4 2 8  deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

    # "8  2 4 262144     d2 4 2 8  deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    # "8  2 4 262144 wlbllm 4 2 8  deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    
    # "8  2 4 262144     d2 2 4 8  deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    # "8  2 4 262144 wlbllm 2 4 8  deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
)
max_cases=100
echo "🏁 Start regression sweep. Only running $max_cases cases."
cases_index=0

for config in "${cases[@]}"; do
    read -r nnodes batch_size microbatch_size num_tokens mode cp_size pp_size tp_size model_path <<< "$config"
    
    export MODE=$mode
    export BATCH_SIZE=$batch_size
    export NNODES=$nnodes
    export NUM_TOKENS=$num_tokens
    export NUM_MICROBATCH=$microbatch_size
    export CP_SIZE=$cp_size
    export PP_SIZE=$pp_size
    export TP_SIZE=$tp_size
    export MODEL_PATH=$model_path

    export ELONGATE_FACTOR=$(($NUM_TOKENS / 65536))
    if [ $ELONGATE_FACTOR -lt 1 ]; then
        ELONGATE_FACTOR=1
    fi
    
    # D2 normal
    echo "🟡 Running config: MODE=$MODE, NNODES=$NNODES, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, NUM_MICROBATCH=$NUM_MICROBATCH, CP_SIZE=$CP_SIZE, PP_SIZE=$PP_SIZE, TP_SIZE=$TP_SIZE, MODE=$MODE, MODEL_PATH=$MODEL_PATH"

    if [ "${DRY_RUN:-0}" == "1" ]; then
        # echo "🔵 DRY_RUN=1, skipping execution"
        continue
    fi
    
    
    if [ $MODE == "d2" ]; then
        bash test_megatron_e2e_pipeline_with_cp.sh
    fi

    if [ $MODE == "wlbllm" ]; then
        bash test_wlb_e2e.sh
    fi

    echo "🟡 Finished running config: MODE=$MODE, NNODES=$NNODES, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, NUM_MICROBATCH=$NUM_MICROBATCH, CP_SIZE=$CP_SIZE, PP_SIZE=$PP_SIZE, TP_SIZE=$TP_SIZE, MODE=$MODE, MODEL_PATH=$MODEL_PATH"
    

    cases_index=$((cases_index + 1))
    if [ $cases_index -gt $max_cases ]; then
        break
    fi
done






# ----------
cases=(
  # (n bs mb t mode     cp pp tp    model_path)
    # "8  2 2 262144 d2     4 2 8"
    # "8  2 2 262144 wlbllm 4 2 8"
    # "8  2 4 262144     d2 4 2 8"
    # "8  2 4 262144 wlbllm 4 2 8"
    
    "8  2 4 262144     d2 4 2 8  deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    "8  2 4 262144 wlbllm 4 2 8  deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

    # "8  2 8 262144     d2 4 2 8  deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    # "8  2 8 262144 wlbllm 4 2 8  deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    
    "8  2 4 262144     d2 2 4 8  deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    "8  2 4 262144 wlbllm 2 4 8  deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

    # "8  2 4 262144     d2 2 4 8"
    # "8  2 4 262144 wlbllm 2 4 8"

    # "8  2 4 262144     d2 2 2 8" # dp 2
    # "8  2 4 262144 wlbllm 2 2 8" # dp 2 

    # "8  2 8 262144     d2 2 2 8" # dp 2
    # "8  2 8 262144 wlbllm 2 2 8" # dp 2
    # "8  2 2 262144 d2     4 2 8"
    # "8  2 2 262144 d2     2 4 8"
    # "8  2 4 262144 d2     4 2 8"
    # "8  2 4 262144 d2     2 4 8"
    # "8  2 8 262144 wlbllm 4 2 8"
    # "8  2 8 262144 wlbllm 2 4 8"
    # "8  2 16 262144 wlbllm 4 2 8"
    # "8  2 16 262144 wlbllm 2 4 8"
    # "4 2 4 262144 d2 2 2 8"
    # "2 2 8 32768 d2 2 2 4"
    # "1 1 16 16384 d2 1 2 4"
    # "1 2 4 32768 d2 1 2 4"
)
cases=(
    # (n bs mb t mode cp pp tp)
    # 🟢 Passed
    # "2 0 2 16384 d2 1 2 8"
    # "4 1 2 65536 d2 2 2 8"
    # "4 1 2 65536 d2 2 2 8"
    # "8 2 2 65536 d2 2 2 8"
    # "16 2 2 65536 d2 4 2 8"
    # "16 1 4 131072 d2 4 4 8"
    # "32 0 8 65536 d2 1 4 8"
    # "32 1 8 16384 d2 2 4 8"
    # "32 1 4 65536 d2 2 4 8"
    # "32 4 8 65536 d2 8 4 8"
    # "32 1 4 65536 d2 2 4 8"
    # "32 1 4 65536 d2 2 4 8"
    # "32 1 4 262144 d2 2 4 8" # we are slower
    
    # 🟡 Running
    # "8 2 4 262144 d2 2 4 8" # we are slower
    # "32 2 4 262144 d2 2 4 8" # we are slower
    # "16 2 2 65536 d2 2 4 8"
    # "32 4 8 65536 d2 4 8 8"

    # 🔴 Failed
    # "32 4 8 262144 d2 2 8 8"
    # "32 4 8 16384 d2 8 4 8"
    
    
    # ⚪ Ready

)