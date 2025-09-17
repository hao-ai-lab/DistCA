# export JOBID=710588

# Define test configurations
declare -a test_configs=(
    # NNODES  NUM_MB  TP  PP  CP  LAYERS  TOKENS  NUM_BATCHES
    # "  32     2       8   2   2   48      131072  1"
    # "  32     2       8   2   2   48      131072  2"
    # "  32     2       8   2   2   48      131072  4"
    # "  32     2       8   2   2   48      131072  8"
    # "  32     2       8   2   2   48      262144  1"
    # "  32     2       8   2   2   48      262144  2"
    # "  32     2       8   2   2   48      262144  4"
    # "  32     2       8   2   2   48      262144  8"
    # "  32     2       8   2   2   48      524288  1"
    # "  32     2       8   2   2   48      524288  2"
    # "  32     2       8   2   2   48      524288  4"
    # "  32     2       8   2   2   48      524288  8"
    # Repeat for other PP_SIZE and CP_SIZE combinations...
)

declare -a success_test_configs=(
    # ✅
    # NNODES  NUM_MB  TP  PP  CP  LAYERS  TOKENS  NUM_BATCHES  UP_SAMPLE_FACTOR  ELONGATE_FACTOR
    "    1     2       2   2   2   8      32768      1             4                  1"
    "    1     2       2   2   2   8      131072     1             4                  2"
    "    1     2       2   2   2   8      32768      2             4                  2"
    "    2     4       2   4   2   8      32768      2             4                  2"
    "    8     4       2   4   2   8      65536      2             4                  2"
    "    8     4       2   4   2   8      65536      2             4                  2"
    "    8     4       2   4   2   8      65536      2             4                  2"
    "    8     4       2   4   2   8      262144     4             4                  4"
)
declare -a failed_test_configs=(
    # NNODES  NUM_MB  TP  PP  CP  LAYERS  TOKENS  NUM_BATCHES  UP_SAMPLE_FACTOR  ELONGATE_FACTOR
    # "    8     4       2   4   2   8      65536      4             4                  2"
    
)
declare -a test_configs=(
    # NNODES  NUM_MB  TP  PP  CP  LAYERS  TOKENS  NUM_BATCHES  UP_SAMPLE_FACTOR  ELONGATE_FACTOR
    "    1     2       2   2   2   16      32768      1             4                  1"
    "    1     2       2   2   2   16      131072     1             4                  2"
    "    1     2       2   2   2   16      32768      2             4                  2"
    "    2     4       2   4   2   32      32768      2             4                  2"
    "    8     4       2   4   2   32      65536      2             4                  2"
    "    8     4       2   4   2   32      65536      2             4                  2"
    "    8     4       2   4   2   32      65536      2             4                  2"
    "    8     4       2   4   2   32      262144     4             4                  4"
    # "    8     4       2   4   4   8      262144     4             4                  4"
    # "    8     4       2   4   2   8      65536      4             4                  2"
    
    # max_seqlen_q_list.append(torch.tensor([max(this_chunk_docs)], dtype=torch.int32).to(device))
    # <class 'ValueError'>: max() iterable argument is empty

    # "    8     4       2   4   2   8      65536      4             4                  2"
    # "    8     4       2   4   4   8      65536      8             4                  2"
    # "    8     4       2   4   4   8      65536      8             4                  2"
    # "    8    2       8   2   2   48      131072      4             4                  2"
    # "    8    2       8   2   2   48      131072      4"
)


export TORCH_SHOW_CPP_STACKTRACES=0
export TORCH_CPP_LOG_LEVEL=ERROR
export TORCH_DISTRIBUTED_DEBUG=OFF

export ENABLE_NSYS=1
export OUTPUT_DIR_PREFIX="/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250911_4dparallel/logs.v1-compare/"
mkdir -p $OUTPUT_DIR_PREFIX
# export MODEL_PATH=codellama/CodeLlama-34b-hf
export MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
export MAX_SAMPLE_ID=5
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2
# export EXPERIMENT_PYTHON_BETTER_TRACEBACK=1
# export EXPERIMENT_LOG_MEMORY_USAGE=1
# export EXPERIMENT_PYTHON_DEBUG_TRACE_CALLS=1

# Run tests for each configuration
for config in "${test_configs[@]}"; do
    read -r nodes mb tp pp cp nlayers num_tokens num_batches up_sample_factor elongate_factor <<< "$config"
    
    export NNODES=$nodes
    export NUM_MICROBATCH=$mb
    export TP_SIZE=$tp
    export PP_SIZE=$pp
    export CP_SIZE=$cp
    export NUM_LAYERS=$nlayers
    export NUM_TOKENS=$num_tokens
    export NUM_BATCHES=$num_batches
    export UP_SAMPLE_FACTOR=$up_sample_factor
    export ELONGATE_FACTOR=$elongate_factor

    
    export MODE=wlbllm
    echo "🟡 Running $MODE test with: NNODES=$nodes NUM_MICROBATCH=$mb TP_SIZE=$tp PP_SIZE=$pp CP_SIZE=$cp NUM_LAYERS=$nlayers NUM_TOKENS=$num_tokens NUM_BATCHES=$num_batches"
    bash test_wlb_e2e.sh
    echo "🟡 Finished running $MODE test with: NNODES=$nodes NUM_MICROBATCH=$mb TP_SIZE=$tp PP_SIZE=$pp CP_SIZE=$cp NUM_LAYERS=$nlayers NUM_TOKENS=$num_tokens NUM_BATCHES=$num_batches"


    export MODE=d2
    echo "🟡 Running $MODE test with: NNODES=$nodes NUM_MICROBATCH=$mb TP_SIZE=$tp PP_SIZE=$pp CP_SIZE=$cp NUM_LAYERS=$nlayers NUM_TOKENS=$num_tokens NUM_BATCHES=$num_batches"
    bash test_megatron_e2e_pipeline_with_cp.sh
    echo "🟡 Finished running $MODE test with: NNODES=$nodes NUM_MICROBATCH=$mb TP_SIZE=$tp PP_SIZE=$pp CP_SIZE=$cp NUM_LAYERS=$nlayers NUM_TOKENS=$num_tokens NUM_BATCHES=$num_batches"

done