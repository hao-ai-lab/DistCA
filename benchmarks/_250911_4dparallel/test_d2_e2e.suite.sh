export JOBID=710588
export ENABLE_NSYS=1
export MODE="d2"
export OUTPUT_DIR_PREFIX="/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250911_4dparallel/d2-4d/"
mkdir -p $OUTPUT_DIR_PREFIX
export MODEL_PATH=codellama/CodeLlama-34b-hf
export MAX_SAMPLE_ID=5
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2

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
    "  32     2       8   2   2   48      524288  4"
    # "  32     2       8   2   2   48      524288  8"
    # Repeat for other PP_SIZE and CP_SIZE combinations...
)

# Run tests for each configuration
for config in "${test_configs[@]}"; do
    read -r nodes mb tp pp cp nlayers num_tokens num_batches <<< "$config"
    echo "Running test with: NNODES=$nodes NUM_MICROBATCH=$mb TP_SIZE=$tp PP_SIZE=$pp CP_SIZE=$cp NUM_LAYERS=$nlayers NUM_TOKENS=$num_tokens NUM_BATCHES=$num_batches"
    
    export NNODES=$nodes
    export NUM_MICROBATCH=$mb
    export TP_SIZE=$tp
    export PP_SIZE=$pp
    export CP_SIZE=$cp
    export NUM_LAYERS=$nlayers
    export NUM_TOKENS=$num_tokens
    export NUM_BATCHES=$num_batches
    export ELONGATE_FACTOR=$((num_tokens / 65536))

    bash test_megatron_e2e_pipeline_with_cp.sh
    echo "Finished running test with: NNODES=$nodes NUM_MICROBATCH=$mb TP_SIZE=$tp PP_SIZE=$pp CP_SIZE=$cp NUM_LAYERS=$nlayers NUM_TOKENS=$num_tokens NUM_BATCHES=$num_batches"
done