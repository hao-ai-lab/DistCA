


export JOBID=710588
export ENABLE_NSYS=1
# export EXPERIMENT_LOG_MEMORY_USAGE=1
export MODE="wlbllm"
export ENABLE_NSYS=1

# Define test configurations
declare -a test_configs=(
    # NNODES  NUM_MB  TP  PP  CP  LAYERS  TOKENS
    # ✅ 
    # "  1      2       2   2   2   8       16384"  # ✅ 
    # "  2      4       2   4   2   8       16384"  # ✅ 
    # "  8      4       8   4   2   8       16384"  # ✅ 
    # "  8      4       8   4   2   32      16384"  # ✅ 
    # "   4     4       1   4   2   8       65536"  # ✅ - labels=None is required
    # "  16     4       8   4   2   48      16384"  # ✅ 
    # "  32     4       8   4   2   48      65536"  # ✅
    # "  32     4       8   4   8   48      65536"  # ✅
    # "  32     8       8   4   8   48      262144"  # ✅
)
declare -a configs=()

# Generate configs using nested loops
for ppxcp in 4 8 16 32; do
    # Calculate pp and cp sizes that multiply to ppxcp
    for pp in 1 2 4 8; do
        cp=$((ppxcp / pp))
        # Only use valid pp/cp combinations
        if [ $((pp * cp)) -eq $ppxcp ]; then
            for tokens in 131072 262144 524288; do # 128K, 256K, 512K
                for batches in 1 2 4 8; do
                    # NNODES  NUM_MB  TP  PP  CP  LAYERS  TOKENS  NUM_BATCHES
                    configs+=("  32     8       8   $pp   $cp   48      $tokens      $batches")
                done
            done
        fi
    done
done

# Run tests for each configuration
for config in "${configs[@]}"; do
    
    read -r nodes mb tp pp cp nlayers num_tokens num_batches<<< "$config"
    echo "Running test with: NNODES=$nodes NUM_MICROBATCH=$mb TP_SIZE=$tp PP_SIZE=$pp CP_SIZE=$cp NUM_LAYERS=$nlayers NUM_TOKENS=$num_tokens NUM_BATCHES=$num_batches"
    NNODES=$nodes NUM_MICROBATCH=$mb TP_SIZE=$tp PP_SIZE=$pp CP_SIZE=$cp NUM_LAYERS=$nlayers NUM_TOKENS=$num_tokens NUM_BATCHES=$num_batches bash test_wlb_e2e.sh
    echo "Finished running test with: NNODES=$nodes NUM_MICROBATCH=$mb TP_SIZE=$tp PP_SIZE=$pp CP_SIZE=$cp NUM_LAYERS=$nlayers NUM_TOKENS=$num_tokens NUM_BATCHES=$num_batches"
done