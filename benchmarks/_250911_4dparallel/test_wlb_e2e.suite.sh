export JOBID=710588
export ENABLE_NSYS=1
export MODE="wlbllm"
export OUTPUT_DIR_PREFIX="/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250911_4dparallel/wlbllm-4d/"
mkdir -p $OUTPUT_DIR_PREFIX
# export EXPERIMENT_LOG_MEMORY_USAGE=1

# Define test configurations
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
    "    8     4       2   4   4   8      262144     4             4                  4"
    # "    8     4       2   4   2   8      65536      4             4                  2"
    
    # max_seqlen_q_list.append(torch.tensor([max(this_chunk_docs)], dtype=torch.int32).to(device))
    # <class 'ValueError'>: max() iterable argument is empty

    # "    8     4       2   4   2   8      65536      4             4                  2"
    # "    8     4       2   4   4   8      65536      8             4                  2"
    # "    8     4       2   4   4   8      65536      8             4                  2"
    # "    8    2       8   2   2   48      131072      4             4                  2"
    # "    8    2       8   2   2   48      131072      4"
)
declare -a production_configs=(
    # NNODES  NUM_MB  TP  PP  CP  LAYERS  TOKENS  NUM_BATCHES UP_SAMPLE_FACTOR  ELONGATE_FACTOR
    "  32     2       8   2   2   48      131072    1            4                 2 "
    # Repeat for other PP_SIZE and CP_SIZE combinations...
)

configs=(${test_configs[@]}) # ${production_configs[@]})

# Run tests for each configuration
for config in "${configs[@]}"; do
    read -r nodes mb tp pp cp nlayers num_tokens num_batches <<< "$config"
    export NNODES=$nodes
    export NUM_MICROBATCH=$mb
    export TP_SIZE=$tp
    export PP_SIZE=$pp
    export CP_SIZE=$cp
    export NUM_LAYERS=$nlayers
    export NUM_TOKENS=$num_tokens
    export NUM_BATCHES=$num_batches
    export MODE=wlbllm
    
    echo "Running $MODE test with: NNODES=$nodes NUM_MICROBATCH=$mb TP_SIZE=$tp PP_SIZE=$pp CP_SIZE=$cp NUM_LAYERS=$nlayers NUM_TOKENS=$num_tokens NUM_BATCHES=$num_batches"
    
    bash test_wlb_e2e.sh
    
    echo "Finished running $MODE test with: NNODES=$nodes NUM_MICROBATCH=$mb TP_SIZE=$tp PP_SIZE=$pp CP_SIZE=$cp NUM_LAYERS=$nlayers NUM_TOKENS=$num_tokens NUM_BATCHES=$num_batches"

done