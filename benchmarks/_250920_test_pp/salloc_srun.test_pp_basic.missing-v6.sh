set -e

# export JOBID=${JOBID:-710588}
TS=$(TZ=America/Los_Angeles date +%m%d_%H%M%S)_PST
export OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250920_test_pp/logs.v16-large-scale-pp--128k-256k

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
# export ENABLE_NSYS=1
# export EXPERIMENT_PYTHON_DEBUG_TRACE_CALLS=1
export EXPERIMENT_LOG_MEMORY_USAGE=0
export EXPERIMENT_SKIP_FA2A_OP=0 # (DebugOnly: Ensure not stuck at fast a2a op)
export EXPERIMENT_SKIP_OPTIMIZER_STEP=1
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=4
export SHOULD_ADD_DEBUG_CASES=0

export EXPERIMENT_0TH_SAMPLE_WARMUP_TIMES=1
export EXPERIMENT_WARMUP_TIMES=0
export EXPERIMENT_REPEAT_TIMES=1

# ------------------------
# 
# ------------------------
# export NCCL_CUMEM_ENABLE=0
# export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6"
# export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:512,expandable_segments:True"
# export PYTORCH_CUDA_ALLOC_CONF="verbose_policy:1"

set -e
export EXPERIMENT_FA2A_SPLIT_SENDRECV=1
export EXPERIMENT_SHOULD_LOG_MEMORY_DURING_WARMUP=1
export EXPERIMENT_ADD_SELECTIVE_CKPT=1
# export PYTORCH_CUDA_ALLOC_CONF="pinned_use_background_threads:True"
# export EXPERIMENT_SHOULD_LOG_MEMORY_DURING_WARMUP=1
export MAX_SAMPLE_ID=10
export ENABLE_NSYS=0

model_config="deepseek-ai/DeepSeek-R1-Distill-Llama-8B 64000 32"

param_configs_cases=()
essential_configs_cases=()

configs_8b_128k__essential=(
    # n  bs   mb   tok      M       cp pp tp comment
    # "8   1   8 131072    d2        4 2  8 'some comment' "
    # "8   1   8 131072    wlbllm    4 2  8 'some comment' "

    # "16  2   8 131072    d2        8 2  8 'some comment' "
    # "16  2   8 131072    wlbllm    8 2  8 'some comment' "

    # "32   1   8 131072    d2        8  4  8 'some comment' "
    # "32   1   8 131072    d2       16  2  8 'some comment' "
    # "32   1   8 131072    wlbllm   16  2  8 'some comment' "
    # "32   2   4 131072    wlbllm    8  2  8 'some comment' "
    # "32   8   1 131072    wlbllm   16  1  8 'some comment' "
    # "32   8   1 131072    d2       32  1  8 'some comment' "

    # "32   2   4 131072    d2       16  2  8 'some comment' "
    # "32   2   4 131072    wlbllm   16  2  8 'some comment' "
    # "32   8   1 131072    wlbllm    2  1  8 'some comment' "
    # "32   8   1 131072    wlbllm    4  1  8 'some comment' "
    # "32   8   1 131072    wlbllm    8  1  8 'some comment' "
    # "32   8   1 131072    wlbllm   16  1  8 'some comment' "
    # "32   8   1 131072    wlbllm   32  1  8 'some comment' "
)
param_configs_cases+=("${configs_8b_128k__essential[@]}")

# configs_8b_256k__essential=(
#     #  n   bs  mb   t        mode    cp pp tp comment
#     "8    2   1 262144    d2       4  2  8 'some comment'  "
#     "8    2   1 262144    wlbllm   4  2  8 'some comment'  "
#     "16   2   1 262144    d2       8  2  8 'some comment'  "
#     "16   2   1 262144    wlbllm   8  2  8 'some comment'  "
#     "32   2   1 262144    d2       16 2  8 'some comment'  "
#     "32   2   1 262144    wlbllm   16 2  8 'some comment'  "
# )
# configs_8b_256k__essential_v1=(
#   #  n   bs  mb   t        mode    cp pp tp comment
#     "8    2   1 262144    d2       8  1  8 'some comment'  "
#     "8    2   1 262144    wlbllm   8  1  8 'some comment'  "
#     "16   2   1 262144    d2       16 1  8 'some comment'  "
#     "16   2   1 262144    wlbllm   16 1  8 'some comment'  "
#     "32   2   1 262144    d2       32 1  8 'some comment'  "
#     "32   2   1 262144    wlbllm   32 1  8 'some comment'  "

# )
configs_8b_256k__essential=(
  #  n   bs  mb   t        mode    cp pp tp comment
    # "8    2   1 262144    d2       8  1  8 'some comment'  "
    # "8    2   1 262144    wlbllm   8  1  8 'some comment'  "
    # "16   2   1 262144    d2       16 1  8 'some comment'  "
    # "16   2   1 262144    wlbllm   16 1  8 'some comment'  "
    # "32   4   1 262144    d2       32 1  8 'some comment'  "
    # "32   4   1 262144    wlbllm   32 1  8 'some comment'  "
    # "32   1   4 262144    d2       16 2  8 'some comment'  "
    # "32   1   4 262144    wlbllm   16 2  8 'some comment'  "

)
# param_configs_cases+=("${configs_8b_256k__essential[@]}")

# configs_8b_512k__essential=(
# #    n bs mb t       mode    cp pp tp comment
#     "16 1  2 524288    d2    8   2  8  'd2-512k-r2--ok '  " # 
#     "16 2  1 524288  wlbllm  16  1  8  'd2-512k-r2 '  " # 
#     "16 2  1 524288  wlbllm  8   1  8  'd2-512k-r2 '  " # 
#     # "16 2  1 524288    d2    16  1  8  'd2-512k-r2 '  " # 
#     # "16 1  2 524288  wlbllm   8  2  8  'd2-512k-r2 '  " # 
    
#     "32 1  4 524288    d2    16  2  8  'd2-512k-r2 '  " # 
#     "32 4  1 524288    d2    32  1  8  'd2-512k-r2 '  " # 
#     "32 4  1 524288  wlbllm  32  1  8  'd2-512k-r2 '  " # 
#     "32 1  4 524288  wlbllm  16  2  8  'd2-512k-r2 '  " # 
#     # "32 2  2 524288    d2    16  2  8  'd2-512k-r2 '  " # 

#     "64 1  8 524288    d2    32  2  8  'd2-512k-r2--ok '  " # 
#     "64 8  1 524288    d2    64  1  8  'd2-512k-r2--ok '  " # 
#     "64 1  8 524288  wlbllm  32  2  8  'd2-512k-r2--ok '  " # 
#     "64 8  1 524288  wlbllm  64  1  8  'd2-512k-r2--ok '  " # 
#     # "64 1  8 524288    d2    16  4  8  'd2-512k-r2--ok '  " # 
#     # "64 2  4 524288    d2    32  2  8  'd2-512k-r2 '  " # 
#     # "64 8  1 524288    d2    64  1  8  'd2-512k-r2 '  " # 
# )

export MAX_SAMPLE_ID=3
export ENABLE_NSYS=1
configs_8b_512k__essential=(
#    n bs mb t       mode   cp pp tp comment

# n8
# tbs = 1
#    n bs mb t       mode   cp pp tp comment
    # "8 2  1 524288    d2    8   1  8  'd2-512k-r2--ok '  " # 
    # "8 2  1 524288  wlbllm  8   1  8  'd2-512k-r2 '  " # 
    # "8 1  1 524288  wlbllm  4   1  8  'd2-512k-r2 '  " # 
# n16
# tbs = 2
#    n  bs mb t       mode   cp pp tp comment
    # "16 2  1 524288    d2    16  1  8  'd2-512k-r2--ok '  " # 
    # "16 2  1 524288  wlbllm  8   1  8  'd2-512k-r2 '  " # 
    
    # "16 2  1 524288  wlbllm  4   1  8  'd2-512k-r2 '  " # 

# n32
# tbs = 4
#    n  bs mb t       mode   cp pp tp comment
    "32 1  4 524288    d2    16  2  8  'd2-512k-r2--ok '  " # 
    "32 1  4 524288  wlbllm  16  2  8  'd2-512k-r2--ok '  " # 
    # "32 4  1 524288    d2    32  1  8  'd2-512k-r2--ok '  " # 
    "32 1  8 524288    d2    16  2  8  'd2-512k-r2--ok '  " # 
    "32 1  8 524288  wlbllm  16  2  8  'd2-512k-r2--ok '  " # 
    # "32 4  1 524288  wlbllm  32  1  8  'd2-512k-r2--ok '  " # 


    # "32 0.5  8 524288  wlbllm  16   2  8  'd2-512k-r2 '  " # 
    # "32 4  1 524288  wlbllm  32  1  8  'd2-512k-r2 '  " # 
    # "32 4  1 524288  wlbllm  8   1  8  'd2-512k-r2 '  " # 


# ----------------------------------------------------

# # n8
# # tbs = 2
#     "8 2  1  524288    d2    8   1  8  'd2-512k-r2--ok '  " # 
#     "8 2  1  524288  wlbllm  8   1  8  'd2-512k-r2 '  " # 
#     "8 2  1  524288  wlbllm  4   1  8  'd2-512k-r2 '  " # 
#     "8 0.5 4 524288  wlbllm  4   2  8  'd2-512k-r2 '  " # 

# # n16
# # tbs = 4

    # "16 2  1 524288    d2    16  1  8  'd2-512k-r2 '  " # 
#     "16 1  2 524288  wlbllm   8  2  8  'd2-512k-r2 '  " # 
    
    # "32 1  4 524288    d2    16  2  8  'd2-512k-r2 '  " # 
    # "32 4  1 524288    d2    32  1  8  'd2-512k-r2 '  " # 
#     "32 4  1 524288  wlbllm  32  1  8  'd2-512k-r2 '  " # 
#     "32 1  4 524288  wlbllm  16  2  8  'd2-512k-r2 '  " # 
    # "32 2  2 524288    d2    16  2  8  'd2-512k-r2 '  " # 

    # "64 1  8 524288    d2    32  2  8  'd2-512k-r2--ok '  " # 
    # "64 8  1 524288    d2    64  1  8  'd2-512k-r2--ok '  " # 
    # "64 1  8 524288  wlbllm  32  2  8  'd2-512k-r2--ok '  " # 
    # "64 8  1 524288  wlbllm  64  1  8  'd2-512k-r2--ok '  " # 
    # # "64 1  8 524288    d2    16  4  8  'd2-512k-r2--ok '  " # 
    # "64 2  4 524288    d2    32  2  8  'd2-512k-r2 '  " # 
    # "64 8  1 524288    d2    64  1  8  'd2-512k-r2 '  " # 
)

# configs_8b_512k__essential=(
# "8 1  1 524288    d2    8   1  8  'd2-512k-r2--ok '  " # 
# )
param_configs_cases+=("${configs_8b_512k__essential[@]}")

for config in "${param_configs_cases[@]}"; do
    cases+=("$model_config $config")
done

model_config="codellama/CodeLlama-34b-hf 131072 48"
param_configs_cases=(
# 34B 128k n8 tbs=2
#    n bs mb t      mode  cp pp tp  comment    
    # "8 4 1 131072    d2   8  1  8  'd2 subopt-mem'  " # 
    # "8 1 4 131072    d2   4  2  8  'd2 subopt-mem'  " # 
    # "8 2 2 131072    d2   4  2  8  'd2 subopt-mem'  " # 
    # "8 1 4 131072    d2   2  4  8  'd2 subopt-mem'  " # 
    # "8 1 2 131072    d2   4  2  8  'd2 subopt-mem'  " # 
    # "8 1 1 131072    d2   8  1  8  'd2 subopt-mem'  " # 
    # "8 1 2 131072    d2   4  2  8  'd2 subopt-mem'  " # 

#     "16 1 4 131072    d2   4  4  8  'd2 subopt-mem'  " # 
#     "16 1 4 131072 wlbllm  4  4  8  'wlbllm subopt-mem'  " # 
#     "16 1 4 131072 wlbllm  8  2  8  'wlbllm subopt-mem'  " # 
#     "16 2 2 131072 wlbllm  4  2  8  'wlbllm subopt-mem'  " # 


# 34B 128k n16 tbs=4
#    n bs mb t       mode  cp pp tp  comment
    # "16 8 1 131072    d2   16  1  8  'd2 subopt-mem'  " # 
    # "16 1 8 131072    d2   8   2  8  'd2 subopt-mem'  " # 
    # "16 2 4 131072    d2   8   2  8  'd2 subopt-mem'  " # 
    # "16 4 2 131072    d2   8   2  8  'd2 subopt-mem'  " # 
    # "16 2 4 131072    d2   4   4  8  'd2 subopt-mem'  " # 


    # "16 2 1 131072    d2   16  1  8  'd2 subopt-mem'  " # 
    # "16 2 1 131072    d2   16  1  8  'd2 subopt-mem'  " # 
    # "16 1 2 131072    d2   8   2  8  'd2 subopt-mem'  " # 
#     "16 1 4 131072    d2   4  4  8  'd2 subopt-mem'  " # 
#     "16 1 4 131072 wlbllm  4  4  8  'wlbllm subopt-mem'  " # 
#     "16 1 4 131072 wlbllm  8  2  8  'wlbllm subopt-mem'  " # 
#     "16 2 2 131072 wlbllm  4  2  8  'wlbllm subopt-mem'  " # 

# 34B 128k n32 tbs=8
#    n bs mb t       mode  cp pp tp  comment
    # "32 16 1 131072    d2   32 1  8  'd2 subopt-mem'  " # 
    # "32 1 16 131072    d2   16 2  8  'd2 subopt-mem'  " # 
    # "32 2  8 131072    d2   16 2  8  'd2 subopt-mem'  " # 
    # "32 4  4 131072    d2   16 2  8  'd2 subopt-mem'  " # 
    # "32 1 16 131072    d2   8  4  8  'd2 subopt-mem'  " # 
    # "32 2  8 131072    d2   8  4  8  'd2 subopt-mem'  " # 
    
    # "32 4  4 131072    d2   8  4  8  'd2 subopt-mem'  " # 
    # "32 2  8 131072    d2   8  4  8  'd2 subopt-mem'  " # 


#     "16 1 4 131072    d2   4  4  8  'd2 subopt-mem'  " # 
#     "16 1 4 131072 wlbllm  4  4  8  'wlbllm subopt-mem'  " # 
#     "16 1 4 131072 wlbllm  8  2  8  'wlbllm subopt-mem'  " # 
#     "16 2 2 131072 wlbllm  4  2  8  'wlbllm subopt-mem'  " # 


# # 34B 256k

#    n bs mb t       mode  cp pp tp comment 
    # "16 1 4 262144    d2   8  4  8  'd2 subopt-mem'  " # 
    # "16 1 4 262144    d2  16  2  8  'd2 subopt-mem'  " # 
    # "16 1 4 262144 wlbllm  8  2  8  'wlbllm subopt-mem'  " # 
    # "16 1 4 262144 wlbllm  4  4  8  'wlbllm subopt-mem'  " # 
    # "16 1 4 262144 wlbllm  2  4  8  'wlbllm subopt-mem'  " # 
    # "16 1 4 262144 wlbllm  2  2  8  'wlbllm subopt-mem'  " # 

# 34B 512k
# #    n bs mb t       mode  cp pp tp comment 
#     "32 1 2 524288    d2   8  2  8  'd2 subopt-mem'  " # 
#     "32 1 2 524288 wlbllm  8  2  8  'wlbllm subopt-mem'  " # 
#     "32 1 2 524288 wlbllm  4  2  8  'wlbllm subopt-mem'  " # 

)

# param_configs_cases=(
# # 34B 128k n8 tbs=2
# #    n bs mb t      mode  cp pp tp  comment    
# #     "8 0.5 4 131072 wlbllm  4  2  8  'wlbllm subopt-mem'  " # 
# #     "8   1 2 131072 wlbllm  4  2  8  'wlbllm subopt-mem'  " # 
# #     "8   2 1 131072 wlbllm  8  1  8  'wlbllm subopt-mem'  " # 

# # # 34B 128k n16 tbs=4
# # #    n bs mb t       mode   cp  pp tp  comment
# #     "16 0.5 8 131072 wlbllm  8   2  8  'wlbllm subopt-mem'  " # 
# #     "16 0.5 8 131072 wlbllm  4   4  8  'wlbllm subopt-mem'  " # 
# #     "16   1 4 131072 wlbllm  8   2  8  'wlbllm subopt-mem'  " # 
# #     "16   1 4 131072 wlbllm  4   2  8  'wlbllm subopt-mem'  " # 
# #     "16   4 1 131072 wlbllm 16   1  8  'wlbllm subopt-mem'  " # 
# #     "16   4 1 131072 wlbllm  8  1  8   'wlbllm subopt-mem'  " # 
# #     "16   4 1 131072 wlbllm  4  1  8   'wlbllm subopt-mem'  " # 

# # # 34B 128k n32 tbs=8
# # #    n  bs  mb t       mode   cp  pp tp  comment
# #     "32 0.5 16 131072 wlbllm  16   2  8  'wlbllm subopt-mem'  " # 
# #     "32 0.5 16 131072 wlbllm   8   4  8  'wlbllm subopt-mem'  " # 
# #     "32   1  8 131072 wlbllm  16   2  8  'wlbllm subopt-mem'  " # 
# #     "32   1  8 131072 wlbllm   8   2  8  'wlbllm subopt-mem'  " # 
# #     "32   2  4 131072 wlbllm  16   2  8  'wlbllm subopt-mem'  " # 
# #     "32   2  4 131072 wlbllm   8   2  8  'wlbllm subopt-mem'  " # 
# #     "32   2  4 131072 wlbllm   4   2  8  'wlbllm subopt-mem'  " # 
# #     "32   8  1 131072 wlbllm  32  1  8  'wlbllm subopt-mem'  " # 
# #     "32   8  1 131072 wlbllm  16  1  8  'wlbllm subopt-mem'  " # 
# #     "32   8  1 131072 wlbllm   8  1  8  'wlbllm subopt-mem'  " # 

# )
# for config in "${param_configs_cases[@]}"; do
#     cases+=("$model_config $config")
# done


dists=(
    "prolong 0.3"
    # "wlbllm 0.0"
)



# -------------------------------
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


max_cases=10000
echo "🏁 Start regression sweep. Only running $max_cases cases."
cases_index=0
    

for sample_config in "${dists[@]}"; do
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

    export RATIO=$(echo "8 / ($NNODES / $BATCH_SIZE)" | bc -l)
    

    export ELONGATE_FACTOR=$(($NUM_TOKENS / 65536))
    if [ $ELONGATE_FACTOR -lt 1 ]; then
        ELONGATE_FACTOR=1
    fi
    
    # D2 normal
    echo "🟡 Running config: MODE=$MODE, NNODES=$NNODES, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, NUM_MICROBATCH=$NUM_MICROBATCH, CP_SIZE=$CP_SIZE, PP_SIZE=$PP_SIZE, TP_SIZE=$TP_SIZE, MODE=$MODE, MODEL_PATH=$MODEL_PATH, MODEL_PATH_normalized=$MODEL_PATH_normalized, NUM_LAYERS=$NUM_LAYERS, ATTN_LINEAR_BREAKPOINT=$ATTN_LINEAR_BREAKPOINT, COMMENT=$COMMENT, RATIO=$RATIO"

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

    echo "🟡 Finished running config: MODE=$MODE, NNODES=$NNODES, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, NUM_MICROBATCH=$NUM_MICROBATCH, CP_SIZE=$CP_SIZE, PP_SIZE=$PP_SIZE, TP_SIZE=$TP_SIZE, MODE=$MODE, MODEL_PATH=$MODEL_PATH, MODEL_PATH_normalized=$MODEL_PATH_normalized, NUM_LAYERS=$NUM_LAYERS, ATTN_LINEAR_BREAKPOINT=$ATTN_LINEAR_BREAKPOINT, COMMENT=$COMMENT, RATIO=$RATIO"
    

    cases_index=$((cases_index + 1))
    if [ $cases_index -gt $max_cases ]; then
        break
    fi

done
done


for i in {1..10}; do
    echo '\a'
    sleep 1
done