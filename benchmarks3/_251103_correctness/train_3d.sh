


export JOBID=1030626
export HEAD_NODE_IP=fs-mbz-gpu-631
# export NNODES=4
export WANDB_API_KEY="02575b6c73e438f9885daa7cf691a45939d26a71"
export ENABLE_WANDB=1
export WANDB_PROJECT=d2-wiki-train
export WANDB_RUN_NAME="llama3b.v4"
export ALLOW_ALL_RANKS_LOSS=1


TS=$(TZ=America/Los_Angeles date +%m%d_%H%M%S)_PST
export OUTPUT_DIR_PREFIX="/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks3/_251103_correctness/logs.v1"

export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1
export EXPERIMENT_LOG_MEMORY_USAGE=1
# export EXPERIMENT_LOG_MEMORY_USAGE=0
export EXPERIMENT_REPEAT_TIMES=1
export EXPERIMENT_WARMUP_TIMES=1
export EXPERIMENT_D2_FLASH_ATTN_SKIP_GET_BACKEND=1 # default 1
export SHOULD_ADD_DEBUG_CASES=0
export EXPERIMENT_SKIP_OPTIMIZER_STEP=0
export EXPERIMENT_FA2A_BARRIER=0
export EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0=0
# export EXPERIMENT_FA2A_BARRIER=1
# export EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0=0 # default 0
export D2_SKIP_FLOAT_CONVERSION=1


DRY_RUN=${DRY_RUN:-0}

# export MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# export MODEL_PATH=codellama/CodeLlama-34b-hf
# export MODEL_PATH=codellama/CodeLlama-34b-hf

export EXPERIMENT_D2_BALANCE_PING_PONG=0

export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO__DISABLE_CHECK=1    


# export TENSOR_DUMP_DIR=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks3/_251103_correctness/logs.v1.tensors
# export TENSOR_DUMP_SUFFIX=d2


# ------------------------------------
# Check and skip success runs
# ------------------------------------
# Distribution Name
# CHANGE_LONG_DOC_RATIO  
# ATTN_LINEAR_BREAKPOINT

# Run one d2 + one wlbllm-cpMax to justify the result.
# - Real datasets with tokens: 'bookcorpus', 'wikitext', 'openwebtext', 'c4', "allenai/c4"
# sample_name="wikitext"
export ENABLE_NSYS=1
sample_name="c4"
change_long_doc_ratio="0.0"
model_path="meta-llama/Llama-3.2-3B"
attn_linear_breakpoint="64000"
num_layers="28"
selective_ckpt="1"
resend_qkv="1"
# batch_size="8"
# batch_size="4"
batch_size="4"
num_tokens="131072"
elongate_factor="2"
export NNODES=4
nnodes=$NNODES
export NPROC_PER_NODE=8
export TP_SIZE=8
# 10,353,901,556 theoretical upper bound
# export MAX_SAMPLE_ID=100000000
# export MAX_SAMPLE_ID=1000
export MAX_SAMPLE_ID=100
export MAX_TOTAL_TOKENS=100000000 # 1 million tokens global budget for data loader
# export MAX_TOTAL_TOKENS=50000000 # 50 million tokens global budget for data loader
# export MAX_TOTAL_TOKENS=100000000 # 100 million tokens global budget for data loader
# export MAX_TOTAL_TOKENS=1000000000 # 1 billion tokens global budget for data loader

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
export VAL_EVERY_N_STEPS=${VAL_EVERY_N_STEPS:-1}   # default: disable validation in this benchmark (0), override if needed

tolerance_factor=0.05
export MODE=d2
export MIN_TOLERANCE_FACTOR=$tolerance_factor
export OUTPUT_DIR_SUFFIX_ADDON="-tol${tolerance_factor}"
eid="d2-cp1-n${NNODES}-b${BATCH_SIZE}-t${NUM_TOKENS}-tol${tolerance_factor}"
echo "🟡 Running d2 with TP_SIZE=$TP_SIZE, NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR, MIN_TOLERANCE_FACTOR=$MIN_TOLERANCE_FACTOR"

bash training_3d.sh

echo "🟡 Finished running d2 with NNODES=$NNODES, JOBID=$JOBID, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, ELONGATE_FACTOR=$ELONGATE_FACTOR, MIN_TOLERANCE_FACTOR=$MIN_TOLERANCE_FACTOR. Not guaranteed to be successful."
echo "\a"