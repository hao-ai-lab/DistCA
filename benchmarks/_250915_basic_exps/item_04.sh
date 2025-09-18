
export MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
export NUM_LAYERS=32
export TP_SIZE=8
export SHOULD_ADD_DEBUG_CASES=1

export EXPERIMENT_LOG_MEMORY_USAGE=1
export SHOULD_PROFILE_MEMORY=1
export OUTPUT_DIR_PREFIX="/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250915_basic_exps/logs.v1-item_04"


for num_tokens in 16384 32768 65536 131072 262144; do
    export NUM_TOKENS=$num_tokens
    export OUTPUT_DIR="${OUTPUT_DIR_PREFIX}/num_tokens_${num_tokens}"
    mkdir -p $OUTPUT_DIR

    torchrun --nnodes=1 --nproc_per_node=8 test_e2e_combined.py --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B --mode wlbllm --replan-iter 0 --batch-size 1 --num-nodes 1 --num-gpus-per-node 8 --num-layers 32 --max-sample-id 1 --tp-size 8 --cp-degree 1 --up-sample-factor 4 --num-tokens $num_tokens --elongate-factor 4 --filter-threshold 65536 --filter-ratio 0.50 --output-dir /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250915_basic_exps/logs.v1-item_04/num_tokens_$num_tokens --should-add-debug-cases --should-profile-memory true
done


