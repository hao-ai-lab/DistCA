cd ~/jd/d2/tests
conda activate jd-d2

export OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250905_large_scale_llama70b/logs/
export MODEL_PATH=Qwen/Qwen2.5-32B
export MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
export MODEL_PATH=Qwen/Qwen2.5-32B
export MODEL_PATH=Qwen/Qwen2.5-32B
export MODEL_PATH=codellama/CodeLlama-34b-hf

export NUM_LAYERS=4 MODE=d2 BATCH_SIZE=1 NUM_TOKENS=32768 MAX_SAMPLE_ID=3
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2
export NNODES=32
export JOBID=697090
export TP_SIZE=8
export ENABLE_NSYS=0
export EXPERIMENT_LOG_MEMORY_USAGE=1
bash test_e2e_combined.salloc.sh

