set -x
# Usage:
# bash test_e2e_combined.salloc.mem.sh
# 
# export CUDA_LAUNCH_BLOCKING=1
export MAX_SAMPLE_ID=2
export TP_SIZE=8 
export PP_SIZE=1
export EXPERIMENT_LOG_MEMORY_USAGE=1
export EXPERIMENT_WARMUP_TIMES=1
export EXPERIMENT_REPEAT_TIMES=1
export SHOULD_ADD_DEBUG_CASES=1
export NNODES=32
export SLURM_NNODES=32
export SLURM_GPUS_ON_NODE=8

export JOBID=677009
export SLURM_JOB_NODELIST="fs-mbz-gpu-[004,036,064,138,041,184,124,144,137,143,153,217,209,272,268,294,341,279,311,369,402,444,488,441,481,460,649,646,753,805,743,880]"

now_ts=$(TZ='America/Los_Angeles' date +%Y%m%d%H%M%S)_PST

# Define arrays for the parameters to loop over
NUM_TOKENS_ARRAY=(65536)
# NUM_TOKENS_ARRAY=(65536)
# NUM_TOKENS_ARRAY=(131072 262144)
# BATCH_SIZE_ARRAY=(32 16 8 4 2 1)
# BATCH_SIZE_ARRAY=(1 4 8 16)
BATCH_SIZE_ARRAY=(32)
NUM_LAYERS_ARRAY=(4)
# NUM_LAYERS_ARRAY=(4)

# Loop over all combinations
for num_tokens in "${NUM_TOKENS_ARRAY[@]}"; do
  for batch_size in "${BATCH_SIZE_ARRAY[@]}"; do
    for num_layers in "${NUM_LAYERS_ARRAY[@]}"; do
      for should_profile_memory in 0; do # for should_profile_memory in 0 1; do
        # export SHOULD_PROFILE_MEMORY=$should_profile_memory
        export NUM_TOKENS=$num_tokens
        export BATCH_SIZE=$batch_size 
        export NUM_LAYERS=$num_layers
        export ELONGATE_FACTOR=$(($num_tokens / 65536))


        export OUTPUT_DIR_PREFIX="/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250901_oom_inspection/logs/${now_ts}/mem.n${NNODES}.n${NUM_TOKENS}.b${BATCH_SIZE}.l${NUM_LAYERS}"

        # for cp_size in 32 16 8 4 2 1; do
        #   if [ $cp_size -gt $NNODES ]; then
        #     continue
        #   fi
        #   export PROFILE_MEMORY_PATH=${OUTPUT_DIR_PREFIX}/wlbllm_cp${cp_size}_1/
        #   OUTPUT_DIR_SUFFIX=wlbllm_cp${cp_size}_${should_profile_memory} \
        #   SHOULD_PROFILE_MEMORY=$should_profile_memory MODE="wlbllm" CP_SIZE=${cp_size} \
        #   bash test_e2e_combined.salloc.sh
        # done

        # 8_65536_32_d2_1_4_1
        export PROFILE_MEMORY_PATH=${OUTPUT_DIR_PREFIX}/d2_b1_1/
        OUTPUT_DIR_SUFFIX=d2_b1_${should_profile_memory} \
        SHOULD_PROFILE_MEMORY=$should_profile_memory MODE="d2" CP_SIZE=1 \
        EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1 \
        bash test_e2e_combined.salloc.sh

      done
    done
  done
done

# Ring a bell when all experiments are done
echo -e '\a\a\a\a'

set +x