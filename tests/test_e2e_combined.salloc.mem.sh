# Usage:
# bash test_e2e_combined.salloc.mem.sh
# 
export CUDA_LAUNCH_BLOCKING=1
export SLURM_NNODES=8
export SLURM_GPUS_ON_NODE=8
export NNODES=8
export MAX_SAMPLE_ID=2
export TP_SIZE=8 
export PP_SIZE=1
export EXPERIMENT_LOG_MEMORY_USAGE=1
export EXPERIMENT_WARMUP_TIMES=1
export EXPERIMENT_REPEAT_TIMES=1
export SHOULD_ADD_DEBUG_CASES=1
export JOBID=668963
export SLURM_JOB_NODELIST="fs-mbz-gpu-[146,317,296,973,536,538,765,885]"

now_ts=$(TZ='America/Los_Angeles' date +%Y%m%d%H%M%S)_PST

# Define arrays for the parameters to loop over
# NUM_TOKENS_ARRAY=(65536 131072 262144)
# NUM_TOKENS_ARRAY=(65536)
NUM_TOKENS_ARRAY=(131072 262144)
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

        # 8_65536_32_d2_1_4_1
        if [ "$should_profile_memory" -eq 1 ]; then
            export SHOULD_PROFILE_MEMORY=${OUTPUT_DIR_PREFIX}/d2_b1_1/
        else
            export SHOULD_PROFILE_MEMORY=""
        fi
        OUTPUT_DIR_SUFFIX=d2_b1_${should_profile_memory} \
        MODE="d2" CP_SIZE=1 \
        EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1 \
        bash test_e2e_combined.salloc-exp.sh

        # 8_65536_32_d2_1_4_4  
        if [ "$should_profile_memory" -eq 1 ]; then
            export SHOULD_PROFILE_MEMORY=${OUTPUT_DIR_PREFIX}/d2_b4_1/
        else
            export SHOULD_PROFILE_MEMORY=""
        fi
        OUTPUT_DIR_SUFFIX=d2_b4_${should_profile_memory} \
        MODE="d2" CP_SIZE=1 \
        EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=4 \
        bash test_e2e_combined.salloc-exp.sh

        # 8_65536_32_d2_1_4_8
        if [ "$should_profile_memory" -eq 1 ]; then
            export SHOULD_PROFILE_MEMORY=${OUTPUT_DIR_PREFIX}/d2_b8_1/
        else
            export SHOULD_PROFILE_MEMORY=""
        fi
        OUTPUT_DIR_SUFFIX=d2_b8_${should_profile_memory} \
        MODE="d2" CP_SIZE=1 \
        EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=8 \
        bash test_e2e_combined.salloc-exp.sh

        # 8_65536_2_wlbllm_8_4
        if [ "$should_profile_memory" -eq 1 ]; then
            export SHOULD_PROFILE_MEMORY=${OUTPUT_DIR_PREFIX}/wlbllm_cp8_1/
        else
            export SHOULD_PROFILE_MEMORY=""
        fi
        OUTPUT_DIR_SUFFIX=wlbllm_cp8_${should_profile_memory} \
        MODE="wlbllm" CP_SIZE=8 \
        bash test_e2e_combined.salloc-exp.sh

        # 8_65536_2_wlbllm_4_4
        if [ "$should_profile_memory" -eq 1 ]; then
            export SHOULD_PROFILE_MEMORY=${OUTPUT_DIR_PREFIX}/wlbllm_cp4_1/
        else
            export SHOULD_PROFILE_MEMORY=""
        fi
        OUTPUT_DIR_SUFFIX=wlbllm_cp4_${should_profile_memory} \
        MODE="wlbllm" CP_SIZE=4 \
        bash test_e2e_combined.salloc-exp.sh

        # 8_65536_2_wlbllm_2_4
        if [ "$should_profile_memory" -eq 1 ]; then
            export SHOULD_PROFILE_MEMORY=${OUTPUT_DIR_PREFIX}/wlbllm_cp2_1/
        else
            export SHOULD_PROFILE_MEMORY=""
        fi
        OUTPUT_DIR_SUFFIX=wlbllm_cp2_${should_profile_memory} \
        MODE="wlbllm" CP_SIZE=2 \
        bash test_e2e_combined.salloc-exp.sh

        # 8_65536_2_wlbllm_1_4
        if [ "$should_profile_memory" -eq 1 ]; then
            export SHOULD_PROFILE_MEMORY=${OUTPUT_DIR_PREFIX}/wlbllm_cp1_1/
        else
            export SHOULD_PROFILE_MEMORY=""
        fi
        OUTPUT_DIR_SUFFIX=wlbllm_cp1_${should_profile_memory} \
        MODE="wlbllm" CP_SIZE=1 \
        bash test_e2e_combined.salloc-exp.sh
      done
    done
  done
done

# Ring a bell when all experiments are done
echo -e '\a\a\a\a'
