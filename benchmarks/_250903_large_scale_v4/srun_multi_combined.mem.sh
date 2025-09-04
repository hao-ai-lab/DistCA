
set -x

export EXPERIMENT_LOG_MEMORY_USAGE=1
export EXPERIMENT_REPEAT_TIMES=1
export EXPERIMENT_WARMUP_TIMES=1

# BATCH_SIZE=32 NUM_LAYERS=32 NUM_TOKENS=65536 ELONGATE_FACTOR=1 MAX_SAMPLE_ID=2 \
# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/srun_one_combined.sh

# 🟢 Pass 128k bs8 sample=2
# EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2 BATCH_SIZE=8 NUM_LAYERS=32 NUM_TOKENS=131072 ELONGATE_FACTOR=2 MAX_SAMPLE_ID=5 \
# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/srun_one_combined.sh
# EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2 BATCH_SIZE=8 NUM_LAYERS=32 NUM_TOKENS=131072 ELONGATE_FACTOR=2 MAX_SAMPLE_ID=5 \
# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/srun_one_combined.sh


# 🟢 Pass 256k bs2 sample=2
# BATCH_SIZE=2 NUM_LAYERS=32 NUM_TOKENS=262144 ELONGATE_FACTOR=4 MAX_SAMPLE_ID=2 \
# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/srun_one_combined.sh

# 🟢 Pass 512k bs1 sample=1
# EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2 BATCH_SIZE=1 NUM_LAYERS=32 NUM_TOKENS=524288 ELONGATE_FACTOR=8 MAX_SAMPLE_ID=2 \
# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/srun_one_combined.sh

# 512k bs1 sample=2 init buffer size 2GB
# EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2 BATCH_SIZE=1 NUM_LAYERS=32 NUM_TOKENS=524288 ELONGATE_FACTOR=8 MAX_SAMPLE_ID=2 \
# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/srun_one_combined.sh


# 🔴 OOM: 1M bs1 sample=1 buffer size 2GB
# EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2 BATCH_SIZE=1 NUM_LAYERS=32 NUM_TOKENS=1048576 ELONGATE_FACTOR=16 MAX_SAMPLE_ID=1 \
# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/srun_one_combined.sh


# Small test case
# EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2 BATCH_SIZE=1 NUM_LAYERS=4 NUM_TOKENS=8192 ELONGATE_FACTOR=1 MAX_SAMPLE_ID=2 \
# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/srun_one_combined.N1.sh

# ⚠️ Hang 256k bs4 - just want to study the memory issue
# SHOULD_PROFILE_MEMORY=1 BATCH_SIZE=4 NUM_LAYERS=4 NUM_TOKENS=262144 ELONGATE_FACTOR=4 MAX_SAMPLE_ID=2 \
# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/srun_one_combined.sh
SHOULD_PROFILE_MEMORY=1 BATCH_SIZE=4 NUM_LAYERS=32 NUM_TOKENS=262144 ELONGATE_FACTOR=4 MAX_SAMPLE_ID=2 \
bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/srun_one_combined.sh


# EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2 SHOULD_PROFILE_MEMORY=1 BATCH_SIZE=4 NUM_LAYERS=4 NUM_TOKENS=262144 ELONGATE_FACTOR=4 MAX_SAMPLE_ID=2 \
# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/srun_one_combined.sh



# --------------
 
# 🟢 Pass 128k bs8
# BATCH_SIZE=8 NUM_LAYERS=32 NUM_TOKENS=131072 ELONGATE_FACTOR=2 MAX_SAMPLE_ID=2 \
# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/srun_one_combined.sh

# ⚠️ Hang 256k bs4
# SHOULD_PROFILE_MEMORY=1 BATCH_SIZE=4 NUM_LAYERS=32 NUM_TOKENS=262144 ELONGATE_FACTOR=4 MAX_SAMPLE_ID=2 \
# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/srun_one_combined.sh


# 🟢 Pass 256k bs2
# BATCH_SIZE=2 NUM_LAYERS=32 NUM_TOKENS=262144 ELONGATE_FACTOR=4 MAX_SAMPLE_ID=2 \
# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/srun_one_combined.sh

# 🟢 Pass 512k bs1 sample=1
# EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=8 BATCH_SIZE=1 NUM_LAYERS=32 NUM_TOKENS=524288 ELONGATE_FACTOR=8 MAX_SAMPLE_ID=1 \
# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/srun_one_combined.sh

# 🟡 Hang 512k bs1 sample=2
# EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2 BATCH_SIZE=1 NUM_LAYERS=32 NUM_TOKENS=524288 ELONGATE_FACTOR=8 MAX_SAMPLE_ID=2 \
# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/srun_one_combined.sh

# 🔴 OOM 1M bs1 sample=1
# EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2 BATCH_SIZE=1 NUM_LAYERS=32 NUM_TOKENS=1048576 ELONGATE_FACTOR=16 MAX_SAMPLE_ID=1 \
# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/srun_one_combined.sh


set +x

echo "Done"
echo "\a\a\a\a"
