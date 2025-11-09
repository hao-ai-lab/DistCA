# Pre-requisites:
# 1. Enable the conda environment "jd-d2"
# 2. Be in the directory of "/mnt/weka/home/yonghao.zhuang/jd/d2/tests" 
# 3. Call this file `bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks3/_251103_correctnes/run.sh`

export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO__DISABLE_CHECK=1    


export TENSOR_DUMP_DIR=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks3/_251103_correctnes/tensors
export TENSOR_DUMP_SUFFIX=d2


# ✅ Pass

# srun -N 1 -G 1 -w fs-mbz-gpu-765 --jobid 1007708 torchrun --nnodes 1 --nproc_per_node 1 test_megatron_layer_correctness.py --world-size 1

# srun -N 1 -G 2 -w fs-mbz-gpu-765 --jobid 1007708 torchrun --nnodes 1 --nproc_per_node 2 test_megatron_layer_correctness.py --world-size 2

# srun -N 1 -G 4 -w fs-mbz-gpu-765 --jobid 1007708 torchrun --nnodes 1 --nproc_per_node 2 test_megatron_layer_correctness.py --world-size 4



# ❌ Fail


# ⚪ Ready
bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks3/_251103_correctness/run_combined.sh