# Pre-requisites:
# 1. Enable the conda environment "jd-d2"
# 2. Be in the directory of "/mnt/weka/home/yonghao.zhuang/jd/d2/tests" 
# 3. Call this file `bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks3/_251104_correctness/run.sh`

export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO__DISABLE_CHECK=1    


export TENSOR_DUMP_DIR=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks3/_251104_correctness/tensors
export TENSOR_DUMP_SUFFIX=d2
# srun -N 1 -G 2 -w fs-mbz-gpu-800 --jobid 1004517 torchrun --nnodes 1 --nproc_per_node 2 test_megatron_layer.py --world-size 2 --dump-debug

srun -N 1 -G 1 -w fs-mbz-gpu-267 --jobid 1004517 torchrun --nnodes 1 --nproc_per_node 1 test_megatron_layer_correctness.py --world-size 1