export NVSHMEM_IB_ENABLE_IBGDA=true

export CUDA_DIR=/mnt/sharefs/software/DeepEP/cuda-12-6
export NCCL_HOME=/usr
export NCCL_LIB=/usr/lib/x86_64-linux-gnu
export NVSHMEM_DIR=/mnt/weka/home/yonghao.zhuang/opt/nvshmem
export NVSHMEM_PREFIX=/mnt/weka/home/yonghao.zhuang/opt/nvshmem
export OPENMPI_DIR=/mnt/weka/home/yonghao.zhuang/opt/openmpi

export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${CUDA_DIR}/lib64:${OPENMPI_DIR}/lib:${NCCL_LIB}/:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:${OPENMPI_DIR}/bin:${CUDA_DIR}/bin:$PATH"


TS=$(TZ=America/Los_Angeles date +%Y%m%d%H%M%S)_PST
srun -N 32 -G 256 --ntasks-per-node=1 --output=/mnt/weka/home/yonghao.zhuang/jd/d2/tests/logs/${TS}/%N.%j.out --error=/mnt/weka/home/yonghao.zhuang/jd/d2/tests/logs/${TS}/%N.%j.out bash -lc "
        set -x
        exec torchrun  --nnodes=32 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=fs-mbz-gpu-004:29500 --rdzv_id=fs-mbz-gpu-004 --max_restarts=0 test_e2e_combined.py --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B --mode d2 --replan-iter 0 --batch-size 1 --num-nodes 32 --num-gpus-per-node 8 --num-layers 32 --max-sample-id 2 --tp-size 8 --cp-degree 1 --up-sample-factor 4 --num-tokens 65536 --elongate-factor 1 --filter-threshold 65536 --filter-ratio 0.50 --output-dir /mnt/weka/home/yonghao.zhuang/jd/d2/tests/logs/${TS} --should-add-debug-cases
    "

# Successfully Running on the following nodes:
#             677325   lowprio d2-inter yonghao.  R      23:50     
# 32 fs-mbz-gpu-[004,036,041,064,124,137-138,143-144,153,184,209,217,268,272,279,294,311,341,369,402,441,444,460,481,488,646,649,743,753,770,880]