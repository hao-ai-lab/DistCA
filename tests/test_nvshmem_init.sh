# set -x
# LOG_DIR="./logs/$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)"

# srun -N 2 --gres=gpu:8 --ntasks-per-node=1 \
#     --output="${LOG_DIR}/%N.%j.%s.out" \
#     --error="${LOG_DIR}/%N.%j.%s.out" \
#     bash -lc '
#         torchrun --nproc_per_node=8 --nnodes=2 --rdzv_backend=c10d --rdzv_endpoint=fs-mbz-gpu-004:29800 --rdzv_id=0000 --max_restarts=0 \
#         test_nvshmem_init.py
#     '
# set +x

export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 
export NVSHMEM_IB_ENABLE_IBGDA=true

export CUDA_DIR=/mnt/sharefs/software/DeepEP/cuda-12-6
export NCCL_HOME=/usr
export NCCL_LIB=/usr/lib/x86_64-linux-gnu
export NVSHMEM_DIR=/mnt/weka/home/yonghao.zhuang/opt/nvshmem
export NVSHMEM_PREFIX=/mnt/weka/home/yonghao.zhuang/opt/nvshmem
export OPENMPI_DIR=/mnt/weka/home/yonghao.zhuang/opt/openmpi

export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${CUDA_DIR}/lib64:${OPENMPI_DIR}/lib:${NCCL_LIB}/:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:${OPENMPI_DIR}/bin:${CUDA_DIR}/bin:$PATH"



set -x
LOG_DIR="./logs/$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)"

srun -N 32 --gres=gpu:8 --ntasks-per-node=1 \
    --output="${LOG_DIR}/%N.%j.%s.out" \
    --error="${LOG_DIR}/%N.%j.%s.out" \
    bash -lc '
        torchrun --nproc_per_node=8 --nnodes=32 --rdzv_backend=c10d --rdzv_endpoint=fs-mbz-gpu-004:29800 --rdzv_id=1000 --max_restarts=0 \
        test_nvshmem_init.py
    '
set +x



# salloc -N 32 -G 256 --ntasks-per-node=1 --cpus-per-task=96 --mem=1440G --time=12:00:00 --exclusive --partition=lowprio --qos=lowprio --job-name=d2-interact --exclude=fs-mbz-gpu-684,fs-mbz-gpu-697,fs-mbz-gpu-286,fs-mbz-gpu-877,fs-mbz-gpu-757,fs-mbz-gpu-806,fs-mbz-gpu-377,fs-mbz-gpu-906,fs-mbz-gpu-168,fs-mbz-gpu-708,fs-mbz-gpu-868,fs-mbz-gpu-223,fs-mbz-gpu-954,fs-mbz-gpu-707,fs-mbz-gpu-805