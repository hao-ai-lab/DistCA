set -x
export NVSHMEM_IB_ENABLE_IBGDA=true

export CUDA_DIR=/mnt/sharefs/software/DeepEP/cuda-12-6
export NCCL_HOME=/usr
export NCCL_LIB=/usr/lib/x86_64-linux-gnu
export NVSHMEM_DIR=/mnt/weka/home/yonghao.zhuang/opt/nvshmem
export NVSHMEM_PREFIX=/mnt/weka/home/yonghao.zhuang/opt/nvshmem
export OPENMPI_DIR=/mnt/weka/home/yonghao.zhuang/opt/openmpi

export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${CUDA_DIR}/lib64:${OPENMPI_DIR}/lib:${NCCL_LIB}/:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:${OPENMPI_DIR}/bin:${CUDA_DIR}/bin:$PATH"


TS=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)_PST
mkdir -p /mnt/weka/home/yonghao.zhuang/jd/d2/tests/logs/${TS}
srun -N 32 -G 256 \
--output="/mnt/weka/home/yonghao.zhuang/jd/d2/tests/logs/${TS}/%N.%j.out" \
--error="/mnt/weka/home/yonghao.zhuang/jd/d2/tests/logs/${TS}/%N.%j.out" \
bash -lc '
	set -x
	hostname
	exec torchrun --nnodes 32 --nproc_per_node 8 --rdzv_backend=c10d --rdzv_endpoint=fs-mbz-gpu-004:29500 --rdzv_id=fs-mbz-gpu-004 \
	test_megatron_e2e_pipeline_with_cp.py  \
	--num-nodes 32 --num-gpus-per-node 8 \
	--pp-size 4 --tp-size 8 \
	--num-microbatch 4 --num-batches 1 --num-tokens 65536 \
	--use-planner --cp-degree 8
'
set +x