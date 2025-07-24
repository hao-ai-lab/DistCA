export CUDA_DIR=/usr/local/cuda
export NCCL_HOME=/usr
export NCCL_LIB=/usr/lib/x86_64-linux-gnu
export NVSHMEM_PREFIX=/workspace/opt/nvshmem-v3.2.5
export NVSHMEM_DIR=/workspace/opt/nvshmem-v3.2.5

export OPENMPI_DIR=/usr/local/openmpi

export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${CUDA_DIR}/lib64:${OPENMPI_DIR}/lib:${NCCL_LIB}/:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:${OPENMPI_DIR}/bin:${CUDA_DIR}/bin:$PATH"

export CUDNN_LIB=/usr/lib/x86_64-linux-gnu
export CUDNN_INCLUDE=/usr/include

export LD_LIBRARY_PATH="${CUDNN_LIB}:$LD_LIBRARY_PATH"
export CPATH="${CUDNN_INCLUDE}:$CPATH"