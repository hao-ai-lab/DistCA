export CUDA_DIR=$HOME/jd/opt/cuda
export NCCL_HOME=$HOME/jd/opt/nccl
export NCCL_LIB=$HOME/jd/opt/nccl/lib
export NVSHMEM_PREFIX=$HOME/jd/opt/nvshmem
export NVSHMEM_DIR=$HOME/jd/opt/nvshmem

export OPENMPI_DIR=$HOME/jd/opt/openmpi

export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${CUDA_DIR}/lib64:${OPENMPI_DIR}/lib:${NCCL_LIB}/:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:${OPENMPI_DIR}/bin:${CUDA_DIR}/bin:$PATH"

export CUDNN_LIB=/usr/lib/x86_64-linux-gnu
export CUDNN_INCLUDE=/usr/include

export LD_LIBRARY_PATH="${CUDNN_LIB}:$LD_LIBRARY_PATH"
export CPATH="${CUDNN_INCLUDE}:$CPATH"