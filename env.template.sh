
# ---------------------------
# Setup paths
# ---------------------------

# -- Setup CUDA
# export CUDA_DIR=$HOME/jd/opt/cuda
export CUDA_DIR=/usr/local/cuda-12.8
export CUDA_HOME=$CUDA_DIR
export PATH="$CUDA_HOME/bin:$PATH"

# -- Setup NCCL
export NCCL_HOME=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nccl/
export NCCL_LIB=$NCCL_HOME/lib
export NCCL_INCLUDE_DIR=$NCCL_HOME/include
export NCCL_LIBRARY_DIR=$NCCL_LIB

# -- Setup NVSHMEM
# Common NVSHMEM installations might be found in:
# - /usr/local/nvshmem        (typical for system-wide install)
# - $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nvshmem/ (conda package path)
# Example: modify as needed based on your installation
export NVSHMEM_PREFIX=/usr/local/nvshmem
export NVSHMEM_DIR=$NVSHMEM_PREFIX
export NVSHMEM_INCLUDE=$NVSHMEM_PREFIX/include

# -- Setup OPENMPI
# Common OPENMPI installations might be found in:
# - /usr/local/openmpi        (typical for system-wide install)
# - $CONDA_PREFIX/lib/python3.12/site-packages/openmpi/ (conda package path)
# Example: modify as needed based on your installation
export OPENMPI_DIR=/usr/local/openmpi
export OPENMPI_INCLUDE=$OPENMPI_DIR/include
export OPENMPI_LIBRARY_DIR=$OPENMPI_DIR/lib

# -- Setup CUDNN
export CUDNN_LIB=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib
export CUDNN_INCLUDE_DIR=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/include
export CUDNN_INCLUDE=$CUDNN_INCLUDE_DIR
export CUDNN_LIBRARY_PATH=$CUDNN_LIB

export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${CUDA_DIR}/lib64:${OPENMPI_DIR}/lib:${NCCL_LIBRARY_DIR}/:$LD_LIBRARY_PATH"
export CPATH="${CUDNN_INCLUDE}:$CPATH"
export PATH="${NVSHMEM_DIR}/bin:${OPENMPI_DIR}/bin:$PATH"
