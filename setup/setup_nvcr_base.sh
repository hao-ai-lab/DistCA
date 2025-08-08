# Instruction to setup the environment inside nvcr container.
# Assume
#     nvcr.io/nvidia/pytorch:25.04-py3

# Remove transformer engine from pip constraint, becuase we're gonna upgrade it
awk '/^transformer_engine/ {print "#" $0; next} 1' /etc/pip/constraint.txt > temp && mv temp /etc/pip/constraint.txt

pip uninstall transformer_engine
cd TransformerEngine
NVTE_FRAMEWORK=pytorch MAX_JOBS=64 NVTE_BUILD_THREADS_PER_JOB=64 pip install --no-build-isolation -v -v -v '.[pytorch]'

cd Megatron-LM
pip install -e .
cd ..

pip install ray
pip install omegaconf
pip install tensordict
pip install transformers


cat << 'EOF' >> ~/.bashrc
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
EOF

source ~/.bashrc