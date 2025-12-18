# Installation

## Overview

We use the following environment in our setup and testing:
- Environment:
    - CUDA 12.8 (PyTorch cu128 wheels)
    - NCCL 2.27.6
    - NVSHMEM 3.2.5
    - OPENMPI 5.0.8
    - PyTorch 2.7.0
- Hardware: NVIDIA H200 GPU
- Interconnect: 
    - Intranode: NVLink
    - Internode: 40GB/s InfiniBand


## Installation

### Step 1: Setup a conda environment:
```bash
# Create a conda environment
conda create -n d2 python=3.12 -y
conda activate d2

# Install pytorch
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

### Step 2: Setup `env.sh` 

We provide a template `env.template.sh` for setting up the environment variables. Please copy it to `env.sh` and customize it according to your environment (e.g. CUDA_DIR, NCCL_HOME, NVSHMEM_PREFIX, OPENMPI_DIR, etc.).
```bash
cp env.template.sh env.sh
source env.sh
```

### Step 3: Install DistCA (`d2`) + dependencies

The commands below assume you are at the **repo root** (i.e. the directory containing `setup.py`, `pretrain_llama.py`, `env.sh`, etc.). 

DistCA has the following external dependencies to install:
- `d2`
- `TransformerEngine`
- `Apex`
- `FlashAttention`
- `Megatron-LM`
- `WLBLLM`

Here, we provide a script to install all dependencies locally. For `TransformerEngine`, `Apex` and `Megatron-LM`, we follow the [Megatron-LM](https://github.com/NVIDIA/Megatron-LM?tab=readme-ov-file#installation) instructions. 


```bash
# (Recommended) make sure env vars are loaded in this shell
source env.sh
```

#### 3.1 Install `d2`

```bash
pip install -e .
pip install -r requirements.txt
```

#### 3.2 Install [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) (required)

```bash
pip install pybind11

git clone https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine
git checkout v2.4
git submodule update --init --recursive

export NVTE_FRAMEWORK=pytorch
export MAX_JOBS=64
export NVTE_BUILD_THREADS_PER_JOB=64

pip install --no-build-isolation -v -v -v '.[pytorch]'
cd ..
```

To check if TransformerEngine is installed correctly, you can run the following command:
```python
import transformer_engine
print(transformer_engine.__version__)
# >>> 2.4.0+3cd6870c
```

If you get error in library loading, review setup in `env.sh` and make sure the libraries (e.g. CUDNN) is properly set.



#### 3.3 Install [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (required)

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
git submodule update --init --recursive
pip install -e .
cd ..
```

To check if Megatron-LM is installed correctly, you can run the following command:
```python
import megatron.core
print(megatron.core.__version__)
# >>> 0.12.1
```

#### 3.4 Install [Apex](https://github.com/NVIDIA/apex) (required)

```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
git submodule update --init --recursive

APEX_CPP_EXT=1 \
APEX_CUDA_EXT=1 \
APEX_FAST_MULTIHEAD_ATTN=1 \
APEX_FUSED_CONV_BIAS_RELU=1 \
pip install -v -v -v --no-build-isolation .
cd ..
```

#### 3.5 Install FlashAttention (required)

Option A (recommended): **prebuilt wheel** (this example is for **CUDA 12.8 + PyTorch 2.7 + Python 3.12**):

```bash
wget https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.18/flash_attn-2.7.4+cu128torch2.7-cp312-cp312-linux_x86_64.whl
pip install ./flash_attn-2.7.4+cu128torch2.7-cp312-cp312-linux_x86_64.whl
```

Option B: build from source (if your CUDA/PyTorch/Python combo does not match a wheel):

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git submodule update --init --recursive
pip install -e .
cd hopper
python setup.py install
cd ..
```

To check if FlashAttention is installed correctly, you can run the following command:
```python
import flash_attn
print(flash_attn.__version__)
# >>> 2.7.4+cu128torch2.7
```

#### 3.6 Build `d2` CUDA extensions (required)

DistCA builds a simple all2all communication library on top of NVSHMEM to support efficient dispatch of attention tasks. 


The following command builds `libas_comm.so` into `d2/runtime/attn_kernels/` (the Python code loads it via `torch.ops.load_library(...)`).

```bash
# If you don't already have them:
pip install ninja cmake

cd csrc
cmake -B build -S ./ -G Ninja -DCMAKE_CUDA_ARCHITECTURES=90a
cmake --build build
cd ..
```

To check if the CUDA extensions are built correctly, you can run the following command:
```python
import d2.runtime.attn_kernels.ops as ops
print(ops.__file__)
# >>> /path/to/d2/runtime/attn_kernels/libas_comm.so
print(ops.nvshmem_init)
```

#### 3.7 (Optional) Build WLBLLM baseline CUDA extension

```bash
cd baseline/wlbllm_original/csrc
cmake -B build -S ./ -G Ninja -DCMAKE_CUDA_ARCHITECTURES=90a
cmake --build build
cd ../../..
```

