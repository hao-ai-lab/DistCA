

conda create -n d2 python=3.12 -y

# Install pytorch
pip install torch torchvision torchaudio

# Install d2
cd d2
pip install -e .
source setup/envvars-login.sh # or set the envvars
cd ..

# Install transformer engine
git clone https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine
git submodule update --init --recursive
NVTE_FRAMEWORK=pytorch MAX_JOBS=64 NVTE_BUILD_THREADS_PER_JOB=64 pip install --no-build-isolation -v -v -v '.[pytorch]'
cd ..

# Install apex
git clone https://github.com/NVIDIA/apex.git
cd apex
git submodule update --init --recursive
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_FAST_MULTIHEAD_ATTN=1 APEX_FUSED_CONV_BIAS_RELU=1 pip install -v -v -v --no-build-isolation .
cd ..

# Install flash-attn
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git submodule update --init --recursive
pip install -e .
cd hopper
python setup.py install
cd ..