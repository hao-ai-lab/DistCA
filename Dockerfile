# DistCA with Transformer Engine: use NGC PyTorch (TE preinstalled since 22.09)
# Ref: README.Installation.md, RUN_STATUS.md, NGC PyTorch 24.12
# 2026: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
# TE: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html

ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:24.12-py3
FROM ${PYTORCH_IMAGE}

# Install build deps and NVSHMEM (pip package for csrc build)
RUN pip install --no-cache-dir ninja cmake \
    && pip install --no-cache-dir nvidia-nvshmem-cu12

# Optional: install flash-attn (can take time; skip if you use a prebuilt wheel later)
# RUN pip install --no-cache-dir wheel && pip install --no-cache-dir flash-attn --no-build-isolation || true

WORKDIR /workspace

# When you run the container, mount DistCA repo and run install script, e.g.:
#   docker run --gpus all -it -v /path/to/DistCA:/workspace/DistCA <image> /workspace/DistCA/scripts/docker_install_and_build.sh
# Then you get a shell with distca + csrc built and can run pretrain_llama.py / benchmark.
