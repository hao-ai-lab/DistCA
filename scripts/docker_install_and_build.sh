#!/usr/bin/env bash
# Run inside NGC PyTorch container with DistCA mounted at /workspace/DistCA.
# Installs distca, Megatron-LM, builds csrc (libas_comm.so). Transformer Engine is preinstalled in the image.

set -e

DISTCA_ROOT="${DISTCA_ROOT:-/workspace/DistCA}"
cd "$DISTCA_ROOT"

# Keep container's torch so Transformer Engine and torchvision stay compatible
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || true)
if [ -n "$TORCH_VERSION" ]; then
  echo "=== Pinning torch==$TORCH_VERSION (container) for installs ==="
  PIP_CONSTRAINT=$(mktemp)
  echo "torch==$TORCH_VERSION" > "$PIP_CONSTRAINT"
  export PIP_CONSTRAINT
  pip_extra="--constraint $PIP_CONSTRAINT"
else
  pip_extra=""
fi

echo "=== Installing distca and requirements ==="
pip install $pip_extra -e .
pip install $pip_extra -r requirements.txt
# Pin transformers to avoid ImportError with NGC torch 2.6.0a0
pip install $pip_extra 'transformers>=4.40,<4.46'

if [ -d Megatron-LM ]; then
  echo "=== Installing Megatron-LM ==="
  pip install $pip_extra -e ./Megatron-LM
fi

# Distca needs transformer_engine.pytorch.attention.dot_product_attention (package structure).
# NGC image may ship an older TE where attention is a single module; repo's TransformerEngine has the right layout.
# Limit parallel jobs to avoid OOM/hang (TE defaults to all cores; set MAX_JOBS=4 if you have limited RAM/WSL2).
export MAX_JOBS="${MAX_JOBS:-4}"
if [ -d TransformerEngine ] && [ -d TransformerEngine/transformer_engine/pytorch/attention/dot_product_attention ]; then
  echo "=== Cleaning TransformerEngine build dir (avoid CMake path mismatch: host vs container) ==="
  rm -rf TransformerEngine/build TransformerEngine/transformer_engine.egg-info
  echo "=== Installing Transformer Engine from repo (MAX_JOBS=$MAX_JOBS); may take 10–30 min ==="
  (cd TransformerEngine && export NVTE_FRAMEWORK=pytorch && export MAX_JOBS && pip install --no-build-isolation -e '.[pytorch]' $pip_extra)
fi

# NGC image flash-attn may lack varlen backward; install full flash-attn for FusedCommAttn.
# With NGC torch (2.6.0a0+...), pip constraint often causes ResolutionImpossible → build from source with --no-deps.
if [ "${INSTALL_FLASH_ATTN:-1}" = "1" ]; then
  echo "=== Installing flash-attn (full build for varlen forward/backward); may take 10–20 min ==="
  if ! pip install $pip_extra 'flash-attn' --no-build-isolation --no-cache-dir; then
    echo "=== pip install flash-attn failed (e.g. ResolutionImpossible with NGC torch); building from source with --no-deps ==="
    FA_SRC="${DISTCA_ROOT}/.build/flash-attention"
    mkdir -p "$(dirname "$FA_SRC")"
    if [ ! -d "$FA_SRC/.git" ]; then
      git clone --depth 1 --branch v2.5.8 https://github.com/Dao-AILab/flash-attention.git "$FA_SRC"
    fi
    (cd "$FA_SRC" && pip install --no-build-isolation --no-deps .) || {
      echo "=== ERROR: flash-attn from-source build failed. See above. Pretrain will fail with varlen backward missing. ==="
      exit 1
    }
  fi
  # Verify varlen backward is available (must match names in fused_comm_attn.py, including _flash_attn_varlen_backward from source build)
  python -c "
import flash_attn.flash_attn_interface as fa
bwd = (getattr(fa, '_wrapped_flash_attn_varlen_backward', None) or getattr(fa, 'flash_attn_varlen_backward', None)
      or getattr(fa, '_flash_attn_varlen_backward', None) or getattr(fa, 'flash_attn_varlen_bwd', None))
assert bwd is not None, 'flash_attn still missing varlen backward (FusedCommAttn will fail)'
print('flash_attn varlen backward OK')
" || { echo "=== ERROR: flash_attn varlen backward still missing. ==="; exit 1; }
fi

# Drop constraint only after all pip installs (including TE, flash-attn) so torch is not upgraded
[ -n "$PIP_CONSTRAINT" ] && rm -f "$PIP_CONSTRAINT"
unset PIP_CONSTRAINT

# NVSHMEM from pip (nvidia-nvshmem-cu12). Handle namespace package (__file__ can be None).
NVSHMEM_PREFIX=$(python -c "
import nvidia.nvshmem
import os
p = getattr(nvidia.nvshmem, '__path__', None)
if p:
    print(os.path.normpath(p[0]))
elif getattr(nvidia.nvshmem, '__file__', None):
    print(os.path.dirname(nvidia.nvshmem.__file__))
else:
    import sys
    for p in sys.path:
        d = os.path.join(p, 'nvidia', 'nvshmem')
        if os.path.isdir(d) and os.path.exists(os.path.join(d, 'lib')):
            print(os.path.normpath(d))
            break
    else:
        raise SystemExit('Could not find NVSHMEM_PREFIX (nvidia.nvshmem)')
")
export NVSHMEM_PREFIX
echo "=== NVSHMEM_PREFIX=$NVSHMEM_PREFIX ==="

# Build csrc (libas_comm.so). Auto-detect GPU arch; override with CMAKE_CUDA_ARCHITECTURES if needed.
if [ -z "${CMAKE_CUDA_ARCHITECTURES:-}" ]; then
  # Auto-detect compute capability from first GPU (e.g. "86" for A100, "90" for H100)
  ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.' 2>/dev/null || true)
  if [ -z "$ARCH" ]; then
    echo "=== WARNING: Could not auto-detect CUDA arch; defaulting to 86 (Ampere). Set CMAKE_CUDA_ARCHITECTURES to override. ==="
    ARCH="86"
  fi
else
  ARCH="$CMAKE_CUDA_ARCHITECTURES"
fi
echo "=== Building csrc (CUDA arch $ARCH) ==="
cd csrc
rm -rf build
cmake -B build -S . -G Ninja \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build
cd ..

echo "=== Verifying distca + libas_comm ==="
python -c "
import distca.runtime.attn_kernels.ops as ops
print('libas_comm:', ops.__file__, 'nvshmem_init:', getattr(ops, 'nvshmem_init', None))
"
echo "=== Verifying Transformer Engine (required: dot_product_attention package) ==="
python -c "
import transformer_engine.pytorch
import transformer_engine.pytorch.attention.dot_product_attention.utils as dpa_utils
print('transformer_engine.pytorch OK, dot_product_attention.utils OK')
"

echo "=== Done. You can run pretrain_llama.py or benchmark. ==="

# Optional: run single-GPU smoke test then exit (for CI / PR verification)
if [ "${1:-}" = "--smoke" ] || [ "${RUN_SMOKE:-0}" = "1" ]; then
  echo "=== Running single-GPU smoke test (scripts/single_gpu_smoke.sh) ==="
  bash "$DISTCA_ROOT/scripts/single_gpu_smoke.sh"
  exit $?
fi

# Optional: run single-GPU benchmark then exit (throughput report)
if [ "${1:-}" = "--benchmark" ] || [ "${RUN_BENCHMARK:-0}" = "1" ]; then
  echo "=== Running single-GPU benchmark (scripts/single_gpu_benchmark.sh) ==="
  bash "$DISTCA_ROOT/scripts/single_gpu_benchmark.sh"
  exit $?
fi

exec bash
