"""
Minimal Megatron-LM test on Modal

This is the simplest possible test - just check if we can install and import Megatron.
"""

import modal

app = modal.App("megatron-minimal-test")

# Minimal image with just Megatron-Core (lighter than full Megatron-LM)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.1.0",
        "packaging",
        "ninja",
        "flash-attn==2.5.0",
        "transformer-engine[pytorch]",
    )
    .run_commands(
        # Clone Megatron-LM (Core version is lighter)
        "cd /root && git clone https://github.com/NVIDIA/Megatron-LM.git",
        "cd /root/Megatron-LM && git checkout core_r0.6.0",
        "cd /root/Megatron-LM && pip install -e .",
    )
)


@app.function(
    image=image,
    gpu="H100",  # DistCA requires H100 (or H200)
    timeout=600,  # 10 minutes
)
def test_megatron_import():
    """Test if we can import Megatron successfully."""
    import sys
    import torch

    print("\n" + "=" * 80)
    print("Testing Megatron-LM on Modal")
    print("=" * 80)

    # Check PyTorch and CUDA
    print(f"\n✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Try importing Megatron
    print("\n" + "-" * 80)
    print("Testing Megatron imports...")
    print("-" * 80)

    try:
        sys.path.insert(0, "/root/Megatron-LM")

        import megatron
        print(f"✅ Megatron imported successfully")

        import megatron.core
        print(f"✅ Megatron Core imported")

        from megatron.core.models.gpt import GPTModel
        print(f"✅ GPT Model imported")

        from megatron.core.transformer import TransformerConfig
        print(f"✅ TransformerConfig imported")

        # Create a tiny model config
        config = TransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        print(f"✅ TransformerConfig created: {config.num_layers} layers, {config.hidden_size} hidden")

        print("\n" + "=" * 80)
        print("✅ SUCCESS: Megatron-LM is working on Modal!")
        print("=" * 80 + "\n")

        return {
            "status": "success",
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.local_entrypoint()
def main():
    """Run the minimal test."""
    print("\n🚀 Testing Megatron-LM on Modal...\n")
    result = test_megatron_import.remote()
    print(f"\n📊 Result: {result}\n")
