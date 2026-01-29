"""
Ultra-minimal test: Just verify Modal GPU access works
"""

import modal

app = modal.App("modal-gpu-test")

@app.function(
    image=modal.Image.debian_slim(python_version="3.10").pip_install("torch==2.1.0"),
    gpu="H100",  # DistCA requires H100 (or H200)
    timeout=300,
)
def test_gpu():
    """Test basic GPU access."""
    import torch

    print("\n" + "=" * 80)
    print("Modal GPU Test")
    print("=" * 80)

    print(f"\n✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ GPU count: {torch.cuda.device_count()}")
        print(f"✅ GPU 0: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        # Quick tensor test
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.mm(x, y)
        print(f"✅ GPU computation works: {z.shape}")

        print("\n" + "=" * 80)
        print("✅ SUCCESS: Modal GPU is working!")
        print("=" * 80 + "\n")

        return {"status": "success", "gpu": torch.cuda.get_device_name(0)}
    else:
        print("\n❌ CUDA not available!")
        return {"status": "error", "message": "No CUDA"}


@app.local_entrypoint()
def main():
    """Run the GPU test."""
    print("\n🚀 Testing Modal GPU access...\n")
    result = test_gpu.remote()
    print(f"\n📊 Result: {result}\n")
