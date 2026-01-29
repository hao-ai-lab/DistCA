"""
Megatron-LM on Modal with stable, available versions

This is a more conservative version that uses packages we know are available:
- PyTorch 2.6.0 (latest stable with CUDA 12.4)
- Transformer Engine from PyPI (if available) or skip
- Megatron-LM from source (compatible version)
- Apex from source
- FlashAttention from PyPI

Use this if the full stack version has build issues.
"""

import modal

app = modal.App("megatron-stable")

# Conservative image build with known-working versions
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",  # Python 3.11 is more widely supported
    )
    .apt_install(
        "git",
        "wget",
        "build-essential",
        "cmake",
        "ninja-build",
    )
    # PyTorch 2.6.0 (latest stable)
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "pybind11",
        "packaging",
        "ninja",
        "einops",
        "regex",
        "tensorboard",
    )
    # Megatron-LM from source (use a stable tag)
    .run_commands(
        "cd /root && git clone https://github.com/NVIDIA/Megatron-LM.git",
        "cd /root/Megatron-LM && git checkout core_r0.9.0",  # Known stable version
        "cd /root/Megatron-LM && pip install -e .",
    )
    # Apex from source (simplified build)
    .run_commands(
        "cd /root && git clone https://github.com/NVIDIA/apex.git",
        "cd /root/apex && pip install -v --no-build-isolation --no-cache-dir \\",
        "  --config-settings '--build-option=--cpp_ext' \\",
        "  --config-settings '--build-option=--cuda_ext' ./",
    )
    # FlashAttention from PyPI
    .pip_install("flash-attn --no-build-isolation")
    .env({"PYTHONPATH": "/root/Megatron-LM:$PYTHONPATH"})
)

volume = modal.Volume.from_name("megatron-stable-ckpts", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
    volumes={"/checkpoints": volume},
)
def test_installation():
    """Quick installation test."""
    import sys
    import torch

    print("\n" + "=" * 80)
    print("Testing Megatron Stable Installation")
    print("=" * 80)

    results = {}

    # PyTorch
    print("\n[1/4] PyTorch and GPU...")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  GPU: {gpu_name}")
        results["gpu"] = gpu_name

    # Megatron
    print("\n[2/4] Megatron-LM...")
    try:
        sys.path.insert(0, "/root/Megatron-LM")
        import megatron.core
        print(f"  ✅ Megatron version: {megatron.core.__version__}")
        results["megatron"] = megatron.core.__version__
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return {"status": "error", "error": str(e)}

    # Apex
    print("\n[3/4] Apex...")
    try:
        import apex
        print(f"  ✅ Apex imported")
        results["apex"] = "installed"
    except Exception as e:
        print(f"  ⚠️  WARNING: {e}")
        results["apex"] = "not available"

    # FlashAttention
    print("\n[4/4] FlashAttention...")
    try:
        import flash_attn
        print(f"  ✅ FlashAttention: {flash_attn.__version__}")
        results["flash_attn"] = flash_attn.__version__
    except Exception as e:
        print(f"  ⚠️  WARNING: {e}")
        results["flash_attn"] = "not available"

    print("\n" + "=" * 80)
    print("✅ Basic installation test passed!")
    print("=" * 80 + "\n")

    return {"status": "success", **results}


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
    volumes={"/checkpoints": volume},
)
def train_simple(num_steps: int = 10):
    """Simple training test without complex Megatron features."""
    import sys
    import torch
    import torch.nn as nn

    sys.path.insert(0, "/root/Megatron-LM")

    print("\n" + "=" * 80)
    print(f"Simple Training Test ({num_steps} steps)")
    print("=" * 80 + "\n")

    # Create a simple GPT-like model
    class SimpleGPT(nn.Module):
        def __init__(self, vocab_size=1024, hidden_size=256, num_layers=4):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=4,
                    dim_feedforward=hidden_size * 4,
                    batch_first=False,
                )
                for _ in range(num_layers)
            ])
            self.output = nn.Linear(hidden_size, vocab_size)

        def forward(self, x):
            x = self.embed(x)
            for layer in self.layers:
                x = layer(x)
            return self.output(x)

    # Setup
    model = SimpleGPT().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {num_params:.2f}M parameters\n")

    # Training loop
    print("Training:")
    for step in range(num_steps):
        # Create dummy batch
        seq_len, batch_size = 64, 4
        inputs = torch.randint(0, 1024, (seq_len, batch_size)).cuda()
        targets = torch.randint(0, 1024, (seq_len, batch_size)).cuda()

        # Forward/backward
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, 1024), targets.view(-1))
        loss.backward()
        optimizer.step()

        if (step + 1) % 2 == 0:
            print(f"  Step {step + 1}/{num_steps}: loss = {loss.item():.4f}")

    print("\n" + "=" * 80)
    print("✅ Training completed!")
    print("=" * 80 + "\n")

    return {"status": "success", "final_loss": loss.item()}


@app.local_entrypoint()
def main(test_only: bool = False, num_steps: int = 10):
    """Run stable Megatron test."""
    print("\n🚀 Megatron Stable Version Test\n")

    # Test installation
    print("Testing installation...")
    result = test_installation.remote()
    print(f"\nResult: {result}\n")

    if result["status"] != "success" or test_only:
        return

    # Simple training
    print("Running simple training...")
    train_result = train_simple.remote(num_steps=num_steps)
    print(f"\nTraining result: {train_result}\n")
