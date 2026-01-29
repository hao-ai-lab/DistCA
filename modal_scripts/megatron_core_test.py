"""
Test Megatron-Core on Modal (lighter than full Megatron-LM)

Megatron-Core is the core library without training scripts - much faster to install.
"""

import modal

app = modal.App("megatron-core-test")

# Install just Megatron-Core (faster than full Megatron-LM)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.0",
        "megatron-core",  # Official PyPI package!
        # Skip transformer-engine for now - not required for basic testing
    )
)


@app.function(
    image=image,
    gpu="H100",  # DistCA requires H100 (or H200)
    timeout=600,
)
def test_megatron_core():
    """Test if we can use Megatron-Core."""
    import torch
    print("\n" + "=" * 80)
    print("Testing Megatron-Core on Modal")
    print("=" * 80)

    # Check PyTorch and CUDA
    print(f"\n✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

    # Test Megatron-Core imports
    print("\n" + "-" * 80)
    print("Testing Megatron-Core imports...")
    print("-" * 80)

    try:
        # Core imports
        from megatron.core import parallel_state
        print(f"✅ parallel_state imported")

        from megatron.core.transformer import TransformerConfig
        print(f"✅ TransformerConfig imported")

        from megatron.core.models.gpt import GPTModel
        print(f"✅ GPTModel imported")

        # Create a tiny model config
        config = TransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        print(f"✅ Config created: {config.num_layers} layers, {config.hidden_size} hidden")

        # Initialize model parallel (required before creating model)
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
        print(f"✅ Parallel state initialized")

        # Create a tiny GPT model
        model = GPTModel(
            config=config,
            transformer_layer_spec=None,  # Use default
            vocab_size=1024,
            max_sequence_length=128,
            pre_process=True,
            post_process=True,
        )
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✅ GPT model created: {num_params:.2f}M parameters")

        # Move to GPU
        model = model.cuda()
        print(f"✅ Model moved to GPU")

        # Test forward pass
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, 1024, (seq_len, batch_size)).cuda()
        position_ids = torch.arange(seq_len).unsqueeze(1).expand(-1, batch_size).cuda()

        with torch.no_grad():
            output = model(input_ids, position_ids)
        print(f"✅ Forward pass works: output shape = {output[0].shape}")

        print("\n" + "=" * 80)
        print("✅ SUCCESS: Megatron-Core is working on Modal!")
        print("=" * 80 + "\n")

        # Clean up parallel state
        parallel_state.destroy_model_parallel()

        return {
            "status": "success",
            "model_params_millions": f"{num_params:.2f}",
            "gpu": torch.cuda.get_device_name(0),
        }

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.local_entrypoint()
def main():
    """Run the Megatron-Core test."""
    print("\n🚀 Testing Megatron-Core on Modal...\n")
    result = test_megatron_core.remote()
    print(f"\n📊 Result: {result}\n")
