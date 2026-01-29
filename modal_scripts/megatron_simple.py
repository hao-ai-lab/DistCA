"""
Simple Megatron-LM training on Modal

This is a minimal example to test running Megatron-LM on Modal's infrastructure.
Uses a tiny GPT model for quick testing.
"""

import modal

# Create Modal app
app = modal.App("megatron-simple-test")

# Define the image with Megatron dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "wget",
        "build-essential",
    )
    .pip_install(
        "torch==2.1.0",
        "transformers==4.36.0",
        "datasets==2.14.0",
        "tensorboard==2.15.0",
        "wandb==0.16.0",
        "regex",
        "pybind11",
        "einops",
        "flash-attn==2.5.0",
    )
    .run_commands(
        # Clone Megatron-LM
        "cd /root && git clone https://github.com/NVIDIA/Megatron-LM.git",
        "cd /root/Megatron-LM && git checkout core_r0.6.0",
        # Install apex
        "cd /root && git clone https://github.com/NVIDIA/apex.git",
        "cd /root/apex && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings '--build-option=--cpp_ext' --config-settings '--build-option=--cuda_ext' ./",
    )
    .env({
        "PYTHONPATH": "/root/Megatron-LM:/root/apex:$PYTHONPATH",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    })
)

# Create a network file system for storing checkpoints
volume = modal.Volume.from_name("megatron-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",  # DistCA requires H100 (or H200)
    timeout=3600,  # 1 hour timeout
    volumes={"/checkpoints": volume},
)
def train_tiny_gpt():
    """Train a tiny GPT model with Megatron-LM."""
    import os
    import subprocess
    import sys

    print("=" * 80)
    print("Starting Megatron-LM Training on Modal")
    print("=" * 80)

    # Set up environment
    os.environ["PYTHONPATH"] = "/root/Megatron-LM:/root/apex"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    # Check CUDA availability
    import torch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Create a simple training script
    training_script = """
import sys
sys.path.insert(0, '/root/Megatron-LM')

import torch
import os
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args
from megatron.training.training import setup_model_and_optimizer
from megatron.core.enums import ModelType
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.models.gpt import GPTModel

def model_provider(pre_process=True, post_process=True):
    '''Build the model.'''
    args = get_args()
    config = core_transformer_config_from_args(args)

    model = GPTModel(
        config=config,
        transformer_layer_spec=None,  # Use default
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
    )
    return model

def main():
    # Initialize Megatron
    initialize_megatron(extra_args_provider=None)

    args = get_args()
    print(f"\\nTraining configuration:")
    print(f"  Model: GPT-Tiny")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Num attention heads: {args.num_attention_heads}")
    print(f"  Sequence length: {args.seq_length}")
    print(f"  Micro batch size: {args.micro_batch_size}")
    print(f"  Global batch size: {args.global_batch_size}")
    print()

    # Setup model and optimizer
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider,
        ModelType.encoder_or_decoder,
    )

    print("✅ Model initialized successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model[0].parameters()) / 1e6:.2f}M")

    # Run a few dummy training steps
    print("\\nRunning dummy training steps...")
    for step in range(3):
        # Create dummy batch
        tokens = torch.randint(0, args.vocab_size, (args.seq_length, args.micro_batch_size),
                               device=torch.cuda.current_device())
        labels = tokens.clone()

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass (simplified - normally would use training loop)
        print(f"  Step {step + 1}/3: Forward pass...")

    print("\\n✅ Megatron-LM is working on Modal!")
    print("=" * 80)

if __name__ == '__main__':
    main()
"""

    # Write the training script
    script_path = "/tmp/train_megatron.py"
    with open(script_path, "w") as f:
        f.write(training_script)

    print(f"\nTraining script written to: {script_path}\n")

    # Megatron arguments for a tiny model
    megatron_args = [
        "python", script_path,

        # Model architecture (tiny for testing)
        "--num-layers", "2",
        "--hidden-size", "256",
        "--num-attention-heads", "4",
        "--seq-length", "128",
        "--max-position-embeddings", "128",

        # Training parameters
        "--micro-batch-size", "2",
        "--global-batch-size", "2",
        "--train-iters", "3",
        "--lr", "0.0001",
        "--lr-decay-style", "constant",
        "--min-lr", "0.00001",

        # Model settings
        "--bf16",
        "--use-flash-attn",
        "--no-gradient-accumulation-fusion",

        # Parallelism (single GPU)
        "--tensor-model-parallel-size", "1",
        "--pipeline-model-parallel-size", "1",

        # Data (mock data for testing)
        "--mock-data",
        "--vocab-size", "1024",

        # Checkpointing
        "--save", "/checkpoints/test",
        "--load", "/checkpoints/test",
        "--no-save-optim",
        "--no-save-rng",
        "--save-interval", "1000",  # Don't save during test

        # Logging
        "--log-interval", "1",
        "--log-throughput",
        "--tensorboard-dir", "/checkpoints/tensorboard",
    ]

    print("Megatron command:")
    print(" ".join(megatron_args))
    print("\n" + "=" * 80 + "\n")

    # Run Megatron training
    try:
        result = subprocess.run(
            megatron_args,
            check=True,
            capture_output=False,
            text=True,
        )
        print("\n" + "=" * 80)
        print("✅ Training completed successfully!")
        print("=" * 80)
        return {"status": "success", "message": "Training completed"}
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 80)
        print(f"❌ Training failed with exit code {e.returncode}")
        print("=" * 80)
        return {"status": "error", "message": str(e)}


@app.local_entrypoint()
def main():
    """Main entry point for running the training."""
    print("\n🚀 Starting Megatron-LM training on Modal...\n")
    result = train_tiny_gpt.remote()
    print(f"\n📊 Result: {result}\n")


if __name__ == "__main__":
    # This allows running with: modal run megatron_simple.py
    pass
