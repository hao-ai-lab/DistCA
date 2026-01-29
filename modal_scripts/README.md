# Running Megatron-LM on Modal

This directory contains Modal scripts for running Megatron-LM (and eventually DistCA) on Modal's serverless infrastructure.

## Prerequisites

1. Modal CLI installed: `pip install modal`
2. Modal account configured: `modal setup`
3. Active workspace (you have: `hao-ai-lab`)

## Scripts

### 1. `megatron_minimal.py` - Minimal Import Test ⭐ Start Here!

The simplest test to verify Megatron can run on Modal.

**What it does:**
- Installs Megatron-LM
- Imports Megatron modules
- Creates a tiny TransformerConfig
- Verifies GPU access

**Run it:**
```bash
cd /Users/mike/Project/GitHub/distca/modal_scripts
modal run megatron_minimal.py
```

**Expected output:**
```
✅ PyTorch version: 2.1.0
✅ CUDA available: True
✅ GPU: NVIDIA A100-SXM4-40GB
✅ Megatron imported successfully
✅ SUCCESS: Megatron-LM is working on Modal!
```

**Time:** ~2-3 minutes (image build) + ~30 seconds (execution)

### 2. `megatron_simple.py` - Simple Training Test

A more complete test that runs actual Megatron training with a tiny model.

**What it does:**
- Sets up full Megatron environment
- Creates a tiny GPT model (2 layers, 256 hidden)
- Runs 3 training steps with mock data
- Uses 1x A100 GPU

**Run it:**
```bash
modal run megatron_simple.py
```

**Expected output:**
```
✅ Model initialized successfully!
Model parameters: 0.52M
Step 1/3: Forward pass...
Step 2/3: Forward pass...
Step 3/3: Forward pass...
✅ Megatron-LM is working on Modal!
```

**Time:** ~5-10 minutes (image build with apex) + ~1-2 minutes (execution)

## Quick Start

### Test 1: Minimal Import (Recommended First)

```bash
# Go to the modal_scripts directory
cd /Users/mike/Project/GitHub/distca/modal_scripts

# Run minimal test
modal run megatron_minimal.py
```

This will:
1. Build a Docker image with Megatron-LM (~2-3 min)
2. Spin up an A100 GPU instance
3. Test imports and GPU access
4. Return success/failure

### Test 2: Simple Training

```bash
# After minimal test succeeds, try training
modal run megatron_simple.py
```

This will:
1. Build a complete Megatron environment
2. Initialize a tiny GPT model
3. Run 3 training steps
4. Show model size and performance

## Troubleshooting

### Image Build Fails

If the image build fails, try:
```bash
# Force rebuild
modal run --force-build megatron_minimal.py
```

### GPU Not Available

Check your Modal plan includes GPU access:
```bash
modal profile current
```

### Import Errors

If Megatron imports fail, check the build logs:
```bash
modal run --debug megatron_minimal.py
```

## Configuration

### GPU Type

**IMPORTANT**: DistCA requires H100 or H200 GPUs!

All scripts use H100:

```python
@app.function(
    gpu="H100",  # Required for DistCA
)
```

**Note**: A100 will NOT work with DistCA. The codebase requires H100/H200 architecture.

### Timeout

Default is 10 minutes (minimal) or 1 hour (simple). Adjust:

```python
@app.function(
    timeout=3600,  # 1 hour in seconds
)
```

### Image Caching

Modal caches built images. To force rebuild:
```bash
modal run --force-build script.py
```

## Next Steps

After these tests work:

1. **Multi-GPU Training** - Test Megatron with TP/PP parallelism
2. **Real Datasets** - Load actual training data
3. **DistCA Integration** - Add DistCA's attention disaggregation
4. **Distributed Modal** - Run across multiple Modal containers

## Cost Estimates

Rough Modal pricing (as of 2024):
- **A100 (40GB)**: ~$1.10/hour
- **Image build**: Free (cached after first build)
- **Storage**: ~$0.10/GB/month

**These tests:**
- Minimal: < $0.02 (~1 minute GPU time)
- Simple: < $0.05 (~2-3 minutes GPU time)

## Files

```
modal_scripts/
├── README.md                    # This file
├── megatron_minimal.py          # Minimal import test (START HERE)
├── megatron_simple.py           # Simple training test
└── (future: distca_modal.py)    # Full DistCA on Modal
```

## Current Status

- [x] Modal account configured (`hao-ai-lab` workspace)
- [x] Minimal import test created
- [x] Simple training test created
- [ ] Tests verified (run the scripts!)
- [ ] Multi-GPU support
- [ ] DistCA integration
- [ ] Production-ready deployment

## Example Output

### Successful Run

```bash
$ modal run megatron_minimal.py

🚀 Testing Megatron-LM on Modal...

✅ Created => megatron-minimal-test
✅ Initialized. View run at https://modal.com/apps/...

================================================================================
Testing Megatron-LM on Modal
================================================================================

✅ PyTorch version: 2.1.0
✅ CUDA available: True
✅ CUDA version: 12.1
✅ GPU: NVIDIA A100-SXM4-40GB
✅ GPU memory: 40.00 GB

--------------------------------------------------------------------------------
Testing Megatron imports...
--------------------------------------------------------------------------------
✅ Megatron imported successfully
✅ Megatron Core imported
✅ GPT Model imported
✅ TransformerConfig imported
✅ TransformerConfig created: 2 layers, 256 hidden

================================================================================
✅ SUCCESS: Megatron-LM is working on Modal!
================================================================================

📊 Result: {'status': 'success', 'pytorch_version': '2.1.0', ...}

✅ App completed. View run at https://modal.com/apps/...
```

## Development Tips

### Testing Locally First

Before deploying to Modal, test the logic locally:
```python
# Add to your script
if __name__ == "__main__":
    # Local test without Modal
    test_megatron_import()
```

### Debugging

Use Modal's logs:
```bash
# Follow logs in real-time
modal run --debug megatron_minimal.py

# View past runs
modal app logs megatron-minimal-test
```

### Iterating Quickly

Modal caches image layers. To iterate:
1. Comment out the slow `.run_commands()` during development
2. Use `.pip_install()` for faster Python packages
3. Build complex dependencies once, save as custom image

## Resources

- **Modal Docs**: https://modal.com/docs
- **Megatron-LM**: https://github.com/NVIDIA/Megatron-LM
- **DistCA**: This repository

## Questions?

Common questions:

**Q: Why start with Megatron instead of DistCA directly?**
A: DistCA builds on Megatron. We need to verify Megatron works first before adding DistCA's complexity.

**Q: Can I use multiple GPUs?**
A: Yes! Change `gpu=modal.gpu.A100(count=4)` for multi-GPU. We'll add proper distributed training in next steps.

**Q: How much will this cost?**
A: These tests are very cheap (< $0.05 each). Modal only charges for GPU time actually used.

**Q: What's the difference between minimal and simple?**
A: Minimal just tests imports (fast). Simple runs actual training (more complete test).

## Let's Test!

Ready to run? Start with:

```bash
cd /Users/mike/Project/GitHub/distca/modal_scripts
modal run megatron_minimal.py
```

This will tell us if Megatron can run on Modal, which is the first step toward deploying DistCA! 🚀
