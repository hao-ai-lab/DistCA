# Megatron Training on Modal - Step-by-Step Guide

## 🎯 Goal

Set up complete Megatron-LM environment on Modal with DistCA-compatible versions and run simple GPT-2 training.

## 📋 What Gets Installed

Exact versions from DistCA `README.Installation.md`:

```
✅ PyTorch 2.7.0 (CUDA 12.4)
✅ Transformer Engine v2.4
✅ Megatron-LM core_v0.12.1
✅ Apex (latest)
✅ FlashAttention 2.7.4
✅ GPU: H100 80GB
```

**Note**: NVSHMEM will be added in a later step (after basic Megatron works)

## 🚀 Quick Start

### Step 1: Test Installation Only (~15-20 min first build)

```bash
cd /Users/mike/Project/GitHub/distca/modal_scripts

# Test that all components install correctly
modal run megatron_full_stack.py --test-only
```

**What this does**:
1. Builds Docker image with all dependencies (cached after first build!)
2. Tests PyTorch + GPU
3. Tests Transformer Engine import
4. Tests Megatron-LM import
5. Tests Apex import
6. Tests FlashAttention import
7. Creates a tiny model and runs forward pass

**Expected output**:
```
[1/6] Testing PyTorch and GPU... ✅
[2/6] Testing Transformer Engine... ✅
[3/6] Testing Megatron-LM... ✅
[4/6] Testing Apex... ✅
[5/6] Testing FlashAttention... ✅
[6/6] Testing model creation... ✅

✅ ALL TESTS PASSED!
```

**If it fails**: Check error messages. Common issues:
- Build timeout (increase timeout in script)
- Memory issues (try smaller MAX_JOBS)
- Version conflicts (check PyPI availability)

---

### Step 2: Run GPT-2 Training (1 GPU) (~5 min)

```bash
# Train GPT-2 Small for 10 steps on 1 GPU
modal run megatron_full_stack.py --num-gpus 1 --num-steps 10
```

**What this does**:
1. Runs installation test (from cache, fast)
2. Sets up GPT-2 Small (117M params, 12 layers)
3. Runs 10 training steps with mock data
4. Uses 1x H100 GPU

**Expected output**:
```
Training GPT-2 (1 GPU(s), 10 steps)
================================================================================

Training configuration:
  Hidden size: 768
  Num layers: 12
  Num attention heads: 12
  Sequence length: 512
  Micro batch size: 2
  Global batch size: 2

✅ Model initialized
   Parameters: 117.00M

Running 10 training steps...
================================================================================

Step 1/10 ✅
Step 2/10 ✅
...
Step 10/10 ✅

✅ Training completed successfully!
```

---

### Step 3: Run GPT-2 Training (2 GPUs) (~5 min)

```bash
# Train with tensor parallelism across 2 GPUs
modal run megatron_full_stack.py --num-gpus 2 --num-steps 10
```

**What this does**:
1. Uses Tensor Parallelism (TP=2)
2. Splits model across 2 H100 GPUs
3. Tests multi-GPU communication
4. Verifies Megatron parallelism works

**Expected output**:
```
Training GPT-2 (2 GPU(s), 10 steps)
================================================================================

Tensor Parallel Size: 2
...
✅ Training completed successfully!
```

---

## 📊 Expected Timeline

| Step | First Time | Subsequent Runs | Cost |
|------|-----------|----------------|------|
| Image Build | 15-20 min | 0 min (cached) | $0 |
| Installation Test | 2-3 min | 2-3 min | $0.10 |
| 1 GPU Training (10 steps) | 5 min | 5 min | $0.30 |
| 2 GPU Training (10 steps) | 5 min | 5 min | $0.60 |

**Total first run**: ~30 minutes, ~$1.00
**Subsequent runs**: ~7 minutes, ~$0.40

## 🐛 Troubleshooting

### Issue 1: Image Build Timeout

**Error**: Build takes > 20 minutes
**Solution**:
```python
# In megatron_full_stack.py, reduce parallelism:
.env({
    "MAX_JOBS": "4",  # Reduce from 8
    "NVTE_BUILD_THREADS_PER_JOB": "2",  # Reduce from 4
})
```

### Issue 2: PyTorch 2.7.0 Not Available

**Error**: `Could not find a version that satisfies the requirement torch==2.7.0`
**Solution**: PyTorch 2.7.0 might not be released yet. Use 2.6.0 or latest:
```python
.pip_install(
    "torch",  # Latest version
    index_url="https://download.pytorch.org/whl/cu124",
)
```

### Issue 3: Transformer Engine Build Fails

**Error**: Build errors during TransformerEngine installation
**Solution**:
1. Check CUDA version compatibility
2. Increase timeout
3. Try without `--no-build-isolation`

### Issue 4: Out of Memory During Training

**Error**: CUDA OOM during training
**Solution**: Reduce batch size:
```python
"--micro-batch-size", "1",  # Reduce from 2
```

### Issue 5: Multi-GPU Communication Fails

**Error**: NCCL errors or hanging during 2-GPU training
**Solution**:
1. Check if Modal supports multi-GPU in single container
2. May need to use Modal's distributed features
3. Start with 1 GPU to isolate issue

## 📝 Understanding the Script

### Image Build Layers

The image is built in layers (Modal caches each):

```python
1. Base: NVIDIA CUDA 12.4 + Python 3.12
2. System packages: git, cmake, build tools
3. PyTorch 2.7.0
4. Transformer Engine v2.4 (SLOW: ~5-10 min)
5. Megatron-LM core_v0.12.1 (FAST: ~1 min)
6. Apex (SLOW: ~5-10 min)
7. FlashAttention 2.7.4 (MEDIUM: ~2-3 min)
```

**Total first build**: 15-20 minutes
**Cached**: instant

### Training Configuration

GPT-2 Small configuration:
```python
Layers: 12
Hidden size: 768
Attention heads: 12
Sequence length: 512
Parameters: ~117M
Vocab size: 50257 (GPT-2 tokenizer)
```

### GPU Usage

**1 GPU mode**:
- Tensor Parallel (TP) = 1
- Model fits entirely on one H100
- Simpler, faster for testing

**2 GPU mode**:
- Tensor Parallel (TP) = 2
- Model split across 2 H100s
- Tests multi-GPU communication
- Tests Megatron parallelism

## 🔍 Verification Checklist

After running, verify:

- [x] Image builds successfully
- [x] All imports work (PyTorch, TE, Megatron, Apex, FlashAttn)
- [x] Can create model
- [x] Forward pass works
- [x] 1 GPU training runs
- [x] 2 GPU training runs
- [x] No CUDA errors
- [x] No communication errors

## ⏭️ Next Steps

Once this works:

1. **Add NVSHMEM** (for DistCA)
   - Install NVSHMEM 3.2.5
   - Build DistCA CUDA extensions
   - Test GPU-to-GPU communication

2. **Test Longer Training**
   - Run 100 steps
   - Monitor GPU utilization
   - Check throughput

3. **Add Real Data**
   - Use actual datasets
   - Test data loading
   - Verify training convergence

4. **Scale Up**
   - Try 4 GPUs
   - Test Pipeline Parallelism
   - Test Data Parallelism

5. **Add DistCA**
   - Integrate DistCA code
   - Test attention disaggregation
   - Run DistCA training

## 💡 Tips

1. **Use `--test-only` first**: Always test installation before training
2. **Start with 1 GPU**: Easier to debug than multi-GPU
3. **Monitor logs**: Modal shows real-time output
4. **Check costs**: Use Modal dashboard to track spending
5. **Iterate quickly**: Modal caches images, so rebuilds are fast

## 📞 Common Commands

```bash
# Test installation only
modal run megatron_full_stack.py --test-only

# Train 1 GPU, 10 steps (default)
modal run megatron_full_stack.py

# Train 2 GPUs, 10 steps
modal run megatron_full_stack.py --num-gpus 2

# Train 1 GPU, 100 steps (longer test)
modal run megatron_full_stack.py --num-steps 100

# Force rebuild image
modal run --force-build megatron_full_stack.py --test-only

# View logs
modal app logs megatron-full-stack
```

## 🎯 Success Criteria

You'll know it works when:

1. ✅ Installation test passes all 6 checks
2. ✅ 1 GPU training completes 10 steps
3. ✅ 2 GPU training completes 10 steps
4. ✅ No CUDA errors
5. ✅ No import errors
6. ✅ GPU utilization looks good

Then you're ready for DistCA! 🚀

---

## 📊 Cost Breakdown

**Image Build** (first time only):
- Time: 15-20 minutes
- Cost: $0 (no GPU during build on Modal)

**Installation Test**:
- Time: 2-3 minutes
- GPU: 1x H100
- Cost: ~$0.10

**1 GPU Training (10 steps)**:
- Time: 5 minutes
- GPU: 1x H100
- Cost: ~$0.30

**2 GPU Training (10 steps)**:
- Time: 5 minutes
- GPU: 2x H100
- Cost: ~$0.60

**Total for complete test**: ~$1.00

**Subsequent runs** (image cached): ~$0.40

Very affordable! 💰
