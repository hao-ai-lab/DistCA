# Getting Megatron (and DistCA) Running on Modal - Complete Guide

## 🎯 Summary of Progress

We've successfully taken the first steps toward running DistCA on Modal! Here's what we accomplished:

### ✅ What's Working
1. **Modal Setup Complete**
   - Modal CLI configured with `hao-ai-lab` workspace
   - GPU access verified (switching to H100)
   - PyTorch 2.1.0 + CUDA 12.1 working

2. **Basic GPU Test Successful**
   - File: `test_gpu.py`
   - Result: ✅ PASSED
   - GPU computation confirmed working

⚠️ **IMPORTANT**: All scripts now updated to use H100 (DistCA requirement!)

### ⚠️ Compatibility Issue Found
- **Megatron-Core from PyPI** (v0.15.2) has compatibility issues with PyTorch 2.1.0/Triton 2.1.0
- Error: Type annotation issue in triton JIT decorator
- **Solution**: Need to use exact versions from DistCA's installation guide

## 📋 Key Findings

### Version Requirements (from DistCA repo)

DistCA uses these **specific** versions:

```bash
Python: 3.12
PyTorch: 2.7.0 (CUDA 12.8)
Transformer Engine: v2.4 (from source)
Megatron-LM: core_v0.12.1 (from source, NOT PyPI)
FlashAttention: 2.7.4
Apex: latest from NVIDIA/apex
```

### Why PyPI Megatron-Core Didn't Work

The PyPI `megatron-core` package (v0.15.2) is:
- ✅ Newer (released recently)
- ❌ Incompatible with older PyTorch/Triton versions
- ❌ Not the version DistCA uses

**Lesson**: For DistCA, we must install from source with exact git tags!

## 🚀 Next Steps: Two Approaches

### Approach A: Quick Megatron Test (Recommended First)

Use compatible versions for quick testing:

```python
# File: megatron_compat_test.py
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.1.0", "packaging", "ninja")
    .run_commands(
        # Install Megatron-LM from source with compatible version
        "git clone https://github.com/NVIDIA/Megatron-LM.git /root/Megatron-LM",
        "cd /root/Megatron-LM && git checkout core_r0.6.0",  # Older, stable version
        "cd /root/Megatron-LM && pip install -e .",
    )
)
```

**Expected time**: 5-10 minutes
**Cost**: < $0.10

### Approach B: Full DistCA Stack (Production)

Match DistCA's exact environment:

```python
# File: distca_full_modal.py
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "build-essential", "cmake")

    # Step 1: PyTorch 2.7.0 with CUDA 12.8
    .pip_install("torch==2.7.0", index_url="https://download.pytorch.org/whl/cu128")

    # Step 2: Transformer Engine v2.4 from source
    .run_commands(
        "git clone https://github.com/NVIDIA/TransformerEngine.git /root/TransformerEngine",
        "cd /root/TransformerEngine && git checkout v2.4",
        "cd /root/TransformerEngine && git submodule update --init --recursive",
        "export NVTE_FRAMEWORK=pytorch && cd /root/TransformerEngine && pip install --no-build-isolation '.[pytorch]'",
    )

    # Step 3: Megatron-LM core_v0.12.1
    .run_commands(
        "git clone https://github.com/NVIDIA/Megatron-LM.git /root/Megatron-LM",
        "cd /root/Megatron-LM && git checkout core_v0.12.1",
        "cd /root/Megatron-LM && git submodule update --init --recursive",
        "cd /root/Megatron-LM && pip install -e .",
    )

    # Step 4: FlashAttention 2.7.4
    .run_commands(
        "pip install flash-attn==2.7.4",  # If available on PyPI
    )
)
```

**Expected time**: 20-30 minutes (first build)
**Cost**: < $0.20

## 📁 Files Created

```
modal_scripts/
├── README.md                      # Comprehensive usage guide
├── GETTING_STARTED.md            # This file - Quick start
├── MODAL_SETUP_SUMMARY.md        # Detailed progress tracker
│
├── test_gpu.py                   # ✅ WORKING - Basic GPU test
├── megatron_core_test.py         # ⚠️  FAILED - Version mismatch
│
├── megatron_minimal.py           # ⏳ TODO - Update with correct versions
├── megatron_simple.py            # ⏳ TODO - Simple training test
└── (future) distca_modal.py      # ⏳ TODO - Full DistCA
```

## 🎓 What We Learned

### 1. Modal GPU Access is Easy
- Simple syntax: `gpu="A100-40GB"`
- Fast provisioning (seconds)
- Reliable CUDA support

### 2. Version Compatibility is Critical
- Can't mix PyPI packages with custom versions
- DistCA requires specific git tags
- PyTorch 2.7.0 not available on PyPI yet (needs --index-url)

### 3. Image Building Strategy
- Modal caches layers (fast subsequent builds)
- Build complex deps once, reuse
- Separate base image from experiments

### 4. NVSHMEM is the Big Question
- DistCA requires NVSHMEM 3.2.5 for CUDA extensions
- Need to investigate if Modal supports this
- May need custom base image or alternative approach

## 🔧 Immediate Next Actions

### Option 1: Test Basic Megatron (1 hour)
1. Create `megatron_stable_test.py` with Megatron core_r0.6.0
2. Run: `modal run megatron_stable_test.py`
3. Verify imports and model creation
4. Test simple forward pass

### Option 2: Build Full Stack (3-4 hours)
1. Create `distca_environment_modal.py` with all dependencies
2. Test each component (TransformerEngine, Megatron, FlashAttn)
3. Investigate NVSHMEM support
4. Test DistCA CUDA extensions

### Option 3: Hybrid Approach (Recommended)
1. ✅ Start with Option 1 to validate Modal + Megatron
2. ⏳ Once working, incrementally add DistCA components
3. ⏳ Test each layer before proceeding
4. ⏳ Document issues as we go

## 💡 Quick Commands

### Test What's Working Now
```bash
cd /Users/mike/Project/GitHub/distca/modal_scripts

# This works perfectly:
modal run test_gpu.py
```

### Check Modal Status
```bash
modal profile list          # Show workspaces
modal profile current       # Current config
modal app list             # Running apps
```

### View Past Runs
```bash
# Visit Modal dashboard:
open https://modal.com/apps/hao-ai-lab
```

## 📊 Cost Estimates

| Test | Duration | GPUs | Cost |
|------|----------|------|------|
| GPU test | 1 min | 1x A100 | $0.02 |
| Megatron import | 3 min | 1x A100 | $0.06 |
| Simple training | 10 min | 1x A100 | $0.20 |
| Multi-GPU test | 30 min | 4x A100 | $2.20 |
| Full DistCA | 1 hour | 8x A100 | $8.80 |

**Total spent so far**: ~$0.08

## 🤔 Open Questions

### 1. NVSHMEM on Modal?
**Question**: Can we install and use NVSHMEM 3.2.5 in Modal containers?
**Impact**: Critical for DistCA CUDA extensions
**Next Steps**:
- Check Modal's base image capabilities
- Try installing NVSHMEM in image build
- Contact Modal support if needed

### 2. Multi-Node Communication?
**Question**: Does Modal support the inter-node communication patterns DistCA needs?
**Impact**: Required for scaling beyond single node
**Next Steps**:
- Test Modal's multi-GPU within single container first
- Investigate Modal's networking capabilities
- Check if InfiniBand-like performance is available

### 3. PyTorch 2.7.0 Availability?
**Question**: How to install PyTorch 2.7.0 on Modal?
**Impact**: Required for exact DistCA match
**Next Steps**:
- Check PyTorch nightly builds
- Use custom index URL for cu128 builds
- May need to install from whl file

## 📝 Recommended Path Forward

### Week 1: Basic Validation
- [ ] Day 1: Fix Megatron import test with core_r0.6.0
- [ ] Day 2: Test basic GPT model creation and forward pass
- [ ] Day 3: Test single-GPU training loop
- [ ] Day 4: Document findings and cost

### Week 2: Full Stack
- [ ] Day 1: Build image with PyTorch 2.7.0
- [ ] Day 2: Add Transformer Engine v2.4
- [ ] Day 3: Add Megatron-LM core_v0.12.1
- [ ] Day 4: Add FlashAttention 2.7.4

### Week 3: DistCA Integration
- [ ] Day 1: Investigate NVSHMEM installation
- [ ] Day 2: Build DistCA CUDA extensions
- [ ] Day 3: Test DistCA attention disaggregation
- [ ] Day 4: Multi-GPU DistCA test

### Week 4: Optimization
- [ ] Day 1: Optimize image build time
- [ ] Day 2: Add checkpointing support
- [ ] Day 3: Multi-node testing
- [ ] Day 4: Documentation and cost analysis

## 🎯 Success Criteria

### Phase 1: Megatron on Modal ✅ (Current)
- [x] Modal GPU access working
- [x] PyTorch + CUDA working
- [ ] Megatron imports successful
- [ ] Can create GPT model
- [ ] Can run forward pass

### Phase 2: Training Working
- [ ] Single-GPU training loop
- [ ] Multi-GPU with TP
- [ ] Multi-GPU with PP
- [ ] Dataset loading

### Phase 3: DistCA Running
- [ ] NVSHMEM working
- [ ] DistCA extensions built
- [ ] Attention disaggregation working
- [ ] Multi-GPU DistCA training

### Phase 4: Production Ready
- [ ] Fast image builds (<5 min)
- [ ] Checkpointing working
- [ ] Multi-node support
- [ ] Cost optimized
- [ ] Documentation complete

## 🔗 Useful Resources

- **Modal Docs**: https://modal.com/docs
- **Modal Dashboard**: https://modal.com/apps/hao-ai-lab
- **DistCA Paper**: https://arxiv.org/abs/2510.18121
- **DistCA Installation**: `/Users/mike/Project/GitHub/distca/README.Installation.md`
- **Megatron-LM**: https://github.com/NVIDIA/Megatron-LM
- **Transformer Engine**: https://github.com/NVIDIA/TransformerEngine

## 💬 Summary

**What worked**: Modal GPU access, PyTorch, basic computation
**What didn't**: PyPI megatron-core with older PyTorch
**Why it matters**: DistCA needs exact version matching
**Next step**: Use Megatron-LM from source with compatible versions
**Time investment**: ~30 mins so far, another 2-4 hours for full stack
**Cost so far**: ~$0.08

## 🚦 Ready to Proceed?

### Quick Win (30 minutes)
```bash
# Let's fix the Megatron test and get it working!
# I can create an updated script with compatible versions
```

### Full Journey (This Week)
```bash
# Build the complete DistCA environment step by step
# Test each component as we go
# Document everything for future use
```

Choose your adventure and let me know how you'd like to proceed! 🚀
