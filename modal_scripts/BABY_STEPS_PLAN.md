# Baby Steps Plan: Megatron → DistCA on Modal

## 🎯 Goal

Get DistCA working on Modal by first validating Megatron works, then adding DistCA components.

## 📋 The Plan

### Phase 1: Megatron Basic Setup ⏳ (START HERE)

**Goal**: Get Megatron running on H100 with all DistCA-required dependencies

**Steps**:
1. ✅ Create full stack image with exact versions
2. ⏳ Test installation (all components import correctly)
3. ⏳ Run simple GPT-2 training (10 steps, 1 GPU)
4. ⏳ Run simple GPT-2 training (10 steps, 2 GPUs)

**Files**:
- `megatron_full_stack.py` - Complete setup with DistCA versions
- `megatron_stable.py` - Fallback with stable versions
- `MEGATRON_TRAINING_GUIDE.md` - How to use

**Commands**:
```bash
# Step 1: Test installation only (15-20 min first build)
modal run megatron_full_stack.py --test-only

# Step 2: Train on 1 GPU (5 min)
modal run megatron_full_stack.py --num-gpus 1 --num-steps 10

# Step 3: Train on 2 GPUs (5 min)
modal run megatron_full_stack.py --num-gpus 2 --num-steps 10
```

**Success Criteria**:
- [x] Image builds successfully
- [ ] All imports work (PyTorch, TE, Megatron, Apex, FlashAttn)
- [ ] Can create GPT-2 model
- [ ] 1 GPU training completes 10 steps
- [ ] 2 GPU training completes 10 steps
- [ ] No CUDA errors

**Estimated Time**: 30-40 minutes
**Estimated Cost**: ~$1.00

---

### Phase 2: Add NVSHMEM ⏳ (AFTER Phase 1)

**Goal**: Get NVSHMEM 3.2.5 working on Modal (required for DistCA)

**Steps**:
1. Research NVSHMEM installation on Modal
2. Add NVSHMEM to image build
3. Test NVSHMEM initialization
4. Test GPU-to-GPU communication via NVSHMEM

**Challenges**:
- NVSHMEM requires specific system setup
- May need custom CUDA paths
- Inter-GPU communication needs NVLink

**Commands** (to be created):
```bash
modal run megatron_with_nvshmem.py --test-nvshmem
```

**Success Criteria**:
- [ ] NVSHMEM 3.2.5 installs
- [ ] Can initialize NVSHMEM
- [ ] Can communicate between GPUs
- [ ] No communication errors

**Estimated Time**: 2-4 hours (investigation + implementation)
**Estimated Cost**: ~$2.00

---

### Phase 3: Build DistCA Extensions ⏳ (AFTER Phase 2)

**Goal**: Build DistCA's custom CUDA extensions

**Steps**:
1. Copy DistCA source to Modal image
2. Build CUDA extensions (libas_comm.so)
3. Test extension loading
4. Test basic operations

**Files**:
- DistCA `csrc/` directory
- CMake build files

**Commands** (to be created):
```bash
modal run distca_build.py --build-extensions
```

**Success Criteria**:
- [ ] DistCA extensions compile
- [ ] Can load libas_comm.so
- [ ] NVSHMEM ops work through extensions
- [ ] No runtime errors

**Estimated Time**: 1-2 hours
**Estimated Cost**: ~$1.00

---

### Phase 4: Run DistCA Training ⏳ (AFTER Phase 3)

**Goal**: Run actual DistCA training with attention disaggregation

**Steps**:
1. Integrate DistCA Python code
2. Run simple DistCA test (1 GPU)
3. Run DistCA with 2 GPUs
4. Run DistCA with 4+ GPUs
5. Verify attention disaggregation works

**Commands** (to be created):
```bash
# 1 GPU test
modal run distca_train.py --num-gpus 1 --num-steps 10

# 2 GPU test
modal run distca_train.py --num-gpus 2 --num-steps 10

# 4 GPU test
modal run distca_train.py --num-gpus 4 --num-steps 10
```

**Success Criteria**:
- [ ] DistCA training runs
- [ ] Attention disaggregation works
- [ ] Multi-GPU scaling works
- [ ] Performance is reasonable

**Estimated Time**: 2-3 hours
**Estimated Cost**: ~$5.00

---

## 📊 Overall Timeline

| Phase | Time | Cost | Status |
|-------|------|------|--------|
| 1. Megatron Setup | 30-40 min | $1.00 | ⏳ **IN PROGRESS** |
| 2. NVSHMEM | 2-4 hours | $2.00 | ⏳ Pending |
| 3. DistCA Extensions | 1-2 hours | $1.00 | ⏳ Pending |
| 4. DistCA Training | 2-3 hours | $5.00 | ⏳ Pending |
| **TOTAL** | **6-10 hours** | **~$9.00** | |

**Note**: Times are for active work. Modal image caching makes subsequent runs much faster!

---

## 🚀 Current Status

### What's Done ✅

1. ✅ Modal setup with H100
2. ✅ Basic GPU test working
3. ✅ All scripts updated for H100
4. ✅ Exact version requirements identified
5. ✅ Full stack script created (`megatron_full_stack.py`)
6. ✅ Stable fallback created (`megatron_stable.py`)
7. ✅ Documentation written
8. ✅ FlashAttention prebuilt wheel integrated

### What's Next ⏳

**IMMEDIATE (Next 5 minutes)**:
```bash
# Start the build and test!
modal run megatron_full_stack.py --test-only
```

This will:
1. Build the Docker image (~15-20 min first time)
2. Test all installations
3. Verify everything works

**If successful**, proceed to training:
```bash
# Train on 1 GPU
modal run megatron_full_stack.py --num-gpus 1 --num-steps 10
```

**If there are errors**:
1. Check error messages
2. Try stable version: `modal run megatron_stable.py --test-only`
3. Debug and fix issues

---

## 🔧 Key Optimizations

### 1. Prebuilt FlashAttention Wheel ✅
**Before**: Building from source takes 10-15 minutes
**After**: Downloading wheel takes 30 seconds
**Savings**: ~14 minutes

### 2. Modal Image Caching ✅
**Before**: Rebuild everything each time
**After**: Cached after first build
**Savings**: 15-20 minutes on subsequent runs

### 3. Parallel Builds
**Before**: Sequential builds
**After**: Parallel jobs (MAX_JOBS=8)
**Savings**: ~5 minutes

### 4. Simplified Testing
**Before**: Full training every time
**After**: `--test-only` flag for quick validation
**Savings**: Variable (5+ minutes per test)

---

## 💡 Tips for Success

### 1. Start Small
- ✅ Test installation first
- ✅ Use `--test-only` flag
- ✅ Verify each component

### 2. Use Cached Images
- First build is slow (15-20 min)
- Subsequent builds are fast (cached)
- Don't `--force-build` unless needed

### 3. Monitor Costs
- Check Modal dashboard
- Most tests are < $1
- Image build is free (no GPU)

### 4. Debug Incrementally
- If installation fails, check which component
- Test components individually if needed
- Use stable version as fallback

### 5. Document Issues
- Save error messages
- Note what worked/didn't work
- Update scripts as you learn

---

## 📝 Version Matrix

### Target (DistCA Requirements)
```
Python: 3.12
PyTorch: 2.7.0 (CUDA 12.8)
Transformer Engine: v2.4
Megatron-LM: core_v0.12.1
Apex: latest
FlashAttention: 2.7.4
NVSHMEM: 3.2.5
GPU: H100/H200
```

### Current (megatron_full_stack.py)
```
Python: 3.12 ✅
PyTorch: 2.7.0 (trying cu124 index) ⚠️
Transformer Engine: v2.4 ✅
Megatron-LM: core_v0.12.1 ✅
Apex: latest ✅
FlashAttention: 2.7.4 (prebuilt wheel) ✅
NVSHMEM: Not yet ⏳
GPU: H100 ✅
```

### Fallback (megatron_stable.py)
```
Python: 3.11 ✅
PyTorch: Latest stable ✅
Megatron-LM: core_r0.9.0 ✅
Apex: latest ✅
FlashAttention: PyPI latest ✅
```

---

## 🎯 Decision Tree

```
Start
  ↓
Run: modal run megatron_full_stack.py --test-only
  ↓
  ├─ SUCCESS → Continue to 1 GPU training
  │             ↓
  │            Run: modal run megatron_full_stack.py --num-gpus 1
  │             ↓
  │             ├─ SUCCESS → Continue to 2 GPU training
  │             │             ↓
  │             │            Run: modal run megatron_full_stack.py --num-gpus 2
  │             │             ↓
  │             │             ├─ SUCCESS → Phase 1 COMPLETE! ✅
  │             │             │             → Move to Phase 2 (NVSHMEM)
  │             │             │
  │             │             └─ FAIL → Debug multi-GPU issues
  │             │                        → Check NCCL, check Modal multi-GPU support
  │             │
  │             └─ FAIL → Debug training issues
  │                        → Check model creation, check CUDA memory
  │
  └─ FAIL → Try stable version
            ↓
           Run: modal run megatron_stable.py --test-only
            ↓
            ├─ SUCCESS → Use stable version for now
            │             → Investigate full stack issues separately
            │
            └─ FAIL → Debug basic issues
                       → Check CUDA, check GPU, check Modal setup
```

---

## 🚦 Ready to Start!

**Next Command**:
```bash
cd /Users/mike/Project/GitHub/distca/modal_scripts
modal run megatron_full_stack.py --test-only
```

**What to watch for**:
1. Image build progress (15-20 min)
2. Each installation test (PyTorch, TE, Megatron, Apex, FlashAttn)
3. Model creation and forward pass
4. Final success message

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

**Then**: Celebrate and move to 1 GPU training! 🎉

---

**Let's do this!** 🚀
