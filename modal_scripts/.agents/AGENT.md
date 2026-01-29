# Agent Guidelines for DistCA Modal Deployment

## 📋 Project Overview

**Goal**: Deploy DistCA (Distributed Core Attention) on Modal with H100 GPUs

**Approach**: Baby steps methodology
1. Phase 1: Get Megatron working on Modal H100
2. Phase 2: Add NVSHMEM support
3. Phase 3: Build DistCA CUDA extensions
4. Phase 4: Run DistCA training

**Current Phase**: Phase 1 - Megatron Setup (IN PROGRESS)

---

## 🎯 Agenda & Priorities

### Phase 1: Megatron Setup ⏳ CURRENT PRIORITY

**Objective**: Get Megatron-LM running on Modal H100 with exact DistCA-compatible versions

**Success Criteria**:
- [x] Image builds successfully
- [ ] All imports work (PyTorch, Transformer Engine, Megatron, Apex, FlashAttention)
- [ ] Can create GPT-2 model
- [ ] 1 GPU training completes 10 steps
- [ ] 2 GPU training completes 10 steps
- [ ] No CUDA errors

**Current Blocker**: Transformer Engine build failing due to cuDNN header path issues

**Target Versions** (from DistCA README.Installation.md):
```
Python: 3.12 ✅
PyTorch: 2.7.0 → 2.6.0 (2.7.0 not released yet) ⚠️
CUDA: 12.4 ✅
Transformer Engine: v2.4 ✅
Megatron-LM: core_v0.12.1 ✅
Apex: latest ✅
FlashAttention: 2.7.4 ⚠️ (building from source)
GPU: H100 ✅
NVSHMEM: 3.2.5 ⏳ (Phase 2)
```

**Priority Actions** (in order):
1. **HIGHEST**: Fix Transformer Engine cuDNN header issue
2. Complete image build successfully
3. Run installation test (`--test-only`)
4. Run 1 GPU training (10 steps)
5. Run 2 GPU training (10 steps)
6. Document working configuration

### Phase 2-4: Future Work

See `BABY_STEPS_PLAN.md` for detailed breakdown.

---

## 🏗️ Preferred Working Style

### 1. Incremental & Methodical

**DO**:
- ✅ Make ONE change at a time
- ✅ Test each change before moving on
- ✅ Document what was tried and why it failed/succeeded
- ✅ Keep build logs for reference
- ✅ Use baby steps - smallest testable unit

**DON'T**:
- ❌ Make multiple changes simultaneously
- ❌ Skip testing intermediate steps
- ❌ Assume things work without verification
- ❌ Delete build logs or progress notes

**Example**:
```
Bad:  Change 5 things, run build, debug failures
Good: Change 1 thing → Test → Document → Next change
```

### 2. Log Everything

**Update `progress.txt` after EVERY attempt**:
- What was tried
- Why it was tried (hypothesis)
- What happened (result)
- What was learned (lesson)
- Next steps

**Format**:
```
ATTEMPT #N: Brief description (STATUS)
Date: YYYY-MM-DD
Script: filename (version)
Build Time: X minutes
Status: ✅ SUCCESS / ❌ FAILED / ⏳ IN PROGRESS

Error: [if failed]
Root Cause: [analysis]
Fix Applied: [code changes]
Lesson Learned: [insight]
```

### 3. Version Control Mindset

**Track versions of everything**:
- Script versions (megatron_full_stack.py v1, v2, v3...)
- Build logs (build_log_v1.txt, build_log_v2.txt...)
- Docker image IDs (im-XsnDdFxwo3X2TH0Z0kSCAo...)

**Keep old versions**:
- Don't overwrite working configurations
- Create new files for experiments
- Use git or numbered versions

### 4. Cost & Time Awareness

**Always estimate before running**:
- Expected build time
- Expected GPU time
- Expected cost

**Track actual costs**:
- Log build duration
- Log GPU hours
- Update cost tracking in progress.txt

**Optimize for cost**:
- Use image caching (Modal's killer feature)
- Use `--test-only` flag for quick validation
- Start with 1 GPU before scaling to multi-GPU
- Don't force-rebuild unless necessary

### 5. Debug Systematically

**When builds fail**:
1. **Find the actual error message** (not just "Failed")
   ```bash
   grep -E "(error:|ERROR|FAILED)" build_log.txt
   ```

2. **Get context** (20 lines before/after error)
   ```bash
   grep -B 20 -A 20 "error:" build_log.txt
   ```

3. **Understand root cause** (not just symptoms)
   - "cudnn.h not found" is symptom
   - "Compiler include paths don't have cuDNN" is root cause

4. **Research if needed**
   - Check official docs
   - Search for similar issues
   - Understand the build system

5. **Form hypothesis**
   - Why did this fail?
   - What would fix it?
   - Are there alternatives?

6. **Test smallest fix first**
   - Try simplest solution
   - Verify it works
   - Then optimize if needed

### 6. Fail Fast, Learn Fast

**Don't wait for long builds to fail**:
- Monitor build progress actively
- Check logs every 2-3 minutes
- Stop builds early if errors appear
- Fix issues before they waste time

**Example**:
```
Bad:  Start 20-min build → Wait → Fail → Restart
Good: Start build → Check at 2min → See error → Stop → Fix → Restart
      (Saves 18 minutes!)
```

### 7. Documentation First

**Before making changes**:
1. Read relevant documentation
2. Understand the system
3. Plan the approach
4. Then code

**Don't**:
- Trial-and-error without understanding
- Blindly copy Stack Overflow solutions
- Guess at configurations

**Do**:
- Read DistCA README.Installation.md for requirements
- Read Transformer Engine build docs
- Read PyTorch cuDNN bundling notes
- Understand what each environment variable does

---

## 🔧 Technical Guidelines

### Modal-Specific

**Image Building**:
```python
# Good: Layer-by-layer with clear separation
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "cmake", "ninja-build")  # System packages
    .pip_install("torch==2.6.0", index_url="...")  # Python packages
    .run_commands("git clone ...", "cd ... && pip install -e .")  # Build from source
    .env({"CUDA_HOME": "/usr/local/cuda"})  # Environment vars
)

# Bad: Everything in one giant run_commands()
.run_commands(
    "apt-get install ... && pip install ... && git clone ... && cd ... && ..."
)
```

**Caching**:
- Modal caches each layer
- Changing early layers invalidates later cache
- Put stable things (apt_install, PyTorch) early
- Put experimental things (custom builds) late

**GPU Selection**:
```python
# DistCA ONLY works on H100 (Hopper architecture)
gpu="H100"  # ✅ Correct

gpu="A100"  # ❌ Wrong - DistCA won't work
```

### Build System Guidelines

**Environment Variables**:
```python
# Set before the build step that needs them
.env({
    "CUDNN_PATH": "/path/to/cudnn",
    "CFLAGS": "-I/path/to/include",
})
.run_commands("pip install package")  # Will use above env vars
```

**Dependencies**:
```python
# Install dependencies BEFORE the package that needs them
.pip_install("wheel", "pybind11", "ninja")  # Dependencies first
.run_commands("pip install package-that-needs-above")  # Package second
```

**Build from Source**:
```python
# Clone → Checkout → Submodules → Build
.run_commands(
    "cd /root && git clone https://github.com/org/repo.git",
    "cd /root/repo && git checkout v1.0.0",  # Specific version
    "cd /root/repo && git submodule update --init --recursive",
    "cd /root/repo && pip install -e .",
)
```

### Debugging Techniques

**For "No such file or directory" errors**:
```bash
# Find where the file actually is
find / -name "cudnn.h" 2>/dev/null

# Check if path is in include paths
echo $CFLAGS
echo $CXXFLAGS
```

**For CMake errors**:
```bash
# Look for CMakeCache.txt to see what CMake found
cat build/CMakeCache.txt | grep CUDNN

# Check CMake error logs
cat build/CMakeFiles/CMakeError.log
```

**For import errors in Python**:
```python
import sys
print(sys.path)  # Where Python looks for packages

import torch
print(torch.__file__)  # Where torch is installed

from nvidia import cudnn
print(cudnn.__path__)  # Where cudnn is installed
```

---

## 📊 Success Metrics

### Phase 1 Success:
- Image builds in < 20 minutes ✅
- Installation test passes all 6 components
- 1 GPU training runs without errors
- 2 GPU training runs with proper tensor parallelism
- Total cost < $1.00
- Reproducible (can rebuild from cached image)

### Documentation Complete:
- progress.txt has all attempts logged
- Working configuration documented
- Known issues documented
- Next steps clear

---

## 🚨 Red Flags & When to Pivot

**Stop and reassess if**:
1. Same error after 3 attempts with different fixes
   → Research more, don't just keep trying random things

2. Build time > 30 minutes
   → Something is wrong, check for infinite loops or huge downloads

3. Cost > $5 for Phase 1
   → Too expensive, find more efficient approach

4. Exact version not available (e.g., PyTorch 2.7.0)
   → Use closest available, document difference, check compatibility

5. Build requires manual intervention
   → Script should be fully automated, redesign approach

**When to use fallback (megatron_stable.py)**:
- After 5 failed attempts with exact versions
- When DistCA-specific versions have blocking issues
- When testing basic Modal functionality
- When learning the system before tackling hard problems

---

## 🎓 Learning & Knowledge Transfer

**Maintain these resources**:
1. `progress.txt` - Running log of all attempts
2. `AGENT.md` - This file, updated with new insights
3. `BABY_STEPS_PLAN.md` - Overall roadmap
4. `MEGATRON_TRAINING_GUIDE.md` - Usage guide
5. Working scripts with comments explaining tricky parts

**When switching contexts** (or handing off to another agent):
- Update progress.txt with current status
- Document current blocker clearly
- List next 3 things to try in order
- Include relevant error messages & log locations

---

## 💡 Quick Reference

**Start a build**:
```bash
modal run megatron_full_stack.py --test-only
```

**Check build progress**:
```bash
tail -f build_log.txt  # If saving to file
# OR check Modal dashboard
```

**Common Modal commands**:
```bash
modal --help
modal app logs megatron-full-stack  # View logs
modal volume ls  # List volumes
modal image list  # List cached images
```

**After successful build**:
```bash
# Phase 1 testing sequence
modal run megatron_full_stack.py --test-only  # Installation test
modal run megatron_full_stack.py --num-gpus 1 --num-steps 10  # 1 GPU
modal run megatron_full_stack.py --num-gpus 2 --num-steps 10  # 2 GPUs
```

---

## 🔄 Workflow Summary

```
1. Identify problem from build logs
2. Research root cause (don't guess!)
3. Form hypothesis about fix
4. Update progress.txt with hypothesis
5. Make ONE targeted change to script
6. Run build (save logs with version number)
7. Monitor actively (check every 2-3 min)
8. Document result in progress.txt
9. If failed: GOTO 1
10. If succeeded: Move to next step
```

**Remember**: Slow is smooth, smooth is fast.
Taking time to understand and document saves time in the long run.

---

## 📞 Current Status

**As of 2026-01-29**:
- Phase: 1 (Megatron Setup)
- Attempt: #3
- Blocker: Transformer Engine - C++ compiler can't find cudnn.h headers
- Next Step: Add CFLAGS/CXXFLAGS to include cuDNN headers
- Build Time So Far: ~25 minutes
- Cost So Far: $0

**Key Insight from Latest Failure**:
PyTorch bundles cuDNN in site-packages, but C++ compilers don't know to
look there. CMake found it (via CUDNN_PATH), but setuptools extension
build doesn't inherit those paths. Need to explicitly add to compiler flags.

---

**Last Updated**: 2026-01-29
**Next Review**: After Attempt #4 completes
