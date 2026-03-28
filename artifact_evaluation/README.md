# DistCA Artifact Evaluation Guide

This repository is being organized for the first two ACM/cTuning artifact badges only:

- `Artifacts Available`
- `Artifacts Evaluated – Functional`

Most of the paper results depend on specialized multi-node hardware and network connectivity, so we do not ask reviewers to reproduce the original large-scale runs during AE.


## Supported Reviewer Workflow

Use this workflow for the `Functional` badge.

### 1. Single-GPU smoke test

This is the primary supported reviewer run path.

```bash
./scripts/run_docker_single_gpu_smoke.sh
```

Expected result:

- the Docker image builds
- `distca` installs
- the CUDA extension builds
- `pretrain_llama.py` completes a 1-GPU, 1-layer smoke run
- the script ends with `Single-GPU smoke test PASSED`

### 2. Optional post-install correctness checks

These tests are useful after the full DistCA environment is installed and the CUDA extension is built:

```bash
python -m pytest tests/test_planner.py tests/test_items.py
```

Expected result:

- both test files pass inside a fully prepared DistCA environment
- they are not the clean-machine entrypoint for the badge workflow

### 3. Source Code Inspection

Reviewers are welcome to inspect the repository to confirm the presence of the system described in the paper. The required components (planner, simulator, distca runtime, and benchmarks) are all included in the repository.

- We do not expect reviewers to regenerate the original large-scale numbers for the first two badges.
- Figures 3, 4, 6, 9, 10, 11, and 12 require the original multi-node setup (8 to 64 nodes) and are provided as author-only reproduction references.

## Resource

### Simple test

- Linux x86_64
- Docker with NVIDIA Container Toolkit
- 1 recent NVIDIA H100 GPU
- enough disk for the container build and Python dependencies

This tier lets reviewers inspect the exact assets behind the paper claims, and optionally exercise the small figure paths, without rerunning the original cluster jobs.

### (Optional) author path for cluster-only figures

Note: This is completely optional and is not required for the first two badges. It covers all figures other than Figure 5. Depending on the figure, the minimum scale ranges from 8x nodes to 64x nodes.

- 8x (or more) nodes with H200 GPU with infiniband access
- NVIDIA DGX H200-class nodes
- Slurm-style launch environment
- 400 Gb/s InfiniBand between nodes

The scripts for the following figures are included in the repository, but reproducing them requires the original hardware scale:

- Figure 12(a): 16 GPUs
- Figure 12(b): 8 nodes (64 GPUs) and 16 nodes (128 GPUs)
- Figure 11: 8 and 16 nodes
- Figure 10: 8, 16, 32, and 64 nodes
- Figure 9: 8, 16, and 32 nodes
- Figure 6: 8 nodes
- Figure 4(a): 8 nodes
- Figure 4(b): 8 nodes
- Figure 3(a): up to 32 network-interconnected nodes
- Figure 3(b): 16 nodes
- Figure 5: 1 GPU

All multi-node figures were developed for the original experimental platform: NVIDIA DGX H200 nodes with 8 H200 GPUs per node, NVLink/NVSwitch within the node, and 400 Gb/s InfiniBand across nodes.

## Submission Checklists Completed

1. Tagged the release used for submission.
2. Archived the snapshot on Zenodo: https://doi.org/10.5281/zenodo.3802578

## Recommended Submission Positioning

State this explicitly in the paper and repository:

- the artifact targets `Available` and `Functional`
- the supported execution path is the single-GPU Docker smoke test
- the planner/item tests are optional post-install checks
- the other figure reproduction paths are included but require the original MBZUAI-style multi-node setup, ranging from 8 nodes to 64 nodes depending on the figure
- full large-scale reruns are optional author-only paths and are not claimed for the badge path
