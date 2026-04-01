# DistCA Artifact Evaluation Guide

This repository is being organized for the first two ACM/cTuning artifact badges only:

- `Artifacts Available`
- `Artifacts Evaluated – Functional`

Most of the paper results depend on specialized multi-node hardware and network connectivity, so we do not ask reviewers to reproduce the original large-scale runs during AE.


## Supported Reviewer Workflow

> Use this workflow for the `Functional` badge. 

We provided Modal scripts that dynamically spin up H100 GPUs to execute our evaluation. You only need to run these entrypoints on your local machine using the API keys provided in the reviewer response:

1. **Baseline Smoke Test (1-layer, 1x H100)**
   ```bash
   modal run artifact_evaluation/01-run_modal_smoke.py
   ```
   *Validates repo installation, csrc compilation, and a simple 1-layer forward-backward pass.*

2. **Memory Validation (16-layers, 20 iterations, 1x H100)**
   ```bash
   modal run artifact_evaluation/02-run_modal_llama_8b_20iter.py
   ```
   *Validates stable training loss over multiple iterations and prevents OOM under constraints.*

3. **Multi-GPU Communication (32-layers, TP=2, 100 iterations, 2x H100)**
   ```bash
   modal run artifact_evaluation/03-run_modal_multi_gpu_tp2.py
   ```
   *Validates Native Tensor Parallelism.*

4. **Multi-Node DistCA (32-layers, TP=8, DP=2, 2 Nodes/16x H100)**
   ```bash
   modal run artifact_evaluation/04-run_modal_multi_node_tp8_dp2.py
   ```
   *Validates multi-node execution via Modal RDMA (Infiniband). DistCA intelligently distributes Core Attention loads across the Data Parallel (DP) ranks implicit in this 16-GPU setup (WORLD_SIZE/(TP*CP) = 16/8 = 2).*

> **Notice**: The first run will build the Docker container and compile NVIDIA TransformerEngine/Megatron-LM, requiring 20-30 minutes. Subsequent invocations will use the heavily cached Docker layer and execute instantly.



### Source Code Inspection

Reviewers are welcome to inspect the repository to confirm the presence of the system described in the paper. The required components (planner, simulator, distca runtime, and benchmarks) are all included in the repository.

- We do not expect reviewers to regenerate the original large-scale numbers for the first two badges.
- Figures 3, 4, 6, 9, 10, 11, and 12 require the original multi-node setup (8 to 64 nodes) and are provided as author-only reproduction references.


