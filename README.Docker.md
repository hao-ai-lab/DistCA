# Docker single-GPU verification (quick smoke / benchmark)

This is the primary supported execution path for the `Artifacts Evaluated – Functional` badge.

If you use Docker, you can verify the setup with a single GPU without Slurm. Two ways:

## 1. Run once then container exits (one-shot, good for CI / quick check)

Build image if needed, install inside the container, run the test, then the container **stops and is removed** (`--rm`).

- **From the host:**
  - Smoke test (minimal run, ~1 batch):  
    `./scripts/run_docker_single_gpu_smoke.sh`
  - Benchmark (multiple steps, reports tokens/s):  
    `./scripts/run_docker_single_gpu_benchmark.sh`

- **Or with raw `docker run`** (replace `<image>` with your image name, e.g. `distca-pytorch:24.12`):
  - `docker run --gpus all --rm --shm-size=2g -v $(pwd):/workspace/DistCA -e DISTCA_ROOT=/workspace/DistCA <image> /workspace/DistCA/scripts/docker_install_and_build.sh --smoke`
  - Same with `--benchmark` instead of `--smoke` for the throughput run.

## 2. Start container, install, then stay inside (interactive shell)

Build image if needed, install, then **drop into a bash shell**; the container **keeps running** until you type `exit`.

- **From the host:**  
  `./scripts/run_docker_benchmark.sh`  
  This runs the install script and then gives you a shell inside the container.

- **Inside that container** you can run smoke or benchmark (and repeat as needed), then exit when done:
  - Smoke: `bash /workspace/DistCA/scripts/single_gpu_smoke.sh`
  - Benchmark: `bash /workspace/DistCA/scripts/single_gpu_benchmark.sh`
  - Exit the container: `exit`

---

Requires one GPU and Docker with NVIDIA Container Toolkit. The install script pins `transformers` and uses `--shm-size=2g` for NCCL/pretrain.
