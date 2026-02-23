"""
DistCA CI Smoke Test on Modal.

Runs the full install-and-build pipeline followed by the single-GPU smoke test
on a Modal H100 (or fallback A100). This proves that distca compiles the C++/CUDA
extensions and completes one end-to-end pretrain_llama.py iteration.

Usage (local):
    modal run ci/modal_smoke_test.py

Usage (CI):
    Requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables.
    modal run ci/modal_smoke_test.py
"""

import modal

# ---------------------------------------------------------------------------
# Image: NGC PyTorch base with NVSHMEM pip package + build deps
# ---------------------------------------------------------------------------
distca_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/pytorch:24.12-py3",
        add_python="3.12",
    )
    .pip_install("ninja", "cmake", "nvidia-nvshmem-cu12", "rich", "omegaconf")
    .pip_install("transformers>=4.40,<4.46")
    # flash-attn is built from source inside the smoke test (slow but correct)
)

app = modal.App("distca-ci-smoke", image=distca_image)

# Mount the entire repo into the container
repo_mount = modal.Mount.from_local_dir(".", remote_path="/workspace/DistCA")

# ---------------------------------------------------------------------------
# Single-GPU smoke test function
# ---------------------------------------------------------------------------
@app.function(
    gpu="H100",
    timeout=1800,  # 30 min for build + smoke
    mounts=[repo_mount],
)
def smoke_test():
    """Install distca, build csrc, then run single-GPU smoke test."""
    import subprocess
    import os
    import sys

    os.chdir("/workspace/DistCA")
    os.environ["DISTCA_ROOT"] = "/workspace/DistCA"
    # Reduce NVSHMEM buffer for 1-GPU config
    os.environ["EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB"] = "0.1"
    # Build for Hopper (H100 = sm_90a)
    os.environ["CMAKE_CUDA_ARCHITECTURES"] = "90a"

    print("=" * 60)
    print("  DistCA CI — install + build + single-GPU smoke test")
    print("=" * 60, flush=True)

    # Step 1: Install & build (docker_install_and_build.sh --smoke runs the
    # smoke test at the end, so we just call it with --smoke directly)
    result = subprocess.run(
        ["bash", "scripts/docker_install_and_build.sh", "--smoke"],
        env=os.environ | {"PATH": os.environ.get("PATH", "")},
        cwd="/workspace/DistCA",
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"\n❌ Smoke test FAILED (exit code {result.returncode})")
        sys.exit(result.returncode)

    print("\n✅ DistCA single-GPU smoke test PASSED on Modal")
    return "PASSED"


# ---------------------------------------------------------------------------
# Planner unit tests (CPU-only, runs fast)
# ---------------------------------------------------------------------------
@app.function(
    gpu="H100",  # need torch.cuda for some planner tests
    timeout=600,
    mounts=[repo_mount],
)
def planner_tests():
    """Run planner unit tests (test_planner.py, test_items.py)."""
    import subprocess
    import os
    import sys

    os.chdir("/workspace/DistCA")
    os.environ["DISTCA_ROOT"] = "/workspace/DistCA"

    # Install distca (no csrc needed for planner tests)
    subprocess.run(
        ["pip", "install", "-e", ".", "--quiet"],
        cwd="/workspace/DistCA",
        check=True,
    )
    subprocess.run(
        ["pip", "install", "-r", "requirements.txt", "--quiet"],
        cwd="/workspace/DistCA",
        check=True,
    )
    subprocess.run(
        ["pip", "install", "transformers>=4.40,<4.46", "--quiet"],
        cwd="/workspace/DistCA",
        check=True,
    )

    print("=" * 60)
    print("  DistCA CI — Planner unit tests")
    print("=" * 60, flush=True)

    failed = False
    for test_file in ["tests/test_planner.py", "tests/test_items.py"]:
        print(f"\n--- Running {test_file} ---", flush=True)
        result = subprocess.run(
            ["python", test_file],
            cwd="/workspace/DistCA",
        )
        if result.returncode != 0:
            print(f"❌ {test_file} FAILED")
            failed = True
        else:
            print(f"✅ {test_file} PASSED")

    if failed:
        sys.exit(1)

    print("\n✅ All planner tests PASSED on Modal")
    return "PASSED"


@app.local_entrypoint()
def main():
    """Run all CI checks."""
    print("🚀 Launching DistCA CI on Modal...\n")

    # Run planner tests first (faster, cheaper)
    print("📋 Running planner unit tests...")
    planner_result = planner_tests.remote()
    print(f"   Planner tests: {planner_result}")

    # Run full single-GPU smoke test
    print("\n🔥 Running single-GPU smoke test...")
    smoke_result = smoke_test.remote()
    print(f"   Smoke test: {smoke_result}")

    print("\n" + "=" * 60)
    print("  ✅ All DistCA CI checks PASSED")
    print("=" * 60)
