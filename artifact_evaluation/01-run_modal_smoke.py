import modal
import sys
import subprocess
from pathlib import Path

# Resolve absolute paths dynamically to support running from any directory
DISTCA_ROOT = Path(__file__).parent.parent.resolve()

# Define the environment image
# We start from your existing Dockerfile, copy the local repository, 
# and run the heavy installation (DistCA + csrc + TE + FlashAttn).
# Note: Since the `run_commands` step is cached by Modal, 
# it will take ~20 mins on the VERY FIRST run only, and subsequent runs will be instant.
image = (
    modal.Image.from_dockerfile(DISTCA_ROOT / "Dockerfile")
    .add_local_dir(DISTCA_ROOT, remote_path="/workspace/DistCA", copy=True)
    .workdir("/workspace/DistCA")
    .run_commands("git clone https://github.com/NVIDIA/TransformerEngine.git && cd TransformerEngine && git checkout v2.4 && git submodule update --init --recursive")
    .run_commands("git clone https://github.com/NVIDIA/Megatron-LM.git && cd Megatron-LM && git checkout core_v0.12.1 && git submodule update --init --recursive")
    # Setting an environment variable so the script knows we are in Modal
    .env({"DISTCA_ROOT": "/workspace/DistCA"})
    # Run the heavy compilation script, but WITHOUT --smoke (so it just installs)
    .run_commands("bash /workspace/DistCA/scripts/docker_install_and_build.sh")
)

app = modal.App("distca-ae-smoke-test")

# Request 1 H100 with a 1-hour timeout. 
# You can change gpu="H100" to gpu="A100" if you want to save credits.
@app.function(
    image=image, 
    gpu="H100",
    timeout=7200 
)
def run_gpu_smoke_test():
    print("=== Starting Single-GPU Smoke Test on Modal ===")
    
    # Execute the single-gpu smoke bash script inside the remote container
    result = subprocess.run(
        ["bash", "/workspace/DistCA/scripts/single_gpu_smoke.sh"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    if result.returncode == 0:
        print("=== Modal Smoke Test Completed Successfully ===")
    else:
        print(f"=== Modal Smoke Test FAILED with exit code {result.returncode} ===")
        sys.exit(result.returncode)

@app.local_entrypoint()
def main():
    print("Submitting the DistCA GPU smoke test to Modal...")
    run_gpu_smoke_test.remote()
