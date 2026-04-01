import modal
import sys
import subprocess
from pathlib import Path

# Resolve absolute paths dynamically to support running from any directory
DISTCA_ROOT = Path(__file__).parent.parent.resolve()

image = (
    modal.Image.from_dockerfile(DISTCA_ROOT / "Dockerfile")
    .add_local_dir(DISTCA_ROOT, remote_path="/workspace/DistCA", copy=True)
    .workdir("/workspace/DistCA")
    .run_commands("git clone https://github.com/NVIDIA/TransformerEngine.git && cd TransformerEngine && git checkout v2.4 && git submodule update --init --recursive")
    .run_commands("git clone https://github.com/NVIDIA/Megatron-LM.git && cd Megatron-LM && git checkout core_v0.12.1 && git submodule update --init --recursive")
    .env({"DISTCA_ROOT": "/workspace/DistCA"})
    .run_commands("bash /workspace/DistCA/scripts/docker_install_and_build.sh")
)

app = modal.App("distca-ae-llama-16layer-20iter")

@app.function(
    image=image, 
    gpu="H100",
    timeout=7200 
)
def run_gpu_llama_20iter_test():
    print("=== Starting LLaMA 8B (16-Layer, 20-Iter) Smoke Test on Modal ===")
    
    inline_bash = """
    sed -i 's/--max-sample-id 1/--max-sample-id 20/g' /workspace/DistCA/scripts/single_gpu_smoke.sh
    sed -i 's/--num-layers 1/--num-layers 16/g' /workspace/DistCA/scripts/single_gpu_smoke.sh
    bash /workspace/DistCA/scripts/single_gpu_smoke.sh
    """
    
    result = subprocess.run(
        ["bash", "-c", inline_bash],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    if result.returncode == 0:
        print("=== Modal LLaMA 16L 20-Iter Test Completed Successfully ===")
    else:
        print(f"=== Modal LLaMA Test FAILED with exit code {result.returncode} ===")
        sys.exit(result.returncode)

@app.local_entrypoint()
def main():
    print("Submitting the DistCA LLaMA 16L 20-Iter smoke test to Modal...")
    run_gpu_llama_20iter_test.remote()
