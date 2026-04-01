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

app = modal.App("distca-ae-llama-32l-tp2-100iter")

@app.function(
    image=image, 
    gpu="H100:2",
    timeout=7200 
)
def run_gpu_llama_tp2_100iter():
    print("=== Starting LLaMA 8B (32-Layer) TP=2 100-Iter Smoke Test ===")
    
    inline_bash = """
    cp /workspace/DistCA/scripts/single_gpu_smoke.sh /workspace/DistCA/scripts/multi_gpu_smoke.sh
    sed -i 's/--nproc_per_node=1/--nproc_per_node=2/g' /workspace/DistCA/scripts/multi_gpu_smoke.sh
    sed -i 's/--num-gpus-per-node 1/--num-gpus-per-node 2/g' /workspace/DistCA/scripts/multi_gpu_smoke.sh
    sed -i 's/--tp-size 1/--tp-size 2/g' /workspace/DistCA/scripts/multi_gpu_smoke.sh
    sed -i 's/--max-sample-id 1/--max-sample-id 100/g' /workspace/DistCA/scripts/multi_gpu_smoke.sh
    sed -i 's/--num-layers 1/--num-layers 32/g' /workspace/DistCA/scripts/multi_gpu_smoke.sh
    
    bash /workspace/DistCA/scripts/multi_gpu_smoke.sh
    """
    
    result = subprocess.run(
        ["bash", "-c", inline_bash],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    if result.returncode == 0:
        print("=== Modal LLaMA 32L TP=2 100-Iter Test Completed Successfully ===")
    else:
        print(f"=== Modal LLaMA Test FAILED with exit code {result.returncode} ===")
        sys.exit(result.returncode)

@app.local_entrypoint()
def main():
    print("Submitting the TP=2 Multi-GPU DistCA test to Modal...")
    run_gpu_llama_tp2_100iter.remote()
