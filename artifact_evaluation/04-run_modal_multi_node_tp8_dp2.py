import modal
import modal.experimental
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

app = modal.App("distca-ae-llama-2node-tp8-dp2")

@app.function(
    image=image, 
    gpu="H100:8", # 8 GPUs per node -> 2 nodes = 16 GPUs
    timeout=3600 
)
@modal.experimental.clustered(size=2, rdma=True)
def run_gpu_llama_multi_node():
    # Use Modal's experimental cluster API to get rank and networking info
    cluster_info = modal.experimental.get_cluster_info()
    rank = cluster_info.rank
    world_size = cluster_info.size
    master_ip = cluster_info.container_ips[0]
    
    print(f"=== Starting LLaMA 8B (32-Layer) 2-Node Test (TP=8, DP=2) on rank {rank}/{world_size} ===")
    
    inline_bash = f"""
    cp /workspace/DistCA/scripts/single_gpu_smoke.sh /workspace/DistCA/scripts/multi_node_smoke.sh
    
    # 1. Update PyTorch Distributed (torchrun) arguments
    sed -i 's/--nproc_per_node=1/--nproc_per_node=8/g' /workspace/DistCA/scripts/multi_node_smoke.sh
    sed -i 's/--nnodes=1/--nnodes={world_size}/g' /workspace/DistCA/scripts/multi_node_smoke.sh
    sed -i 's/--rdzv_endpoint=127.0.0.1:29500/--rdzv_endpoint={master_ip}:29500 --node_rank={rank}/g' /workspace/DistCA/scripts/multi_node_smoke.sh
    
    # 2. Update Megatron/DistCA arguments
    sed -i 's/--num-gpus-per-node 1/--num-gpus-per-node 8/g' /workspace/DistCA/scripts/multi_node_smoke.sh
    sed -i 's/--num-nodes 1/--num-nodes {world_size}/g' /workspace/DistCA/scripts/multi_node_smoke.sh
    sed -i 's/--tp-size 1/--tp-size 8/g' /workspace/DistCA/scripts/multi_node_smoke.sh
    sed -i 's/--max-sample-id 1/--max-sample-id 100/g' /workspace/DistCA/scripts/multi_node_smoke.sh
    sed -i 's/--num-layers 1/--num-layers 32/g' /workspace/DistCA/scripts/multi_node_smoke.sh
    sed -i 's/--cp-size 1/--cp-size 1/g' /workspace/DistCA/scripts/multi_node_smoke.sh # DP implicitly becomes WORLD_SIZE/(TP*PP*CP) = 16/(8*1*1) = 2
    
    # Enable Infiniband (RDMA) and NVSHMEM for multi-node
    export NVSHMEM_BOOTSTRAP=mpi
    export NVSHMEM_IB_ENABLE_IBGDA=true
    export NCCL_IB_DISABLE=0
    
    bash /workspace/DistCA/scripts/multi_node_smoke.sh
    """
    
    result = subprocess.run(
        ["bash", "-c", inline_bash],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    if result.returncode == 0:
        print(f"=== Modal LLaMA 32L 2-Node Test Completed Successfully on rank {rank} ===")
    else:
        print(f"=== Modal LLaMA Test FAILED with exit code {result.returncode} on rank {rank} ===")
        sys.exit(result.returncode)

@app.local_entrypoint()
def main():
    print("Submitting the TP=8 DP=2 Multi-Node DistCA test to Modal...")
    run_gpu_llama_multi_node.remote()
