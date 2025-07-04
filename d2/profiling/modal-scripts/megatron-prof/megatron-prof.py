import modal
import os


image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.05-py3")
    # .pip_install("megatron-core[dev]")
    # .pip_install("megatron")
    # .pip_install("flash-attn")
    .run_commands(
        "cd /root && git clone https://github.com/NVIDIA/Megatron-LM.git",
        "cd /root/Megatron-LM && pip install .",
        "apt-get update && apt-get install tree",
    )
    .add_local_dir(
        ".",
        remote_path="/root/megatron-prof",
    )
    
)

app = modal.App("megatron-prof", image=image)

@app.function(
    gpu="L40S:1",
    # cloud="oci",
)
# @modal.experimental.clustered(size=N_NODES, rdma=True)
def megatron_prof():
    
    os.chdir("/root/Megatron-LM/")
    os.system("pwd")
    os.system("tree -L 1")
    
    from torch.distributed.run import parse_args, run

    N_NODES = 1
    N_PROC_PER_NODE = 1
    node_rank = 0 # cluster_info.rank
    main_ip_addr = "localhost"
    main_port = 6001
    
    
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    args = [
        f"--nnodes={N_NODES}",
        f"--nproc-per-node={N_PROC_PER_NODE}",
        f"--node-rank={node_rank}",
        f"--master-addr={main_ip_addr}",
        f"--master-port={main_port}",
        "/root/Megatron-LM/pretrain_gpt.py",
        "--tensor-model-parallel-size", "1",
        "--pipeline-model-parallel-size", "1",
        # GPT args
        "--num-layers", "4",
        "--hidden-size", "768",
        "--num-attention-heads", "12",
        "--seq-length", "256",
        "--max-position-embeddings", "256",
        "--micro-batch-size", "1",
        "--global-batch-size", "1",
        "--lr", "0.0005",
        "--train-iters", "1",
        "--lr-decay-iters", "150000",
        "--lr-decay-style", "cosine",
        "--lr-warmup-iters", "1",
        "--weight-decay", ".1",
        "--adam-beta2", ".999",
        "--fp16",
        "--log-interval", "1",
        "--save-interval", "4",
        "--eval-interval", "4",
        "--eval-iters", "4",
        # vocab args
        "--vocab-file", "/root/megatron-prof/vocab.json",
        "--merge-file", "/root/megatron-prof/merges.txt",
        "--save", "/root/gpt-checkpoint",
        "--load", "/root/gpt-checkpoint",
        "--mock-data",
        "--logging-level", "0",
        # Tensorboard args
        "--tensorboard-dir", "/root/tensorboard-logs/",
    ]
    print(f"Running torchrun with args: {' '.join(args)}")
    run(parse_args(args))
    pass


@app.local_entrypoint()
def main():
    megatron_prof.remote()