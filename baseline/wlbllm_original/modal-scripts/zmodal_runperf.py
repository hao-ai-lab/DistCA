import modal
import os


# image_with_backward = (
#     modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.01-py3")
# )

image = (
    
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm-flash-attn")
    .pip_install("tqdm")
    .add_local_dir("../wlbllm", "/workspace/WLB-LLM-CP", ignore=[
        "*.pyc", "__pycache__", ".git", "modal-scripts/"
    ])
)

app = modal.App(name="WLB-LLM-CP")


def run_perf(
    world_size,
    seq_len=1024, # unit length
    total_length_k=4, # K
    tp:int=1
):
    import os
    os.chdir("/workspace/WLB-LLM-CP")

    import sys
    sys.path.append("/workspace/WLB-LLM-CP")

    import torch.multiprocessing as mp
    from argparse import Namespace
    from cp_performance_compare import (run, parser)


    num_heads = 64

    avg_doc_len = seq_len / (total_length_k * 1024)
    args = Namespace(
        context_length=total_length_k,
        batch_size=1,
        num_heads=num_heads // tp,
        head_dim=128,
        avg_doc_len=avg_doc_len,
        std_doc_len=0,
        cp_size=world_size,
        fix_seed=1,
        include_backward=False,
        gqa=16,
    )

    manager = mp.Manager()
    return_dict = manager.dict()

    mp.spawn(
        run,
        nprocs=world_size,
        args=(world_size, args, return_dict),
        join=True,
    )
    
    num_seq_in_batch = total_length_k * 1024 // seq_len
    return_dict = dict(return_dict)

    per_seq_latency = [
        return_dict[i]["per_seq_latency"] for i in range(world_size)
    ]
    per_seq_allgather_latency = [
        return_dict[i]["perseq_allgather_time"] for i in range(world_size)
    ]
    per_seq_attn_latency = [
        return_dict[i]["perseq_attn_time"] for i in range(world_size)
    ]
    per_doc_latency = [
        return_dict[i]["per_doc_latency"] for i in range(world_size)
    ]
    per_doc_allgather_latency = [
        return_dict[i]["perdoc_allgather_time"] for i in range(world_size)
    ]
    per_doc_attn_latency = [
        return_dict[i]["perdoc_attn_time"] for i in range(world_size)
    ]
    per_seq_latency = max(per_seq_latency) / num_seq_in_batch
    per_doc_latency = max(per_doc_latency) / num_seq_in_batch
    per_seq_allgather_latency = max(per_seq_allgather_latency) / num_seq_in_batch
    per_seq_attn_latency = max(per_seq_attn_latency) / num_seq_in_batch
    per_doc_allgather_latency = max(per_doc_allgather_latency) / num_seq_in_batch
    per_doc_attn_latency = max(per_doc_attn_latency) / num_seq_in_batch
    
    return_dict.update(dict(
        per_seq_latency=per_seq_latency,
        per_doc_latency=per_doc_latency,
        per_seq_allgather_latency=per_seq_allgather_latency,
        per_seq_attn_latency=per_seq_attn_latency,
        per_doc_allgather_latency=per_doc_allgather_latency,
        per_doc_attn_latency=per_doc_attn_latency,
        num_seq_in_batch= num_seq_in_batch,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=args.head_dim,
        cp=args.cp_size,
        tp=tp,
        comment="tp runs with reduced num_heads, cp is the same as world_size. Time measured in ms."
    ))

    print(return_dict)
    return return_dict


@app.function(image=image, gpu="A100:1", timeout=60)
def run_perf_A100_1(*args, **kwargs):
    return run_perf(1, *args, **kwargs)

@app.function(image=image, gpu="A100:2", timeout=60)
def run_perf_A100_2(*args, **kwargs):
    return run_perf(2, *args, **kwargs)

@app.function(image=image, gpu="A100:4", timeout=60)
def run_perf_A100_4(*args, **kwargs):
    return run_perf(4, *args, **kwargs)

@app.function(image=image, gpu="A100:8", timeout=60)
def run_perf_A100_8(*args, **kwargs):
    return run_perf(8, *args, **kwargs)

@app.function(image=image, gpu="H100:1", timeout=60)
def run_perf_H100_1(*args, **kwargs):
    return run_perf(1, *args, **kwargs)

@app.function(image=image, gpu="H100:2", timeout=60)
def run_perf_H100_2(*args, **kwargs):
    return run_perf(2, *args, **kwargs)

@app.function(image=image, gpu="H100:4", timeout=60)
def run_perf_H100_4(*args, **kwargs):
    return run_perf(4, *args, **kwargs)

@app.function(image=image, gpu="H100:8", timeout=60)
def run_perf_H100_8(*args, **kwargs):
    return run_perf(8, *args, **kwargs)


def run(
    gpu_type,
    world_size, 
    seq_len=1024, # unit length
    total_length_k=64, # K
    tp:int=1
):
    kwargs = dict(
        seq_len=seq_len,
        total_length_k=total_length_k,
        tp=tp,
    )
    if gpu_type == "A100":
        if world_size == 1:
            return run_perf_A100_1.remote(**kwargs)
        elif world_size == 2:
            return run_perf_A100_2.remote(**kwargs)
        elif world_size == 4:
            return run_perf_A100_4.remote(**kwargs)
        elif world_size == 8:
            return run_perf_A100_8.remote(**kwargs)
    elif gpu_type == "H100":
        if world_size == 1:
            return run_perf_H100_1.remote(**kwargs)
        elif world_size == 2:
            return run_perf_H100_2.remote(**kwargs)
        elif world_size == 4:
            return run_perf_H100_4.remote(**kwargs)
        elif world_size == 8:
            return run_perf_H100_8.remote(**kwargs)
    else:
        raise ValueError(f"Invalid GPU type: {gpu_type}")


@app.local_entrypoint()
def main(
    cp_size:int = 2,
):
    K = 1024
    min_seq_len = 64
    max_seq_len = 64 * K
    total_length_k = 64
    factor = 2
    gpu_type = "H100"

    import json
    file = open(f"zmodal_runperf_{gpu_type}_{cp_size}.jsonl", "a")
    seq_len = min_seq_len
    while seq_len <= max_seq_len:
        for tp in [1, 2, 4, 8]:
            result = run(
                gpu_type=gpu_type,
                world_size=cp_size,
                seq_len=seq_len,
                total_length_k=total_length_k,
                tp=tp,
            )
            print(result, flush=True)
            file.write(json.dumps(result) + "\n")
            file.flush()
        seq_len *= factor
    file.close()

"""
modal run zmodal_runperf.py --cp-size 2
"""