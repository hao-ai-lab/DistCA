# allreduce_bench.py
import modal

# ---------- Modal boilerplate ----------
app = modal.App("allreduce-bench")

# Base image with PyTorch + CUDA already linked against NCCL
image = (
    modal.Image.debian_slim()
    .pip_install("torch==2.3.0")   # use the latest stable that matches CUDA 12
)

def run_allreduce(local_rank, world_size, iters, tensor_elems, ret):
    import os, time, torch, torch.multiprocessing as mp
    # ----- NCCL setup -----
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=local_rank,
        init_method="env://",
    )

    # ----- Benchmark -----
    tensor = torch.randn(tensor_elems, device="cuda", dtype=torch.float16)
    torch.distributed.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / iters  # avg per iteration

    # Only let rank-0 report
    if local_rank == 0:
        ret["latency_ms"] = elapsed_ms

    torch.distributed.destroy_process_group()


def run_allgather(local_rank, world_size, iters, tensor_elems, ret):
    import os, time, torch, torch.multiprocessing as mp
    # ----- NCCL setup -----
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=local_rank,
        init_method="env://",
    )
    
    # ----- Benchmark -----
    tensors = []
    for i in range(world_size):
        tensors.append(
            (torch.ones(tensor_elems, device="cuda", dtype=torch.float16) * i)
        )
    tensor = tensors[local_rank]
    torch.distributed.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        torch.distributed.all_gather(tensors, tensor)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / iters  # avg per iteration
    
    # Only let rank-0 report
    if local_rank == 0:
        ret["latency_ms"] = elapsed_ms

    torch.distributed.destroy_process_group()
    

def allreduce_worker(world_size: int, iters: int, tensor_elems: int):
    """
    Spawned from local_entrypoint with .options(gpu="TYPE:COUNT").
    Runs a single-process NCCL benchmark (rank == local GPU index).
    Returns average latency in milliseconds.
    """
    import os, time, torch, torch.multiprocessing as mp

    manager = mp.Manager()
    shared = manager.dict()
    mp.spawn(run_allreduce, args=(world_size, iters, tensor_elems, shared),
             nprocs=world_size)
    return shared["latency_ms"]


def allgather_worker(world_size: int, iters: int, tensor_elems: int):
    import os, time, torch, torch.multiprocessing as mp
    manager = mp.Manager()
    shared = manager.dict()
    mp.spawn(run_allgather, args=(world_size, iters, tensor_elems, shared),
             nprocs=world_size)
    return shared["latency_ms"]


@app.function(image=image, gpu="A100:2", timeout=60)
def allreduce_worker_A100_2(iters: int, tensor_elems: int):
    return allreduce_worker(2, iters, tensor_elems)

@app.function(image=image, gpu="A100:4", timeout=60)
def allreduce_worker_A100_4(iters: int, tensor_elems: int):
    return allreduce_worker(4, iters, tensor_elems)

@app.function(image=image, gpu="A100:8", timeout=60)
def allreduce_worker_A100_8(iters: int, tensor_elems: int):
    return allreduce_worker(8, iters, tensor_elems)

@app.function(image=image, gpu="H100:2", timeout=60)
def allreduce_worker_H100_2(iters: int, tensor_elems: int):
    return allreduce_worker(2, iters, tensor_elems)

@app.function(image=image, gpu="H100:4", timeout=60)
def allreduce_worker_H100_4(iters: int, tensor_elems: int):
    return allreduce_worker(4, iters, tensor_elems)

@app.function(image=image, gpu="H100:8", timeout=60)
def allreduce_worker_H100_8(iters: int, tensor_elems: int):
    return allreduce_worker(8, iters, tensor_elems)

@app.function(image=image, gpu="A100:2", timeout=60)
def allgather_worker_A100_2(iters: int, tensor_elems: int):
    return allgather_worker(2, iters, tensor_elems)

@app.function(image=image, gpu="A100:4", timeout=60)
def allgather_worker_A100_4(iters: int, tensor_elems: int):
    return allgather_worker(4, iters, tensor_elems)

@app.function(image=image, gpu="A100:8", timeout=60)
def allgather_worker_A100_8(iters: int, tensor_elems: int):
    return allgather_worker(8, iters, tensor_elems)

@app.function(image=image, gpu="H100:2", timeout=60)
def allgather_worker_H100_2(iters: int, tensor_elems: int):
    return allgather_worker(2, iters, tensor_elems)

@app.function(image=image, gpu="H100:4", timeout=60)
def allgather_worker_H100_4(iters: int, tensor_elems: int):
    return allgather_worker(4, iters, tensor_elems)

@app.function(image=image, gpu="H100:8", timeout=60)
def allgather_worker_H100_8(iters: int, tensor_elems: int):
    return allgather_worker(8, iters, tensor_elems)


def gpu_func_maping(gpu_type: str, world_size: int, op: str):
    if op == "allreduce":
        if gpu_type == "A100":
            if world_size == 2:
                return allreduce_worker_A100_2
            elif world_size == 4:
                return allreduce_worker_A100_4
            elif world_size == 8:
                return allreduce_worker_A100_8
        elif gpu_type == "H100":
            if world_size == 2:
                return allreduce_worker_H100_2
            elif world_size == 4:
                return allreduce_worker_H100_4
            elif world_size == 8:
                return allreduce_worker_H100_8
        else:
            raise ValueError(f"Invalid GPU type: {gpu_type}")
    elif op == "allgather":
        if gpu_type == "A100":
            if world_size == 2:
                return allgather_worker_A100_2
            elif world_size == 4:
                return allgather_worker_A100_4
            elif world_size == 8:
                return allgather_worker_A100_8
        elif gpu_type == "H100":
            if world_size == 2:
                return allgather_worker_H100_2
            elif world_size == 4:
                return allgather_worker_H100_4
            elif world_size == 8:
                return allgather_worker_H100_8
        else:
            raise ValueError(f"Invalid GPU type: {gpu_type}")
    else:
        raise ValueError(f"Invalid operation: {op}")


# ---------- CLI entrypoint ----------
@app.local_entrypoint()
def main(
    gpu_type: str = "H100",
    op: str = "allgather",
    world_size: int = 8,
    iters: int = 10,
    min_tensor_elems: int = 2 ** 5,  # 32 elements -> 64 bytes data
    max_tensor_elems: int = 2 ** 34, # 16T elements -> 32 TB data
    step_tensor_elems: int = 2,
):
    tensor_elems = min_tensor_elems
    func = gpu_func_maping(gpu_type, world_size, op)

    filename = f"network-{op}-{gpu_type}-{world_size}.csv"
    import os
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            pass

    with open(filename, "a") as f:
        def print_dual(*args):
            print(*args, file=f, flush=True)
            print(*args, flush=True)
            return
        
        print_dual(f"gpu_type,world_size,op,nelem,dtype,latency(ms),throughput(nelem_per_ms)")
        while tensor_elems <= max_tensor_elems:
            latency = (
                func.remote(iters, tensor_elems)
            )
            throughput_gbps = (
                tensor_elems / latency
            )
            print_dual(
                f"{gpu_type},{world_size},{op},{tensor_elems},fp16,{latency:.2f},{throughput_gbps:.1f}"
            )
            tensor_elems *= step_tensor_elems