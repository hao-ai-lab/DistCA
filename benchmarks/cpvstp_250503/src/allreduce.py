import torch
import torch.distributed as dist
import time
import argparse
import numpy as np
import json
from torch.distributed import ReduceOp
from rich import print

from pathlib import Path

myrank = 0
def print_if_rank_0(msg):
    global myrank
    if myrank == 0:
        print(msg)


def setup_distributed():
    """Initialize distributed environment"""
    global myrank
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
        myrank = dist.get_rank()
    return dist.get_rank(), dist.get_world_size()

def benchmark_allreduce(tensor_size, num_iterations=100):
    """Benchmark allreduce operation for a given tensor size"""
    rank, world_size = setup_distributed()
    
    # Create tensor
    tensor = torch.rand(tensor_size, device=f'cuda:{rank}')
    
    # Warmup
    for _ in range(10):
        dist.all_reduce(tensor, op=ReduceOp.SUM)
    
    torch.distributed.barrier()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.time()
        dist.all_reduce(tensor, op=ReduceOp.SUM)
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
    
    torch.distributed.barrier()
    
    # Calculate statistics
    mean_time = np.mean(times) * 1000  # Convert to milliseconds
    std_time = np.std(times) * 1000
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000
    data_size_mb = tensor.element_size() * tensor.nelement() / (1024*1024)
    
    result = {
        "tensor_size": tensor_size,
        "data_size_mb": round(data_size_mb, 2),
        "world_size": world_size,
        "iterations": num_iterations,
        "mean_time_ms": round(mean_time, 3),
        "std_time_ms": round(std_time, 3),
        "min_time_ms": round(min_time, 3),
        "max_time_ms": round(max_time, 3)
    }
    return result

def main():
    parser = argparse.ArgumentParser(description='Benchmark allreduce operations')
    parser.add_argument('--sizes', type=int, nargs='+', 
                        default=[2 ** k for k in range(1, 34 + 1)],
                        help='List of tensor sizes to benchmark')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations per size')
    args = parser.parse_args()
    
    rank, world_size = setup_distributed()
    
    print_if_rank_0(args)
    
    # do a warmup
    for _ in range(3):
        dist.all_reduce(torch.rand(1024, device=f'cuda:{rank}'), op=ReduceOp.SUM)
    torch.distributed.barrier()

    output_file = Path(__file__).parent.parent / 'results' / f'allreduce.w{world_size}.jsonl'
    
    if rank == 0:
        print_if_rank_0({
            "benchmark_start": True,
            "world_size": world_size,
            "num_sizes": len(args.sizes),
            "iterations": args.iterations
        })
    
    for size in args.sizes:
        iterations = args.iterations
        if size > 2 ** 28:
            iterations = 5
        result = benchmark_allreduce(size, iterations)
        if rank == 0:
            print_if_rank_0(json.dumps(result))
            with open(output_file, 'a') as f:
                f.write(json.dumps(result) + '\n')

if __name__ == "__main__":
    main()
"""
torchrun --nproc_per_node=4 allreduce.py 
"""