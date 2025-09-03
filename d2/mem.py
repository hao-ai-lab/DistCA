import torch
import torch.distributed

import os


memory_usage = []


def log_memory_usage(message: str):
    
    global memory_usage

    if os.getenv("EXPERIMENT_LOG_MEMORY_USAGE", "0") != "1":
        return
    # Check if `CUDA_LAUNCH_BLOCKING` is set
    if os.getenv("CUDA_LAUNCH_BLOCKING", "0") != "1":
        if not hasattr(log_memory_usage, "_warned_cuda_launch_blocking"):
            print("⚠️ Warning: CUDA_LAUNCH_BLOCKING=0 may affect memory logging accuracy")
            log_memory_usage._warned_cuda_launch_blocking = True
    
    if not torch.distributed.is_initialized():
        return
    
    rank = torch.distributed.get_rank()
    # if rank % 8 != 0:
    #     return

    device = torch.cuda.current_device()
    allocated_cur = torch.cuda.memory_allocated(device) / (1024 ** 2) # MB
    allocated_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    total_alloc = (allocated_cur + torch.cuda.memory_reserved(device) / (1024 ** 2))

    print(f"Ⓜ️ [{message}] Allocated: {(allocated_cur/ 1024):.2f} GB | "
          f"Peak: {(allocated_peak/ 1024):.2f} GB | "
          f"Total alloc (approx): {(total_alloc/ 1024):.2f} GB")

    memory_usage.append({
        "message": message,
        "allocated_cur": allocated_cur,
        "allocated_peak": allocated_peak,
        "total_alloc": total_alloc,
    })


def get_memory_usage():
    global memory_usage
    return memory_usage