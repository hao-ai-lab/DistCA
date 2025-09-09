import pynvml
import torch
import torch.distributed
import json
import os
import inspect

def get_caller_info(depth=1):
    frame = inspect.currentframe()
    try:
        for _ in range(depth):
            frame = frame.f_back
        return frame.f_code.co_filename, frame.f_lineno
    finally:
        del frame  # avoid reference cycles


memory_usage = []
memory_usage_log_file = None


def set_memory_usage_log_file(x: str):
    global memory_usage_log_file
    memory_usage_log_file = x


def get_pynvml_gpu_memory_usage(device):   
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used = info.used / (1024**2)
    pynvml.nvmlShutdown()
    return used
    

def get_torch_cuda_memory_usage(device):
    
    allocated_cur = torch.cuda.memory_allocated(device) / (1024 ** 2) # MB
    allocated_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    total_alloc = (allocated_cur + torch.cuda.memory_reserved(device) / (1024 ** 2))
    return allocated_cur, allocated_peak, total_alloc

def log_memory_usage(message: str, force: bool = False):
    EXPERIMENT_LOG_MEMORY_USAGE = os.getenv("EXPERIMENT_LOG_MEMORY_USAGE", "0")
    # print(f"Ⓜ️", EXPERIMENT_LOG_MEMORY_USAGE)
    # return
    if EXPERIMENT_LOG_MEMORY_USAGE != "1":
        return
    
    global memory_usage
    
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

    caller_info = get_caller_info(3)
    message += f" ({caller_info[0]}:{caller_info[1]})"

    device = torch.cuda.current_device()
    allocated_cur, allocated_peak, total_alloc = get_torch_cuda_memory_usage(device)

    # also use nvidia-smi to get the allocated memory
    pynvml_gpu_memory_usage = get_pynvml_gpu_memory_usage(device)
    

    print(f"Ⓜ️ [{message}] Allocated: {(allocated_cur/ 1024):.2f} GB | "
          f"Peak: {(allocated_peak/ 1024):.2f} GB | "
          f"Total alloc (approx): {(total_alloc/ 1024):.2f} GB | "
          f"nvidia-smi reported usage: {(pynvml_gpu_memory_usage/1024):.2f} GB")
    
    new_entry = {
        "message": message,
        "allocated_cur": allocated_cur,
        "allocated_peak": allocated_peak,
        "total_alloc": total_alloc,
        "pynvml_gpu_memory_usage": pynvml_gpu_memory_usage,
    }
    memory_usage.append(new_entry)
    
    global memory_usage_log_file
    if memory_usage_log_file is not None:
        with open(memory_usage_log_file, 'a') as f:
            f.write(json.dumps(new_entry) + "\n")


def get_memory_usage():
    global memory_usage
    return memory_usage