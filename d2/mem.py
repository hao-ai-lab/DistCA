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


def log_memory_usage(message: str, force: bool = False, comment: str = None):
    EXPERIMENT_LOG_MEMORY_USAGE = os.getenv("EXPERIMENT_LOG_MEMORY_USAGE", "0")
    # print(f"Ⓜ️", EXPERIMENT_LOG_MEMORY_USAGE)
    # return
    if not force:
        if EXPERIMENT_LOG_MEMORY_USAGE != "1":
            return
    
    global memory_usage
    
    # Check if `CUDA_LAUNCH_BLOCKING` is set
    if os.getenv("CUDA_LAUNCH_BLOCKING", "0") != "1":
        if not hasattr(log_memory_usage, "_warned_cuda_launch_blocking"):
            print("⚠️ Warning: CUDA_LAUNCH_BLOCKING=0 may affect memory logging accuracy")
            log_memory_usage._warned_cuda_launch_blocking = True
    
    if not force:
        if not torch.distributed.is_initialized():
            return
    
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
          f"nvidia-smi reported usage: {(pynvml_gpu_memory_usage/1024):.2f} GB | "
          f"comment: {comment}")
    
    new_entry = {
        "message": message,
        "allocated_cur": allocated_cur,
        "allocated_peak": allocated_peak,
        "total_alloc": total_alloc,
        "pynvml_gpu_memory_usage": pynvml_gpu_memory_usage,
        "comment": comment,
    }
    memory_usage.append(new_entry)
    
    global memory_usage_log_file
    if memory_usage_log_file is not None:
        with open(memory_usage_log_file, 'a') as f:
            f.write(json.dumps(new_entry) + "\n")

from contextlib import contextmanager

@contextmanager
def log_memory_usage_context():
    old_env_var = os.environ.get("EXPERIMENT_LOG_MEMORY_USAGE", "0")
    os.environ["EXPERIMENT_LOG_MEMORY_USAGE"] = "1"
    print(f"🟡 Setting EXPERIMENT_LOG_MEMORY_USAGE to 1 temporarily")
    yield
    os.environ["EXPERIMENT_LOG_MEMORY_USAGE"] = old_env_var
    print(f"🟡 Setting EXPERIMENT_LOG_MEMORY_USAGE back to {old_env_var}")


def enable_memory_usage_logging(memory_usage_dir: str):
    os.makedirs(memory_usage_dir, exist_ok=True)
    rank = os.environ.get("RANK", "0")
    memory_usage_log_file = os.path.join(memory_usage_dir, f"mem.rank{rank}.log.jsonl")
    with open(memory_usage_log_file, 'w') as f:
        pass
    set_memory_usage_log_file(memory_usage_log_file)
    pass



def get_memory_usage():
    global memory_usage
    return memory_usage


import glob
import pandas as pd
import matplotlib.pyplot as plt


def plot_memory_usage_figure(memory_usage_dir: str):
    # list the memory usage log files in the directory, 
    # read each file and plot the memory usage
    memory_usage_log_files = glob.glob(os.path.join(memory_usage_dir, "mem.rank*"))
    
    # allocated_cur
    for rank, memory_usage_log_file in enumerate(memory_usage_log_files):
        with open(memory_usage_log_file, 'r') as f:
            memory_usage = [json.loads(line) for line in f]
            df = pd.DataFrame(memory_usage)
            plt.plot(df.index, df['allocated_cur'], label=f'Rank {rank}', alpha=0.5)
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(memory_usage_dir, "memory_usage.allocated_cur.png"))
    return