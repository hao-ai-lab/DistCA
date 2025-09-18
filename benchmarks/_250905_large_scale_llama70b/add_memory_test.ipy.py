import pynvml
import torch

def get_gpu_memory_usage(gpuid=None):
    if gpuid is None:
        gpuid = torch.cuda.current_device()
        
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    memory_info = []
    
    if gpuid >= 0:
        # Get memory info for specific GPU
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpuid)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_info.append({
            'device': gpuid,
            'total': info.total / (1024**3),  # Convert to GB
            'used': info.used / (1024**3),
            'free': info.free / (1024**3)
        })
    else:
        # Get memory info for all GPUs
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_info.append({
                'device': i,
                'total': info.total / (1024**3),  # Convert to GB
                'used': info.used / (1024**3),
                'free': info.free / (1024**3)
            })
    
    pynvml.nvmlShutdown()
    return memory_info

def print_gpu_memory_usage(gpuid=None):
    memory_info = get_gpu_memory_usage(gpuid)
    for gpu in memory_info:
        print(f"GPU {gpu['device']}:")
        print(f"  Total memory: {gpu['total']:.1f} GB")
        print(f"  Used memory:  {gpu['used']:.1f} GB")
        print(f"  Free memory:  {gpu['free']:.1f} GB")
        print()
