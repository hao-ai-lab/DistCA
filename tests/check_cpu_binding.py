import os, re, socket, torch

def read_status_field(field: str) -> str:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return line.split(":",1)[1].strip()
    return "N/A"

def check_cpu_binding():
    """Check and print CPU binding information for the current process.
    
    Args:
        env: Optional dictionary of environment variables. If not provided, uses os.environ.
    """
    # Get CPU and memory binding info
    aff = read_status_field("Cpus_allowed_list")
    mems = read_status_field("Mems_allowed_list")
    return aff, mems