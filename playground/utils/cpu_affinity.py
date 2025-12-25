"""
CPU core affinity utilities for distributed training.
"""
import os
from typing import List, Optional

import psutil


def set_cpu_affinity(
    local_rank: int,
    ncpu_per_proc: int = 16,
    logger=None,
) -> List[int]:
    """
    Bind the current process to a specific set of CPU cores based on local_rank.

    This helps reduce CPU contention in multi-GPU training by assigning
    each GPU process to a dedicated set of cores.

    Args:
        local_rank: The local rank of this process (0-indexed per node).
        ncpu_per_proc: Number of CPU cores to allocate per process.
        logger: Optional logger for info messages.

    Returns:
        List of core indices assigned to this process.

    Example:
        # local_rank=0, ncpu_per_proc=16 -> cores [0, 1, ..., 15]
        # local_rank=1, ncpu_per_proc=16 -> cores [16, 17, ..., 31]
    """
    p = psutil.Process(os.getpid())
    start_core = local_rank * ncpu_per_proc
    end_core = (local_rank + 1) * ncpu_per_proc
    core_list = list(range(start_core, end_core))
    p.cpu_affinity(core_list)

    if logger is not None:
        logger.info(f"Set cpu_affinity (local_rank {local_rank}): {core_list}")

    return core_list


def get_cpu_affinity() -> List[int]:
    """
    Get the current CPU affinity of this process.

    Returns:
        List of core indices this process is bound to.
    """
    p = psutil.Process(os.getpid())
    return p.cpu_affinity()


