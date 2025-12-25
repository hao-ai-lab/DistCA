"""
Theoretical memory and compute estimation utilities for distributed training.

This module provides tools to estimate memory usage and FLOPS for training runs.
"""
import logging
from dataclasses import dataclass
from typing import Optional

import torch

from .logging import get_logger

# =============================================================================
# Constants
# =============================================================================

NUM_BYTES_IN_GIGABYTE = 1024 * 1024 * 1024

# GPU peak FLOPS (approximate values for common GPUs in BF16/FP16 Tensor Core)
GPU_PEAK_TFLOPS = {
    "H100_SXM": 989.0,
    "H100_PCIe": 756.0,
    "H200": 989.0,  # Same compute as H100 SXM
    "A100": 312.0,
    "default": 312.0,  # Conservative estimate
}


# =============================================================================
# GPU Detection
# =============================================================================

def get_gpu_peak_tflops(device_id: int = 0, logger: Optional[logging.Logger] = None) -> tuple[str, float]:
    """Get GPU name and estimated peak BF16/FP16 TFLOPS.
    
    Args:
        device_id: CUDA device ID to query.
        logger: Logger for warnings. If None, uses module logger.
        
    Returns:
        Tuple of (gpu_name, peak_tflops)
    """
    log = logger or get_logger()
    gpu_name = torch.cuda.get_device_name(device_id)
    
    if "H100" in gpu_name:
        if "SXM" in gpu_name:
            peak_tflops = GPU_PEAK_TFLOPS["H100_SXM"]
        else:  # PCIe or other H100 variant
            peak_tflops = GPU_PEAK_TFLOPS["H100_PCIe"]
    elif "H200" in gpu_name:
        peak_tflops = GPU_PEAK_TFLOPS["H200"]
    elif "A100" in gpu_name:
        peak_tflops = GPU_PEAK_TFLOPS["A100"]
    else:
        peak_tflops = GPU_PEAK_TFLOPS["default"]
        log.warning(
            f"Unknown GPU '{gpu_name}', using conservative peak FLOPS estimate "
            f"of {peak_tflops} TFLOP/s"
        )
    
    return gpu_name, peak_tflops


# =============================================================================
# Memory Estimation
# =============================================================================

@dataclass
class MemoryEstimate:
    """Theoretical memory usage estimates in GB."""
    weight_and_optimizer_gb: float
    activation_gb: float
    total_gb: float


def estimate_memory(
    args,
    num_microbatches: int,
    verbose: bool = True,
    logger: Optional[logging.Logger] = None,
) -> MemoryEstimate:
    """Compute theoretical memory usage estimates.
    
    Args:
        args: Megatron arguments object.
        num_microbatches: Number of microbatches.
        verbose: Whether to print verbose output from megatron functions.
        logger: Logger for output. If None, uses module logger.
        
    Returns:
        MemoryEstimate with weight/optimizer, activation, and total memory in GB.
    """
    from megatron.training.theoretical_memory_usage import (
        compute_weight_and_optimizer_memory,
        compute_activation_memory,
    )
    
    weight_and_optimizer_gb = (
        compute_weight_and_optimizer_memory(args, verbose=verbose)
        / NUM_BYTES_IN_GIGABYTE
    )
    
    activation_gb = (
        compute_activation_memory(args, num_microbatches=num_microbatches, verbose=verbose)
        / NUM_BYTES_IN_GIGABYTE
    )
    
    return MemoryEstimate(
        weight_and_optimizer_gb=weight_and_optimizer_gb,
        activation_gb=activation_gb,
        total_gb=weight_and_optimizer_gb + activation_gb,
    )


def log_memory_estimate(
    args,
    num_microbatches: int,
    verbose: bool = True,
    logger: Optional[logging.Logger] = None,
) -> MemoryEstimate:
    """Compute and log theoretical memory usage estimates.
    
    Args:
        args: Megatron arguments object.
        num_microbatches: Number of microbatches.
        verbose: Whether to print verbose output from megatron functions.
        logger: Logger for output. If None, uses module logger.
        
    Returns:
        MemoryEstimate with the computed values.
    """
    log = logger or get_logger()
    
    estimate = estimate_memory(args, num_microbatches, verbose=verbose, logger=logger)
    
    log.info(f"=== Theoretical Memory Report ===")
    log.info(f"  Weight + optimizer memory: {estimate.weight_and_optimizer_gb:.2f} GB")
    log.info(f"  Activation memory: {estimate.activation_gb:.2f} GB")
    log.info(f"  Total memory: {estimate.total_gb:.2f} GB")
    log.info(f"  Note: Activation memory is only accurate with sequence_parallel + selective recompute")
    log.info(f"=================================")
    
    return estimate


# =============================================================================
# FLOPS Estimation
# =============================================================================

@dataclass
class FlopsEstimate:
    """Theoretical FLOPS estimates."""
    flops_per_forward_pass: float  # FLOPs for one forward pass (one micro-batch)
    flops_per_iteration_per_gpu: float  # FLOPs per GPU per iteration (fwd+bwd)
    total_flops_per_iteration: float  # Total FLOPs across all GPUs per iteration
    
    # Derived values in TFLOP
    tflops_per_forward: float
    tflops_per_iteration_per_gpu: float
    tflops_per_iteration_total: float
    
    # Batch info
    micro_batch_size: int
    num_microbatches: int
    global_batch_size: int
    
    # Parallelism info
    tp: int
    pp: int
    cp: int
    dp: int


def estimate_flops(
    args,
    num_microbatches: int,
    micro_batch_size: int,
    dp_world_size: int,
    tp: int,
    pp: int,
    cp: int,
) -> FlopsEstimate:
    """Compute theoretical FLOPS estimates.
    
    Args:
        args: Megatron arguments object.
        num_microbatches: Number of microbatches.
        micro_batch_size: Micro batch size.
        dp_world_size: Data parallel world size.
        tp: Tensor parallel size.
        pp: Pipeline parallel size.
        cp: Context parallel size.
        
    Returns:
        FlopsEstimate with computed values.
    """
    from megatron.training.training import num_floating_point_operations
    
    global_batch_size = micro_batch_size * num_microbatches * dp_world_size
    
    # Calculate FLOPs per forward pass for one micro-batch
    flops_per_forward_pass = num_floating_point_operations(args, micro_batch_size)
    
    # For training: forward + backward ≈ 3x forward (backward is ~2x forward)
    flops_per_iteration_per_gpu = flops_per_forward_pass * 3
    
    # Total FLOPs across all GPUs per iteration (considering all microbatches and DP)
    total_flops_per_iteration = flops_per_forward_pass * 3 * num_microbatches * dp_world_size
    
    return FlopsEstimate(
        flops_per_forward_pass=flops_per_forward_pass,
        flops_per_iteration_per_gpu=flops_per_iteration_per_gpu,
        total_flops_per_iteration=total_flops_per_iteration,
        tflops_per_forward=flops_per_forward_pass / 1e12,
        tflops_per_iteration_per_gpu=flops_per_iteration_per_gpu / 1e12,
        tflops_per_iteration_total=total_flops_per_iteration / 1e12,
        micro_batch_size=micro_batch_size,
        num_microbatches=num_microbatches,
        global_batch_size=global_batch_size,
        tp=tp,
        pp=pp,
        cp=cp,
        dp=dp_world_size,
    )


def log_flops_estimate(
    args,
    num_microbatches: int,
    micro_batch_size: int,
    dp_world_size: int,
    tp: int,
    pp: int,
    cp: int,
    device_id: int = 0,
    logger: Optional[logging.Logger] = None,
) -> FlopsEstimate:
    """Compute and log theoretical FLOPS estimates.
    
    Args:
        args: Megatron arguments object.
        num_microbatches: Number of microbatches.
        micro_batch_size: Micro batch size.
        dp_world_size: Data parallel world size.
        tp: Tensor parallel size.
        pp: Pipeline parallel size.
        cp: Context parallel size.
        device_id: CUDA device ID for GPU info.
        logger: Logger for output. If None, uses module logger.
        
    Returns:
        FlopsEstimate with computed values.
    """
    log = logger or get_logger()
    
    estimate = estimate_flops(
        args, num_microbatches, micro_batch_size, dp_world_size, tp, pp, cp
    )
    
    gpu_name, gpu_peak_tflops = get_gpu_peak_tflops(device_id, logger=log)
    
    log.info(f"=== Theoretical FLOPS Report ===")
    log.info(f"  Batch size config: micro_batch={estimate.micro_batch_size}, "
             f"num_microbatches={estimate.num_microbatches}, "
             f"global_batch={estimate.global_batch_size}")
    log.info(f"  Parallelism: TP={estimate.tp}, PP={estimate.pp}, CP={estimate.cp}, DP={estimate.dp}")
    log.info(f"  FLOPs per forward pass (1 micro-batch): "
             f"{estimate.flops_per_forward_pass:.2e} ({estimate.tflops_per_forward:.2f} TFLOP)")
    log.info(f"  FLOPs per iteration per GPU (fwd+bwd): "
             f"{estimate.flops_per_iteration_per_gpu:.2e} ({estimate.tflops_per_iteration_per_gpu:.2f} TFLOP)")
    log.info(f"  Total FLOPs per iteration (all GPUs): "
             f"{estimate.total_flops_per_iteration:.2e} ({estimate.tflops_per_iteration_total:.2f} TFLOP)")
    log.info(f"  GPU: {gpu_name}")
    log.info(f"  GPU peak BF16 TFLOP/s (estimated): {gpu_peak_tflops:.1f}")
    log.info(f"  To compute MFU at runtime: MFU = (TFLOP per iteration / time_sec) / {gpu_peak_tflops:.1f}")
    log.info(f"================================")
    
    return estimate

