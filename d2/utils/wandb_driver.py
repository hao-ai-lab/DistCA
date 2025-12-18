"""
Wandb driver for training_3d.py

This module handles all wandb-related functionality for logging training metrics.
"""

import os
from typing import Optional, Dict, Any

# Try to import wandb (optional dependency)
try:
    import wandb
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key is not None:
        wandb.login(key=api_key)
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class WandbDriver:
    """Driver class for managing wandb logging."""
    
    def __init__(self):
        self.run = None
        self.enabled = False
        self.allow_all_ranks = False
        self.memory_step = 0  # Separate step counter for memory usage
    
    def initialize(
        self,
        enable_wandb: bool,
        rank: int,
        project: str = "d2-training",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        allow_all_ranks: bool = False,
    ) -> bool:
        """
        Initialize wandb run.
        
        Args:
            enable_wandb: Whether to enable wandb logging
            rank: Process rank (only rank 0 initializes wandb)
            project: Wandb project name
            run_name: Wandb run name (auto-generated if None)
            config: Configuration dictionary to log
            allow_all_ranks: If True, allow all ranks to log (not just rank 0)
            
        Returns:
            True if wandb is successfully initialized, False otherwise
        """
        self.enabled = False
        self.allow_all_ranks = allow_all_ranks
        
        if not enable_wandb:
            return False
        
        # Only rank 0 initializes wandb run
        if rank != 0:
            # If allow_all_ranks is True, non-zero ranks can still log to the run
            # but they need to join the existing run (handled in log method)
            return allow_all_ranks
        
        if not WANDB_AVAILABLE:
            print(f"⚠️  [Rank {rank}] Wandb not installed. Skipping wandb logging. Install with: pip install wandb")
            return False
        
        if run_name is None:
            from datetime import datetime
            run_name = f"d2_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            self.run = wandb.init(
                project=project,
                name=run_name,
                config=config or {},
            )
            self.enabled = True
            print(f"🟢 [Rank {rank}] Wandb initialized: project={project}, run_name={run_name}")
            if allow_all_ranks:
                print(f"🟡 [Rank {rank}] All ranks enabled for wandb logging")
            return True
        except Exception as e:
            print(f"⚠️  [Rank {rank}] Failed to initialize wandb: {e}")
            return False
    
    def log(
        self,
        sample_id: int,
        duration_ms: float,
        iteration_time_ms: float,
        loss: Optional[float] = None,
        rank: int = 0,
        **kwargs
    ):
        """
        Log metrics to wandb.
        
        Args:
            sample_id: Sample/iteration ID (used as step, not logged as metric)
            duration_ms: Total duration in milliseconds
            iteration_time_ms: Iteration time in milliseconds
            loss: Loss value (optional)
            rank: Process rank
            **kwargs: Additional metrics to log
        """
        # Only rank 0 logs to wandb by default, unless allow_all_ranks is True
        if not self.allow_all_ranks and rank != 0:
            return
        
        if not self.enabled or self.run is None:
            return
        
        log_dict = {
            "duration_ms": duration_ms,
            "iteration_time_ms": iteration_time_ms,
            **kwargs
        }
        
        if loss is not None:
            log_dict["loss"] = float(loss)
        
        # Add rank suffix if logging from multiple ranks
        if self.allow_all_ranks and rank != 0:
            log_dict = {f"{k}_rank{rank}": v for k, v in log_dict.items()}
        
        try:
            wandb.log(log_dict, step=sample_id)
        except Exception as e:
            print(f"⚠️  [Rank {rank}] Failed to log to wandb: {e}")
    
    def log_memory(
        self,
        allocated_cur_gb: float,
        allocated_peak_gb: float,
        total_alloc_gb: float,
        pynvml_gpu_memory_usage_gb: float,
        rank: int = 0,
    ):
        """
        Log memory usage metrics to wandb with a separate step counter.
        
        Args:
            allocated_cur_gb: Current allocated memory in GB
            allocated_peak_gb: Peak allocated memory in GB
            total_alloc_gb: Total allocated memory in GB
            pynvml_gpu_memory_usage_gb: PyNVML GPU memory usage in GB
            rank: Process rank
        """
        # Only rank 0 logs to wandb by default, unless allow_all_ranks is True
        if not self.allow_all_ranks and rank != 0:
            return
        
        if not self.enabled or self.run is None:
            return
        
        # Prefix metrics with 'memory/' so they appear in a separate section in W&B UI
        log_dict = {
            "memory/allocated_cur_gb": allocated_cur_gb,
            "memory/allocated_peak_gb": allocated_peak_gb,
            "memory/total_alloc_gb": total_alloc_gb,
            "memory/pynvml_gpu_gb": pynvml_gpu_memory_usage_gb,
        }
        
        # Add rank suffix if logging from multiple ranks
        if self.allow_all_ranks and rank != 0:
            log_dict = {f"{k}_rank{rank}": v for k, v in log_dict.items()}
        
        try:
            wandb.log(log_dict, step=self.memory_step)
            self.memory_step += 1
        except Exception as e:
            print(f"⚠️  [Rank {rank}] Failed to log memory to wandb: {e}")
    
    def print_loss(
        self,
        sample_id: int,
        loss: Optional[float],
        rank: int,
        allow_all_ranks: bool = False,
    ):
        """
        Print loss value (for console output from all ranks if enabled).
        
        Args:
            sample_id: Sample/iteration ID
            loss: Loss value
            rank: Process rank
            allow_all_ranks: If True, print from all ranks; otherwise only rank 0
        """
        if not allow_all_ranks and rank != 0:
            return
        
        if loss is not None:
            print(f"📉 [Rank {rank}] Sample {sample_id}: loss = {loss:.6f}")
        else:
            print(f"📉 [Rank {rank}] Sample {sample_id}: loss = N/A")
    
    def finish(self, rank: int = 0):
        """Finish the wandb run."""
        if not self.enabled or self.run is None:
            return
        
        try:
            wandb.finish()
            print(f"🟢 [Rank {rank}] Wandb run finished successfully")
        except Exception as e:
            print(f"⚠️  [Rank {rank}] Failed to finish wandb run: {e}")
        finally:
            self.enabled = False
            self.run = None

