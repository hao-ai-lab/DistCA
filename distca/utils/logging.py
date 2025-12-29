"""
Logging, directory setup, and debugging utilities for distributed training.
"""
import logging
import os
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, List, Optional

# =============================================================================
# ANSI Colors
# =============================================================================

_COLOR_MAP = {
    logging.DEBUG: "\033[36m",     # Cyan
    logging.INFO: "\033[32m",      # Green
    logging.WARNING: "\033[33m",   # Yellow
    logging.ERROR: "\033[31m",     # Red
    logging.CRITICAL: "\033[35m",  # Magenta
}
_COLOR_RESET = "\033[0m"

# =============================================================================
# Module-level state
# =============================================================================

_rank: Optional[int] = None
_world_size: Optional[int] = None
_logger: Optional[logging.Logger] = None
_original_stdout: Optional[IO] = None
_original_stderr: Optional[IO] = None
_rank_log_handle: Optional[IO] = None
_rank_log_file: Optional[Path] = None  # Path to current rank's log file


# =============================================================================
# Accessors
# =============================================================================

def get_rank() -> int:
    """Return the current process rank. Must call setup_logging first or pass rank explicitly."""
    if _rank is None:
        raise RuntimeError("Logging not initialized. Call setup_logging() first.")
    return _rank


def get_world_size() -> int:
    """Return the world size. Must call setup_logging first or pass world_size explicitly."""
    if _world_size is None:
        raise RuntimeError("Logging not initialized. Call setup_logging() first.")
    return _world_size


def get_logger() -> logging.Logger:
    """Return the configured logger. Must call setup_logging first."""
    if _logger is None:
        raise RuntimeError("Logging not initialized. Call setup_logging() first.")
    return _logger


# =============================================================================
# Log Paths & Directory Setup
# =============================================================================

@dataclass
class LogPaths:
    """Container for all log-related paths."""
    log_root_dir: Path
    data_cache_path: Path
    ckpt_path: Path
    tensorboard_path: Path
    rank_logs_path: Path


class TeeStream:
    """Write to both an original stream (stdout/stderr) and a file."""
    
    def __init__(self, file, original_stream):
        self.file = file
        self.original_stream = original_stream
    
    def write(self, data):
        self.original_stream.write(data)
        self.file.write(data)
    
    def flush(self):
        self.original_stream.flush()
        self.file.flush()


def setup_log_directories(
    rank: int,
    barrier_fn,
    base_dir: Optional[str] = None,
    current_log_dir: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> LogPaths:
    """
    Set up logging directories and redirect stdout for distributed training.
    
    Also adds a FileHandler to the module logger so all ranks log to their
    respective rank log files.
    
    This function checks for DISTCA_LOG_* environment variables first, which allows
    the shell script to pre-create directories (e.g., for nsys profiling) and share
    the same paths with Python. If env vars are not set, falls back to defaults.
    
    Environment variables (set by shell script):
        DISTCA_LOG_TIMESTAMP: Timestamp string (e.g., "20251225_082754")
        DISTCA_LOG_BASE_DIR: Base directory for logs (e.g., "/path/to/logs")
        DISTCA_LOG_ROOT_DIR: Full path to log root (e.g., "/path/to/logs/20251225_082754")
        DISTCA_LOG_LATEST_LINK: Path for symlink to latest log dir
    
    Args:
        rank: The rank of the current process.
        barrier_fn: A callable to synchronize across ranks (e.g., torch.distributed.barrier).
        base_dir: Base directory for logs. Overridden by DISTCA_LOG_BASE_DIR if set.
        current_log_dir: Path for symlink. Overridden by DISTCA_LOG_LATEST_LINK if set.
        timestamp: Timestamp string. Overridden by DISTCA_LOG_TIMESTAMP if set.
    
    Returns:
        LogPaths dataclass containing all the log paths.
    """
    global _original_stdout, _original_stderr, _rank_log_handle
    
    # Check for environment variables from shell script
    env_timestamp = os.environ.get("DISTCA_LOG_TIMESTAMP")
    env_base_dir = os.environ.get("DISTCA_LOG_BASE_DIR")
    env_log_root = os.environ.get("DISTCA_LOG_ROOT_DIR")
    env_latest_link = os.environ.get("DISTCA_LOG_LATEST_LINK")
    
    # If DISTCA_LOG_ROOT_DIR is set, the shell already created everything
    shell_created_dirs = env_log_root is not None
    
    # Resolve paths - prefer env vars, then function args, then defaults
    if env_log_root:
        log_root_dir = Path(env_log_root).absolute()
    else:
        if timestamp is None:
            timestamp = env_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        if base_dir is None:
            base_dir = env_base_dir or "logs"
        log_root_dir = Path(f"{base_dir}/{timestamp}").absolute()
    
    data_cache_path = (log_root_dir / "data_cache").absolute()
    ckpt_path = (log_root_dir / "checkpoints").absolute()
    tensorboard_path = (log_root_dir / "tensorboard").absolute()
    rank_logs_path = (log_root_dir / "rank_logs").absolute()
    
    if current_log_dir is None:
        current_log_dir = env_latest_link or "logs-latest"
    current_log_link = Path(current_log_dir).absolute()
    
    # Rank 0 creates directories (skip if shell already created them)
    if rank == 0 and not shell_created_dirs:
        for path in [log_root_dir, data_cache_path, ckpt_path, tensorboard_path, rank_logs_path]:
            if path.exists() and path.is_dir():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
        
        # Create/update symlink to point to the latest log directory
        if current_log_link.is_symlink() or current_log_link.exists():
            current_log_link.unlink()
        current_log_link.symlink_to(log_root_dir)
    
    # Wait for rank 0 to finish creating directories
    barrier_fn()
    
    global _rank_log_file
    
    # Redirect stdout and stderr to per-rank log files
    # Rank 0: tee to both console and file; Other ranks: file only
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    rank_log_file = rank_logs_path / f"rank_{rank}.log"
    _rank_log_file = rank_log_file  # Store for use by redirect_external_loggers
    _rank_log_handle = open(rank_log_file, 'w', buffering=1)  # line-buffered
    
    if rank == 0:
        sys.stdout = TeeStream(_rank_log_handle, _original_stdout)
        sys.stderr = TeeStream(_rank_log_handle, _original_stderr)
    else:
        sys.stdout = _rank_log_handle
        sys.stderr = _rank_log_handle
    
    # Add file handler to the logger for all ranks
    # This ensures logger.info() etc. go to the rank log file
    if _logger is not None:
        file_handler = logging.FileHandler(rank_log_file)
        file_handler.setFormatter(RankFormatter(rank=rank, use_color=False))
        file_handler.setLevel(_logger.level)
        _logger.addHandler(file_handler)
    
    paths = LogPaths(
        log_root_dir=log_root_dir,
        data_cache_path=data_cache_path,
        ckpt_path=ckpt_path,
        tensorboard_path=tensorboard_path,
        rank_logs_path=rank_logs_path,
    )
    
    # Print the paths (rank 0 only since print goes to console for rank 0)
    if rank == 0:
        print(f"log_root_dir: {paths.log_root_dir}")
        print(f"data_cache_path: {paths.data_cache_path}")
        print(f"ckpt_path: {paths.ckpt_path}")
        print(f"tensorboard_path: {paths.tensorboard_path}")
        print(f"rank_logs_path: {paths.rank_logs_path}")
    
    return paths


def restore_streams():
    """Restore original stdout/stderr and close log file handles."""
    global _original_stdout, _original_stderr, _rank_log_handle
    
    if _original_stdout is not None:
        sys.stdout = _original_stdout
    
    if _original_stderr is not None:
        sys.stderr = _original_stderr
    
    if _rank_log_handle is not None:
        _rank_log_handle.close()
        _rank_log_handle = None


# Keep the old name for backwards compatibility
restore_stdout = restore_streams


def redirect_external_loggers(
    logger_names: List[str] = None,
    level: int = logging.INFO,
    use_rank_format: bool = True,
) -> List[logging.Logger]:
    """
    Redirect external loggers (e.g., Megatron's) to the same rank log file.
    
    This function adds a FileHandler to specified loggers so their output
    goes to the same rank_X.log file as your main logs. It also optionally
    adds a console handler for rank 0.
    
    Call this AFTER setup_log_directories() has been called.
    
    Args:
        logger_names: List of logger names to redirect. If None, defaults to
            ["megatron"] which captures all megatron.* loggers.
        level: Logging level to set for these loggers.
        use_rank_format: Whether to use the RankFormatter (includes rank info).
    
    Returns:
        List of configured loggers.
    
    Example:
        # Redirect all Megatron logs to rank log files
        redirect_external_loggers(["megatron"])
        
        # Redirect specific loggers
        redirect_external_loggers([
            "megatron.core",
            "megatron.core.distributed",
        ])
    """
    if _rank_log_file is None:
        raise RuntimeError(
            "Log directories not set up. Call setup_log_directories() first."
        )
    
    if logger_names is None:
        logger_names = ["megatron"]
    
    configured_loggers = []
    rank = _rank if _rank is not None else 0
    
    for name in logger_names:
        ext_logger = logging.getLogger(name)
        ext_logger.setLevel(level)
        
        # Remove existing handlers to avoid duplicates
        ext_logger.handlers = []
        
        # Don't propagate to root logger (we handle everything explicitly)
        ext_logger.propagate = False
        
        # Add file handler - all ranks write to their respective log files
        file_handler = logging.FileHandler(_rank_log_file)
        if use_rank_format:
            file_handler.setFormatter(RankFormatter(rank=rank, use_color=False))
        else:
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s",
                datefmt="%H:%M:%S"
            ))
        file_handler.setLevel(level)
        ext_logger.addHandler(file_handler)
        
        # Add console handler for rank 0 only (matches your existing pattern)
        if rank == 0:
            console_handler = logging.StreamHandler(_original_stdout or sys.stdout)
            if use_rank_format:
                console_handler.setFormatter(RankFormatter(rank=rank, use_color=True))
            else:
                console_handler.setFormatter(logging.Formatter(
                    "%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S"
                ))
            console_handler.setLevel(level)
            ext_logger.addHandler(console_handler)
        
        configured_loggers.append(ext_logger)
    
    return configured_loggers


# =============================================================================
# Python Logging Setup
# =============================================================================

class RankFormatter(logging.Formatter):
    """Formatter that includes rank info and optional ANSI colors."""

    def __init__(self, fmt=None, datefmt=None, rank=None, use_color=True):
        self.rank = rank
        # Default to a structured format; allow override via fmt.
        default_fmt = "%(asctime)s [Rank %(rank)s] %(levelname)s %(message)s"
        basefmt = fmt if fmt else default_fmt
        super().__init__(basefmt, datefmt or "%H:%M:%S")
        self.use_color = use_color

    def format(self, record):
        # Copy to avoid mutating the original record when colorizing.
        record = logging.makeLogRecord(record.__dict__.copy())
        record.rank = self.rank
        formatted = super().format(record)
        if self.use_color:
            color = _COLOR_MAP.get(record.levelno)
            if color:
                formatted = f"{color}{formatted}{_COLOR_RESET}"
        return formatted


def setup_logging(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    level: int = logging.INFO,
    use_color: bool = True,
    logger_name: str = __name__,
    console_ranks: Optional[List[int]] = None,
) -> logging.Logger:
    """
    Initialize logging with rank-aware formatting.

    Args:
        rank: Process rank. If None, reads from RANK env var.
        world_size: Total number of processes. If None, reads from WORLD_SIZE env var.
        level: Logging level (default: INFO).
        use_color: Whether to use ANSI colors in output.
        logger_name: Name for the logger.
        console_ranks: List of ranks permitted to log to console. If None, all ranks
            log to console. If an empty list, no ranks log to console. Common usage:
            [0] for only rank 0, [0, 1] for ranks 0 and 1, etc.

    Returns:
        Configured logger instance.
    """
    global _rank, _world_size, _logger

    # Resolve rank and world_size
    if rank is None:
        _rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    else:
        _rank = rank

    if world_size is None:
        _world_size = int(os.environ.get("WORLD_SIZE", 1))
    else:
        _world_size = world_size

    _logger = logging.getLogger(logger_name)
    _logger.handlers = []
    _logger.setLevel(level)
    _logger.propagate = False  # Prevent duplicate logs from root logger

    # Determine if this rank should log to console
    console_ranks = console_ranks or [0]
    should_log_to_console = (console_ranks is None) or (_rank in console_ranks)
    if should_log_to_console:
        handler = logging.StreamHandler()
        handler.setFormatter(RankFormatter(rank=_rank, use_color=use_color))
        _logger.addHandler(handler)

    return _logger


# =============================================================================
# Debugging Utilities
# =============================================================================

def log_tensor_stats(tensor, name: str = "tensor", preview: int = 0, logger: logging.Logger = None):
    """
    Log quick stats for a tensor: shape, device, dtype, zeros, nans/infs, min/max/mean/std.
    Optionally log a small preview of the flattened values (first `preview` elements).

    Args:
        tensor: PyTorch tensor to inspect.
        name: Display name for the tensor.
        preview: Number of elements to preview (0 = no preview).
        logger: Logger to use. If None, uses the module logger.
    """
    import torch

    log = logger or get_logger()

    with torch.no_grad():
        total_elems = tensor.numel()
        zero_count = (tensor == 0).sum().item()

        if tensor.is_floating_point() or tensor.is_complex():
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            finite_mask = torch.isfinite(tensor)
            finite_vals = tensor[finite_mask]
            stats_source = finite_vals
        else:
            nan_count = 0
            inf_count = 0
            finite_vals = tensor
            # Mean/std on integer/boolean tensors require casting.
            stats_source = finite_vals.float()

        if finite_vals.numel() > 0:
            min_val = finite_vals.min().item()
            max_val = finite_vals.max().item()
            mean_val = stats_source.mean().item() if stats_source.numel() else float("nan")
            std_val = stats_source.std().item() if stats_source.numel() else float("nan")
        else:
            min_val = max_val = mean_val = std_val = float("nan")

        log.info(
            f"{name} -> shape: {tuple(tensor.shape)}, dtype: {tensor.dtype}, device: {tensor.device}, "
            f"total: {total_elems}, zeros: {zero_count}, nan: {nan_count}, inf: {inf_count}, "
            f"min: {min_val:.6f}, max: {max_val:.6f}, mean: {mean_val:.6f}, std: {std_val:.6f}"
        )
        if preview > 0 and total_elems > 0:
            preview_vals = tensor.flatten()[:preview].detach().cpu().tolist()
            log.info(f"{name} preview first {preview}: {preview_vals}")


def log_module(module, name: str = None, preview: int = 0, logger: logging.Logger = None):
    """
    Log module repr, parameters, buffers, and state_dict keys. Optional tensor previews.

    Args:
        module: PyTorch nn.Module to inspect.
        name: Display name for the module. Defaults to class name.
        preview: Number of elements to preview for each param/buffer (0 = no preview).
        logger: Logger to use. If None, uses the module logger.
    """
    log = logger or get_logger()
    mod_name = name or module.__class__.__name__

    log.info(f"{mod_name} repr:\n{module}")
    params = [
        (n, tuple(p.shape), str(p.dtype), p.device.type, p.requires_grad)
        for n, p in module.named_parameters(recurse=True)
    ]
    buffers = [
        (n, tuple(b.shape), str(b.dtype), b.device.type)
        for n, b in module.named_buffers(recurse=True)
    ]
    log.info(f"{mod_name} parameters: {params}")
    log.info(f"{mod_name} buffers: {buffers}")
    log.info(f"{mod_name} state_dict keys: {list(module.state_dict().keys())}")

    # Optional previews for module outputs of interest
    if preview > 0:
        for n, p in module.named_parameters(recurse=True):
            log_tensor_stats(p, name=f"{mod_name}.param.{n}", preview=preview, logger=log)
        for n, b in module.named_buffers(recurse=True):
            log_tensor_stats(b, name=f"{mod_name}.buffer.{n}", preview=preview, logger=log)


@contextmanager
def time_it(name: str, logger: logging.Logger = None):
    """
    Context manager to measure and log elapsed time for a code block.

    Args:
        name: Label for the timed block.
        logger: Logger to use. If None, uses the module logger.

    Usage:
        with time_it("model forward"):
            output = model(input)
    """
    log = logger or get_logger()
    start_time = time.time()
    yield
    end_time = time.time()
    log.info(f"{name} took {end_time - start_time:.4f} seconds")
