"""
NaN Detection Hooks for PyTorch Models

This module provides utilities to register forward hooks on PyTorch models
to detect NaN values during forward passes.
"""

import logging
import torch

# ANSI color codes
_COLOR_BLUE = "\033[34m"
_COLOR_GREEN = "\033[32m"
_COLOR_RED = "\033[31m"
_COLOR_RESET = "\033[0m"


def _colorize_text(text, color_code):
    """Add ANSI color codes to text"""
    return f"{color_code}{text}{_COLOR_RESET}"


def _check_weights_for_zeros(module, logger, name):
    """Check if module weights are all zeros"""
    if hasattr(module, 'weight') and module.weight is not None:
        weight = module.weight
        zero_count = (weight == 0).sum().item()
        zero_ratio = zero_count / weight.numel()
        if zero_ratio > 0.5:  # More than 50% zeros
            msg = (
                f"[ZERO WEIGHTS] {name}: weight has {zero_count}/{weight.numel()} zeros "
                f"({zero_ratio:.4f} ratio), shape={weight.shape}"
            )
            logger.warning(_colorize_text(msg, _COLOR_RED))
        if zero_ratio == 1.0:  # All zeros
            msg = (
                f"[ALL ZERO WEIGHTS] {name}: weight is completely zero! "
                f"shape={weight.shape}, dtype={weight.dtype}"
            )
            logger.error(_colorize_text(msg, _COLOR_RED))


def _get_tensor_stats(tensor, name="tensor", show_start_end=False, num_elements=5):
    """
    Get statistics for a tensor.
    
    Args:
        tensor: The tensor to analyze
        name: Name for logging
        show_start_end: If True, show first and last few elements
        num_elements: Number of elements to show at start/end
    
    Returns:
        Dictionary with statistics
    """
    try:
        stats = {
            'shape': tensor.shape,
            'dtype': tensor.dtype,
        }
        
        # Check if tensor supports NaN/Inf (floating point or complex)
        is_floating = tensor.dtype.is_floating_point or tensor.dtype.is_complex
        if is_floating:
            stats['nan_count'] = torch.isnan(tensor).sum().item()
            stats['inf_count'] = torch.isinf(tensor).sum().item()
        else:
            stats['nan_count'] = 0
            stats['inf_count'] = 0
        
        # Count zeros (exact zero, not just close to zero)
        stats['zero_count'] = (tensor == 0).sum().item()
        
        # Compute stats only if tensor has elements
        if tensor.numel() > 0:
            # Calculate zero ratio
            stats['zero_ratio'] = stats['zero_count'] / tensor.numel()
            # For integer tensors, we can still compute min/max but not mean/std
            stats['min'] = tensor.min().item()
            stats['max'] = tensor.max().item()
            
            # Mean and std only work for floating point/complex types
            if is_floating:
                # Use nan-aware operations for safety
                finite_mask = torch.isfinite(tensor)
                if finite_mask.any():
                    finite_tensor = tensor[finite_mask]
                    stats['mean'] = finite_tensor.mean().item()
                    stats['std'] = finite_tensor.std().item()
                else:
                    stats['mean'] = float('nan')
                    stats['std'] = float('nan')
            else:
                # For integer tensors, compute mean/std by converting to float
                try:
                    float_tensor = tensor.float()
                    stats['mean'] = float_tensor.mean().item()
                    stats['std'] = float_tensor.std().item()
                except Exception:
                    stats['mean'] = 'N/A (integer tensor)'
                    stats['std'] = 'N/A (integer tensor)'
            
            if show_start_end:
                # Flatten for easier indexing
                flat = tensor.flatten()
                num_show = min(num_elements, flat.numel())
                stats['start_elements'] = flat[:num_show].tolist()
                stats['end_elements'] = flat[-num_show:].tolist()
        else:
            stats['min'] = float('nan')
            stats['max'] = float('nan')
            stats['mean'] = float('nan')
            stats['std'] = float('nan')
        
        return stats
    except Exception as e:
        return {
            'shape': tensor.shape if hasattr(tensor, 'shape') else 'unknown',
            'dtype': tensor.dtype if hasattr(tensor, 'dtype') else 'unknown',
            'error': str(e)
        }


def _format_tensor_stats(stats, prefix=""):
    """Format tensor statistics as a string"""
    if 'error' in stats:
        return f"{prefix}Error computing stats: {stats['error']}"
    
    lines = [
        f"{prefix}shape={stats['shape']}, dtype={stats['dtype']}",
    ]
    
    # Format numeric stats, handling NaN values and non-numeric types
    def format_num(val):
        if isinstance(val, str):
            return val
        if isinstance(val, float) and (val != val or val == float('inf') or val == float('-inf')):
            return f"{val}"
        if isinstance(val, (int, float)):
            return f"{val:.6f}"
        return str(val)
    
    if 'min' in stats and 'max' in stats and 'mean' in stats and 'std' in stats:
        lines.append(
            f"{prefix}min={format_num(stats['min'])}, max={format_num(stats['max'])}, "
            f"mean={format_num(stats['mean'])}, std={format_num(stats['std'])}"
        )
    
    if 'nan_count' in stats and 'inf_count' in stats:
        zero_info = ""
        if 'zero_count' in stats:
            if 'zero_ratio' in stats:
                zero_info = f", zero_count={stats['zero_count']} ({stats['zero_ratio']:.4f})"
            else:
                zero_info = f", zero_count={stats['zero_count']}"
        lines.append(f"{prefix}nan_count={stats['nan_count']}, inf_count={stats['inf_count']}{zero_info}")
    
    if 'start_elements' in stats:
        lines.append(f"{prefix}start={stats['start_elements']}")
        lines.append(f"{prefix}end={stats['end_elements']}")
    
    return "; ".join(lines)


def register_nan_detection_hooks(
    model, 
    logger=None, 
    raise_on_nan=False, 
    check_inputs=False,
    log_all_modules=False,
    log_tensor_stats=False,
    tensor_stats_elements=5
):
    """
    Register forward hooks on all modules to detect NaN values.
    
    Args:
        model: The model to register hooks on
        logger: Logger instance for output (defaults to module logger)
        raise_on_nan: If True, raise exception when NaN is detected (default: False)
        check_inputs: If True, also check inputs to each module (default: False)
        log_all_modules: If True, log all modules even when they don't have NaN (default: False)
        log_tensor_stats: If True, print tensor statistics (min, max, mean, std, start/end) (default: False)
        tensor_stats_elements: Number of elements to show at start/end when log_tensor_stats=True (default: 5)
    
    Returns:
        Tuple of (hooks, nan_detected):
            - hooks: List of hook handles that can be used to remove hooks later
            - nan_detected: Dictionary with 'value' key indicating if NaN was detected
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    hooks = []
    nan_detected = {'value': False}  # Use dict to allow modification in nested function
    
    def make_nan_hook(name, module_name):
        """Create a hook function for a specific module"""
        def hook_fn(module, input, output):
            has_nan = False
            
            # Check inputs if requested
            if check_inputs:
                for i, inp in enumerate(input):
                    if isinstance(inp, torch.Tensor):
                        if torch.isnan(inp).any():
                            nan_count = torch.isnan(inp).sum().item()
                            msg = (
                                f"[NaN DETECTED] INPUT[{i}] to {name} ({module_name}): "
                                f"shape={inp.shape}, nan_count={nan_count}, "
                                f"nan_ratio={nan_count/inp.numel():.4f}, "
                                f"min={inp.min().item():.6f}, max={inp.max().item():.6f}"
                            )
                            logger.error(_colorize_text(msg, _COLOR_RED))
                            if log_tensor_stats:
                                stats = _get_tensor_stats(inp, f"INPUT[{i}]", show_start_end=True, num_elements=tensor_stats_elements)
                                msg = f"[NaN DETECTED] INPUT[{i}] stats: {_format_tensor_stats(stats, '  ')}"
                                logger.error(_colorize_text(msg, _COLOR_RED))
                            nan_detected['value'] = True
                            has_nan = True
                            if raise_on_nan:
                                raise RuntimeError(f"NaN detected in input[{i}] to {name}")
                        elif log_all_modules or log_tensor_stats:
                            # Log non-NaN inputs if requested
                            if log_tensor_stats:
                                try:
                                    stats = _get_tensor_stats(inp, f"INPUT[{i}]", show_start_end=True, num_elements=tensor_stats_elements)
                                    if 'error' not in stats:
                                        msg = f"[OK] INPUT[{i}] to {name} ({module_name}): {_format_tensor_stats(stats, '  ')}"
                                        logger.info(_colorize_text(msg, _COLOR_GREEN))
                                    else:
                                        # Fallback to simple logging if stats computation fails
                                        msg = f"[OK] INPUT[{i}] to {name} ({module_name}): shape={inp.shape}, dtype={inp.dtype}, no NaN"
                                        logger.info(_colorize_text(msg, _COLOR_GREEN))
                                except Exception as e:
                                    msg = f"[OK] INPUT[{i}] to {name} ({module_name}): shape={inp.shape}, dtype={inp.dtype}, no NaN (stats error: {e})"
                                    logger.info(_colorize_text(msg, _COLOR_GREEN))
                            elif log_all_modules:
                                msg = f"[OK] INPUT[{i}] to {name} ({module_name}): shape={inp.shape}, no NaN"
                                logger.info(_colorize_text(msg, _COLOR_GREEN))
            
            # Check outputs
            if isinstance(output, torch.Tensor):
                # Check for all-zero outputs (potential issue)
                zero_count = (output == 0).sum().item()
                zero_ratio = zero_count / output.numel() if output.numel() > 0 else 0.0
                if zero_ratio == 1.0:  # All zeros
                    msg = (
                        f"[ALL ZERO OUTPUT] {name} ({module_name}): "
                        f"shape={output.shape}, dtype={output.dtype}"
                    )
                    logger.error(_colorize_text(msg, _COLOR_RED))
                    # Check weights if this is a linear layer
                    _check_weights_for_zeros(module, logger, name)
                    # Check inputs
                    if check_inputs and len(input) > 0:
                        for i, inp in enumerate(input):
                            if isinstance(inp, torch.Tensor):
                                inp_zero_count = (inp == 0).sum().item()
                                inp_zero_ratio = inp_zero_count / inp.numel() if inp.numel() > 0 else 0.0
                                msg = (
                                    f"[ALL ZERO OUTPUT] INPUT[{i}] to {name}: "
                                    f"zero_ratio={inp_zero_ratio:.4f}, "
                                    f"shape={inp.shape}, dtype={inp.dtype}"
                                )
                                logger.error(_colorize_text(msg, _COLOR_RED))
                
                if torch.isnan(output).any():
                    nan_count = torch.isnan(output).sum().item()
                    msg = (
                        f"[NaN DETECTED] OUTPUT from {name} ({module_name}): "
                        f"shape={output.shape}, nan_count={nan_count}, "
                        f"nan_ratio={nan_count/output.numel():.4f}, "
                        f"min={output.min().item():.6f}, max={output.max().item():.6f}, "
                        f"mean={output.nanmean().item():.6f}"
                    )
                    logger.error(_colorize_text(msg, _COLOR_RED))
                    if log_tensor_stats:
                        stats = _get_tensor_stats(output, "OUTPUT", show_start_end=True, num_elements=tensor_stats_elements)
                        msg = f"[NaN DETECTED] OUTPUT stats: {_format_tensor_stats(stats, '  ')}"
                        logger.error(_colorize_text(msg, _COLOR_RED))
                    nan_detected['value'] = True
                    has_nan = True
                    if raise_on_nan:
                        raise RuntimeError(f"NaN detected in output from {name}")
                elif log_all_modules or log_tensor_stats:
                    # Log non-NaN outputs if requested
                    if log_tensor_stats:
                        try:
                            stats = _get_tensor_stats(output, "OUTPUT", show_start_end=True, num_elements=tensor_stats_elements)
                            if 'error' not in stats:
                                msg = f"[OK] OUTPUT from {name} ({module_name}): {_format_tensor_stats(stats, '  ')}"
                                logger.info(_colorize_text(msg, _COLOR_GREEN))
                            else:
                                # Fallback to simple logging if stats computation fails
                                msg = f"[OK] OUTPUT from {name} ({module_name}): shape={output.shape}, dtype={output.dtype}, no NaN"
                                logger.info(_colorize_text(msg, _COLOR_GREEN))
                        except Exception as e:
                            msg = f"[OK] OUTPUT from {name} ({module_name}): shape={output.shape}, dtype={output.dtype}, no NaN (stats error: {e})"
                            logger.info(_colorize_text(msg, _COLOR_GREEN))
                    elif log_all_modules:
                        msg = f"[OK] OUTPUT from {name} ({module_name}): shape={output.shape}, no NaN"
                        logger.info(_colorize_text(msg, _COLOR_GREEN))
            elif isinstance(output, (tuple, list)):
                for i, out in enumerate(output):
                    # Skip None values (e.g., bias when skip_bias_add=True)
                    if out is None:
                        continue
                    if isinstance(out, torch.Tensor):
                        # Check for all-zero outputs (potential issue)
                        zero_count = (out == 0).sum().item()
                        zero_ratio = zero_count / out.numel() if out.numel() > 0 else 0.0
                        if zero_ratio == 1.0:  # All zeros
                            # Determine if this is likely a bias tensor (1D with small size)
                            is_likely_bias = len(out.shape) == 1 and out.shape[0] < 10000
                            bias_note = " (likely bias tensor)" if is_likely_bias else ""
                            
                            msg = (
                                f"[ALL ZERO OUTPUT] OUTPUT[{i}] from {name} ({module_name}): "
                                f"shape={out.shape}, dtype={out.dtype}{bias_note}"
                            )
                            logger.error(_colorize_text(msg, _COLOR_RED))
                            
                            # Only check weights/inputs if this is NOT likely a bias tensor
                            # (bias tensors being zero is less concerning)
                            if not is_likely_bias:
                                # Check weights if this is a linear layer
                                _check_weights_for_zeros(module, logger, name)
                                # Check inputs
                                if check_inputs and len(input) > 0:
                                    for j, inp in enumerate(input):
                                        if isinstance(inp, torch.Tensor):
                                            inp_zero_count = (inp == 0).sum().item()
                                            inp_zero_ratio = inp_zero_count / inp.numel() if inp.numel() > 0 else 0.0
                                            msg = (
                                                f"[ALL ZERO OUTPUT] INPUT[{j}] to {name}: "
                                                f"zero_ratio={inp_zero_ratio:.4f}, "
                                                f"shape={inp.shape}, dtype={inp.dtype}"
                                            )
                                            logger.error(_colorize_text(msg, _COLOR_RED))
                            else:
                                msg = (
                                    f"[ALL ZERO OUTPUT] OUTPUT[{i}] appears to be a bias tensor. "
                                    f"If skip_bias_add=True, this is expected and not a problem."
                                )
                                logger.warning(_colorize_text(msg, _COLOR_RED))
                        
                        if torch.isnan(out).any():
                            nan_count = torch.isnan(out).sum().item()
                            msg = (
                                f"[NaN DETECTED] OUTPUT[{i}] from {name} ({module_name}): "
                                f"shape={out.shape}, nan_count={nan_count}, "
                                f"nan_ratio={nan_count/out.numel():.4f}, "
                                f"min={out.min().item():.6f}, max={out.max().item():.6f}, "
                                f"mean={out.nanmean().item():.6f}"
                            )
                            logger.error(_colorize_text(msg, _COLOR_RED))
                            if log_tensor_stats:
                                stats = _get_tensor_stats(out, f"OUTPUT[{i}]", show_start_end=True, num_elements=tensor_stats_elements)
                                msg = f"[NaN DETECTED] OUTPUT[{i}] stats: {_format_tensor_stats(stats, '  ')}"
                                logger.error(_colorize_text(msg, _COLOR_RED))
                            nan_detected['value'] = True
                            has_nan = True
                            if raise_on_nan:
                                raise RuntimeError(f"NaN detected in output[{i}] from {name}")
                        elif log_all_modules or log_tensor_stats:
                            # Log non-NaN outputs if requested
                            if log_tensor_stats:
                                try:
                                    stats = _get_tensor_stats(out, f"OUTPUT[{i}]", show_start_end=True, num_elements=tensor_stats_elements)
                                    if 'error' not in stats:
                                        msg = f"[OK] OUTPUT[{i}] from {name} ({module_name}): {_format_tensor_stats(stats, '  ')}"
                                        logger.info(_colorize_text(msg, _COLOR_GREEN))
                                    else:
                                        # Fallback to simple logging if stats computation fails
                                        msg = f"[OK] OUTPUT[{i}] from {name} ({module_name}): shape={out.shape}, dtype={out.dtype}, no NaN"
                                        logger.info(_colorize_text(msg, _COLOR_GREEN))
                                except Exception as e:
                                    msg = f"[OK] OUTPUT[{i}] from {name} ({module_name}): shape={out.shape}, dtype={out.dtype}, no NaN (stats error: {e})"
                                    logger.info(_colorize_text(msg, _COLOR_GREEN))
                            elif log_all_modules:
                                msg = f"[OK] OUTPUT[{i}] from {name} ({module_name}): shape={out.shape}, no NaN"
                                logger.info(_colorize_text(msg, _COLOR_GREEN))
            
            # Log module entry if log_all_modules is enabled and no NaN was found
            if log_all_modules and not has_nan:
                msg = f"[MODULE] {name} ({module_name}): executed successfully, no NaN"
                logger.info(_colorize_text(msg, _COLOR_BLUE))
        
        return hook_fn
    
    def register_hooks_recursive(module, prefix=""):
        """Recursively register hooks on all submodules"""
        # Skip if module is not a PyTorch module (e.g., list, tuple, etc.)
        if not isinstance(module, torch.nn.Module):
            logger.warning(f"Skipping {prefix}: not a PyTorch module (type: {type(module)})")
            return
        
        # Register hook on current module
        module_name = type(module).__name__
        full_name = f"{prefix}.{module_name}" if prefix else module_name
        
        try:
            hook = module.register_forward_hook(make_nan_hook(full_name, module_name))
            hooks.append(hook)
        except Exception as e:
            logger.warning(f"Failed to register hook on {full_name}: {e}")
            return
        
        # Recursively register on children
        try:
            for name, child in module.named_children():
                child_prefix = f"{full_name}.{name}" if prefix else name
                logger.info(f"Registering hooks on {child_prefix}")
                register_hooks_recursive(child, child_prefix)
        except Exception as e:
            logger.warning(f"Failed to register hooks on children of {full_name}: {e}")
    
    # Unwrap model if needed
    try:
        from distca.utils.megatron_test_utils import unwrap_model
    except ImportError:
        # Fallback if distca is not available
        def unwrap_model(model):
            return model
    
    unwrapped_model = unwrap_model(model)
    
    # Handle both single model and list of models (for virtual pipeline parallelism)
    if isinstance(unwrapped_model, list):
        logger.info(f"Model is a list with {len(unwrapped_model)} modules, registering hooks on all")
        for i, model_module in enumerate(unwrapped_model):
            logger.info(f"Registering hooks on model module {i}")
            register_hooks_recursive(model_module, prefix=f"model[{i}]")
    else:
        # Single model
        register_hooks_recursive(unwrapped_model)
    
    logger.info(f"Registered {len(hooks)} NaN detection hooks on model")
    
    return hooks, nan_detected


def remove_hooks(hooks, logger=None):
    """
    Remove all registered hooks.
    
    Args:
        hooks: List of hook handles returned from register_nan_detection_hooks
        logger: Logger instance for output (defaults to module logger)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    for hook in hooks:
        hook.remove()
    logger.info(f"Removed {len(hooks)} hooks")

