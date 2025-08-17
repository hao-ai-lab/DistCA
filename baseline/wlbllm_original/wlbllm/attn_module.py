try:
    from flash_attn.flash_attn_interface import (
        flash_attn_varlen_func, 
        _flash_attn_varlen_forward,
        _flash_attn_varlen_backward,
    )
except ImportError:
    print("flash_attn not found, using vllm-flash-attn")
    from vllm_flash_attn.flash_attn_interface import (
        flash_attn_varlen_func, 
        _flash_attn_varlen_forward,
        _flash_attn_varlen_backward,
    )

__all__ = [
    "flash_attn_varlen_func",
    "_flash_attn_varlen_forward",
    "_flash_attn_varlen_backward",
]