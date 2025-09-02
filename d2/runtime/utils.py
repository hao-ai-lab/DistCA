import torch

_CUDA_INT4_BYTES = 16


def size_pad_by_int4(hidden_size: int, itemsize: int):
    """
    Args:
        hidden_size: num elements
        itemsize: of each element
    Returns:
        hidden_size_pad: padded num elements
        pad_size: padding size, in number of elements
    """
    hidden_bytes = hidden_size * itemsize
    if hidden_bytes % _CUDA_INT4_BYTES != 0:
        hidden_bytes += _CUDA_INT4_BYTES - (hidden_bytes % _CUDA_INT4_BYTES)
    hidden_size_pad = hidden_bytes // itemsize
    pad_size = hidden_size_pad - hidden_size
    return hidden_size_pad, pad_size


def prepend_zero_fn(tensor: torch.Tensor, dim: int=0):
    zero = torch.zeros_like(tensor.select(dim, 0)).unsqueeze(dim)
    return torch.cat([zero, tensor], dim=dim)


def exclusive_cumsum(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Cumsum but excluding itself."""
    cumsum = tensor.cumsum(dim=dim)
    zero = torch.zeros_like(tensor.select(dim, 0))
    return torch.cat([zero.unsqueeze(dim), cumsum.narrow(dim, 0, cumsum.size(dim) - 1)], dim=dim)
