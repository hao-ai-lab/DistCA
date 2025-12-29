from typing import Dict, List, Optional

import torch

#### Tool functions for splitting and gathering args ####
def _split_tensor(x: Optional[torch.Tensor], num_splits: int):
    if x is None:
        return (None,) * num_splits
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"_split_tensor expected Tensor or None, got {type(x)}: {x}")
    if x.dim() == 0:
        # Scalar tensor - return num_splits copies
        return (x,) * num_splits
    if x.shape[0] < num_splits:
        raise ValueError(f"Cannot split tensor with shape[0]={x.shape[0]} into {num_splits} parts")
    # Use chunk to guarantee exactly num_splits chunks
    chunks = x.chunk(num_splits, dim=0)
    return tuple(chunks)

def repack_args(args: List[List[torch.Tensor]], num_splits: int):
    assert all(len(a) == num_splits for a in args)
    return [
        [a[i] for a in args]
        for i in range(num_splits)
    ]

def _repack_dicts(args: Dict[str, List[torch.Tensor]], num_splits: int):
    # Check which values don't have the expected length
    bad_keys = [k for k, v in args.items() if len(v) != num_splits]
    if bad_keys:
        bad_info = {k: (len(args[k]), type(args[k])) for k in bad_keys}
        raise AssertionError(
            f"Length of args must be {num_splits}, but got mismatched lengths. "
            f"Problematic keys: {bad_info}"
        )
    return [
        {k: a[i] for k, a in args.items()}
        for i in range(num_splits)
    ]

def splits_all(tensors: List[torch.Tensor], num_splits: int):
    splits = [_split_tensor(t, num_splits) for t in tensors]
    return repack_args(splits, num_splits)

def split_all_dict(tensors: Dict[str, torch.Tensor], num_splits: int):
    splits = {}
    for k, v in tensors.items():
        try:
            splits[k] = _split_tensor(v, num_splits)
        except Exception as e:
            raise ValueError(f"Error splitting tensor {k}: {e}") from e
    # splits = {k: _split_tensor(v, num_splits) for k, v in tensors.items()}
    return _repack_dicts(splits, num_splits)

def gather_tensor(tensors: List[torch.Tensor], num_splits: int):
    assert len(tensors) == num_splits
    if any(t is None for t in tensors):
        assert all(t is None for t in tensors), "None tensors in gather_tensor"
        return None
    return torch.cat(tensors, dim=0)
####
