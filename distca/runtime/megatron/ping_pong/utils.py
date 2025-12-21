from typing import Dict, List, Optional

import torch

#### Tool functions for splitting and gathering args ####
def _split_tensor(x: Optional[torch.Tensor], num_splits: int):
    if x is None:
        return (None,) * num_splits
    return x.split(x.shape[0] // num_splits, dim=0)

def repack_args(args: List[List[torch.Tensor]], num_splits: int):
    assert all(len(a) == num_splits for a in args)
    return [
        [a[i] for a in args]
        for i in range(num_splits)
    ]

def _repack_dicts(args: Dict[str, List[torch.Tensor]], num_splits: int):
    assert all(len(a) == num_splits for a in args.values()), f"Length of args must be {num_splits}, but got {len(args.values())}"
    return [
        {k: a[i] for k, a in args.items()}
        for i in range(num_splits)
    ]

def splits_all(tensors: List[torch.Tensor], num_splits: int):
    splits = [_split_tensor(t, num_splits) for t in tensors]
    return repack_args(splits, num_splits)

def split_all_dict(tensors: Dict[str, torch.Tensor], num_splits: int):
    splits = {k: _split_tensor(v, num_splits) for k, v in tensors.items()}
    return _repack_dicts(splits, num_splits)

def gather_tensor(tensors: List[torch.Tensor], num_splits: int):
    assert len(tensors) == num_splits
    if any(t is None for t in tensors):
        assert all(t is None for t in tensors), "None tensors in gather_tensor"
        return None
    return torch.cat(tensors, dim=0)
####
