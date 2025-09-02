import os

import torch

from wlbllm.utils import doc_shard


_lib_path = os.path.join(os.path.dirname(__file__), "libwlb_shuffle.so")
torch.ops.load_library(_lib_path)
_ops = torch.ops.wlbmemcpy_kernels


def _exclusive_cumsum(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Cumsum but excluding itself."""
    cumsum = tensor.cumsum(dim=dim)
    zero = torch.zeros_like(tensor.select(dim, 0))
    return torch.cat([zero.unsqueeze(dim), cumsum.narrow(dim, 0, cumsum.size(dim) - 1)], dim=dim)


def prepare_metadata(doc_shards: list[list[doc_shard]], device: torch.device,
                     cp_size: int, chunk_size: int):
    assert len(doc_shards) == cp_size * 2
    # shape (num_chunks, num_docs)
    chunk_shard_lens = torch.tensor([
        [ds.shard_len if ds is not None else 0 for ds in cds]
        for cds in doc_shards
    ], dtype=torch.uint64, device=device)

    # compute the current shard's offset at dst layout
    intra_chunk_offset = _exclusive_cumsum(chunk_shard_lens, dim=1)
    # here we use chunk_size instead of sum(shard_lens_on_rank) because there are padding values.
    chunk_offset = chunk_size * torch.arange(0, cp_size * 2, device=device, dtype=torch.uint64)
    shard_src_offset = intra_chunk_offset + chunk_offset.unsqueeze(1)

    # return value is in the (docs, chunks) layout
    chunk_shard_lens = chunk_shard_lens.transpose(0, 1).contiguous()
    shard_src_offset = shard_src_offset.transpose(0, 1).contiguous()

    # total number of tokens. This removes all padding
    num_tokens = int(chunk_shard_lens.sum().item())
    return chunk_shard_lens, shard_src_offset, num_tokens


def prepare_tensor(k_tensor_list: list[torch.Tensor], context_length: int, cp_size: int):
    chunk_size = context_length // (2 * cp_size)
    assert len(k_tensor_list) == cp_size
    k_tensors = [torch.split(k, chunk_size, dim=0) for k in k_tensor_list]
    # sort by the chunk index
    k_tensors = [ks[0] for ks in k_tensors] + list(reversed([ks[1] for ks in k_tensors]))
    # concat to shape (num_chunks, chunk_size, hidden)
    k_tensors = [k.unsqueeze(0) for k in k_tensors]
    k_tensor = torch.cat(k_tensors, dim=0).contiguous()
    return k_tensor


def gathered_shuffled_switch(
    gathered_tensor: torch.Tensor, shuffled_tensor: torch.Tensor,
    shard_lens: torch.Tensor, shard_gathered_offsets: torch.Tensor, is_from_gathered: bool
):
    _ops.wlb_shuffle_memcpy(gathered_tensor, shuffled_tensor,
                            shard_lens, shard_gathered_offsets,
                            is_from_gathered)


def _shuffle_per_doc_cp(context_length: int, tensor_list, cp_size: int,
                        chunk_shard_lens: torch.Tensor,
                        shard_src_offset: torch.Tensor, num_tokens: int):
    gathered_tensor = prepare_tensor(tensor_list, context_length, cp_size)
    out_shape = (num_tokens,) + gathered_tensor.shape[2:]
    gathered_tensor = gathered_tensor.flatten(start_dim=2)
    shuffled_tensor = torch.empty(out_shape, dtype=gathered_tensor.dtype, device=gathered_tensor.device)
    shuffled_tensor = shuffled_tensor.flatten(start_dim=1)

    gathered_shuffled_switch(gathered_tensor, shuffled_tensor, chunk_shard_lens, shard_src_offset,
                             is_from_gathered=True)
    shuffled_tensor = shuffled_tensor.reshape(out_shape)
    return shuffled_tensor


def kv_shuffle_for_per_doc_cp_fast(
    context_length, k_tensor_list, v_tensor_list, cp_size,
    chunk_shard_lens, shard_src_offset, num_tokens
):
    k_global = _shuffle_per_doc_cp(context_length, k_tensor_list, cp_size, chunk_shard_lens, shard_src_offset, num_tokens)
    if v_tensor_list is not None:
        v_global = _shuffle_per_doc_cp(context_length, v_tensor_list, cp_size, chunk_shard_lens, shard_src_offset, num_tokens)
    else:
        v_global = None
    return k_global, v_global


def _unshuffle_per_doc_cp(context_length: int, grad_shuffled: torch.Tensor, cp_size: int,
                          chunk_shard_lens: torch.Tensor, shard_src_offset: torch.Tensor):
    chunk_size = context_length // (2 * cp_size)
    gathered_tensor_shape = (cp_size * 2, chunk_size) + grad_shuffled.shape[1:]
    gathered_tensor = torch.empty(gathered_tensor_shape, dtype=grad_shuffled.dtype, device=grad_shuffled.device)
    grad_shuffled = grad_shuffled.flatten(start_dim=1)
    gathered_tensor = gathered_tensor.flatten(start_dim=1)
    gathered_shuffled_switch(
        gathered_tensor, grad_shuffled, chunk_shard_lens, shard_src_offset,
        is_from_gathered=False
    )
    gathered_tensor = gathered_tensor.reshape(gathered_tensor_shape)
    per_rank_tensor = [
        torch.concat((gathered_tensor[rank], gathered_tensor[2 * cp_size - 1 - rank])).contiguous()
        for rank in cp_size
    ]
    return per_rank_tensor


def kv_unshuffle_for_per_doc_cp_fast(
    context_length, grad_k, grad_v, cp_size,
    chunk_shard_lens, shard_src_offset
):
    grad_k_list = _unshuffle_per_doc_cp(context_length, grad_k, cp_size, chunk_shard_lens, shard_src_offset)
    if grad_v is not None:
        grad_v_list = _unshuffle_per_doc_cp(context_length, grad_v, cp_size, chunk_shard_lens, shard_src_offset)
    else:
        grad_v_list = None
    return grad_k_list, grad_v_list
