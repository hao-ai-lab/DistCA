
import torch
from d2.runtime.attn_kernels.ops import nvshmem_get_unique_id, nvshmem_alloc_empty_unique_id, DispatcherWrapper

from d2.runtime.inplace_metadata import (
    Metadata, compute_metadata, compute_metadata_kv, compute_attn_layout_seqlens, exclusive_cumsum
)


def compute_e2e_metadata(
    mlp_seq_len: torch.Tensor,  # shape of (world_size, max_num_local_seqs)
    mlp_num_seqs: torch.Tensor,
    mlp_q_dispatch: torch.Tensor,  # shape of (world_size, max_num_local_seqs)
    kv_to_q_mapping: torch.Tensor,
    kv_to_q_rank: torch.Tensor,
    kv_context_size: torch.Tensor,
    q_to_num_kv_seq: torch.Tensor,
    q_to_num_kv_token: torch.Tensor,
    return_intermediate: bool = False
):
    """
    High level functions to compute all required metadata.
    """
    fwd_metadata_q, rev_metadata_q, q_intermediates = compute_metadata(
        mlp_seq_len, mlp_q_dispatch, return_intermediate=True
    )

    max_num_local_seqs = mlp_q_dispatch.shape[1]
    _, q_seq_to_dst, num_received_seqs_q = q_intermediates
    fwd_metadata_kv, rev_metadata_kv, kv_intermediates = compute_metadata_kv(
        kv_to_q_mapping, kv_to_q_rank, kv_context_size,
        q_to_num_kv_seq, q_to_num_kv_token, mlp_seq_len, mlp_num_seqs,
        mlp_q_dispatch, q_seq_to_dst, max_num_local_seqs,
        return_intermediate=True
    )

    (
        cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv,
        num_local_seqs_recv
    ) = compute_attn_layout_seqlens(
        mlp_seq_len, q_to_num_kv_token, mlp_q_dispatch, shard_to_tuple=True
    )
    fa_params = (cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)

    ret = fwd_metadata_q, rev_metadata_q, fwd_metadata_kv, rev_metadata_kv, fa_params
    if return_intermediate:
        ret += (q_intermediates + kv_intermediates,)
    return ret




import rich
# VERBOSE = True
VERBOSE = False
DEBUG = False

def print_if_verbose(*args, **kwargs):
    if VERBOSE:
        rich.print(*args, **kwargs)


def create_qkv_dispatch_2cp_with_explicit_sharding(
    world_size: int,
    seq_lens: torch.Tensor,
    cp_dst: torch.Tensor,
    cp_shards: torch.Tensor,
):
    """
    Create qkv with explicit sequence sharding.

    ----
    Arguments

    @param world_size: int
        The number of ranks in the context-parallel world.

    @param seq_lens: torch.Tensor
        seq_lens[src_rank, seq_id] = The length of the sequence.
        shape: (world_size, num_seqs)

    @param cp_dst: torch.Tensor
        cp_dst[src_rank, seq_id, shard_id] = dst_rank
            The destination rank of this sequence shard from (src_rank, seq_id)'s shard_id.
            The sequence shard's length is defined in `cp_shards`, a tensor of the same shape.
            If dst_rank == -1, it is a mask.
        shape: (world_size, num_seqs, max_cp_degree)

    @param cp_shards: torch.Tensor
        cp_shards[src_rank, seq_id, shard_id] = length
            The length of the sequence shard to send to `cp_dst[src_rank, seq_id, shard_id]`.
            If value == -1, it is a mask.
        shape: (world_size, num_seqs, max_cp_degree)

    ----
    Return values
    
    @return fwd_q_metadata: Metadata
    @return rev_q_metadata: Metadata
    @return fwd_k_metadata: Metadata
    @return rev_k_metadata: Metadata
    @return attention_metadata: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    
    ----
    Usage

    See test_util_cp_metadata.py for usage.

    """
    # init sequence    
    num_seqs: torch.Tensor = (seq_lens > 0).sum(dim=1).max()
    # max_num_seqs = num_seqs.max()
    
    cp_num = (cp_dst > -1).sum(dim=2)
    max_cp_degree: int = max(cp_num.max(), 1)
    
    # assert on seq_lens[rank, seq_id] = sequence length
    assert seq_lens.shape == (world_size, num_seqs)
    assert seq_lens.dtype == torch.int64

    # assert on cp_dst
    assert cp_dst.shape == (world_size, num_seqs, max_cp_degree)
    assert cp_shards.shape == (world_size, num_seqs, max_cp_degree)


    num_cp_shards = cp_num.sum(dim=1)
    pad_len = torch.max(num_cp_shards)
    cp_seq_lens = torch.zeros(world_size, pad_len, dtype=torch.int64)
    cp_query_dst = torch.ones(world_size, pad_len, dtype=torch.int64) * -1
    kv_to_q_mapping = torch.ones((world_size, pad_len, max_cp_degree, 2), dtype=torch.int64) * -1
    kv_to_q_rank = torch.ones((world_size, pad_len, max_cp_degree), dtype=torch.int64) * -1
    kv_context_size = torch.zeros((world_size, pad_len), dtype=torch.int64)
    num_kv_to_q = torch.zeros((world_size, pad_len), dtype=torch.int64)

    num_cul_cp_shards = exclusive_cumsum(cp_num, dim=1)


    for i in range(world_size):
        cp_seq_lens_local = []
        cp_query_dst_local = []
        kv_to_q_mapping_local = []
        kv_to_q_rank_local = []
        kv_context_size_local = []
        num_kv_to_q_local = []

        for j in range(num_seqs):
            num_cp = int((cp_num[i, j]).item())

            if num_cp <= 1:
                seq_len = seq_lens[i, j]
                if seq_len <= 0:
                    continue
                
                seq_shard_len = seq_len // num_cp
                _seq_shard_len = seq_shard_len.reshape(1,).repeat(num_cp)
            else:
                _seq_shard_len = cp_shards[i, j, :num_cp]

            # else:
            #     _seq_shard_len = seq_shard_len.reshape(1,).repeat(num_cp)
            #     unit_to_transfer = _seq_shard_len[0] // 2
            #     _seq_shard_len[0] -= unit_to_transfer
            #     _seq_shard_len[-1] += unit_to_transfer
            #     pass

            cp_seq_lens_local.append(_seq_shard_len)
            print_if_verbose(f"_seq_shard_len\n", _seq_shard_len, "\n")
            
            cp_query_dst_local.append(cp_dst[i, j, :num_cp].flatten())
            print_if_verbose(f"cp_dst[i, j, :num_cp].flatten():\n", cp_dst[i, j, :num_cp].flatten(), "\n")
            
            row_indices = torch.arange(num_cp).view(-1, 1)
            print_if_verbose(f"row_indices:\n", row_indices, "\n")
            
            col_indices = torch.arange(max_cp_degree).view(1, -1)
            print_if_verbose(f"col_indices:\n", col_indices, "\n")
            
            mask = col_indices < (num_cp - row_indices)
            print_if_verbose(f"mask:\n", mask, "\n")
            
            kv_to_q_mapping_seq = torch.empty((num_cp, max_cp_degree, 2), dtype=torch.int64)
            kv_to_q_mapping_seq[..., 0] = torch.where(mask, i, -1)
            print_if_verbose(f"kv_to_q_mapping_seq[..., 0]:\n", kv_to_q_mapping_seq[..., 0], "\n")
            
            vals_ch1 = row_indices + col_indices + num_cul_cp_shards[i, j]
            print_if_verbose(f"vals_ch1:\n", vals_ch1, "\n")
            
            kv_to_q_mapping_seq[..., 1] = torch.where(mask, vals_ch1, -1)
            print_if_verbose(f"kv_to_q_mapping_seq[..., 1]:\n", kv_to_q_mapping_seq[..., 1], "\n")
            
            kv_to_q_mapping_local.append(kv_to_q_mapping_seq)
            
            #### Compute kv_to_q_rank (Index of this KV to the query's dst).
            # < In: num_cp, max_cp_degree, mask
            # > Out: rank indices for KV to query communication with masking
            kv_to_q_rank_seq = torch.arange(num_cp).view(-1, 1).repeat(1, max_cp_degree) * mask + (mask.int() - 1)
            print_if_verbose(f"kv_to_q_rank_seq:\n", kv_to_q_rank_seq, "\n")
            kv_to_q_rank_local.append(kv_to_q_rank_seq)
            #### Compute kv context size (For this kv, how many tokens are in the context).
            # TODO(GindaChen): `kv_context_size_seq` - Insert the proper kv_context_size_seq for the context parallel size
            if num_cp <= 1:
                kv_context_size_seq = torch.arange(num_cp) * seq_shard_len
            else:
                kv_context_size_seq = exclusive_cumsum(_seq_shard_len, dim=0)
            print_if_verbose(f"kv_context_size_seq:\n", kv_context_size_seq, "\n")
            kv_context_size_local.append(kv_context_size_seq)
            #### Compute num_kv_to_q (For this kv, how many shards are in the context).
            num_kv_to_q_seq = torch.arange(num_cp) + 1
            print_if_verbose(f"num_kv_to_q_seq:\n", num_kv_to_q_seq, "\n")
            num_kv_to_q_local.append(num_kv_to_q_seq)
            
            # breakpoint()

        if len(cp_seq_lens_local) == 0:
            continue
    
        cp_seq_lens_local = torch.cat(cp_seq_lens_local, dim=0)
        cp_query_dst_local = torch.cat(cp_query_dst_local, dim=0)
        kv_to_q_mapping_local = torch.cat(kv_to_q_mapping_local, dim=0)
        kv_to_q_rank_local = torch.cat(kv_to_q_rank_local, dim=0)
        kv_context_size_local = torch.cat(kv_context_size_local, dim=0)
        num_kv_to_q_local = torch.cat(num_kv_to_q_local, dim=0)

        seq_shards = cp_seq_lens_local.shape[0]
        assert cp_seq_lens_local.shape == (seq_shards,)
        assert cp_query_dst_local.shape == (seq_shards,)
        assert kv_to_q_mapping_local.shape == (seq_shards, max_cp_degree, 2)
        assert kv_to_q_rank_local.shape == (seq_shards, max_cp_degree)
        assert kv_context_size_local.shape == (seq_shards,)
        assert num_kv_to_q_local.shape == (seq_shards,)

        cp_seq_lens[i, :seq_shards] = cp_seq_lens_local
        cp_query_dst[i, :seq_shards] = cp_query_dst_local
        kv_to_q_mapping[i, :seq_shards] = kv_to_q_mapping_local
        kv_to_q_rank[i, :seq_shards] = kv_to_q_rank_local
        kv_context_size[i, :seq_shards] = kv_context_size_local
        num_kv_to_q[i, :seq_shards] = num_kv_to_q_local

    print_if_verbose(f"cp_seq_lens: ", cp_seq_lens)
    print_if_verbose(f"cp_query_dst: ", cp_query_dst)
    print_if_verbose(f"kv_to_q_mapping: ", kv_to_q_mapping)
    print_if_verbose(f"kv_to_q_rank: ", kv_to_q_rank)
    print_if_verbose(f"kv_context_size: ", kv_context_size)
    print_if_verbose(f"num_kv_to_q: ", num_kv_to_q)
    print_if_verbose(f"num_cp_shards: ", num_cp_shards)

    num_total_kv_to_q = kv_context_size + cp_seq_lens
    print_if_verbose(f"num_total_kv_to_q: ", num_total_kv_to_q)

    fwd_q_metadata, rev_q_metadata, intermediates = compute_metadata(
        cp_seq_lens, cp_query_dst, return_intermediate=True
    )
    _, q_seq_to_dst, _ = intermediates
    fwd_k_metadata, rev_k_metadata = compute_metadata_kv(
        kv_to_q_mapping, kv_to_q_rank, kv_context_size, num_kv_to_q,
        num_total_kv_to_q, cp_seq_lens, num_cp_shards, cp_query_dst,
        q_seq_to_dst.squeeze(2), pad_len
    )
    attention_metadata = compute_attn_layout_seqlens(
        cp_seq_lens, num_total_kv_to_q, cp_query_dst
    )

    
    if DEBUG: breakpoint()

    print_if_verbose(f"fwd_q_metadata: ", fwd_q_metadata)
    print_if_verbose(f"rev_q_metadata: ", rev_q_metadata)
    print_if_verbose(f"fwd_k_metadata: ", fwd_k_metadata)
    print_if_verbose(f"rev_k_metadata: ", rev_k_metadata)
    print_if_verbose(f"attention_metadata: ", attention_metadata)

    if DEBUG: breakpoint()
    
    return (
        fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata, attention_metadata
    )
