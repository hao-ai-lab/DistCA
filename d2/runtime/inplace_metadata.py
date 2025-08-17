"""
Metadata about dispatching strategy.
NOTE: with fast all2all display, this strategy is only logical.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from megatron.core.packed_seq_params import PackedSeqParams

"""
NOTE:
There is an implicit global index for all sequence shards.
The index is a flatten (rank, seq_shard_id, dispatch_id) on FFN layout.
For query, dispatch_id is 0.

For query, Attention layout, the local order of the shards is determined by
their global index.

For key-value, Attention layout, the local order is determined by the query's
order. Hence, it needs to know a Q-shard and KV-shard correlation.
"""

def exclusive_cumsum(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Cumsum but excluding itself."""
    cumsum = tensor.cumsum(dim=dim)
    zero = torch.zeros_like(tensor.select(dim, 0))
    return torch.cat([zero.unsqueeze(dim), cumsum.narrow(dim, 0, cumsum.size(dim) - 1)], dim=dim)


def mask_by_neg(tensor: torch.Tensor, mask: torch.Tensor):
    """Apply the mask with masking value -1."""
    return tensor * mask.bool() + mask.int() - 1


def index_put_with_neg_padding_1d(
    tensor: torch.Tensor, src: torch.Tensor, index: torch.Tensor
):
    """
    Handling the case that index value -1 means padding, which should not
    write to the tensor.
    """
    tensor_shape = tensor.shape
    tensor = tensor.flatten()
    src = src.flatten()
    index = index.flatten()
    tensor = torch.concat([tensor, torch.zeros([1], device=tensor.device, dtype=tensor.dtype)], dim=0)
    tensor.index_put_((index,), src, accumulate=False)
    return tensor[:-1].reshape(tensor_shape)  # remove the padding value at the end


def prepend_zero_fn(tensor: torch.Tensor, dim: int=0):
    zero = torch.zeros_like(tensor.select(dim, 0)).unsqueeze(dim)
    return torch.cat([zero, tensor], dim=dim)


@dataclass
class Metadata:
    # Rank, offset, and length described local sequence index.
    dst_rank: torch.Tensor
    dst_offset: torch.Tensor
    seq_len: torch.Tensor
    # num tokens received from each rank.
    num_recv_tokens: torch.Tensor
    # NOTE: used for kv gradient communication.
    # This is a sequence-scale mask, recording whether a sequence is received
    # or should be skipped.
    seq_recv_mask: Optional[torch.Tensor] = None
    # NOTE: used for kv gradient communication.
    # This records the length of all sequences received to this rank.
    recv_seq_lens: Optional[torch.Tensor] = None
    # Number of sequences on each rank. This is used when this Metadata
    # describes all ranks in the world, and thus is padded to the one with
    # the most sequences. Otherwise, this should be None.
    num_seqs: Optional[torch.Tensor] = None
    world_size: int = None
    normalized: bool = False
    num_total_recv_tokens: Union[int, tuple[int]] = None

    def get_slice(self, rank: int):
        assert self.world_size is not None
        assert self.num_seqs is not None

        num_seqs = self.num_seqs[rank]
        return Metadata(
            dst_rank=self.dst_rank[rank][:num_seqs],
            dst_offset=self.dst_offset[rank][:num_seqs],
            seq_len=self.seq_len[rank][:num_seqs],
            num_recv_tokens=self.num_recv_tokens[rank], # this is of shape (world_size,)
            seq_recv_mask=self.seq_recv_mask[rank] if self.seq_recv_mask is not None else None,
            recv_seq_lens=self.recv_seq_lens[rank] if self.recv_seq_lens is not None else None,
            normalized=self.normalized,
            num_total_recv_tokens=self.num_total_recv_tokens[rank],
        )

    def normalize_dtype(self):
        return Metadata(
            dst_rank=self.dst_rank.to(torch.int32),
            dst_offset=self.dst_offset.to(torch.uint32),
            seq_len=self.seq_len.to(torch.uint32),
            num_recv_tokens=self.num_recv_tokens.to(torch.uint64),
            seq_recv_mask=self.seq_recv_mask.to(torch.uint32) if self.seq_recv_mask is not None else None,
            recv_seq_lens=self.recv_seq_lens.to(torch.uint32) if self.recv_seq_lens is not None else None,
            num_seqs=self.num_seqs.to(torch.int32) if self.num_seqs is not None else None,
            world_size=self.world_size,
            normalized=True,
            num_total_recv_tokens=self.num_total_recv_tokens,
        )

    def cuda(self):
        return Metadata(
            dst_rank=self.dst_rank.cuda().contiguous(),
            dst_offset=self.dst_offset.cuda().contiguous(),
            seq_len=self.seq_len.cuda().contiguous(),
            num_recv_tokens=self.num_recv_tokens.cuda().contiguous(),
            seq_recv_mask=self.seq_recv_mask.cuda().contiguous() if self.seq_recv_mask is not None else None,
            recv_seq_lens=self.recv_seq_lens.cuda().contiguous() if self.recv_seq_lens is not None else None,
            num_seqs=self.num_seqs.cuda().contiguous() if self.num_seqs is not None else None,
            world_size=self.world_size,
            normalized=self.normalized,
            num_total_recv_tokens=self.num_total_recv_tokens,
        )


@torch.no_grad()
def compute_metadata(
    seq_len: torch.Tensor,  # shape of (world_size, max_num_local_seqs)
    global_dispatch: torch.Tensor,  # shape of (world_size, max_num_local_seqs, max_cp_degree)
    return_intermediate: bool = False,
):
    """
    Given a dispatch plan, this function assigns the query tensor's attention layout.
    Args:
        seq_len (Tensor): shape (world_size, max_num_local_seq_shards). The length of each sequence.
        global_dispatch (Tensor): shape (world_size, max_num_local_seq_shards, max_cp_degree). value -1 is padding.
        The decision tensor. Recording the ranks that each sequence is dispatched to.
    Returns:
        fwd_metadata, rev_metadata: Metadata for forward and reverse communication.
    """
    world_size = global_dispatch.shape[0]
    max_num_local_seqs = global_dispatch.shape[1]
    assert seq_len.shape == (world_size, max_num_local_seqs)
    assert global_dispatch.dtype == torch.int64
    assert seq_len.dtype == torch.int64

    # Query. No max cp degree dimension.
    dispatch_shape = global_dispatch.shape
    if global_dispatch.dim() == 2:
        global_dispatch = global_dispatch.unsqueeze(-1)
    # max_cp_degree = global_dispatch.shape[2]

    ######## Forward side metadata
    seq_to_dst = F.one_hot(global_dispatch + 1, num_classes=world_size + 1)[:, :, :, 1:]
    # (world_size, max_num_local_seqs, max_cp_degree, world_size) [i][j][k][l]:
    # for each sequence ([i][j]), for each dispatch ([k]), the number of sequence sent to rank [l] (either 0 or seq_len[i][j])
    tokens_to_dst_per_dispatch = seq_to_dst * seq_len.unsqueeze(-1).unsqueeze(-1)
    # all tokens by global index i, sending to rank j
    tokens_to_dst = tokens_to_dst_per_dispatch.reshape(-1, world_size)
    seq_begin_offset = tokens_to_dst.cumsum(dim=0) - tokens_to_dst
    # masking and back to the original shape
    seq_begin_offset = seq_begin_offset.reshape(*seq_to_dst.shape) * seq_to_dst
    seq_begin_offset = seq_begin_offset.sum(dim=-1)
    # compute the sequence level offset
    # (do not consider number of tokens. This is only used for reverse communication's indexing in scatter_)
    seq_to_dst_flatten = seq_to_dst.reshape(-1, world_size)
    seq_offset = seq_to_dst_flatten.cumsum(dim=0) - seq_to_dst_flatten
    seq_offset = seq_offset.reshape(*seq_to_dst.shape) * seq_to_dst
    seq_offset = seq_offset.sum(dim=-1)

    # number of tokens received from each rank ([i][j] means tokens received at rank i from rank j), last column is the total number.
    num_recv_tokens = tokens_to_dst_per_dispatch.reshape(world_size, -1, world_size).sum(dim=1).transpose(0, 1)
    num_recv_tokens = torch.cat([num_recv_tokens, num_recv_tokens.sum(dim=1, keepdim=True)], dim=1)

    ######## reverse side metadata:
    seq_rank_expanded = torch.arange(world_size, device=global_dispatch.device).reshape(world_size, 1, 1).expand_as(global_dispatch)
    # seq_offset = seq_len.cumsum(dim=1) - seq_len # NOTE: this is the final offset after deduplication.
    # expanding by max_cp_degree to allow concurrently receiving from multiple ranks
    seq_len_expanded = seq_len.unsqueeze(2).expand_as(global_dispatch).transpose(1, 2)
    seq_offset_expanded = seq_len_expanded.reshape(world_size, -1).cumsum(dim=1).reshape(seq_len_expanded.shape) - seq_len_expanded
    seq_offset_expanded = seq_offset_expanded.transpose(1, 2)
    seq_len_expanded = seq_len_expanded.transpose(1, 2)
    # reverse side communication
    num_received_seqs = seq_to_dst.reshape(-1, world_size).sum(0)
    max_rev_seqs = num_received_seqs.max()
    rev_dst_rank = torch.zeros(world_size, max_rev_seqs, dtype=torch.int64, device=global_dispatch.device)
    rev_dst_offset = torch.zeros(world_size, max_rev_seqs, dtype=torch.int64, device=global_dispatch.device)
    rev_seq_len = torch.zeros(world_size, max_rev_seqs, dtype=torch.int64, device=global_dispatch.device)

    # Create valid mask for non-padding entries
    valid_mask = global_dispatch != -1
    # Flatten all tensors for vectorized processing
    # flatten to [world_size * max_num_local_seqs * max_cp_degree]
    valid_mask_flat = valid_mask.flatten()
    global_dispatch_flat = global_dispatch.flatten()
    seq_rank_expanded_flat = seq_rank_expanded.flatten()
    seq_offset_expanded_flat = seq_offset_expanded.flatten()
    seq_len_expanded_flat = seq_len_expanded.flatten()

    # Use flatten valid indices to choose the valid sequence dispatch entries
    valid_indices = torch.where(valid_mask_flat)[0] # ranging from 0 to world_size * max_num_local_seqs * max_cp_degree - 1
    # from global dispatch id to the dst rank
    valid_dst_ranks = global_dispatch_flat[valid_indices]
    # from global dispatch id to the src rank
    valid_src_ranks = seq_rank_expanded_flat[valid_indices]
    valid_src_offsets = seq_offset_expanded_flat[valid_indices]
    valid_src_seq_lens = seq_len_expanded_flat[valid_indices]
    dst_seq_local_offset = seq_offset.flatten()[valid_indices]

    # Compute global destination (flatten) indices for scatter operation
    # For each valid entry, compute where it should go in the reverse tensors
    global_dst_indices = valid_dst_ranks * max_rev_seqs + dst_seq_local_offset

    rev_dst_rank.view(-1).scatter_(0, global_dst_indices, valid_src_ranks)
    rev_dst_offset.view(-1).scatter_(0, global_dst_indices, valid_src_offsets)
    rev_seq_len.view(-1).scatter_(0, global_dst_indices, valid_src_seq_lens)

    # masking
    rev_dst_rank = rev_dst_rank.masked_fill(
        num_received_seqs.unsqueeze(1) <= torch.arange(max_rev_seqs, device=rev_dst_rank.device), -1)
    rev_dst_offset = rev_dst_offset.masked_fill(
        num_received_seqs.unsqueeze(1) <= torch.arange(max_rev_seqs, device=rev_dst_offset.device), 0)
    rev_seq_len = rev_seq_len.masked_fill(
        num_received_seqs.unsqueeze(1) <= torch.arange(max_rev_seqs, device=rev_seq_len.device), 0)
    # number of tokens received in the reverse communication. This equals the number of tokens sent in the forward communication.
    rev_num_received_tokens = tokens_to_dst_per_dispatch.reshape(world_size, -1, world_size).sum(dim=1)
    rev_num_received_tokens = torch.cat([rev_num_received_tokens, rev_num_received_tokens.sum(dim=1, keepdim=True)], dim=1)

    # If this is kv (has cp degree), we add the sequence-cp mask to the reverse communication metadata.
    seq_recv_mask = None
    if len(dispatch_shape) == 3:
        seq_recv_mask = global_dispatch.reshape(dispatch_shape) >= 0
    num_seqs = (seq_len > 0).sum(dim=1)
    num_seqs_rev = (rev_seq_len > 0).sum(dim=1)

    fwd_metadata = Metadata(
        dst_rank=global_dispatch.reshape(dispatch_shape),
        dst_offset=seq_begin_offset.reshape(dispatch_shape),
        seq_len=seq_len,
        num_recv_tokens=num_recv_tokens,
        num_seqs=num_seqs,
        world_size=world_size,
        num_total_recv_tokens=num_recv_tokens[:, -1].tolist(),
    )
    rev_metadata = Metadata(
        dst_rank=rev_dst_rank,
        dst_offset=rev_dst_offset,
        seq_len=rev_seq_len,
        num_recv_tokens=rev_num_received_tokens,
        seq_recv_mask=seq_recv_mask,
        recv_seq_lens=seq_len if seq_recv_mask is not None else None,
        num_seqs=num_seqs_rev,
        world_size=world_size,
        num_total_recv_tokens=rev_num_received_tokens[:, -1].tolist(),
    )

    # NOTE: use this for the fast alltoall dispatch layout.
    intermediates = (
        tokens_to_dst_per_dispatch, seq_to_dst, num_received_seqs
    )
    if return_intermediate:
        return fwd_metadata, rev_metadata, intermediates
    return fwd_metadata, rev_metadata


@torch.no_grad()
def compute_attn_layout_seqlens(
    seq_shard_len: torch.Tensor, seq_shard_cumsum: torch.Tensor,
    dispatch: torch.Tensor, prepend_zero: bool=True, shard_to_tuple: bool=False
):
    """
    Compute the cu_seqlens_q and cu_seqlens_kv for the attention layout.
    NOTE: both inputs and outputs are the global value, so it has `world_size` dimension.
    Args:
        seq_shard_len: shape (world_size, max_num_local_seqs), length of each sequence shard.
          A sequence shard is the unit sending to a rank.
        seq_shard_cumsum: shape (world_size, max_num_local_seqs). Cumulative number of tokens
          for the sequence shard's context. This is to compute the KV seqlens of the shard.
        dispatch: shape (world_size, max_num_local_seqs). The rank that each sequence
          shard is dispatched to. The value is -1 for padding.
        prepend_zero: bool, whether to prepend a zero to the cu_seqlens.
          flash attention requires tha prefix zero.
        shard_to_tuple: bool, whether to return the output as a tuple of tensors, and apply the actual receive length.
    """
    world_size = dispatch.shape[0]
    max_num_local_seqs = dispatch.shape[1]
    assert dispatch.dim() == 2
    assert seq_shard_len.shape == (world_size, max_num_local_seqs)
    assert seq_shard_cumsum.shape == (world_size, max_num_local_seqs)
    assert dispatch.dtype == torch.int64
    assert seq_shard_len.dtype == torch.int64
    assert seq_shard_cumsum.dtype == torch.int64
    # dispatch[i, j] = the rank that sequence [i,j] is dispatched to.
    flatten_dispatch = dispatch.flatten()

    flatten_dispatch_one_hot = F.one_hot(flatten_dispatch + 1, num_classes=world_size + 1)[:, 1:]
    # shape: (world_size, seq_len, world_size)
    local_indices_flat_one_hot = (flatten_dispatch_one_hot.cumsum(dim=0) - 1) * flatten_dispatch_one_hot
    # local_indices_flat_one_hot[shard_i, rank_j] = if shard_id is sent to rank j, then it is the index in that rank j, otherwise 0. 
    local_indices_flat = local_indices_flat_one_hot.sum(dim=1).reshape(-1)
    # if dispatch[i, j] = k, then local_indices_flat[i, j, k] = l means sequence [i,j] is at out_sequence [k,l]
    # out_seqlens_q[k, l] = seq_shard_len[i, j]
    max_num_seq = int(local_indices_flat.max().item() + 1)
    scatter_index = flatten_dispatch * max_num_seq + local_indices_flat
    scatter_index = mask_by_neg(scatter_index, flatten_dispatch >= 0)

    src_seqlens = seq_shard_len.flatten()
    src_seq_lens_kv = seq_shard_cumsum.flatten()

    out_seqlens_q = torch.zeros(world_size * max_num_seq, dtype=torch.int64, device=dispatch.device)
    out_seqlens_kv = torch.zeros(world_size * max_num_seq, dtype=torch.int64, device=dispatch.device)

    out_seqlens_q = index_put_with_neg_padding_1d(
        out_seqlens_q, src_seqlens, scatter_index
    )
    out_seqlens_kv = index_put_with_neg_padding_1d(
        out_seqlens_kv, src_seq_lens_kv, scatter_index
    )

    out_seqlens_q = out_seqlens_q.reshape(world_size, max_num_seq)
    out_seqlens_kv = out_seqlens_kv.reshape(world_size, max_num_seq)

    # TODO: Maybe we just take flatten_dispatch_one_hot and take a sum -> to get the max sequence length.
    num_local_seqs_recv = local_indices_flat_one_hot.max(dim=0)[0] + 1

    cu_seqlens_q = out_seqlens_q.cumsum(dim=1)
    cu_seqlens_kv = out_seqlens_kv.cumsum(dim=1)
    max_seqlen_q = out_seqlens_q.max(dim=1)[0]
    max_seqlen_kv = out_seqlens_kv.max(dim=1)[0]
    if prepend_zero:
        cu_seqlens_q = prepend_zero_fn(cu_seqlens_q, dim=1)
        cu_seqlens_kv = prepend_zero_fn(cu_seqlens_kv, dim=1)
    if shard_to_tuple:
        sq_len_extra = 1 if prepend_zero else 0
        cu_seqlens_q = tuple(
            cu_seqlens_q[i][:num_local_seqs_recv[i] + sq_len_extra]
            for i in range(world_size)
        )
        cu_seqlens_kv = tuple(
            cu_seqlens_kv[i][:num_local_seqs_recv[i] + sq_len_extra]
            for i in range(world_size)
        )
        # NOTE: max_seqlen does not need to shard to tuples because
        # they are of the same length (1,) for each rank.

    return cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, num_local_seqs_recv


def compute_metadata_kv(
    kv_to_q_mapping: torch.Tensor,
    kv_to_q_rank: torch.Tensor,
    kv_context_size: torch.Tensor,
    q_to_num_kv_seq: torch.Tensor,
    q_to_num_kv_token: torch.Tensor,
    seq_len: torch.Tensor,
    num_seqs: torch.Tensor,
    # query metadata
    q_dispatch: torch.Tensor,
    q_seq_to_dst: torch.Tensor,
    max_num_local_seqs: int,
    return_intermediate: bool=False
):
    """
    Given the query's dispatch plan and a mapping from key-value to query,
    this function computes forward and reverse communication metadata for key-value.
    Args:
        kv_to_q_mapping (Tensor): shape (world_size, max_num_local_seqs, max_cp_degree, 2).
            The mapping from this key-value dispatch to a query index. The value is -1 for padding.
            The last dimension describes the query's world_size and local_index.
        kv_to_q_rank (Tensor): shape (world_size, max_num_local_seqs, max_cp_degree).
            The rank among all kvs that mapping to the same query. The value is -1 for padding.
        kv_context_size (Tensor): shape (world_size, max_num_local_seqs).
            The number of tokens in this document, before this key-value shard.
            NOTE: this is like the token_wise kv_to_q_rank
        q_to_num_kv_seq (Tensor): shape (world_size, max_num_local_seqs).
            The number of key-value shards that map to the query.
        q_to_num_kv_token (Tensor): shape (world_size, max_num_local_seqs).
            The total number of key-value tokens that map to the query.
            NOTE: this is like the token_wise q_to_num_kv_seq
            NOTE: This is on q's perspective, while kv_context_size is on the KV's perspective. Are they always the same?
        q_dispatch (Tensor): shape (world_size, max_num_local_seqs).
            The query's dispatch plan. The value is -1 for padding.
        q_seq_to_dst (Tensor): intermediate value, equals one_hot_remove_padding(q_dispatch)
    """
    world_size = kv_to_q_mapping.shape[0]
    max_cp_degree = kv_to_q_mapping.shape[2]
    assert kv_to_q_mapping.shape == (world_size, max_num_local_seqs, max_cp_degree, 2)
    assert kv_to_q_rank.shape == (world_size, max_num_local_seqs, max_cp_degree)
    assert kv_context_size.shape == (world_size, max_num_local_seqs)
    assert q_to_num_kv_seq.shape == (world_size, max_num_local_seqs)
    assert q_dispatch.shape == (world_size, max_num_local_seqs)
    assert q_seq_to_dst.shape == (world_size, max_num_local_seqs, world_size)
    assert seq_len.shape == (world_size, max_num_local_seqs)
    assert num_seqs.shape == (world_size,)
    ######## Forward
    # 1. compute the dst rank for each kv dispatch, on the attention layout.
    # kv_dst_rank[rank, s_id, c_id] = q_dispatch[kv_to_q_mapping[rank, s_id, c_id, 0], kv_to_q_mapping[rank, s_id, c_id, 1]]
    kv_valid_mask = kv_to_q_mapping[..., 0] >= 0
    kv_to_q_mapping_flatten = kv_to_q_mapping[..., 0] * max_num_local_seqs + kv_to_q_mapping[..., 1]

    kv_dst_rank = (q_dispatch.flatten()[kv_to_q_mapping_flatten].reshape(kv_valid_mask.shape) * kv_valid_mask +
                   # NOTE: this is to make sure all padding values get a rank -1.
                   (kv_valid_mask.int() - 1))

    # 2. compute the local sequence id for each kv dispatch, on the attention layout.
    # shape (num_global_seqs, world_size): 0 if q not sending to that rank, else q_seq_to_dst[global_idx]
    num_kv_seq_to_dst = (q_seq_to_dst * q_to_num_kv_seq.unsqueeze(-1)).reshape(-1, world_size)
    # shape (num_global_seqs, world_size): 0 if q not sending to that rank, else the begin (sequence level) offset of the kv shard.
    # On the dst layout, the first kv shard of **this query**'s sequence id
    query_dst_kv_seq_id = exclusive_cumsum(num_kv_seq_to_dst, dim=0) * q_seq_to_dst.bool().reshape(-1, world_size)
    query_dst_kv_seq_id = query_dst_kv_seq_id.sum(dim=-1).reshape(world_size, max_num_local_seqs)

    # shape (world_size, max_num_local_seqs, max_cp_degree): the dst rank's seq idx for this src kv replica
    kv_dst_seq_id = query_dst_kv_seq_id.flatten()[kv_to_q_mapping_flatten].reshape(kv_valid_mask.shape) * kv_valid_mask
    kv_dst_seq_id = kv_dst_seq_id + kv_to_q_rank
    # 3. compute the dst offset for each kv dispatch, on the attention layout.
    num_token_to_dst = (q_seq_to_dst * q_to_num_kv_token.unsqueeze(-1)).reshape(-1, world_size)
    # for each query shard, the number of tokens before it starts.
    query_dst_kv_token_id = exclusive_cumsum(num_token_to_dst, dim=0) * q_seq_to_dst.bool().reshape(-1, world_size)
    query_dst_kv_token_id = query_dst_kv_token_id.sum(dim=-1).reshape(world_size, max_num_local_seqs)

    # get the inter-query-group offset for each kv shard.
    kv_dst_token_offset = query_dst_kv_token_id.flatten()[kv_to_q_mapping_flatten].reshape(kv_valid_mask.shape)
    # add the intra-query-group offset for each kv shard.
    kv_dst_token_offset = (kv_dst_token_offset + kv_context_size.unsqueeze(-1)) * kv_valid_mask
    # 4. compute the number of tokens received for kv shards.
    num_send_tokens = num_token_to_dst.reshape(world_size, -1, world_size).sum(dim=1)
    num_recv_tokens = num_send_tokens.transpose(0, 1)
    num_total_recv_tokens = num_recv_tokens.sum(dim=1)
    num_recv_tokens = torch.concat(
        [num_recv_tokens, num_total_recv_tokens.unsqueeze(1)], dim=1
    )
    fwd_metadata = Metadata(
        kv_dst_rank, kv_dst_token_offset, seq_len,
        num_recv_tokens=num_recv_tokens, num_seqs=num_seqs,
        world_size=world_size, num_total_recv_tokens=num_total_recv_tokens.tolist()
    )
    ######## Backward
    num_seq_bwd = num_kv_seq_to_dst.sum(dim=0)
    max_num_local_seqs_rev = int(num_seq_bwd.max().item())
    rev_seq_valid_mask = (torch.arange(max_num_local_seqs_rev).view(1, -1).repeat(world_size, 1)
                          < num_seq_bwd.unsqueeze(1))
    # NOTE: we order metadata by their local seq id (kv_dst_seq_id).
    # shape: (world_size, max_num_local_seqs, max_cp_degree)
    # global sequence id for the kv_dst tensor.
    kv_dst_global_seq_id = kv_dst_seq_id + kv_dst_rank * max_num_local_seqs_rev
    src_rank_expand = torch.arange(world_size).view(-1, 1, 1).expand_as(kv_dst_global_seq_id)
    kv_dst_global_seq_id = mask_by_neg(
        kv_dst_global_seq_id, kv_valid_mask
    )

    rev_kv_dst_rank = torch.empty(
        (world_size, max_num_local_seqs_rev), dtype=torch.int64, device=kv_dst_rank.device
    )
    rev_kv_dst_rank = index_put_with_neg_padding_1d(
        rev_kv_dst_rank, src_rank_expand, kv_dst_global_seq_id
    )
    rev_kv_dst_rank = mask_by_neg(rev_kv_dst_rank, rev_seq_valid_mask)

    # NOTE: this is the offset written to the global buffer.
    # NOTE: we have use layout (cp_degree, num_token, hidden) for the gradient tensor,
    # because it makes copying a cp repeat consecutive.
    # shape (world_size, max_num_local_seqs, max_cp_degree)
    inter_replica_offset = (
        (torch.arange(max_cp_degree)).reshape(1, -1) *
         seq_len.sum(1).reshape(-1, 1)
    ).unsqueeze(1)
    intra_replica_offset = exclusive_cumsum(seq_len, dim=1).unsqueeze(-1)
    src_kv_offset = (
        inter_replica_offset + intra_replica_offset
    )

    src_kv_grad_buffer_offset = torch.zeros_like(rev_kv_dst_rank)
    src_kv_grad_buffer_offset = index_put_with_neg_padding_1d(
        src_kv_grad_buffer_offset, src_kv_offset, kv_dst_global_seq_id
    )
    # sequence lengths
    rev_kv_seqlen = torch.zeros_like(rev_kv_dst_rank)
    src_kv_seqlen = fwd_metadata.seq_len.unsqueeze(-1).repeat(1, 1, max_cp_degree)
    rev_kv_seqlen = index_put_with_neg_padding_1d(
        rev_kv_seqlen, src_kv_seqlen, kv_dst_global_seq_id
    )
    # num_token_to_dst is the forward size tokens sent to the dst, which equals token received during backward.
    rev_kv_num_recv_tokens = num_send_tokens
    rev_total_recv_tokens = rev_kv_num_recv_tokens.sum(dim=1, keepdim=True)
    rev_kv_num_recv_tokens = torch.cat(
        [rev_kv_num_recv_tokens, rev_total_recv_tokens], dim=1
    )

    bwd_metadata = Metadata(
        rev_kv_dst_rank,
        src_kv_grad_buffer_offset,
        rev_kv_seqlen,
        rev_kv_num_recv_tokens,
        seq_recv_mask=kv_valid_mask,
        recv_seq_lens=fwd_metadata.seq_len,
        num_seqs=num_seq_bwd,
        world_size=world_size,
        num_total_recv_tokens=rev_total_recv_tokens.flatten().tolist()
    )
    if return_intermediate:
        return fwd_metadata, bwd_metadata, (
            kv_dst_global_seq_id,
        )
    return fwd_metadata, bwd_metadata


def mlp_layout_packed_params(seq_lens: torch.Tensor):
    """
    Compute the MLP layout packed_seq_params. MLP layout guarantees seqlens_q == seqlens_kv.
    This is mainly for RoPE.
    NOTE: this is the seq lens on the local rank.
    """
    cu_seqlens = prepend_zero_fn(seq_lens.cumsum(dim=0))
    max_seqlen = seq_lens.max().item() # int
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )
    return packed_seq_params


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
    if q_seq_to_dst.dim() == 4:
        q_seq_to_dst = q_seq_to_dst.squeeze(2)
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

