from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from d2.runtime.inplace_metadata import Metadata, exclusive_cumsum, index_put_with_neg_padding_1d


@dataclass
class FastAlltoAllMetadata:
    fa2a_metadata: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    # List of (world_size,) tensors, each of shape (num_sequences,). If a slice, no world_size dimension.
    # metadata on the sender side:
    #   the offset to copy each sequence to the buffer, ordered by the sender's sequence idx
    #       NOTE: for kv sender's sequence idx, to make it directly available to the reverse operation,
    #         we use the (cp_id, seq_idx) order instead of the (seq_idx, cp_id) order
    #   the offset to copy each sequence from the buffer, ordered by the recver's sequence idx
    send_memcpy_metadata: Sequence[Union[torch.Tensor, List[torch.Tensor]]]
    recv_memcpy_metadata: Sequence[Union[torch.Tensor, List[torch.Tensor]]]
    def get_slice(self, rank):
        """
        Returns the metadata for the given rank.
        """
        fa2a_metadata = tuple(t[rank] for t in self.fa2a_metadata)
        send_memcpy_metadata = tuple(t[rank] for t in self.send_memcpy_metadata)
        recv_memcpy_metadata = tuple(t[rank] for t in self.recv_memcpy_metadata)
        return FastAlltoAllMetadata(
            fa2a_metadata, send_memcpy_metadata, recv_memcpy_metadata,
        )


def _filter_nonzero_by_mask(tensor: torch.Tensor, mask: torch.Tensor, num_received: torch.Tensor) -> torch.Tensor:
    tensor = tensor.flatten(start_dim=1)
    mask = mask.flatten(start_dim=1).bool()
    condensed = tensor[mask]
    output_tuple = torch.split(condensed, num_received.tolist())
    return output_tuple


def compute_forward_qkv_a2a_layout_meatadata(
    # q intermediate metadata
    tokens_to_dst_per_dispatch_q: torch.Tensor,
    seq_to_dst_mask_q: torch.Tensor,
    # num_recv_tokens * bytes_q
    recver_transfer_sz_q: torch.Tensor,
    recver_transfer_sz_kv: torch.Tensor,
    # num_recv sequences
    num_recv_seqs_q: torch.Tensor,
    num_recv_seqs_kv: torch.Tensor,
    # kv dispatch metadata
    kv_dst_rank: torch.Tensor,  # shape (world_size, num_local_seqs, cp_degree)
    kv_seq_len: torch.Tensor,  # shape (world_size, num_local_seqs)
    kv_dst_global_seq_id: torch.Tensor,  # shape (world_size, num_local_seqs, cp_degree)
    num_send_tokens_kv: torch.Tensor,
    bytes_q: int, bytes_kv: int,
):
    """
    Returns:
    1. fwd_fa2a_metadata: metadata used for fast all-to-all;
    2. fwd_send_memcpy_metadata: metadata for memcpy on the sender side;
    3. fwd_recv_memcpy_metadata: metadata for memcpy on the receiver side.
    """
    assert tokens_to_dst_per_dispatch_q.ndim == 3
    world_size = tokens_to_dst_per_dispatch_q.shape[0]
    max_num_send_seq = tokens_to_dst_per_dispatch_q.shape[1]
    assert tokens_to_dst_per_dispatch_q.shape == (world_size, max_num_send_seq, world_size)
    assert kv_dst_rank.ndim == 3
    max_cp_degree = kv_dst_rank.shape[2]
    assert kv_dst_rank.shape == (world_size, max_num_send_seq, max_cp_degree)
    assert kv_seq_len.shape == (world_size, max_num_send_seq)
    assert kv_dst_global_seq_id.shape == (world_size, max_num_send_seq, max_cp_degree)
    assert num_recv_seqs_q.shape == (world_size,)
    assert num_recv_seqs_kv.shape == (world_size,)
    seq_q_shape = (world_size, max_num_send_seq)
    seq_kv_shape_T = (world_size, max_cp_degree, max_num_send_seq)

    # For each sequence, the bytes to copy to the destination.
    # shape: (world_size, max_num_local_seqs, max_cp_degree, world_size)
    # NOTE: max_cp_degree is 1 for q
    bytes_to_dst_per_dispatch_q = tokens_to_dst_per_dispatch_q * bytes_q
    # (world_size, world_size) [i][j]: bytes send from rank i to rank j
    bytes_to_dst_q = bytes_to_dst_per_dispatch_q.reshape(world_size, -1, world_size).sum(dim=1)
    bytes_to_dst_kv = num_send_tokens_kv * bytes_kv
    bytes_to_dst = bytes_to_dst_q + bytes_to_dst_kv * 2 # 2 for k and v

    # one hot mask describing if a kv shard is dispatched to a rank, at this replica
    # shape (world_size, max_cp_degree, max_num_local_seqs, world_size)
    dst_per_dispatch_kv = F.one_hot(
        kv_dst_rank + 1, num_classes=world_size + 1
    )[..., 1:].permute(0, 2, 1, 3)
    # NOTE: sequence index here is (cp_id, seq_id)
    seq_dispatch_mask_kv = dst_per_dispatch_kv.reshape(world_size, -1, world_size)
    bytes_to_dst_per_dispatch_kv = (
        dst_per_dispatch_kv * kv_seq_len.view(world_size, 1, -1, 1) * bytes_kv
    )

    #### Forward metadata - FastA2A
    sender_send_disp = exclusive_cumsum(bytes_to_dst, dim=1)
    sender_transfer_sz = bytes_to_dst
    # [i][j] before transpose: at i, offset in bytes to place data from j
    recver_transfer_sz = recver_transfer_sz_q + recver_transfer_sz_kv * 2  # 2 for k and v
    sender_recv_disp = exclusive_cumsum(recver_transfer_sz,  dim=1).transpose(0, 1)
    fwd_fa2a_metadata = (
        sender_send_disp, sender_transfer_sz, sender_recv_disp, recver_transfer_sz
    )

    #### Forward layout transfer metadata - Sender side
    # NOTE: this mask provides an implicit order: the dim 1 index of the mask equals the
    # sequence index on the sender side.
    # shape: (world_size, max_num_local_seqs, world_size)
    seq_dispatch_mask_q = seq_to_dst_mask_q.reshape(world_size, -1, world_size).bool()

    # intra_dst_rank_offset [src][k][dst]: at rank src, the dispatch k's intra-recv-rank offset (rank dst).
    # rank i: | send to rank 0 | send to rank 1 | ... 
    #         | send to rank j: seq j0, seq j1, seq j2, ... |
    # for seq j2, this offset equals size(seq j0) + size(seq j1)
    # sequence index here is (cp_id, seq_id)
    seq_intra_dst_rank_offset_q = exclusive_cumsum(
        bytes_to_dst_per_dispatch_q.reshape(world_size, -1, world_size), dim=1
    )
    seq_intra_dst_rank_offset_kv = exclusive_cumsum(
        bytes_to_dst_per_dispatch_kv.reshape(world_size, -1, world_size), dim=1
    )
    seq_offset_q = seq_intra_dst_rank_offset_q + sender_send_disp.unsqueeze(1)
    # kv sender memcpy:
    # for i in cp_rank: for j in dst_id: if dst_rank[i][j] >= 0: copy seq[j] to offset[i][j]
    # the same looping logic applies to recv memcpy as well.

    # NOTE: this does not compress padding values.
    # shape (src_rank, cp_degree * num_local_seqs, dst_rank(onehot))
    seq_offset_k = seq_intra_dst_rank_offset_kv + (sender_send_disp + bytes_to_dst_q).unsqueeze(1)
    seq_offset_v = seq_intra_dst_rank_offset_kv + (sender_send_disp + bytes_to_dst_q + bytes_to_dst_kv).unsqueeze(1)
    # shrink the dst_rank dimension (one_hot, so add after mask)
    seq_offset_q = (seq_offset_q * seq_dispatch_mask_q).sum(dim=-1).reshape(seq_q_shape)
    seq_offset_k = (seq_offset_k * seq_dispatch_mask_kv).sum(dim=-1).reshape(seq_kv_shape_T)
    seq_offset_v = (seq_offset_v * seq_dispatch_mask_kv).sum(dim=-1).reshape(seq_kv_shape_T)
    # offset to copy each data to their destination.
    fwd_send_memcpy_metadata = tuple(
        tuple(tt.squeeze(0) for tt in torch.split(t, 1, dim=0))
        for t in (seq_offset_q, seq_offset_k, seq_offset_v)
    )

    #### Forward layout transfer metadata - Receiver side
    # [dst][src][k]: at rank dst, for all tensors coming from src,
    # the dispatch k's intra-src-rank offset on the attention receiver buffer.
    # rank i: | recv from rank 0 | recv from rank 1 | ... 
    #         | recv from rank j: seq j0, seq j1, seq j2, ... |
    # for seq j2, this offset equals size(seq j0) + size(seq j1)
    seq_intra_src_rank_offset_q = seq_intra_dst_rank_offset_q.permute(2, 0, 1)
    seq_intra_src_rank_offset_kv = seq_intra_dst_rank_offset_kv.permute(2, 0, 1)
    seq_inter_src_rank_offset = exclusive_cumsum(recver_transfer_sz, dim=1)
    seq_recv_offset_q = (
        seq_intra_src_rank_offset_q +
        seq_inter_src_rank_offset.unsqueeze(-1)
    )
    # shape (world_size(onehot), world_size, max_num_local_seqs * max_cp_degree)
    seq_recv_offset_k = seq_intra_src_rank_offset_kv + (
        seq_inter_src_rank_offset + recver_transfer_sz_q).unsqueeze(-1)
    seq_recv_offset_v = seq_intra_src_rank_offset_kv + (
        seq_inter_src_rank_offset + recver_transfer_sz_q + recver_transfer_sz_kv
    ).unsqueeze(-1)

    seq_dispatch_mask_recv_q = seq_dispatch_mask_q.permute(2, 0, 1)
    seq_dispatch_mask_recv_kv = seq_dispatch_mask_kv.permute(2, 0, 1)

    seq_recv_offset_q = _filter_nonzero_by_mask(
        seq_recv_offset_q, seq_dispatch_mask_recv_q, num_recv_seqs_q
    )
    # NOTE: for q, this order is the same as the final output sequence order. (i.e. the (dst_rank, (src_rank, seq_id)) order)
    # However, for kv, this is not the order we need. The current order is (world_id, cp_id, seq_id) on the src side,
    # we should pick it by the correct order using kv_dst_seq_id
    # shape: (world_size, cp_degree * num_recv_seqs_kv)
    seq_recv_offset_k = (seq_recv_offset_k * seq_dispatch_mask_recv_kv).sum(dim=0)
    seq_recv_offset_v = (seq_recv_offset_v * seq_dispatch_mask_recv_kv).sum(dim=0)

    max_num_recv_seqs_kv = int(num_recv_seqs_kv.max().item())
    kv_dst_global_seq_id_cp_sq = kv_dst_global_seq_id.permute(0, 2, 1)
    seq_recv_offset_k_compact = torch.zeros(
        (world_size, max_num_recv_seqs_kv),
        dtype=seq_recv_offset_k.dtype,
        device=seq_recv_offset_k.device
    )
    seq_recv_offset_v_compact = seq_recv_offset_k_compact.clone()
    # recv_offset_k.flatten()[i] <= kv_dst_global_seq_id_cp_sq[]
    seq_recv_offset_k_compact = index_put_with_neg_padding_1d(
        seq_recv_offset_k_compact, seq_recv_offset_k, kv_dst_global_seq_id_cp_sq
    )
    seq_recv_offset_k_compact = tuple(
        seq_recv_offset_k_compact[rank, :num_recv_seqs_kv[rank]] for rank in range(world_size)
    )
    seq_recv_offset_v_compact = index_put_with_neg_padding_1d(
        seq_recv_offset_v_compact, seq_recv_offset_v, kv_dst_global_seq_id_cp_sq
    )
    seq_recv_offset_v_compact = tuple(
        seq_recv_offset_v_compact[rank, :num_recv_seqs_kv[rank]] for rank in range(world_size)
    )

    fwd_recv_memcpy_metadata = (
        seq_recv_offset_q, seq_recv_offset_k_compact, seq_recv_offset_v_compact
    )

    return FastAlltoAllMetadata(
        fwd_fa2a_metadata, fwd_send_memcpy_metadata, fwd_recv_memcpy_metadata
    )


def compute_backward_qkv_a2a_layout_metadata(
    fwd_metadata: FastAlltoAllMetadata
):
    # during backward, the tensor is copied back to the original location.
    send_memcpy_metadata = fwd_metadata.recv_memcpy_metadata
    recv_memcpy_metadata = fwd_metadata.send_memcpy_metadata
    # the tensor is sent back to the original location
    fwd_sender_send_disp, fwd_sender_transfer_sz, fwd_sender_recv_disp, fwd_recver_transfer_sz = fwd_metadata.fa2a_metadata

    # fwd bytes received from each rank -> bwd bytes sent to each rank
    bwd_sender_transfer_sz = fwd_recver_transfer_sz
    bwd_recver_transfer_sz = fwd_sender_transfer_sz
    # fwd_sender_send_disp: [i][j]: rank i send to rank j, offset at rank i
    # fwd_sender_recv_disp: [i][j]: rank i receive from rank j, offset at rank j
    # bwd_sender_send_disp: [j][i]: rank j send to rank i, offset at rank j
    #  == fwd offset that rank j recv from rank i, offset at rank j == fwd_sender_recv_disp.T
    bwd_sender_send_disp = fwd_sender_recv_disp.transpose(0, 1)
    bwd_sender_recv_disp = fwd_sender_send_disp.transpose(0, 1)
    bwd_fa2a_metadata = (
        bwd_sender_send_disp, bwd_sender_transfer_sz, bwd_sender_recv_disp,
        bwd_recver_transfer_sz
    )
    return FastAlltoAllMetadata(
        bwd_fa2a_metadata, send_memcpy_metadata, recv_memcpy_metadata
    )


def compute_backward_attn_out_a2a_layout_metadata(
    tokens_to_dst_per_dispatch_q: torch.Tensor,
    seq_to_dst_mask_q: torch.Tensor,
    recver_transfer_sz_q: torch.Tensor,
    bytes_q: int, world_size: int,
    seq_q_shape: torch.Size
):
    """
    Unlike qkv, the backward is easier to compute. We do it first, then the forward.
    """
    bytes_to_dst_per_dispatch_q = tokens_to_dst_per_dispatch_q * bytes_q
    bytes_to_dst_q = bytes_to_dst_per_dispatch_q.reshape(world_size, -1, world_size).sum(dim=1)
    bytes_to_dst = bytes_to_dst_q
    sender_send_disp = exclusive_cumsum(bytes_to_dst, dim=1)
    sender_transfer_sz = bytes_to_dst
    recver_transfer_sz = recver_transfer_sz_q
    sender_recv_disp = exclusive_cumsum(recver_transfer_sz,  dim=1).transpose(0, 1)
    bwd_fa2a_metadata = (
        sender_send_disp, sender_transfer_sz, sender_recv_disp, recver_transfer_sz
    )

    seq_dispatch_mask_q = seq_to_dst_mask_q.reshape(world_size, -1, world_size).bool()
    seq_intra_dst_rank_offset_q = exclusive_cumsum(
        bytes_to_dst_per_dispatch_q.reshape(world_size, -1, world_size), dim=1
    )
    seq_offset_q = seq_intra_dst_rank_offset_q + sender_send_disp.unsqueeze(1)
    seq_offset_q = (seq_offset_q * seq_dispatch_mask_q).sum(dim=-1).reshape(seq_q_shape)


# FIXME: fix below.
def compute_a2a_layout_metadata(
    fwd_metadata_q: Metadata, fwd_metadata_kv: Metadata,
    hidden_size_q: int, hidden_size_kv: int, itemsize: int,
    tokens_to_dst_per_dispatch_q: torch.Tensor,
    tokens_to_dst_per_dispatch_kv: torch.Tensor,
    seq_to_dst_mask_q: torch.Tensor,
    seq_to_dst_mask_kv: torch.Tensor,
    num_rev_seqs_q: int, num_rev_seqs_kv: int
):
    """
    FastAlltoAll layout on both send and receive side is as:
    | q_seq_0_rank_0, q_seq_1_rank_0 ... | k_seq_0_rank_0, ... | v_seq_0_rank_0, ...
    | q_seq_0_rank_1, ...
    (This enables a higher send throughput, as the data is contiguous.)
    This function compute:
    1. the begin offset of each sequence in this layout, on the send side;
    2. the begin offset of each sequence in this layout, on the receive side;

    NOTE: hidden_size_kv is the hidden size for tensor K or V, so there is an extra *2 factor
    """
    bytes_q = hidden_size_q * itemsize
    bytes_kv = hidden_size_kv * itemsize

    world_size = fwd_metadata_q.world_size

    seq_mask = fwd_metadata_q.seq_len > 0
    seq_mask_kv = fwd_metadata_kv.seq_len > 0
    assert torch.allclose(seq_mask, seq_mask_kv)
    assert fwd_metadata_kv.dst_rank.ndim == 3
    assert fwd_metadata_kv.world_size == world_size

    dst_rank = fwd_metadata_q.dst_rank.unsqueeze(-1)
    dst_rank_kv = fwd_metadata_kv.dst_rank
    assert dst_rank.ndim == 3

    #### QKV Forward/backward metadata
    qkv_fwd_metadata = compute_forward_qkv_a2a_layout_meatadata(
        tokens_to_dst_per_dispatch_q,
        tokens_to_dst_per_dispatch_kv,
        seq_to_dst_mask_q,
        seq_to_dst_mask_kv,
        (fwd_metadata_q.num_recv_tokens * bytes_q)[:, :-1],
        (fwd_metadata_kv.num_recv_tokens * bytes_kv)[:, :-1],
        fwd_metadata_q.dst_offset * bytes_q,
        fwd_metadata_kv.dst_offset * bytes_kv,
        bytes_q, bytes_kv, world_size,
        num_rev_seqs_q, num_rev_seqs_kv,
        dst_rank.shape, dst_rank_kv.shape
    )
    qkv_bwd_metadata = compute_backward_qkv_a2a_layout_metadata(
        qkv_fwd_metadata
    )

    #### attn_out Forward/backward metadata

    # now dst_offset is the offset of each sequence to copy data back
