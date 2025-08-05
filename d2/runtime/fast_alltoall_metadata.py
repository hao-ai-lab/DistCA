from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from d2.runtime.inplace_metadata import compute_e2e_metadata, exclusive_cumsum, index_put_with_neg_padding_1d, Metadata

_Tensor_Or_Tensor_List = Union[torch.Tensor, Sequence[torch.Tensor]]


def _get_ragged_seqlen(seq_len: torch.Tensor, num_seqs: torch.Tensor):
    world_size = seq_len.shape[0]
    assert num_seqs.shape == (world_size,)
    return tuple(
        seq_len[i, :num_seqs[i]] for i in range(world_size)
    )


@dataclass
class SeqLens:
    send_seqlens: _Tensor_Or_Tensor_List
    recv_seqlens: _Tensor_Or_Tensor_List

    def get_slice(self, rank):
        return SeqLens(
            self.send_seqlens[rank], self.recv_seqlens[rank],
        )

    def normalize(self):
        if isinstance(self.send_seqlens, torch.Tensor):
            return SeqLens(
                self.send_seqlens.cuda().to(torch.int64).contiguous(),
                self.recv_seqlens.cuda().to(torch.int64).contiguous()
            )
        return SeqLens(
            tuple(t.cuda().to(torch.int64).contiguous() for t in self.send_seqlens),
            tuple(t.cuda().to(torch.int64).contiguous() for t in self.recv_seqlens)
        )

    @staticmethod
    def get_seqlens(
        fwd_metadata: Metadata, bwd_metadata: Metadata
    ):
        fwd_seq_lens = _get_ragged_seqlen(
            fwd_metadata.seq_len, fwd_metadata.num_seqs
        )
        bwd_seq_lens = _get_ragged_seqlen(
            bwd_metadata.seq_len, bwd_metadata.num_seqs
        )
        return SeqLens(fwd_seq_lens, bwd_seq_lens)


@dataclass
class LogicalShape:
    """
    Logical shape for input and output tensors. By logical,
    KV send shape is (cp_degree, num_tokens, hidden_size);
    Other tensors have the same shape as physical shape.
    """
    send_shape: Union[torch.Size, Sequence[torch.Size]]
    recv_shape: Union[torch.Size, Sequence[torch.Size]]
    def get_slice(self, rank):
        return LogicalShape(
            self.send_shape[rank], self.recv_shape[rank],
        )

    @staticmethod
    def get_shape(
        mlp_to_attn_metadata: Metadata, hidden_size: int,
        mlp_num_tokens: int
    ):
        world_size = mlp_to_attn_metadata.world_size
        assert mlp_to_attn_metadata.dst_rank.shape[0] == world_size
        if mlp_to_attn_metadata.dst_rank.ndim == 2:
            token_layout = (mlp_num_tokens,)
        else:
            assert mlp_to_attn_metadata.dst_rank.ndim == 3
            max_cp = mlp_to_attn_metadata.dst_rank.shape[2]
            token_layout = (max_cp, mlp_num_tokens)

        send_shape = tuple(
            token_layout + (hidden_size,)
            for _ in range(world_size)
        )
        recv_shape = tuple(
            (nt, hidden_size)
            for nt in mlp_to_attn_metadata.num_total_recv_tokens
        )
        return LogicalShape(send_shape, recv_shape)


@dataclass
class FastAlltoAllMetadata:
    """
    NOTE: FastAlltoAllMetadata has some duplicated fields with Metadata.
    With FastAlltoAll enabled, the original Metadata is not used.
    """
    # sender_send_offset, sender_transfer_sz, sender_recv_offset, recver_transfer_sz
    fa2a_metadata: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    # List of (world_size,) tensors, each of shape (num_sequences,). If a slice, no world_size dimension.
    # metadata on the sender side:
    #   the offset to copy each sequence to the buffer, ordered by the sender's sequence idx
    #       NOTE: for kv sender's sequence idx, to make it directly available to the reverse operation,
    #         we use the (cp_id, seq_idx) order instead of the (seq_idx, cp_id) order
    #   the offset to copy each sequence from the buffer, ordered by the recver's sequence idx
    send_memcpy_metadata: Sequence[_Tensor_Or_Tensor_List]
    recv_memcpy_metadata: Sequence[_Tensor_Or_Tensor_List]
    my_rank_send_offset: Union[int, List[int]]
    my_rank_recv_offset: Union[int, List[int]]
    my_rank_send_sz: Union[int, List[int]]
    seq_lens: Sequence[SeqLens]
    # Num received / send tokens for each tensor (i.e. (q,k) or (attn,))/
    # This is to construct the recv buffer size.
    # NOTE: for Q/KV backward, this is just a placeholder and we don't use it.
    tensor_shape: Sequence[LogicalShape]
    # List of kv replica mask for each rank. (or this rank)
    # shape is (num_local_seqs, cp_degree).
    kv_replica_mask: Optional[_Tensor_Or_Tensor_List] = None
    # Debug setting
    single_stream: bool = False

    def get_slice(self, rank):
        """
        Returns the metadata for the given rank.
        """
        fa2a_metadata = tuple(t[rank] for t in self.fa2a_metadata)
        send_memcpy_metadata = tuple(t[rank] for t in self.send_memcpy_metadata)
        recv_memcpy_metadata = tuple(t[rank] for t in self.recv_memcpy_metadata)
        seq_lens = tuple(
            sl.get_slice(rank) for sl in self.seq_lens
        )
        tensor_shape = tuple(
            ts.get_slice(rank) for ts in self.tensor_shape
        )
        return FastAlltoAllMetadata(
            fa2a_metadata, send_memcpy_metadata, recv_memcpy_metadata,
            self.my_rank_send_offset[rank],
            self.my_rank_recv_offset[rank],
            self.my_rank_send_sz[rank],
            seq_lens,
            tensor_shape,
            kv_replica_mask=(
                self.kv_replica_mask[rank] if self.kv_replica_mask is not None else None
            ),
            single_stream=self.single_stream,
        )

    def normalize(self):
        """To device and transfer dtype."""
        return FastAlltoAllMetadata(
            tuple(t.cuda().to(torch.uint64).contiguous() for t in self.fa2a_metadata),
            tuple(t.cuda().to(torch.int64).contiguous() for t in self.send_memcpy_metadata),
            tuple(t.cuda().to(torch.int64).contiguous() for t in self.recv_memcpy_metadata),
            self.my_rank_send_offset,
            self.my_rank_recv_offset,
            self.my_rank_send_sz,
            tuple(t.normalize() for t in self.seq_lens),
            self.tensor_shape,
            kv_replica_mask=(
                self.kv_replica_mask.cuda().to(torch.int8).contiguous()
                if self.kv_replica_mask is not None else None
            ),
            single_stream=self.single_stream,
        )


def _filter_nonzero_by_mask(tensor: torch.Tensor, mask: torch.Tensor, num_received: torch.Tensor) -> torch.Tensor:
    tensor = tensor.flatten(start_dim=1)
    mask = mask.flatten(start_dim=1).bool()
    condensed = tensor[mask]
    output_tuple = torch.split(condensed, num_received.tolist())
    return output_tuple


def _get_diag(tensor: torch.Tensor, world_size: int):
    assert tensor.shape == (world_size, world_size)
    return torch.diagonal(tensor).flatten().cpu().tolist()


def _get_my_rank_from_metadata(fa2a_metadata: Sequence[torch.Tensor]):
    (sender_send_disp, sender_transfer_sz,
        sender_recv_disp, recver_transfer_sz) = fa2a_metadata
    world_size = sender_send_disp.shape[0]
    return {
        "my_rank_send_offset": _get_diag(sender_send_disp, world_size),
        "my_rank_recv_offset": _get_diag(sender_recv_disp, world_size),
        "my_rank_send_sz": _get_diag(sender_transfer_sz, world_size)
    }


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
    num_seqs: torch.Tensor,
    # kv dispatch metadata
    kv_dst_rank: torch.Tensor,  # shape (world_size, num_local_seqs, cp_degree)
    kv_seq_len: torch.Tensor,  # shape (world_size, num_local_seqs)
    kv_dst_global_seq_id: torch.Tensor,  # shape (world_size, num_local_seqs, cp_degree)
    num_send_tokens_kv: torch.Tensor,
    bytes_q: int, bytes_kv: int,
    # These values are passed only to be added to the FastAlltoAllMetadata at the end.
    seqlens: Sequence[SeqLens],
    tensor_shape: Sequence[LogicalShape],
    kv_replica_mask: Sequence[torch.Tensor],
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
    # trim by num_seqs.
    fwd_send_memcpy_metadata = tuple(
        tuple(
            tt.squeeze(0)[..., :num_seqs[i]]
            for i, tt in enumerate(torch.split(t, 1, dim=0))
        )
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

    my_rank_vals = _get_my_rank_from_metadata(fwd_fa2a_metadata)

    return FastAlltoAllMetadata(
        fwd_fa2a_metadata, fwd_send_memcpy_metadata, fwd_recv_memcpy_metadata,
        **my_rank_vals, seq_lens=seqlens, tensor_shape=tensor_shape,
        kv_replica_mask=kv_replica_mask,
    )


def compute_reverse_a2a_layout_metadata(
    fwd_metadata: FastAlltoAllMetadata
):
    # TODO: as bwd values are mainly the same as fwd values
    # we should only store those that are different.

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

    bwd_seqlens = tuple(
        SeqLens(seq_len.recv_seqlens, seq_len.send_seqlens)
        for seq_len in fwd_metadata.seq_lens
    )
    bwd_tensor_shape = tuple(
        LogicalShape(ts.recv_shape, ts.send_shape)
        for ts in fwd_metadata.tensor_shape
    )

    my_rank_vals = _get_my_rank_from_metadata(bwd_fa2a_metadata)
    return FastAlltoAllMetadata(
        bwd_fa2a_metadata, send_memcpy_metadata, recv_memcpy_metadata,
        **my_rank_vals, seq_lens=bwd_seqlens, tensor_shape=bwd_tensor_shape,
        kv_replica_mask=fwd_metadata.kv_replica_mask,
    )


def compute_backward_attn_out_a2a_layout_metadata(
    tokens_to_dst_per_dispatch_q: torch.Tensor,
    seq_to_dst_mask_q: torch.Tensor,
    recver_transfer_sz_q: torch.Tensor,
    num_recv_seqs_q: torch.Tensor,
    bytes_q: int,
    seq_lens: Sequence[SeqLens],
    tensor_shape: Sequence[LogicalShape],
):
    """
    Unlike qkv, the backward is easier to compute. We do it first, then the forward.
    """
    assert tokens_to_dst_per_dispatch_q.ndim == 3
    world_size = tokens_to_dst_per_dispatch_q.shape[0]
    max_num_send_seq = tokens_to_dst_per_dispatch_q.shape[1]
    assert tokens_to_dst_per_dispatch_q.shape == (world_size, max_num_send_seq, world_size)
    assert num_recv_seqs_q.shape == (world_size,)
    seq_q_shape = (world_size, max_num_send_seq)

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
    bwd_send_memcpy_metadata = (
        tuple(t.squeeze(0) for t in torch.split(seq_offset_q, 1, dim=0)),
    )

    seq_intra_src_rank_offset_q = seq_intra_dst_rank_offset_q.permute(2, 0, 1)
    seq_inter_src_rank_offset = exclusive_cumsum(recver_transfer_sz, dim=1)
    seq_recv_offset_q = (
        seq_intra_src_rank_offset_q +
        seq_inter_src_rank_offset.unsqueeze(-1)
    )

    seq_dispatch_mask_recv_q = seq_dispatch_mask_q.permute(2, 0, 1)

    seq_recv_offset_q = _filter_nonzero_by_mask(
        seq_recv_offset_q, seq_dispatch_mask_recv_q, num_recv_seqs_q
    )

    bwd_recv_memcpy_metadata = (seq_recv_offset_q,)
    my_rank_vals = _get_my_rank_from_metadata(bwd_fa2a_metadata)
    return FastAlltoAllMetadata(
        bwd_fa2a_metadata, bwd_send_memcpy_metadata, bwd_recv_memcpy_metadata,
        **my_rank_vals, seq_lens=seq_lens, tensor_shape=tensor_shape,
        kv_replica_mask=None,
    )
    

def compute_fa2a_metadata_from_logical_metadata(
    fwd_metadata_q: Metadata,
    bwd_metadata_q: Metadata,
    fwd_metadata_kv: Metadata,
    bwd_metadata_kv: Metadata,
    intermediates,
    mlp_num_tokens: int,
    hidden_size_q: int,
    hidden_size_k: int,
    element_size: int,  # dtype's size
):
    (
        tokens_to_dst_per_dispatch_q, q_seq_to_dst,
        num_received_seqs_q, kv_dst_global_seq_id
    ) = intermediates

    bytes_q = element_size * hidden_size_q
    bytes_k = element_size * hidden_size_k

    recver_transfer_sz_q = (
        fwd_metadata_q.num_recv_tokens * bytes_q
    )[..., :-1]
    recver_transfer_sz_kv = (
        fwd_metadata_kv.num_recv_tokens * bytes_k
    )[..., :-1]

    num_recv_seqs_q = bwd_metadata_q.num_seqs
    num_recv_seqs_kv = bwd_metadata_kv.num_seqs
    num_seqs = fwd_metadata_q.num_seqs

    num_send_tokens_kv = bwd_metadata_kv.num_recv_tokens[..., :-1]
    seqlens = [SeqLens.get_seqlens(fwd_metadata_q, bwd_metadata_q),
               SeqLens.get_seqlens(fwd_metadata_kv, bwd_metadata_kv)]
    tensor_shape = [
        LogicalShape.get_shape(fwd_metadata_q, hidden_size_q, mlp_num_tokens),
        LogicalShape.get_shape(fwd_metadata_kv, hidden_size_k, mlp_num_tokens),
    ]
    kv_replica_mask = fwd_metadata_kv.dst_rank >= 0
    kv_replica_mask = tuple(
        kv_replica_mask[i][:num_seq].to(torch.int8)
        for i, num_seq in enumerate(num_seqs)
    )

    qkv_fwd_fa2a_metadata = compute_forward_qkv_a2a_layout_meatadata(
        tokens_to_dst_per_dispatch_q.squeeze(2), q_seq_to_dst,
        recver_transfer_sz_q, recver_transfer_sz_kv,
        num_recv_seqs_q, num_recv_seqs_kv, num_seqs,
        fwd_metadata_kv.dst_rank, fwd_metadata_kv.seq_len,
        kv_dst_global_seq_id, num_send_tokens_kv,
        bytes_q, bytes_k,
        seqlens, tensor_shape, kv_replica_mask,
    )
    qkv_bwd_fa2a_metadata = compute_reverse_a2a_layout_metadata(
        qkv_fwd_fa2a_metadata,
    )

    # the backward seqlens of attn out equals that of forward q.
    seqlens = [seqlens[0]]
    tensor_shape = [tensor_shape[0]]
    attn_out_bwd_fa2a_metadata = compute_backward_attn_out_a2a_layout_metadata(
        tokens_to_dst_per_dispatch_q.squeeze(2), q_seq_to_dst,
        recver_transfer_sz_q, num_recv_seqs_q, bytes_q,
        seq_lens=seqlens, tensor_shape=tensor_shape,
    )
    attn_out_fwd_fa2a_metadata = compute_reverse_a2a_layout_metadata(
        attn_out_bwd_fa2a_metadata,
    )
    fa2a_metadata = (
        qkv_fwd_fa2a_metadata,
        qkv_bwd_fa2a_metadata,
        attn_out_fwd_fa2a_metadata,
        attn_out_bwd_fa2a_metadata,
    )
    return fa2a_metadata


def compute_e2e_fa2a_metadata(
    mlp_seq_len: torch.Tensor,  # shape of (world_size, max_num_local_seqs)
    mlp_num_seqs: torch.Tensor,
    mlp_q_dispatch: torch.Tensor,  # shape of (world_size, max_num_local_seqs)
    kv_to_q_mapping: torch.Tensor,
    kv_to_q_rank: torch.Tensor,
    kv_context_size: torch.Tensor,
    q_to_num_kv_seq: torch.Tensor,
    q_to_num_kv_token: torch.Tensor,
    hidden_size_q: int,
    hidden_size_k: int,
    element_size: int,  # dtype's size
):
    (
        fwd_metadata_q, bwd_metadata_q, fwd_metadata_kv, bwd_metadata_kv, fa_params, intermediates
    ) = compute_e2e_metadata(
        mlp_seq_len, mlp_num_seqs,
        mlp_q_dispatch,
        kv_to_q_mapping,
        kv_to_q_rank,
        kv_context_size,
        q_to_num_kv_seq,
        q_to_num_kv_token,
        return_intermediate=True
    )
    mlp_num_tokens = mlp_seq_len.sum(dim=1).max().item()
    fa2a_metadata = compute_fa2a_metadata_from_logical_metadata(
        fwd_metadata_q, bwd_metadata_q, fwd_metadata_kv, bwd_metadata_kv,
        intermediates, mlp_num_tokens, hidden_size_q, hidden_size_k,
        element_size
    )
    return (
        fwd_metadata_q, bwd_metadata_q, fwd_metadata_kv, bwd_metadata_kv, fa_params, fa2a_metadata
    )
