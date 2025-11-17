from dataclasses import dataclass
import os
from typing import Optional, Sequence

from megatron.core.packed_seq_params import PackedSeqParams
import torch

from d2.runtime.metadata import (
    AlltoAllMetadata, LogicalShape, SeqLens, DedupGradSumMetadata,
    compute_reverse_a2a_layout_metadata, _get_my_rank_from_metadata,
)
from d2.runtime.shard_info import ShardInfo
from d2.runtime.utils import size_pad_by_int4, prepend_zero_fn, exclusive_cumsum


#### Helper classes to store intermediate states.
@dataclass(frozen=True, eq=True)
class _ShardID:
    doc_id: int
    logical_id: int
@dataclass(frozen=True, eq=True)
class _ShardPos:
    rank: int
    id_on_rank: int
@dataclass(frozen=True, eq=True)
class _ShardCommInfo:
    id_on_linear_rank: int
    id_on_attn_rank: int
    shard_glob_id: _ShardID
    replica_id: Optional[int]  # None for q, replica id for k/v

_ShardCommInfoList = list[_ShardCommInfo]

@dataclass(frozen=True, eq=True)
class _PairCommInfo:
    """Comm info from a rank to another rank."""
    shards: _ShardCommInfoList
    dedup_shards: _ShardCommInfoList
    is_main_copy: list[bool]  # for shards
    main_copy_id: list[int]  # for shards

_ShardList = list[_ShardID]
_PerRankShards = list[_ShardList]
_AllDocInfo = tuple[tuple[ShardInfo]]
_PerPairShardInfo = list[list[list[_ShardCommInfo]]]
_PerPairCommInfo = list[list[_PairCommInfo]]


def get_per_token_bytes(
    hidden_size_q: int,
    hidden_size_kv: int,
    lse_size_in_hidden_dtype: int,
    element_size: int,
    is_resend_qkv_in_bwd: bool,
    is_send_lse_in_fwd: bool,
):
    """Get the number of bytes per token."""
    assert not (is_send_lse_in_fwd and is_resend_qkv_in_bwd), "cannot be both fwd(send lse) and bwd(resend qkv)"
    # compute per token hidden size
    q_bytes = element_size * (
        # if resend qkv in bwd, will send q, attn_out, attn_out_grad, lse
        hidden_size_q * 3 + lse_size_in_hidden_dtype
        if is_resend_qkv_in_bwd else hidden_size_q
    )
    q_bytes_pad, _ = size_pad_by_int4(q_bytes, 1)
    k_bytes = element_size * hidden_size_kv
    assert k_bytes == size_pad_by_int4(k_bytes, 1)[0], "does not support padding kv now"

    attn_out_bytes = element_size * (
        # if will resend in bwd, 
        hidden_size_q + lse_size_in_hidden_dtype
        if is_send_lse_in_fwd else hidden_size_q
    )
    attn_out_bytes_pad, _ = size_pad_by_int4(attn_out_bytes, 1)
    return q_bytes_pad, k_bytes, attn_out_bytes_pad


def get_fa2a_metadata(send_bytes: torch.Tensor):
    """Get the fa2a metadata from the send_bytes matrix (index (i,j) means bytes from i to j)."""
    # TODO: remove the local communication, which is the major part.
    # This allows a much smaller nsys buffer.
    sender_send_disp = exclusive_cumsum(send_bytes, dim=1)
    sender_transfer_sz = send_bytes
    # (i,j) -> on i, send_bytes received from j
    recver_transfer_sz = send_bytes.transpose(0, 1).contiguous()
    # (i,j) -> on i, offset for buffer from j
    recver_recv_disp = exclusive_cumsum(recver_transfer_sz, dim=1)
    sender_recv_disp = recver_recv_disp.transpose(0, 1).contiguous()
    return (sender_send_disp, sender_transfer_sz, sender_recv_disp, recver_transfer_sz)


#### Tensor shape.
def _total_tokens(comm_infos: list[_ShardCommInfo], doc_info: _AllDocInfo):
    """Sum the number of tokens."""
    num_tokens = 0
    for comm_info in comm_infos:
        shard_id = comm_info.shard_glob_id
        shard_len = doc_info[shard_id.doc_id][shard_id.logical_id].shard_len
        num_tokens += shard_len
    return num_tokens


def _get_logical_shape(
    num_token_per_rank: torch.Tensor,
    world_size: int,
    hidden: int,
    is_kv_linear: bool = False,
    kv_max_cp: Optional[list[int]] = None,
):
    """Get the logical shape of the tensor on each rank."""
    assert num_token_per_rank.shape == (world_size,)
    num_total_tokens: list[int] = [
        num_token_per_rank[rank].item() for rank in range(world_size)
    ]
    return [
        (kv_max_cp[rank], nt, hidden) if is_kv_linear else (nt, hidden)
        for rank, nt in enumerate(num_total_tokens)
    ]


def _get_input_output_shape(linear_to_attn_q: _PerPairShardInfo,
                            linear_to_attn_k: _PerPairCommInfo,
                            max_num_dst_on_rank: list[int],
                            doc_info: _AllDocInfo,
                            q_bytes: int, k_bytes: int, element_size: int,):
    world_size = len(linear_to_attn_q)
    # count the number of final output tokens
    rank_send_tokens = torch.zeros((world_size,), dtype=torch.int64)
    rank_out_tokens_q = torch.zeros((world_size,), dtype=torch.int64)
    rank_out_tokens_k = torch.zeros((world_size,), dtype=torch.int64)
    # count the number of tokens communicated.
    pair_comm_tokens_q = torch.zeros((world_size, world_size), dtype=torch.int64)
    pair_comm_tokens_k = torch.zeros((world_size, world_size), dtype=torch.int64)

    for linear_rank in range(world_size):
        for attn_rank in range(world_size):
            q_comm_tokens = _total_tokens(
                linear_to_attn_q[linear_rank][attn_rank], doc_info
            )
            rank_send_tokens[linear_rank] += q_comm_tokens
            rank_out_tokens_q[attn_rank] += _total_tokens(
                linear_to_attn_q[linear_rank][attn_rank], doc_info
            )
            rank_out_tokens_k[attn_rank] += _total_tokens(
                linear_to_attn_k[linear_rank][attn_rank].shards, doc_info
            )
            pair_comm_tokens_q[linear_rank, attn_rank] = q_comm_tokens
            pair_comm_tokens_k[linear_rank, attn_rank] = _total_tokens(
                linear_to_attn_k[linear_rank][attn_rank].dedup_shards, doc_info
            )

    # logical shape
    q_hidden = q_bytes // element_size
    k_hidden = k_bytes // element_size
    q_send_shape = _get_logical_shape(rank_send_tokens, world_size, q_hidden,)
    q_recv_shape = _get_logical_shape(rank_out_tokens_q, world_size, q_hidden,)
    k_send_shape = _get_logical_shape(
        rank_send_tokens, world_size, k_hidden,
        is_kv_linear=True, kv_max_cp=max_num_dst_on_rank
    )
    k_recv_shape = _get_logical_shape(rank_out_tokens_k, world_size, k_hidden,)
    q_shape = LogicalShape(q_send_shape, q_recv_shape)
    k_shape = LogicalShape(k_send_shape, k_recv_shape)
    return q_shape, k_shape, pair_comm_tokens_q, pair_comm_tokens_k


#### Seqlens
def _collect_per_rank_shard_lens(
    doc_info: _AllDocInfo,
    on_rank_shard_id: _PerRankShards,
):
    """Get the seqlens of each shard on each rank."""
    world_size = len(on_rank_shard_id)
    seqlens = []
    for rank in range(world_size):
        rank_seqlens = []
        for shard_id in on_rank_shard_id[rank]:
            shard_info = doc_info[shard_id.doc_id][shard_id.logical_id]
            rank_seqlens.append(shard_info.shard_len)
        seqlens.append(torch.tensor(rank_seqlens, dtype=torch.int32))
    return seqlens


def _get_seqlens(doc_info: _AllDocInfo,
                 linear_shards_on_rank: _PerRankShards,
                 attn_q_shards_on_rank: _PerRankShards,
                 attn_k_shards_on_rank: _PerRankShards):
    """
    Compute the seqlens for a2a metadata.
    NOTE: this should only be used for k shards not deduped.
    """
    linear_seqlens = _collect_per_rank_shard_lens(doc_info, linear_shards_on_rank)
    attn_q_seqlens = _collect_per_rank_shard_lens(doc_info, attn_q_shards_on_rank)
    attn_k_seqlens = _collect_per_rank_shard_lens(doc_info, attn_k_shards_on_rank)
    linear_to_attn_seqlens_q = SeqLens(linear_seqlens, attn_q_seqlens)
    linear_to_attn_seqlens_k = SeqLens(linear_seqlens, attn_k_seqlens)
    return linear_to_attn_seqlens_q, linear_to_attn_seqlens_k
####


### Dedup items
def _dedup_k_shard(comm_infos: list[_ShardCommInfo]):
    """See _dedup_k_shard_world."""
    seen = set()
    shard_main_copy_id: dict[_ShardID, int] = {}
    id_to_main_copy_id = []
    is_main_copy = []
    dedup_shards = []
    for info_id, info in enumerate(comm_infos):
        shard_id = info.shard_glob_id
        if shard_id not in seen:
            dedup_shards.append(info)
            shard_main_copy_id[shard_id] = info_id
            seen.add(shard_id)
            is_main_copy.append(True)
        else:
            is_main_copy.append(False)
        id_to_main_copy_id.append(shard_main_copy_id[shard_id])
    return _PairCommInfo(comm_infos, dedup_shards, is_main_copy, id_to_main_copy_id)


def _get_dedup_grad_sum_metadata(
    attn_shard_lens: torch.Tensor,
    attn_shard_comm_info: list[_PairCommInfo],
):
    """
    Compute the grad sum metadata for dedup k/v shards on one rank.
    """
    assert attn_shard_lens.ndim == 1
    num_seq = attn_shard_lens.shape[0]

    # assert len(attn_shard_ids) == num_seq
    # max_num_copies = 

    copy_indices = [[] for _ in range(num_seq)]
    main_copy_mask = torch.zeros((num_seq,), dtype=torch.bool, device=attn_shard_lens.device)
    # Record the copy relations
    # NOTE: this might be confusing. We have two indices:
    # 1. the index of this shard on the attn layout. This is used for
    # `attn_shard_lens`, `copy_indices`, etc. This is obtained by
    # `id_on_attn_rank` of each shard.
    # 2. the index within elements of _PairCommInfo. This is obtained
    # by `_PairCommInfo.main_copy_id` or enumerating _PairCommInfo's
    # `is_main_copy` and `shards`.
    for pair_comm_info in attn_shard_comm_info:
        shards = pair_comm_info.shards
        for sid, shard in enumerate(pair_comm_info.shards):
            seq_idx = shard.id_on_attn_rank
            is_main_copy = pair_comm_info.is_main_copy[sid]
            if not is_main_copy:
                main_copy_id = pair_comm_info.main_copy_id[sid]
                main_copy = shards[main_copy_id]
                # check they are the same copy
                assert main_copy.shard_glob_id == shard.shard_glob_id
                assert main_copy.replica_id == shard.replica_id
                assert main_copy.id_on_linear_rank == shard.id_on_linear_rank
                # record this idx
                copy_indices[main_copy.id_on_attn_rank].append(seq_idx)
            main_copy_mask[seq_idx] = is_main_copy
    # transform to tensors
    num_copies = torch.tensor(
        [len(cis) for cis in copy_indices], dtype=torch.int32, device=attn_shard_lens.device
    )
    max_num_copies = max(1, num_copies.max().item())
    # start token id of each copy
    copy_start_id = torch.zeros(
        (num_seq, max_num_copies), dtype=torch.int64, device=attn_shard_lens.device
    )
    start_token_idx = exclusive_cumsum(attn_shard_lens, dim=0)
    for seq_idx, cis in enumerate(copy_indices):
        for cid, copy_idx in enumerate(cis):
            copy_start_id[seq_idx, cid] = start_token_idx[copy_idx]
    return DedupGradSumMetadata(num_copies, copy_start_id, main_copy_mask)


#### Offset
def _assign_offsets(
    cur_offset_send: int,
    cur_offset_recv: int,
    send_offset: list[torch.Tensor],
    recv_offset: list[torch.Tensor],
    doc_info: Sequence[Sequence[ShardInfo]],
    token_bytes: int,
    linear_rank: int,
    attn_rank: int,
    shards: _PairCommInfo | list[_ShardCommInfo],
    is_kv_linear: bool = False,
):
    """Assign the offsets for each shard in the send and recv buffer."""
    if is_kv_linear:
        assert isinstance(shards, _PairCommInfo)
        comm_infos = shards.shards
    else:
        assert isinstance(shards, list)
        comm_infos = shards

    for idx, comm_info in enumerate(comm_infos):
        id_on_linear_rank = comm_info.id_on_linear_rank
        id_on_attn_rank = comm_info.id_on_attn_rank
        shard_id = comm_info.shard_glob_id
        replica_id = comm_info.replica_id

        if is_kv_linear:
            is_main_copy = shards.is_main_copy[idx]
            if not is_main_copy:
                main_copy_id = shards.main_copy_id[idx]
                main_copy = comm_infos[main_copy_id]
                assert main_copy.shard_glob_id == shard_id
                assert main_copy.replica_id == replica_id

                recv_offset[attn_rank][id_on_attn_rank] = recv_offset[attn_rank][main_copy.id_on_attn_rank]
                continue

        if is_kv_linear:
            send_offset[linear_rank][replica_id, id_on_linear_rank] = cur_offset_send
        else:
            send_offset[linear_rank][id_on_linear_rank] = cur_offset_send
        recv_offset[attn_rank][id_on_attn_rank] = cur_offset_recv

        shard_len = doc_info[shard_id.doc_id][shard_id.logical_id].shard_len
        cur_offset_send += shard_len * token_bytes
        cur_offset_recv += shard_len * token_bytes
    return cur_offset_send, cur_offset_recv


def get_attn_metadata(
    seqlens_q: Sequence[torch.Tensor],
    seqlens_kv: Sequence[torch.Tensor] = None,
    get_packed_seq_params: bool=False
):
    """Get attn metadata from seqlens of each rank"""
    return_list = True
    if seqlens_kv is None:
        seqlens_kv = seqlens_q
    if isinstance(seqlens_q, torch.Tensor):
        assert isinstance(seqlens_kv, torch.Tensor)
        seqlens_q = [seqlens_q]
        seqlens_kv = [seqlens_kv]
        return_list = False

    world_size = len(seqlens_q)
    assert len(seqlens_kv) == world_size
    world_metadata = []
    for r in range(world_size):
        seqlen_q = seqlens_q[r]
        assert seqlen_q.ndim == 1
        cu_seqlens_q = prepend_zero_fn(seqlen_q.cumsum(dim=0))
        max_seqlen_q = seqlen_q.max().item()
        seqlen_kv = seqlens_kv[r]
        assert seqlen_kv.shape == seqlen_q.shape
        cu_seqlens_kv = prepend_zero_fn(seqlen_kv.cumsum(dim=0))
        max_seqlen_kv = seqlen_kv.max().item()
        out = dict(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv
        )
        if get_packed_seq_params:
            out = PackedSeqParams(qkv_format="thd", **out)
        world_metadata.append(out)
    if return_list:
        return world_metadata
    else:
        return world_metadata[0]


def _from_planner_output(
    world_size: int,
    scheduler_output: Sequence[Sequence[ShardInfo]],
    q_bytes: int,
    k_bytes: int,
    element_size: int,
    compute_attn_out_metadata: bool,
    attn_out_bytes: int,
):
    """From the shard info to fast all2all metadata of each rank."""

    scheduler_output = [
        sorted(doc_so, key=lambda shard: shard.logical_sid)
        for doc_so in scheduler_output
    ]
    num_doc = len(scheduler_output)
    linear_shards_on_rank: _PerRankShards = [[] for _ in range(world_size)]
    attn_q_shards_on_rank: _PerRankShards = [[] for _ in range(world_size)]
    attn_k_shards_on_rank: _PerRankShards = [[] for _ in range(world_size)]
    shard_pos_linear_layout: list[list[_ShardPos]] = [[] for _ in range(num_doc)]
    attn_q_seqlens_on_rank: list[list[int]] = [[] for _ in range(world_size)]
    attn_k_seqlens_on_rank: list[list[int]] = [[] for _ in range(world_size)]

    # linear_rank -> attn_rank -> list of each shard's info
    linear_to_attn_q: _PerPairShardInfo = [
        [[] for _ in range(world_size)] for _ in range(world_size)
    ]
    linear_to_attn_k_raw: _PerPairShardInfo = [
        [[] for _ in range(world_size)] for _ in range(world_size)
    ]
    # number of times a k shard is sent
    k_dsts: list[list[list[int]]] = [
        [[] for _ in range(len(scheduler_output[did]))] for did in range(num_doc)
    ]
    # Record shards on each rank in the linear and attn layout.
    # Record the linear_to_attn_q, linear_to_attn_k_raw, and num_send_k
    for doc_id, doc in enumerate(scheduler_output):
        context_len = 0
        for shard_id, shard in enumerate(doc):
            linear_rank = shard.rid
            attn_rank = shard.dispatch_rid
            # get the id of this shard, on both linear and attn layout.
            id_linear_rank = len(linear_shards_on_rank[linear_rank])
            id_attn_rank_q = len(attn_q_shards_on_rank[attn_rank])

            shard_linear_pos = _ShardPos(linear_rank, id_linear_rank)
            shard_pos_linear_layout[doc_id].append(shard_linear_pos)

            # query (or query + attn + attn grad + lse)
            shard_glob_id = _ShardID(doc_id, shard_id)
            linear_shards_on_rank[linear_rank].append(shard_glob_id)
            attn_q_shards_on_rank[attn_rank].append(shard_glob_id)
            # handle seqlens
            context_len += shard.shard_len
            attn_q_seqlens_on_rank[attn_rank].append(shard.shard_len)
            attn_k_seqlens_on_rank[attn_rank].append(context_len)
            # id_linear_rank -> buffer_id -> id_attn_rank_q
            linear_to_attn_q[linear_rank][attn_rank].append(
                _ShardCommInfo(id_linear_rank, id_attn_rank_q, shard_glob_id, None)
            )

            for k_shard_id in range(shard_id + 1):
                # Add a communication for each context k/v shard.
                # get the k shard pos and id.
                id_attnk_rank_k = len(attn_k_shards_on_rank[attn_rank])
                k_shard_glob_id = _ShardID(doc_id, k_shard_id)
                attn_k_shards_on_rank[attn_rank].append(k_shard_glob_id)

                k_shard_pos_linear = shard_pos_linear_layout[doc_id][k_shard_id]
                k_linear_rank = k_shard_pos_linear.rank
                id_linear_rank_k = k_shard_pos_linear.id_on_rank
                assert k_linear_rank == doc[k_shard_id].rid
                # all sends of this k shard to the same attn rank shares a replica.
                dsts = k_dsts[doc_id][k_shard_id]
                if attn_rank not in dsts:
                    dsts.append(attn_rank)
                replica_id = dsts.index(attn_rank)

                linear_to_attn_k_raw[k_linear_rank][attn_rank].append(
                    _ShardCommInfo(id_linear_rank_k, id_attnk_rank_k, k_shard_glob_id, replica_id)
                )
    linear_to_attn_k = [
        [
            _dedup_k_shard(linear_to_attn_k_raw[linear_rank][attn_rank])
            for attn_rank in range(world_size)
        ] for linear_rank in range(world_size)
    ]
    num_dst_k_on_rank: list[list[int]] = [
        [len(k_dsts[sid.doc_id][sid.logical_id]) for sid in linear_shards_on_rank[l_rank]]
        for l_rank in range(world_size)
    ]
    max_num_dst_on_rank: list[int] = [max(ns) for ns in num_dst_k_on_rank]

    #### Input and Output shape
    q_shape, k_shape, pair_comm_tokens_q, pair_comm_tokens_k = _get_input_output_shape(
        linear_to_attn_q, linear_to_attn_k,
        max_num_dst_on_rank, scheduler_output,
        q_bytes, k_bytes, element_size
    )

    #### A2A: total bytes and offsets
    # number of bytes from i to j
    linear_to_attn_qkv_bytes = (
        pair_comm_tokens_q * q_bytes + pair_comm_tokens_k * k_bytes * 2
    )
    # fast a2a metadata
    linear_to_attn_qkv_fa2a_metadata = get_fa2a_metadata(
        linear_to_attn_qkv_bytes
    )

    # qkv linear to attn metadata
    # init
    n_shards_send = [len(linear_shards_on_rank[l_rank]) for l_rank in range(world_size)]
    q_offset_sends = [
        torch.zeros((n_shards_send[l_rank],), dtype=torch.int64)
        for l_rank in range(world_size)
    ]
    k_offset_sends = [
        torch.ones((max_num_dst_on_rank[l_rank], n_shards_send[l_rank]), dtype=torch.int64) * -1
        for l_rank in range(world_size)
    ]
    v_offset_sends = [ko.clone() for ko in k_offset_sends]
    kv_replica_masks = [
        torch.zeros((n_shards_send[l_rank], max_num_dst_on_rank[l_rank]), dtype=torch.int8)
        for l_rank in range(world_size)
    ]
    q_offset_recvs = [
        torch.zeros((len(attn_q_shards_on_rank[a_rank]),), dtype=torch.int64)
        for a_rank in range(world_size)
    ]
    k_offset_recvs = [
        torch.zeros((len(attn_k_shards_on_rank[a_rank]),), dtype=torch.int64)
        for a_rank in range(world_size)
    ]
    v_offset_recvs = [torch.zeros_like(ko) for ko in k_offset_recvs]

    sender_send_disp = linear_to_attn_qkv_fa2a_metadata[0]
    sender_recv_disp = linear_to_attn_qkv_fa2a_metadata[2]
    # assign offsets in the buffer.
    for l_rank in range(world_size):
        for sid, shard_num_send in enumerate(num_dst_k_on_rank[l_rank]):
            kv_replica_masks[l_rank][sid, :shard_num_send] = 1
        for a_rank in range(world_size):
            cur_offset_send = sender_send_disp[l_rank][a_rank].item()
            cur_offset_recv = sender_recv_disp[l_rank][a_rank].item()
            cur_offset_send, cur_offset_recv = _assign_offsets(
                cur_offset_send, cur_offset_recv,
                q_offset_sends, q_offset_recvs, scheduler_output,
                q_bytes, l_rank, a_rank, linear_to_attn_q[l_rank][a_rank],
            )
            cur_offset_send, cur_offset_recv = _assign_offsets(
                cur_offset_send, cur_offset_recv,
                k_offset_sends, k_offset_recvs, scheduler_output,
                k_bytes, l_rank, a_rank, linear_to_attn_k[l_rank][a_rank],
                is_kv_linear=True,
            )
            cur_offset_send, cur_offset_recv = _assign_offsets(
                cur_offset_send, cur_offset_recv,
                v_offset_sends, v_offset_recvs, scheduler_output,
                k_bytes, l_rank, a_rank, linear_to_attn_k[l_rank][a_rank],
                is_kv_linear=True,
            )
    qkv_my_rank_fa2a_metadata = _get_my_rank_from_metadata(linear_to_attn_qkv_fa2a_metadata)

    # seqlens
    (linear_to_attn_seqlens_q,
     linear_to_attn_seqlens_k) = _get_seqlens(
        scheduler_output,
        linear_shards_on_rank,
        attn_q_shards_on_rank,
        attn_k_shards_on_rank,
    )
    qkv_linear_to_attn = AlltoAllMetadata(
        linear_to_attn_qkv_fa2a_metadata,
        send_memcpy_metadata=(
            tuple(q_offset_sends),
            tuple(k_offset_sends),
            tuple(v_offset_sends),
        ),
        recv_memcpy_metadata=(
            tuple(q_offset_recvs),
            tuple(k_offset_recvs),
            tuple(v_offset_recvs),
        ),
        **qkv_my_rank_fa2a_metadata,
        seq_lens=(linear_to_attn_seqlens_q, linear_to_attn_seqlens_k),
        tensor_shape=(q_shape, k_shape),
        kv_replica_mask=tuple(kv_replica_masks)
    )

    ## Flash Attn metadata
    attn_q_seqlens_on_rank = [
        torch.tensor(v, dtype=torch.int32) for v in attn_q_seqlens_on_rank
    ]
    attn_k_seqlens_on_rank = [
        torch.tensor(v, dtype=torch.int32) for v in attn_k_seqlens_on_rank
    ]
    attn_metadata = get_attn_metadata(
        attn_q_seqlens_on_rank, attn_k_seqlens_on_rank
    )

    ## Bwd metadata
    qkv_grad_attn_to_linear = compute_reverse_a2a_layout_metadata(
        qkv_linear_to_attn
    )
    dedup_metadata = []
    for a_rank in range(world_size):
        attn_k_shard_lens = linear_to_attn_seqlens_k.recv_seqlens[a_rank]
        rank_dedup = _get_dedup_grad_sum_metadata(
            attn_k_shard_lens,
            [linear_to_attn_k[l_rank][a_rank] for l_rank in range(world_size)]
        )
        dedup_metadata.append(rank_dedup)
    qkv_grad_attn_to_linear.kv_grad_send_dedup = dedup_metadata

    ## Attn out metadata
    if compute_attn_out_metadata:
        # fa2a metadata
        out_grad_linear_to_attn_bytes = (
            pair_comm_tokens_q * attn_out_bytes
        )
        linear_to_attn_out_grad_fa2a_metadata = get_fa2a_metadata(
            out_grad_linear_to_attn_bytes
        )
        # my rank fa2a metadata
        out_grad_my_rank_fa2a_metadata = _get_my_rank_from_metadata(
            linear_to_attn_out_grad_fa2a_metadata
        )
        # memcpy metadata
        out_grad_offset_sends = [torch.zeros_like(qo) for qo in q_offset_sends]
        out_grad_offset_recvs = [torch.zeros_like(qo) for qo in q_offset_recvs]
        sender_send_disp = linear_to_attn_out_grad_fa2a_metadata[0]
        sender_recv_disp = linear_to_attn_out_grad_fa2a_metadata[2]
        for l_rank in range(world_size):
            for a_rank in range(world_size):
                cur_offset_send = sender_send_disp[l_rank][a_rank].item()
                cur_offset_recv = sender_recv_disp[l_rank][a_rank].item()
                cur_offset_send, cur_offset_recv = _assign_offsets(
                    cur_offset_send, cur_offset_recv,
                    out_grad_offset_sends, out_grad_offset_recvs, scheduler_output,
                    attn_out_bytes, l_rank, a_rank, linear_to_attn_q[l_rank][a_rank],
                )
        # seqlen
        out_grad_seqlen = [linear_to_attn_seqlens_q]
        # shape
        out_hidden = attn_out_bytes // element_size
        pair_comm_tokens_out_grad = pair_comm_tokens_q
        out_grad_send_shape = _get_logical_shape(
            pair_comm_tokens_out_grad.sum(dim=1), world_size, out_hidden,
        )
        out_grad_recv_shape = _get_logical_shape(
            pair_comm_tokens_out_grad.sum(dim=0), world_size, out_hidden,
        )
        out_grad_linear_to_attn = AlltoAllMetadata(
            fa2a_metadata=linear_to_attn_out_grad_fa2a_metadata,
            send_memcpy_metadata=(tuple(out_grad_offset_sends),),
            recv_memcpy_metadata=(tuple(out_grad_offset_recvs),),
            **out_grad_my_rank_fa2a_metadata,
            seq_lens=out_grad_seqlen,
            tensor_shape=[LogicalShape(out_grad_send_shape, out_grad_recv_shape)],
            kv_replica_mask=None,
        )
        out_attn_to_linear = compute_reverse_a2a_layout_metadata(
            out_grad_linear_to_attn
        )
        return (
            qkv_linear_to_attn, qkv_grad_attn_to_linear,
            out_attn_to_linear, out_grad_linear_to_attn,
            attn_metadata,
        )
    else:
        return qkv_linear_to_attn, qkv_grad_attn_to_linear, attn_metadata


def from_planner_output(
    world_size: int,
    scheduler_output: Sequence[Sequence[ShardInfo]],
    hidden_size_q: int,
    hidden_size_kv: int,
    lse_size_in_hidden_dtype: int,
    element_size: int,
    is_pipeline_tick: bool,
    is_resend_qkv: bool = False,
):
    """NOTE: for PP, this only computes the Forward"""
    if is_resend_qkv:
        assert not is_pipeline_tick, "is_resend_qkv is for Non-PP but resend QKV to make memory balanced"
    is_send_lse_in_fwd = is_pipeline_tick or is_resend_qkv

    q_bytes, k_bytes, out_bytes = get_per_token_bytes(
        hidden_size_q, hidden_size_kv, lse_size_in_hidden_dtype, element_size,
        is_resend_qkv_in_bwd=False, is_send_lse_in_fwd=is_send_lse_in_fwd,
    )
    (qkv_linear_to_attn, qkv_grad_attn_to_linear, out_attn_to_linear,
     out_grad_linear_to_attn, attn_metadata) = _from_planner_output(
        world_size, scheduler_output, q_bytes, k_bytes, element_size,
        compute_attn_out_metadata=True, attn_out_bytes=out_bytes,
    )

    if os.getenv("D2_DEBUG_PRINT_METADATA", "0") == "1":
        import rich
        # qkv_linear_to_attn, qkv_grad_attn_to_linear, out_attn_to_linear,
        #  out_grad_linear_to_attn, attn_metadata
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank % 8 == 1:
            rich.print(f"from_planner_output: qkv_linear_to_attn=", qkv_linear_to_attn)
            rich.print(f"from_planner_output: qkv_grad_attn_to_linear=", qkv_grad_attn_to_linear)
            rich.print(f"from_planner_output: out_attn_to_linear=", out_attn_to_linear)
            rich.print(f"from_planner_output: out_grad_linear_to_attn=", out_grad_linear_to_attn)
            rich.print(f"from_planner_output: attn_metadata=", attn_metadata)

    if is_pipeline_tick:
        # Force them to None to avoid being misused.
        qkv_grad_attn_to_linear = None
        out_grad_linear_to_attn = None
    if is_resend_qkv:
        out_grad_linear_to_attn, qkv_grad_attn_to_linear, _ = backward_from_planner_output(
            world_size, scheduler_output, hidden_size_q, hidden_size_kv,
            lse_size_in_hidden_dtype, element_size
        )
    return (
        qkv_linear_to_attn, qkv_grad_attn_to_linear,
        out_attn_to_linear, out_grad_linear_to_attn,
        attn_metadata,
    )


def backward_from_planner_output(
    world_size: int,
    scheduler_output_bwd: Optional[Sequence[Sequence[ShardInfo]]],
    hidden_size_q: int,
    hidden_size_kv: int,
    lse_size_in_hidden_dtype: int,
    element_size: int,
):
    """
    Compute the backward communication metadatga for pipeline parallel. (resend qkv)
    """
    # backward 1: out_grad & out & q & k & v
    q_bytes, k_bytes, _ = get_per_token_bytes(
        hidden_size_q, hidden_size_kv, lse_size_in_hidden_dtype, element_size,
        is_resend_qkv_in_bwd=True, is_send_lse_in_fwd=False,
    )
    qkv_resend_and_out_grad_linear_to_attn, _, bwd_attn_metadata = _from_planner_output(
        world_size, scheduler_output_bwd, q_bytes, k_bytes, element_size,
        compute_attn_out_metadata=False, attn_out_bytes=None
    )
    # backward 2: q_grad, k_grad, v_grad
    q_bytes, k_bytes, _ = get_per_token_bytes(
        hidden_size_q, hidden_size_kv, lse_size_in_hidden_dtype, element_size,
        is_resend_qkv_in_bwd=False, is_send_lse_in_fwd=False,
    )
    _, qkv_grad_attn_to_linear, bwd_attn_metadata = _from_planner_output(
        world_size, scheduler_output_bwd, q_bytes, k_bytes, element_size,
        compute_attn_out_metadata=False, attn_out_bytes=None
    )

    if os.getenv("D2_DEBUG_PRINT_METADATA", "0") == "1":
        import rich
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank % 8 == 1:
            rich.print(f"backward_from_planner_output: qkv_resend_and_out_grad_linear_to_attn=", qkv_resend_and_out_grad_linear_to_attn.__better_print__())
            rich.print(f"backward_from_planner_output: qkv_grad_attn_to_linear=", qkv_grad_attn_to_linear.__better_print__())
            for i, metadata in enumerate(bwd_attn_metadata):
                rich.print(f"backward_from_planner_output: bwd_attn_metadata[{i}]=", metadata)

    return qkv_resend_and_out_grad_linear_to_attn, qkv_grad_attn_to_linear, bwd_attn_metadata
