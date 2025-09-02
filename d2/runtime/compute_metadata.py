from typing import Optional, Sequence, Tuple

from megatron.core.packed_seq_params import PackedSeqParams
import torch

from d2.runtime.fast_alltoall_metadata import (
    FastAlltoAllMetadata, LogicalShape, SeqLens,
    compute_reverse_a2a_layout_metadata,
    exclusive_cumsum, size_pad_by_int4, _get_my_rank_from_metadata,
)
from d2.runtime.inplace_metadata import prepend_zero_fn
from d2.runtime.shard_info import ShardInfo


# (doc_id, logical_id)
_ShardID = Tuple[int, int]
# (rank, id_on_rank)
_ShardPos = Tuple[int, int]
# (id_on_linear_rank, id_on_attn_rank, _ShardID, replica_id_for_kv)
_ShardCommInfo = Tuple[int, int, _ShardID, Optional[int]]


def send_bytes_to_fa2a_metadata(send_bytes: torch.Tensor):
    sender_send_disp = exclusive_cumsum(send_bytes, dim=1)
    sender_transfer_sz = send_bytes
    # (i,j) -> on i, send_bytes received from j
    recver_transfer_sz = send_bytes.transpose(0, 1).contiguous()
    # (i,j) -> on i, offset for buffer from j
    recver_recv_disp = exclusive_cumsum(recver_transfer_sz, dim=1)
    sender_recv_disp = recver_recv_disp.transpose(0, 1).contiguous()
    return (sender_send_disp, sender_transfer_sz, sender_recv_disp, recver_transfer_sz)


def get_per_token_bytes(
    hidden_size_q: int,
    hidden_size_kv: int,
    lse_size_in_hidden_dtype: int,
    element_size: int,
    is_resend_qkv_in_bwd: bool,
    is_send_lse_in_fwd: bool,
):
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


def _get_logical_shape(
    num_token_to_ranks: torch.Tensor,
    world_size: int,
    hidden: int,
    is_kv_linear: bool = False,
    kv_max_cp: Optional[list[int]] = None,
):
    assert num_token_to_ranks.shape == (world_size, world_size)
    num_total_tokens: list[int] = [
        num_token_to_ranks[rank].sum().item() for rank in range(world_size)
    ]
    return [
        (kv_max_cp[rank], nt, hidden) if is_kv_linear else (nt, hidden)
        for rank, nt in enumerate(num_total_tokens)
    ]


def _get_seqlens(
    doc_info: Sequence[Sequence[ShardInfo]],
    on_rank_doc_id: list[list[_ShardID]],
):
    world_size = len(on_rank_doc_id)
    seqlens = []
    for rank in range(world_size):
        rank_seqlens = []
        for doc_id, logical_id in on_rank_doc_id[rank]:
            shard_info = doc_info[doc_id][logical_id]
            rank_seqlens.append(shard_info.shard_len)
        seqlens.append(torch.tensor(rank_seqlens, dtype=torch.int32))
    return seqlens


def _assign_offsets(
    cur_offset_send: int,
    cur_offset_recv: int,
    send_offset: list[torch.Tensor],
    recv_offset: list[torch.Tensor],
    doc_info: Sequence[Sequence[ShardInfo]],
    token_bytes: int,
    linear_rank: int,
    attn_rank: int,
    comm_infos: list[_ShardCommInfo],
    is_kv_linear: bool = False,
):
    for comm_info in comm_infos:
        linear_rank_id, attn_rank_id, shard_glob_id, replica_id = comm_info
        if is_kv_linear:
            send_offset[linear_rank][replica_id, linear_rank_id] = cur_offset_send
        else:
            send_offset[linear_rank][linear_rank_id] = cur_offset_send
        recv_offset[attn_rank][attn_rank_id] = cur_offset_recv

        doc_id, shard_id = shard_glob_id
        shard_len = doc_info[doc_id][shard_id].shard_len
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
    linear_shards_on_rank: list[list[_ShardID]] = [[] for _ in range(world_size)]
    attn_q_shards_on_rank: list[list[_ShardID]] = [[] for _ in range(world_size)]
    attn_k_shards_on_rank: list[list[_ShardID]] = [[] for _ in range(world_size)]
    shard_pos_linear_layout: list[list[_ShardPos]] = [[] for _ in range(num_doc)]
    attn_q_seqlens_on_rank: list[list[int]] = [[] for _ in range(world_size)]
    attn_k_seqlens_on_rank: list[list[int]] = [[] for _ in range(world_size)]

    linear_to_attn_q: list[list[list[_ShardCommInfo]]] = [
        [[] for _ in range(world_size)] for _ in range(world_size)
    ]
    linear_to_attn_k: list[list[list[_ShardCommInfo]]] = [
        [[] for _ in range(world_size)] for _ in range(world_size)
    ]
    num_send_k: list[list[int]] = [
        [0 for _ in range(len(scheduler_output[did]))] for did in range(num_doc)
    ]
    for doc_id, doc in enumerate(scheduler_output):
        context_len = 0
        for shard_id, shard in enumerate(doc):
            linear_rank = shard.rid
            attn_rank = shard.dispatch_rid
            # id among all shards on this rank
            linear_rank_id = len(linear_shards_on_rank[linear_rank])
            attn_rank_id_q = len(attn_q_shards_on_rank[attn_rank])

            shard_linear_pos: _ShardPos = (linear_rank, linear_rank_id)
            shard_pos_linear_layout[doc_id].append(shard_linear_pos)

            # query (or query + attn + attn grad + lse)
            shard_glob_id: _ShardID = (doc_id, shard_id)
            linear_shards_on_rank[linear_rank].append(shard_glob_id)
            attn_q_shards_on_rank[attn_rank].append(shard_glob_id)
            # handle seqlens
            context_len += shard.shard_len
            attn_q_seqlens_on_rank[attn_rank].append(shard.shard_len)
            attn_k_seqlens_on_rank[attn_rank].append(context_len)
            # linear_rank_id -> buffer_id -> attn_rank_id_q
            linear_to_attn_q[linear_rank][attn_rank].append(
                (linear_rank_id, attn_rank_id_q, shard_glob_id, None)
            )

            for k_shard_id in range(shard_id + 1):
                attn_rank_id_k = len(attn_k_shards_on_rank[attn_rank])
                k_shard_glob_id: _ShardID = (doc_id, k_shard_id)
                attn_k_shards_on_rank[attn_rank].append(k_shard_glob_id)

                k_shard_pos_linear = shard_pos_linear_layout[doc_id][k_shard_id]
                k_linear_rank, k_linear_rank_id = k_shard_pos_linear
                assert k_linear_rank == doc[k_shard_id].rid
                replica_id = num_send_k[doc_id][k_shard_id]
                linear_to_attn_k[k_linear_rank][attn_rank].append(
                    (k_linear_rank_id, attn_rank_id_k, k_shard_glob_id, replica_id)
                )
                num_send_k[doc_id][k_shard_id] += 1

    # seqlens
    linear_seqlens = _get_seqlens(scheduler_output, linear_shards_on_rank)
    attn_q_seqlens = _get_seqlens(scheduler_output, attn_q_shards_on_rank)
    attn_k_seqlens = _get_seqlens(scheduler_output, attn_k_shards_on_rank)

    # num tokens from a rank to another rank
    linear_to_attn_num_tokens_q = torch.zeros((world_size, world_size), dtype=torch.int64)
    linear_to_attn_num_tokens_k = torch.zeros((world_size, world_size), dtype=torch.int64)
    for linear_rank in range(world_size):
        for attn_rank in range(world_size):
            for _, _, shard_id, _ in linear_to_attn_q[linear_rank][attn_rank]:
                doc_id, shard_id = shard_id
                shard_len = scheduler_output[doc_id][shard_id].shard_len
                linear_to_attn_num_tokens_q[linear_rank, attn_rank] += shard_len

            for _, _, shard_id, _ in linear_to_attn_k[linear_rank][attn_rank]:
                doc_id, shard_id = shard_id
                shard_len = scheduler_output[doc_id][shard_id].shard_len
                linear_to_attn_num_tokens_k[linear_rank, attn_rank] += shard_len

    num_send_k_on_rank: list[list[int]] = [
        [num_send_k[did][sid] for did, sid in linear_shards_on_rank[l_rank]]
        for l_rank in range(world_size)
    ]
    max_cp_on_ranks: list[int] = [
        max(num_send_k_on_rank[l_rank]) for l_rank in range(world_size)
    ]
    # logical shape
    q_hidden = q_bytes // element_size
    k_hidden = k_bytes // element_size
    q_send_shape = _get_logical_shape(
        linear_to_attn_num_tokens_q, world_size, q_hidden,
    )
    q_recv_shape = _get_logical_shape(
        linear_to_attn_num_tokens_q.T, world_size, q_hidden,
    )
    # we use _q instead of _k here because _k has duplication
    k_send_shape = _get_logical_shape(
        linear_to_attn_num_tokens_q, world_size, k_hidden,
        is_kv_linear=True, kv_max_cp=max_cp_on_ranks
    )
    k_recv_shape = _get_logical_shape(
        linear_to_attn_num_tokens_k.T, world_size, k_hidden,
    )

    # number of bytes from i to j
    linear_to_attn_qkv_bytes = (
        linear_to_attn_num_tokens_q * q_bytes +
        linear_to_attn_num_tokens_k * k_bytes * 2
    )
    # fast a2a metadata
    linear_to_attn_qkv_fa2a_metadata = send_bytes_to_fa2a_metadata(
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
        torch.ones((max_cp_on_ranks[l_rank], n_shards_send[l_rank]), dtype=torch.int64) * -1
        for l_rank in range(world_size)
    ]
    v_offset_sends = [ko.clone() for ko in k_offset_sends]
    kv_replica_masks = [
        torch.zeros((n_shards_send[l_rank], max_cp_on_ranks[l_rank]), dtype=torch.int8)
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
    for l_rank in range(world_size):
        for sid, shard_num_send in enumerate(num_send_k_on_rank[l_rank]):
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

    linear_to_attn_seqlens_q = SeqLens(linear_seqlens, attn_q_seqlens)
    linear_to_attn_seqlens_k = SeqLens(linear_seqlens, attn_k_seqlens)
    qkv_linear_to_attn = FastAlltoAllMetadata(
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
        tensor_shape=(
            LogicalShape(q_send_shape, q_recv_shape),
            LogicalShape(k_send_shape, k_recv_shape),
        ),
        kv_replica_mask=tuple(kv_replica_masks)
    )
    qkv_grad_attn_to_linear = compute_reverse_a2a_layout_metadata(
        qkv_linear_to_attn
    )

    attn_q_seqlens_on_rank = [
        torch.tensor(v, dtype=torch.int32) for v in attn_q_seqlens_on_rank
    ]
    attn_k_seqlens_on_rank = [
        torch.tensor(v, dtype=torch.int32) for v in attn_k_seqlens_on_rank
    ]
    attn_metadata = get_attn_metadata(
        attn_q_seqlens_on_rank, attn_k_seqlens_on_rank
    )
    if compute_attn_out_metadata:
        # fa2a metadata
        out_grad_linear_to_attn_bytes = (
            linear_to_attn_num_tokens_q * attn_out_bytes
        )
        linear_to_attn_out_grad_fa2a_metadata = send_bytes_to_fa2a_metadata(
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
        out_grad_seqlen = [SeqLens(linear_seqlens, attn_q_seqlens)]
        # shape
        out_hidden = attn_out_bytes // element_size
        out_grad_send_shape = _get_logical_shape(
            linear_to_attn_num_tokens_q, world_size, out_hidden,
        )
        out_grad_recv_shape = _get_logical_shape(
            linear_to_attn_num_tokens_q.T, world_size, out_hidden,
        )
        out_grad_linear_to_attn = FastAlltoAllMetadata(
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
    if is_pipeline_tick:
        # Force them to None to avoid being misused.
        qkv_grad_attn_to_linear = None
        out_grad_linear_to_attn = None
    if is_resend_qkv:
        qkv_grad_attn_to_linear, out_grad_linear_to_attn, _ = backward_from_planner_output(
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
    return qkv_resend_and_out_grad_linear_to_attn, qkv_grad_attn_to_linear, bwd_attn_metadata
