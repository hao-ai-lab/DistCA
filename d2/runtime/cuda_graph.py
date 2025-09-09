from dataclasses import dataclass
from enum import Enum

import torch

from d2.runtime.metadata import AlltoAllMetadata, SeqLens


#### For cuda graph
class AlltoAllPhase(Enum):
    FWD_QKV = 1
    BWD_QKV = 2
    FWD_ATTN = 3
    FWD_ATTN_LSE = 4
    BWD_ATTN = 5
    BWD_QKV_ATTN_LSE = 6

    def comm_non_repeat_only(self):
        return self in (AlltoAllPhase.FWD_ATTN, AlltoAllPhase.BWD_ATTN, AlltoAllPhase.FWD_ATTN_LSE)


@dataclass(eq=True, frozen=True)    # frozen to make it hashable
class AlltoAllMetadataRecord:
    num_seqs_pad: int
    max_cp_degree_pad: int
    phase: AlltoAllPhase
    ping_pong_id: int
    microbatch_id: int
####


def _create_static_metadata(
    num_seqs_pad: int, max_cp_degree_pad: int,
    phase: AlltoAllPhase, world_size: int,
):
    # whether kv will be sent
    comm_non_repeat_only = phase.comm_non_repeat_only()
    static_metadata = AlltoAllMetadata(
        fa2a_metadata=tuple(
            torch.zeros((world_size, world_size), dtype=torch.uint64) for _ in range(4)
        ),
        send_memcpy_metadata=tuple(
            torch.zeros((num_seqs_pad,), dtype=torch.int64),
            *(
                torch.zeros((max_cp_degree_pad, num_seqs_pad,), dtype=torch.int64)
                for _ in range(2 if not comm_non_repeat_only else 0)
            ),
        ),
        recv_memcpy_metadata=tuple(
            torch.zeros((num_seqs_pad,), dtype=torch.int64),
            *(
                torch.zeros((max_cp_degree_pad, num_seqs_pad,), dtype=torch.int64)
                for _ in range(2 if not comm_non_repeat_only else 0)
            ),
        ),
        my_rank_send_offset=0,
        my_rank_recv_offset=0,
        my_rank_send_sz=0,
        seq_lens=tuple(SeqLens(
            torch.zeros((num_seqs_pad,), dtype=torch.int64),
            torch.zeros((num_seqs_pad,), dtype=torch.int64),
        ) for _ in range(1 if comm_non_repeat_only else 2)),
        tensor_shape=tuple(None for _ in range(1 if comm_non_repeat_only else 2)),
        kv_replica_mask=(
            None if comm_non_repeat_only
            else torch.zeros((num_seqs_pad, max_cp_degree_pad), dtype=torch.int8)
        ),
        single_stream=None,
    ).normalize()
    return static_metadata


def _copy_to_static_metadata(
    metadata: AlltoAllMetadata, dst_static_metadata: AlltoAllMetadata,
    send_num_seqs: list[int], recv_num_seqs: list[int], max_cp_degree: int | None
):
    def copy_to_head(src: torch.Tensor, dst: torch.Tensor):
        dst = dst.flatten()
        src = src.flatten()
        dst[:src.numel()].copy_(src, non_blocking=True)

    for src, tgt in zip(metadata.fa2a_metadata, dst_static_metadata.fa2a_metadata):
        tgt.copy_(src, non_blocking=True)
    for i, src, tgt in enumerate(
        zip(metadata.send_memcpy_metadata, dst_static_metadata.send_memcpy_metadata)
    ):
        if i == 0:
            assert (send_num_seqs[i],) == src.shape
        else:
            assert (max_cp_degree, send_num_seqs[i]) == src.shape
        copy_to_head(src, tgt)
    for src, tgt in zip(metadata.recv_memcpy_metadata, dst_static_metadata.recv_memcpy_metadata):
        if i == 0:
            assert (recv_num_seqs[i],) == src.shape
        else:
            assert (max_cp_degree, recv_num_seqs[i]) == src.shape
        copy_to_head(src, tgt)
    for src_seqlen, tg_seqlen in zip(metadata.seq_lens, dst_static_metadata.seq_lens):
        copy_to_head(src_seqlen.send_seqlens, tg_seqlen.send_seqlens)
        copy_to_head(src_seqlen.recv_seqlens, tg_seqlen.recv_seqlens)
    if metadata.kv_replica_mask is not None:
        assert dst_static_metadata.kv_replica_mask is not None
        copy_to_head(metadata.kv_replica_mask, dst_static_metadata.kv_replica_mask)
    dst_static_metadata.my_rank_send_offset = metadata.my_rank_send_offset
    dst_static_metadata.my_rank_recv_offset = metadata.my_rank_recv_offset
    dst_static_metadata.my_rank_send_sz = metadata.my_rank_send_sz
    dst_static_metadata.tensor_shape = metadata.tensor_shape
    dst_static_metadata.single_stream = metadata.single_stream
    dst_static_metadata.send_num_seqs = send_num_seqs
    dst_static_metadata.recv_num_seqs = recv_num_seqs
    dst_static_metadata.max_cp_degree = max_cp_degree
    return dst_static_metadata


class StaticMetadataStorage:
    """
    Store static AlltoAllMetadata. This also checks that for each iteration,
    a static metadata is only copied once.
    """
    def __init__(self):
        self._storage: dict[AlltoAllMetadataRecord, AlltoAllMetadata] = {}
        self._used_keys: set[AlltoAllMetadataRecord] = set()

    def create_metadata(
        self, num_seqs_pad: int, max_cp_degree_pad: int,
        phase: AlltoAllPhase, ping_pong_id: int, microbatch_id: int,
        world_size: int,
    ):
        key = AlltoAllMetadataRecord(
            num_seqs_pad=num_seqs_pad, max_cp_degree_pad=max_cp_degree_pad,
            phase=phase, ping_pong_id=ping_pong_id, microbatch_id=microbatch_id
        )
        assert key not in self._storage
        metadata = _create_static_metadata(
            num_seqs_pad, max_cp_degree_pad, phase, world_size
        )
        self._storage[key] = metadata

    def copy_to_static_metadata(
        self, metadata: AlltoAllMetadata, num_seqs_pad: int, max_cp_degree_pad: int,
        phase: AlltoAllPhase, ping_pong_id: int, microbatch_id: int,
    ):
        key = AlltoAllMetadataRecord(
            num_seqs_pad=num_seqs_pad, max_cp_degree_pad=max_cp_degree_pad,
            phase=phase, ping_pong_id=ping_pong_id, microbatch_id=microbatch_id
        )
        comm_non_repeat_only = phase.comm_non_repeat_only()
        send_num_seqs = [sl.send_seqlens.shape[0] for sl in metadata.seq_lens]
        recv_num_seqs = [sl.recv_seqlens.shape[0] for sl in metadata.seq_lens]
        max_cp_degree = None if comm_non_repeat_only else metadata.kv_replica_mask.shape[1]
        assert all(ns <= num_seqs_pad for ns in send_num_seqs)
        assert all(ns <= num_seqs_pad for ns in recv_num_seqs)
        assert max_cp_degree is None or max_cp_degree <= max_cp_degree_pad

        assert key not in self._used_keys, f"static metadata already assigned value {key}"
        static_metadata = self._storage[key]
        self._used_keys.add(key)

        if comm_non_repeat_only:
            pass
        static_metadata = _copy_to_static_metadata(
            metadata, static_metadata, send_num_seqs, recv_num_seqs, max_cp_degree
        )
        return static_metadata

    def reset_iter(self):
        self._used_keys.clear()


metadata_storage = StaticMetadataStorage()
