# NOTE: this file should be deprecated:
# We have a new all2all implementation, and a new kv metadata generation.

import math
from typing import List, Tuple, Optional

import ray
import torch

from d2.runtime.attn_kernels.ops import (
    dispatch_no_cp_tensor, dispatch_kv_backward, dispatch_qkv,
    nvshmem_barrier_all, nvshmem_get_unique_id, DispatcherWrapper,
)
from d2.runtime.inplace_metadata import compute_metadata, Metadata

from test_util import gen_seq_lens
from test_comm_metadata import orchestrate_simulate


@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.dispatcher = None

    def init_comm(self,
        uid: torch.Tensor, stride_q: int, stride_kv: int, max_tokens_query: int, max_tokens_key_value: int,
        use_fast: bool=False
    ):
        DispatcherWrapper.init(
            q_stride=stride_q,
            kv_stride=stride_kv,
            max_tokens_query=max_tokens_query,
            max_tokens_key_value=max_tokens_key_value,
            rank=self.rank,
            world_size=self.world_size,
            uid=uid,
        )
        self.nvshmem_initialized = True

    def test_qkv(self, seed: int,
                 fwd_q_metadata: Metadata, rev_q_metadata: Metadata,
                 fwd_kv_metadata: Metadata, rev_kv_metadata: Metadata,
                 tensor_q: torch.Tensor, tensor_kv: torch.Tensor,
                 ):
        fwd_q_metadata = fwd_q_metadata.cuda().normalize_dtype()
        rev_q_metadata = rev_q_metadata.cuda().normalize_dtype()
        fwd_kv_metadata = fwd_kv_metadata.cuda().normalize_dtype()
        rev_kv_metadata = rev_kv_metadata.cuda().normalize_dtype()
        tensor_q = tensor_q.cuda()
        tensor_kv = tensor_kv.cuda()

        hidden_size_q = tensor_q.shape[-1]
        hidden_size_kv = tensor_kv.shape[-1]
        torch.manual_seed(seed)

        # of shape (num_seqs, cp_degree)
        cp_degree = fwd_kv_metadata.dst_rank.shape[1]
        num_tokens_q = tensor_q.shape[0]
        num_tokens_kv = tensor_kv.shape[0]
        assert num_tokens_q == num_tokens_kv
        num_tokens = num_tokens_q

        # forward communication buffer
        num_received_tokens_q = fwd_q_metadata.num_recv_tokens[-1]
        num_received_tokens_kv = fwd_kv_metadata.num_recv_tokens[-1]
        dst_tensor_q = torch.zeros(
            (num_received_tokens_q, hidden_size_q), dtype=tensor_q.dtype, device=tensor_q.device
        )
        torch.cuda.synchronize()
        dst_tensor_kv = torch.zeros(
            (num_received_tokens_kv, hidden_size_kv), dtype=tensor_kv.dtype, device=tensor_kv.device
        )
        dispatch_qkv(
            DispatcherWrapper.get_instance(),
            tensor_q, dst_tensor_q, fwd_q_metadata,
            tensor_kv, dst_tensor_kv, fwd_kv_metadata,
        )
        nvshmem_barrier_all()
        torch.cuda.synchronize()
        # print(f"rank {self.rank} fwd done", flush=True)

        # reverse communication buffer
        back_tensor_shape_q = (num_tokens, hidden_size_q,)
        back_tensor_shape_kv = (num_tokens * cp_degree, hidden_size_kv,)
        back_tensor_q = torch.zeros(
            back_tensor_shape_q, dtype=tensor_q.dtype, device=tensor_q.device
        )
        back_tensor_kv = torch.zeros(
            back_tensor_shape_kv, dtype=tensor_kv.dtype, device=tensor_kv.device
        ).contiguous()
        dispatch_no_cp_tensor(
            DispatcherWrapper.get_instance(),
            dst_tensor_q, back_tensor_q, rev_q_metadata,
        )
        nvshmem_barrier_all()

        dispatch_kv_backward(
            DispatcherWrapper.get_instance(),
            dst_tensor_kv, back_tensor_kv, rev_kv_metadata,
        )
        nvshmem_barrier_all()
        torch.cuda.synchronize()
        # print(f"rank {self.rank} bwd communication done", flush=True)

        back_tensor_kv = back_tensor_kv.reshape(
            (cp_degree, num_tokens, hidden_size_kv)
        )
        dedup_mask = torch.zeros_like(back_tensor_kv)
        tot_tokens = 0
        for j in range(fwd_kv_metadata.seq_len.shape[0]):
            seq_len = fwd_kv_metadata.seq_len[j].int()
            for k in range(cp_degree):
                if fwd_kv_metadata.dst_rank[j, k] >= 0:
                    dedup_mask[k, tot_tokens:tot_tokens+seq_len] = 1
                    assert rev_kv_metadata.seq_recv_mask[j, k] == 1
                else:
                    assert rev_kv_metadata.seq_recv_mask[j, k] == 0
            tot_tokens += seq_len

        back_tensor_kv_dedup = back_tensor_kv.sum(dim=0) / dedup_mask.sum(dim=0)
        torch.testing.assert_close(
            back_tensor_kv_dedup.unsqueeze(0).repeat(cp_degree, 1, 1) * dedup_mask, back_tensor_kv
        )
        torch.testing.assert_close(tensor_q, back_tensor_q)
        torch.testing.assert_close(tensor_kv, back_tensor_kv_dedup)
        torch.cuda.synchronize()
        return {
            "dst_q": dst_tensor_q,
            "dst_kv": dst_tensor_kv,
            "rev_q": back_tensor_q,
            "rev_kv": back_tensor_kv_dedup,
        }


def create_testcase_qkv(
    seed: int, world_size: int, num_tokens: int, max_cp_degree: int, num_seqs: int,
    seqlen_multiple: int=1,
) -> Tuple[Metadata, Metadata, Metadata, Metadata, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a random sequence parallel dispatch plan that:
    - Generate sequences of different lengths. The total number of tokens at each rank
      is the same.
    - Generate a CP degree for each sequence (under a max_cp_degree).
    - Generate CP ranks for each CP shard of a sequence.
    - Create metadata accordingly.
    """
    torch.manual_seed(seed)
    ### Init dispatch plan ###
    assert num_tokens % (max_cp_degree * seqlen_multiple) == 0
    _num_tokens_shard = num_tokens // (max_cp_degree * seqlen_multiple)
    seq_lens = gen_seq_lens(world_size, num_seqs, _num_tokens_shard).long()
    # make sure each sequence is divisible by max_cp_degree.
    seq_lens *= max_cp_degree * seqlen_multiple
    # the CP for each sequence.
    log_cp_num = torch.randint(0, int(math.log2(max_cp_degree)) + 1, (world_size, num_seqs))
    cp_num = torch.pow(2, log_cp_num)
    # CP dst for each sequence. Masking by cp_num.
    cp_dst_helper = torch.rand((world_size, num_seqs, world_size)).argsort(dim=2)
    cp_dst = cp_dst_helper[:, :, :max_cp_degree]
    # For each (i, j), set cp_dst[i, j, cp_num[i, j]:] = -1
    mask = torch.arange(max_cp_degree).expand(world_size, num_seqs, max_cp_degree)
    cp_num_expanded = cp_num.unsqueeze(-1)
    mask = mask >= cp_num_expanded
    cp_dst[mask] = -1

    pad_len = torch.max(cp_num.sum(dim=1))
    cp_seq_lens = torch.zeros(world_size, pad_len, dtype=torch.int64)
    cp_query_dst = torch.ones(world_size, pad_len, dtype=torch.int64) * -1
    cp_kv_dst = torch.ones(world_size, pad_len, max_cp_degree, dtype=torch.int64) * -1
    cp_dst_kv_len = torch.zeros(world_size, pad_len, dtype=torch.int64) * -1
    for i in range(world_size):
        cp_seq_lens_local = []
        cp_query_dst_local = []
        cp_kv_dst_local = []
        cp_dst_kv_len_local = []

        for j in range(num_seqs):
            num_cp = cp_num[i, j]
            seq_len = seq_lens[i, j]
            seq_shard_len = seq_len // num_cp

            cp_seq_lens_local.append(seq_shard_len.reshape(1,).repeat(num_cp))
            cp_query_dst_local.append(cp_dst[i, j, :num_cp].flatten())
            seq_cp_dst_kv = cp_dst[i, j].reshape(1, -1).repeat(num_cp, 1)
            # kv total length of each CP shard.
            cp_dst_kv_len_local.append(
                seq_shard_len.reshape(1,) * torch.arange(
                    1, num_cp + 1, device=seq_shard_len.device, dtype=seq_shard_len.dtype
                )
            )
            # triangular mask of seq_cp_dst_kv. Mask value is always 0 but we want it -1.
            # FIXME: the following line leads to a number mismatch.
            kv_dst_mask = ((
                torch.arange(max_cp_degree).view(1, -1) # keep j > i and j <= a
                >= torch.arange(num_cp.item()).view(-1, 1)
            ) & (
                torch.arange(max_cp_degree).view(1, -1) < num_cp
            ))
            seq_cp_dst_kv_tril = (seq_cp_dst_kv + 1) * kv_dst_mask - 1
            cp_kv_dst_local.append(seq_cp_dst_kv_tril)
        
        cp_seq_lens_local = torch.cat(cp_seq_lens_local, dim=0)
        cp_query_dst_local = torch.cat(cp_query_dst_local, dim=0)
        cp_kv_dst_local = torch.cat(cp_kv_dst_local, dim=0)
        cp_dst_kv_len_local = torch.cat(cp_dst_kv_len_local, dim=0)
        # shape check:
        seq_shards = cp_seq_lens_local.shape[0]
        assert cp_seq_lens_local.shape == (seq_shards,)
        assert cp_query_dst_local.shape == (seq_shards,)
        assert cp_kv_dst_local.shape == (seq_shards, max_cp_degree)
        assert cp_dst_kv_len_local.shape == (seq_shards,)

        cp_seq_lens[i, :seq_shards] = cp_seq_lens_local
        cp_query_dst[i, :seq_shards] = cp_query_dst_local
        cp_kv_dst[i, :seq_shards, :] = cp_kv_dst_local
        cp_dst_kv_len[i, :seq_shards] = cp_dst_kv_len_local

    ### Init metadata ###
    fwd_q_metadata, rev_q_metadata = compute_metadata(cp_seq_lens, cp_query_dst)
    fwd_kv_metadata, rev_kv_metadata = compute_metadata(cp_seq_lens, cp_kv_dst)

    return (
        fwd_q_metadata, rev_q_metadata, fwd_kv_metadata, rev_kv_metadata,
        cp_kv_dst, cp_seq_lens, cp_query_dst, cp_dst_kv_len, seq_lens,
    )


def create_answer(
    fwd_q_metadata: Metadata, fwd_kv_metadata: Metadata, rev_q_metadata: Metadata, rev_kv_metadata: Metadata,
    cp_kv_dst: torch.Tensor, cp_seq_lens: torch.Tensor,
    world_size: int, num_tokens: int, hidden_size_q: int, hidden_size_kv: int, max_cp_degree: int,
):

    ### Init tensor ###
    tensor_q = torch.randn(world_size, num_tokens, hidden_size_q, dtype=torch.float16)
    tensor_kv = torch.randn(world_size, num_tokens, hidden_size_kv, dtype=torch.float16)

    ### Init output tensor ###
    max_recv_tokens_q = fwd_q_metadata.num_recv_tokens.max()
    max_recv_tokens_kv = fwd_kv_metadata.num_recv_tokens.max()
    output_tensor_q = torch.zeros((world_size, max_recv_tokens_q, hidden_size_q), dtype=tensor_q.dtype, device=tensor_q.device)
    output_tensor_kv = torch.zeros((world_size, max_recv_tokens_kv, hidden_size_kv), dtype=tensor_kv.dtype, device=tensor_kv.device)

    ### Run orchestrate ###
    output_tensor_q = orchestrate_simulate(tensor_q, output_tensor_q, fwd_q_metadata)
    output_tensor_kv = orchestrate_simulate(tensor_kv, output_tensor_kv, fwd_kv_metadata)

    back_tensor_q = torch.zeros((world_size, num_tokens, hidden_size_q), dtype=output_tensor_q.dtype, device=output_tensor_q.device)
    back_tensor_kv = torch.zeros((world_size, num_tokens * max_cp_degree, hidden_size_kv), dtype=output_tensor_kv.dtype, device=output_tensor_kv.device)

    back_tensor_q = orchestrate_simulate(output_tensor_q, back_tensor_q, rev_q_metadata)
    back_tensor_kv = orchestrate_simulate(output_tensor_kv, back_tensor_kv, rev_kv_metadata)

    assert torch.allclose(back_tensor_q, tensor_q)

    assert max_cp_degree > 1
    back_tensor_kv = back_tensor_kv.reshape(
        (world_size, max_cp_degree, num_tokens, hidden_size_kv)
    )

    back_tensor_mask = torch.zeros_like(back_tensor_kv)
    for i in range(world_size):
        tok = 0
        for j in range(cp_kv_dst.shape[1]):
            for k in range(max_cp_degree):
                if cp_kv_dst[i, j, k] >= 0:
                    back_tensor_mask[i, k, tok:tok+cp_seq_lens[i, j]] = 1
            tok += cp_seq_lens[i, j]

    back_tensor_kv_dedup = back_tensor_kv.sum(dim=1) / (back_tensor_mask != 0).sum(dim=1)

    torch.testing.assert_close(back_tensor_kv_dedup, tensor_kv)

    torch.testing.assert_close(
        back_tensor_kv_dedup.unsqueeze(1).repeat(1, max_cp_degree, 1, 1) * (back_tensor_mask != 0), back_tensor_kv
    )
    return tensor_q, tensor_kv, output_tensor_q, output_tensor_kv, back_tensor_q, back_tensor_kv_dedup


@torch.no_grad()
def test_qkv(
    seed, world_size, num_tokens, num_seqs, max_cp_degree, workers: List[ray.ObjectRef], hidden_size_q: int, hidden_size_kv: int,
):
    (
        fwd_q_metadata, rev_q_metadata,
        fwd_kv_metadata, rev_kv_metadata,
        cp_kv_dst, cp_seq_lens, cp_query_dst, cp_dst_kv_len, seq_lens,
    ) = create_testcase_qkv(
        seed, world_size, num_tokens, max_cp_degree, num_seqs,
    )
    (tensor_q, tensor_kv, output_tensor_q, output_tensor_kv, back_tensor_q, back_tensor_kv_dedup) = create_answer(
        fwd_q_metadata, fwd_kv_metadata, rev_q_metadata, rev_kv_metadata,
        cp_kv_dst, cp_seq_lens,
        world_size, num_tokens, hidden_size_q, hidden_size_kv, max_cp_degree
    )
    assert len(workers) == world_size
    refs = []

    for rank, worker in enumerate(workers):
        tensor_q_slice = tensor_q[rank]
        tensor_kv_slice = tensor_kv[rank]
        fwd_q_metadata_slice = fwd_q_metadata.get_slice(rank)
        fwd_kv_metadata_slice = fwd_kv_metadata.get_slice(rank)
        rev_q_metadata_slice = rev_q_metadata.get_slice(rank)
        rev_kv_metadata_slice = rev_kv_metadata.get_slice(rank)
        ref = worker.test_qkv.remote(
            seed,
            fwd_q_metadata_slice, rev_q_metadata_slice,
            fwd_kv_metadata_slice, rev_kv_metadata_slice,
            tensor_q_slice, tensor_kv_slice,
        )
        refs.append(ref)

    results = ray.get(refs)
    ### Test send dst tensor's correctness ###
    dst_q_tensor_outs = [r["dst_q"] for r in results]
    # The first dimension of each value in the list should be padded to the same as dst_tensor's dim 1
    max_dim1 = output_tensor_q.shape[1]
    dst_q_tensor_outs = [
        torch.cat([r, torch.zeros(max_dim1 - r.shape[0], *r.shape[1:], dtype=r.dtype, device=r.device)], dim=0).unsqueeze(0)
        for r in dst_q_tensor_outs
    ]
    dst_q_tensor_out = torch.cat(dst_q_tensor_outs, dim=0)
    assert dst_q_tensor_out.shape == output_tensor_q.shape
    torch.testing.assert_close(dst_q_tensor_out, output_tensor_q.to(dst_q_tensor_out.device))

    dst_kv_tensor_outs = [r["dst_kv"] for r in results]
    max_dim1 = output_tensor_kv.shape[1]
    dst_kv_tensor_outs = [
        torch.cat([r, torch.zeros(max_dim1 - r.shape[0], *r.shape[1:], dtype=r.dtype, device=r.device)], dim=0).unsqueeze(0)
        for r in dst_kv_tensor_outs
    ]
    dst_kv_tensor_out = torch.cat(dst_kv_tensor_outs, dim=0)
    assert dst_kv_tensor_out.shape == output_tensor_kv.shape
    torch.testing.assert_close(dst_kv_tensor_out, output_tensor_kv.to(dst_kv_tensor_out.device))

    ### Test backward reverse's correctness ###
    back_q_tensor_outs = [r["rev_q"].unsqueeze(0) for r in results]
    back_kv_tensor_outs = [r["rev_kv"].unsqueeze(0) for r in results]
    back_q_tensor_out = torch.cat(back_q_tensor_outs, dim=0)
    back_kv_tensor_out = torch.cat(back_kv_tensor_outs, dim=0)
    torch.testing.assert_close(back_q_tensor_out, back_tensor_q.to(back_q_tensor_out.device))
    torch.testing.assert_close(back_kv_tensor_out, back_tensor_kv_dedup.to(back_kv_tensor_out.device))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--max-cp-degree", type=int, default=2)
    parser.add_argument("--hidden-size-query", type=int, default=256)
    parser.add_argument("--hidden-size-kv", type=int, default=128)
    parser.add_argument("--num-seqs", type=int, default=3)
    args = parser.parse_args()
    ray.init()
    workers = [
        Worker.remote(i, args.world_size)
        for i in range(args.world_size)
    ]

    uid = nvshmem_get_unique_id()
    stride_q = args.hidden_size_query * torch.float16.itemsize
    stride_kv = args.hidden_size_kv * torch.float16.itemsize
    max_tokens_query = args.num_tokens * args.world_size
    max_tokens_key_value = args.num_tokens * args.world_size
    ray.get([
        worker.init_comm.remote(uid, stride_q, stride_kv, max_tokens_query, max_tokens_key_value)
        for worker in workers
    ])

    print("init done")
    test_qkv(
        0, args.world_size, args.num_tokens, args.num_seqs, args.max_cp_degree,
        workers, args.hidden_size_query, args.hidden_size_kv,
    )
    print("test done")
