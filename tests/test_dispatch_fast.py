"""
Launch command:

NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
torchrun --nnodes 1 --nproc_per_node $NUM_WORKER test_dispatch_fast.py \
    --world-size $NUM_WORKER

NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
torchrun --nnodes 1 --nproc_per_node 2 test_dispatch_fast.py \
    --world-size 2
"""
import torch

from d2.runtime.attn_kernels.ops import (
    nvshmem_barrier_all
)
# TODO: test fast_a2a_attn_out and its metadata by sending q_attn_layout back to q_mlp_layout via attn_out metadata.
from d2.runtime.attn_kernels.dispatch import (
    fast_a2a_qkv
)
from d2.runtime.inplace_metadata import Metadata
from d2.runtime.fast_alltoall_metadata import FastAlltoAllMetadata, compute_fa2a_metadata_from_logical_metadata

from test_comm_metadata import orchestrate_simulate
from test_util import BaseWorker, create_qkv_dispatch, init_worker_torch_distributed


class Worker(BaseWorker):

    def run_qkv(
        self, fa2a_metadata_fwd: FastAlltoAllMetadata,
        fa2a_metadata_rev: FastAlltoAllMetadata,
        tensor_q: torch.Tensor, tensor_kv: torch.Tensor,
        dispatch_mask: torch.Tensor,
    ):
        tensor_q = tensor_q.cuda()
        tensor_kv = tensor_kv.cuda()
        dispatch_mask = dispatch_mask.cuda().to(torch.int8)
        tensor_k = tensor_kv[:, :tensor_kv.shape[-1] // 2]
        tensor_v = tensor_kv[:, tensor_kv.shape[-1] // 2:]

        fa2a_metadata_fwd = fa2a_metadata_fwd.normalize()
        fa2a_metadata_rev = fa2a_metadata_rev.normalize()

        dtype = tensor_q.dtype
        device = tensor_q.device

        dst_tensor_q = torch.zeros(
            fa2a_metadata_fwd.tensor_shape[0].recv_shape,
            dtype=dtype, device=device
        )
        dst_tensor_k = torch.zeros(
            fa2a_metadata_fwd.tensor_shape[1].recv_shape,
            dtype=dtype, device=device
        )
        dst_tensor_v = dst_tensor_k.clone()
        fast_a2a_qkv(
            tensor_q, tensor_k, tensor_v,
            fa2a_metadata_fwd.kv_replica_mask,
            dst_tensor_q, dst_tensor_k, dst_tensor_v,
            fa2a_metadata_fwd.seq_lens[0].send_seqlens,
            fa2a_metadata_fwd.seq_lens[1].send_seqlens,
            fa2a_metadata_fwd.seq_lens[0].recv_seqlens,
            fa2a_metadata_fwd.seq_lens[1].recv_seqlens,
            *(fa2a_metadata_fwd.send_memcpy_metadata),
            *(fa2a_metadata_fwd.recv_memcpy_metadata),
            *(fa2a_metadata_fwd.fa2a_metadata),
            fa2a_metadata_fwd.my_rank_send_offset,
            fa2a_metadata_fwd.my_rank_recv_offset,
            fa2a_metadata_fwd.my_rank_send_sz,
            is_fwd=True
        )

        nvshmem_barrier_all()
        torch.cuda.synchronize()
        print(f"rank {self.rank} fwd done", flush=True)

        # reverse communication buffer
        back_tensor_q = torch.zeros(
            fa2a_metadata_rev.tensor_shape[0].recv_shape,
            dtype=dtype, device=device
        )
        back_tensor_k: torch.Tensor = torch.zeros(
            fa2a_metadata_rev.tensor_shape[1].recv_shape,
            dtype=dtype, device=device
        )
        back_tensor_v = back_tensor_k.clone()

        nvshmem_barrier_all()

        fast_a2a_qkv(
            dst_tensor_q, dst_tensor_k, dst_tensor_v,
            fa2a_metadata_rev.kv_replica_mask,
            back_tensor_q, back_tensor_k, back_tensor_v,
            fa2a_metadata_rev.seq_lens[0].send_seqlens,
            fa2a_metadata_rev.seq_lens[1].send_seqlens,
            fa2a_metadata_rev.seq_lens[0].recv_seqlens,
            fa2a_metadata_rev.seq_lens[1].recv_seqlens,
            *(fa2a_metadata_rev.send_memcpy_metadata),
            *(fa2a_metadata_rev.recv_memcpy_metadata),
            *(fa2a_metadata_rev.fa2a_metadata),
            fa2a_metadata_rev.my_rank_send_offset,
            fa2a_metadata_rev.my_rank_recv_offset,
            fa2a_metadata_rev.my_rank_send_sz,
            is_fwd=False
        )
        nvshmem_barrier_all()
        torch.cuda.synchronize()
        print(f"rank {self.rank} bwd communication done", flush=True)

        torch.testing.assert_close(tensor_q, back_tensor_q)
        dst_tensor_kv = torch.cat([dst_tensor_k, dst_tensor_v], dim=-1)
        back_tensor_kv = torch.cat([back_tensor_k, back_tensor_v], dim=-1)
        torch.cuda.synchronize()
        return {
            "dst_q": dst_tensor_q,
            "dst_kv": dst_tensor_kv,
            "rev_q": back_tensor_q,
            "rev_kv": back_tensor_kv,
        }


def create_answer(
    fwd_q_metadata: Metadata, fwd_kv_metadata: Metadata,
    rev_q_metadata: Metadata, rev_kv_metadata: Metadata,
    world_size: int, num_tokens: int, max_cp_degree: int,
    hidden_size_q: int, hidden_size_k: int,
):
    cp_kv_dst = fwd_kv_metadata.dst_rank
    cp_seq_lens = fwd_kv_metadata.seq_len
    hidden_size_kv = hidden_size_k * 2
    dtype = torch.float16
    device = "cpu"

    ### Init tensor ###
    tensor_q = torch.randn(
        world_size, num_tokens, hidden_size_q,
        dtype=dtype, device=device
    )
    tensor_kv = torch.randn(
        world_size, num_tokens, hidden_size_kv,
        dtype=dtype, device=device
    )

    ### Init output tensor ###
    max_recv_tokens_q = fwd_q_metadata.num_recv_tokens.max()
    max_recv_tokens_kv = fwd_kv_metadata.num_recv_tokens.max()
    output_tensor_q = torch.zeros((world_size, max_recv_tokens_q, hidden_size_q), dtype=dtype, device=device)
    output_tensor_kv = torch.zeros((world_size, max_recv_tokens_kv, hidden_size_kv), dtype=dtype, device=device)

    ### Run orchestrate ###
    output_tensor_q = orchestrate_simulate(tensor_q, output_tensor_q, fwd_q_metadata)
    output_tensor_kv = orchestrate_simulate(tensor_kv, output_tensor_kv, fwd_kv_metadata)

    back_tensor_q = torch.zeros((world_size, num_tokens, hidden_size_q), dtype=dtype, device=device)
    back_tensor_kv = torch.zeros(
        (world_size, num_tokens * max_cp_degree, hidden_size_kv),
        dtype=dtype, device=device)

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
    return tensor_q, tensor_kv, output_tensor_q, output_tensor_kv, back_tensor_q, back_tensor_kv


def create_test_case(world_size: int, total_seq_len: int, num_seqs: int,
                     max_cp_degree: int, hidden_size_q: int, hidden_size_k: int):
    (fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata,
     _, intermediates
     ) = create_qkv_dispatch(
        world_size, total_seq_len, num_seqs, max_cp_degree, return_intermediate=True
    )

    # create answers:
    (
        tensor_q, tensor_kv, output_tensor_q, output_tensor_kv,
        back_tensor_q, back_tensor_kv
    ) = create_answer(
        fwd_q_metadata, fwd_k_metadata,
        rev_q_metadata, rev_k_metadata,
        world_size, total_seq_len, max_cp_degree,
        hidden_size_q, hidden_size_k,
    )
    element_size = tensor_q.dtype.itemsize

    fa2a_metadata = compute_fa2a_metadata_from_logical_metadata(
        fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata,
        intermediates, total_seq_len, hidden_size_q, hidden_size_k,
        element_size
    )
    (fa2a_metadata_qkv_fwd, fa2a_metadata_qkv_rev,
     fa2a_metadata_attn_out_fwd, fa2a_metadata_attn_out_rev) = fa2a_metadata
    print("metadata compute done.")
    return (
        tensor_q, tensor_kv, output_tensor_q, output_tensor_kv,
        back_tensor_q, back_tensor_kv,
        fwd_q_metadata, fwd_k_metadata,
        fa2a_metadata_qkv_fwd, fa2a_metadata_qkv_rev,
        fa2a_metadata_attn_out_fwd, fa2a_metadata_attn_out_rev
    )


@torch.no_grad()
def test_qkv(
    seed, world_size, total_seq_len, num_seqs, max_cp_degree,
    worker: Worker, hidden_size_q: int, hidden_size_k: int,
):
    torch.manual_seed(seed)
    (
        tensor_q, tensor_kv, output_tensor_q, output_tensor_kv,
        back_tensor_q, back_tensor_kv,
        fwd_q_metadata, fwd_k_metadata,
        fa2a_metadata_qkv_fwd, fa2a_metadata_qkv_rev,
        fa2a_metadata_attn_out_fwd, fa2a_metadata_attn_out_rev
    ) = create_test_case(
        world_size, total_seq_len, num_seqs, max_cp_degree,
        hidden_size_q, hidden_size_k
    )

    rank = worker.rank
    q_slice = tensor_q[rank]
    kv_slice = tensor_kv[rank]
    k_slice = kv_slice[:, :hidden_size_k]
    v_slice = kv_slice[:, hidden_size_k:]
    num_tokens_q_dst = fwd_q_metadata.num_total_recv_tokens[rank]
    num_tokens_kv_dst = fwd_k_metadata.num_total_recv_tokens[rank]
    dispatch_mask = (
        fwd_k_metadata.get_slice(rank).dst_rank >= 0
    ).to(torch.int8)
    fa2a_metadata_fwd_slice = fa2a_metadata_qkv_fwd.get_slice(rank)
    fa2a_metadata_rev_slice = fa2a_metadata_qkv_rev.get_slice(rank)

    # run this communication
    out_dict = worker.run_qkv(
        fa2a_metadata_fwd_slice, fa2a_metadata_rev_slice,
        q_slice, kv_slice, dispatch_mask
    )
    dst_q = out_dict["dst_q"]
    out_q_shard = output_tensor_q[rank, :num_tokens_q_dst]
    out_kv = out_dict["dst_kv"]
    out_kv_shard = output_tensor_kv[rank, :num_tokens_kv_dst]
    rev_q = out_dict["rev_q"]
    rev_q_shard = back_tensor_q[rank]
    rev_kv = out_dict["rev_kv"]
    rev_kv_shard = back_tensor_kv[rank]

    torch.testing.assert_close(dst_q, out_q_shard.to(dst_q.device))
    torch.testing.assert_close(out_kv, out_kv_shard.to(dst_q.device))
    torch.testing.assert_close(rev_q, rev_q_shard.to(dst_q.device))
    torch.testing.assert_close(rev_kv, rev_kv_shard.to(dst_q.device))


def test(args):
    stride_q = args.hidden_size_query * torch.float16.itemsize
    stride_kv = args.hidden_size_kv * torch.float16.itemsize
    world_size = args.world_size
    max_tokens_query = args.num_tokens * world_size
    max_tokens_key_value = args.num_tokens * world_size
    max_cp_degree = args.max_cp_degree

    buffer_size = (
        stride_q * max_tokens_query +
        stride_kv * max_tokens_key_value * max_cp_degree * 2
    )
    worker = init_worker_torch_distributed(
        world_size, buffer_size, Worker
    )
    print("init done.")
    test_qkv(
        args.seed, world_size, args.num_tokens, args.num_seqs, max_cp_degree,
        worker, args.hidden_size_query, args.hidden_size_kv
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--max-cp-degree", type=int, default=2)
    parser.add_argument("--hidden-size-query", type=int, default=64)
    parser.add_argument("--hidden-size-kv", type=int, default=16)
    parser.add_argument("--num-seqs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    test(args)
