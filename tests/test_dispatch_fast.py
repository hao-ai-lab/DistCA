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
from d2.runtime.compute_metadata import from_planner_output
from d2.runtime.fast_alltoall_metadata import FastAlltoAllMetadata

from test_util import (
    BaseWorker, init_worker_torch_distributed,
    random_shard_info_linear_layout_dp
)
from test_shard_info_to_fa2a import simulate_all2all


class Worker(BaseWorker):

    def run_qkv(
        self, fa2a_metadata_fwd: FastAlltoAllMetadata,
        fa2a_metadata_rev: FastAlltoAllMetadata,
        tensor_q: torch.Tensor, tensor_kv: torch.Tensor,
    ):
        tensor_q = tensor_q.cuda()
        tensor_kv = tensor_kv.cuda()
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
    fwd_qkv_metadata: FastAlltoAllMetadata,
    bwd_qkv_metadata: FastAlltoAllMetadata,
    world_size: int, num_tokens: int, max_cp_degree: int,
    hidden_size_q: int, hidden_size_k: int, dtype: torch.dtype,
):
    device = "cpu"

    ### Init tensor ###
    src_qs = torch.randn(
        world_size, num_tokens, hidden_size_q,
        dtype=dtype, device=device
    )
    src_ks = torch.randn(
        world_size, num_tokens, hidden_size_k,
        dtype=dtype, device=device
    )
    src_vs = torch.randn(
        world_size, num_tokens, hidden_size_k,
        dtype=dtype, device=device
    )
    element_size = dtype.itemsize

    ### Init output tensor ###
    dst_qs, dst_ks, dst_vs = simulate_all2all(
        src_qs, src_ks, src_vs, fwd_qkv_metadata, element_size, hidden_size_q, hidden_size_k,
        is_from_linear_layout=True,
    )
    rev_qs, rev_ks, rev_vs = simulate_all2all(
        dst_qs, dst_ks, dst_vs, bwd_qkv_metadata, element_size, hidden_size_q, hidden_size_k,
        is_from_linear_layout=False,
    )
    kv_mask = fwd_qkv_metadata.kv_replica_mask
    src_kvs = [
        torch.concat((src_ks[r], src_vs[r]), dim=-1)
        for r in range(world_size)
    ]
    dst_kvs = [
        torch.concat((dst_ks[r], dst_vs[r]), dim=-1)
        for r in range(world_size)
    ]
    rev_kvs = [
        torch.concat((rev_ks[r], rev_vs[r]), dim=-1)
        for r in range(world_size)
    ]

    return src_qs, src_kvs, dst_qs, dst_kvs, rev_qs, rev_kvs


def create_test_case(seed: int, world_size: int, total_seq_len: int, num_docs: int,
                     max_cp_degree: int, hidden_size_q: int, hidden_size_k: int):
    dtype = torch.float16
    planner_output, doc_lens = random_shard_info_linear_layout_dp(
        world_size, num_docs, total_seq_len, seed=seed,max_num_shard=max_cp_degree,
    )
    lse_size = 0
    element_size = dtype.itemsize
    (fa2a_metadata_qkv_fwd, fa2a_metadata_qkv_rev,
     fa2a_metadata_attn_out_fwd, fa2a_metadata_attn_out_rev, _) = from_planner_output(
        world_size, planner_output, hidden_size_q, hidden_size_k,
        lse_size, element_size, is_pipeline_tick=False
    )

    # create answers:
    (
        tensor_q, tensor_kv, output_tensor_q, output_tensor_kv,
        back_tensor_q, back_tensor_kv
    ) = create_answer(
        fa2a_metadata_qkv_fwd, fa2a_metadata_qkv_rev,
        world_size, total_seq_len, max_cp_degree,
        hidden_size_q, hidden_size_k, dtype,
    )

    print("metadata compute done.")
    return (
        tensor_q, tensor_kv, output_tensor_q, output_tensor_kv,
        back_tensor_q, back_tensor_kv,
        fa2a_metadata_qkv_fwd, fa2a_metadata_qkv_rev,
        fa2a_metadata_attn_out_fwd, fa2a_metadata_attn_out_rev
    )


@torch.no_grad()
def test_qkv(
    seed: int, world_size: int, total_seq_len: int, num_docs: int,
    max_cp_degree: int, worker: Worker, hidden_size_q: int, hidden_size_k: int,
):
    torch.manual_seed(seed)
    (
        tensor_q, tensor_kv, output_tensor_q, output_tensor_kv,
        back_tensor_q, back_tensor_kv,
        fa2a_metadata_qkv_fwd, fa2a_metadata_qkv_rev, _, _
    ) = create_test_case(
        seed, world_size, total_seq_len, num_docs, max_cp_degree,
        hidden_size_q, hidden_size_k
    )

    rank = worker.rank
    q_slice = tensor_q[rank]
    kv_slice = tensor_kv[rank]
    fa2a_metadata_fwd_slice = fa2a_metadata_qkv_fwd.get_slice(rank)
    fa2a_metadata_rev_slice = fa2a_metadata_qkv_rev.get_slice(rank)

    # run this communication
    out_dict = worker.run_qkv(
        fa2a_metadata_fwd_slice, fa2a_metadata_rev_slice,
        q_slice, kv_slice
    )
    dst_q = out_dict["dst_q"]
    out_q_shard = output_tensor_q[rank]
    out_kv = out_dict["dst_kv"]
    out_kv_shard = output_tensor_kv[rank]
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
        args.seed, world_size, args.num_tokens, args.num_docs, max_cp_degree,
        worker, args.hidden_size_query, args.hidden_size_kv
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--max-cp-degree", type=int, default=4)
    parser.add_argument("--hidden-size-query", type=int, default=64)
    parser.add_argument("--hidden-size-kv", type=int, default=16)
    parser.add_argument("--num-docs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    test(args)
