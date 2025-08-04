"""
Profiling:

NVSHMEM_IB_ENABLE_IBGDA=true NVSHMEM_DEBUG=DEBUG NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
    nsys profile -o pingpang_layer_%p.nsys-rep -t cuda,nvtx \
torchrun --nnodes 1 --nproc_per_node 2 test_pingpang_layer.py \
    --world-size 2 \
    --profile \
    --num-query-heads 8 --num-heads 32 --hidden-size 4096 --num-tokens 8192

Correctness:

NVSHMEM_IB_ENABLE_IBGDA=true NVSHMEM_DEBUG=DEBUG NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
torchrun --nnodes 1 --nproc_per_node 2 test_pingpang_layer.py \
    --world-size 2 \
    --num-query-heads 8 --num-heads 32 --hidden-size 4096 --num-tokens 8192
"""

import argparse

import torch

from d2.runtime.inplace_metadata import mlp_layout_packed_params
from d2.runtime.megatron_patch.packed_seq_params import PingPangPackedSeqParams, PingPangSingleStepPackedSeqParams
from d2.runtime.fast_alltoall_metadata import compute_fa2a_metadata_from_logical_metadata

from test_util import create_qkv_dispatch
from test_megatron_layer import MegatronLayerWorker, init_megatron_test


class PingPangLayerWorker(MegatronLayerWorker):
    def __init__(self, rank: int, world_size: int):
        super().__init__(rank, world_size)
        self.stream = torch.cuda.Stream()

    def forward_ping_pang(self, tensor_input: torch.Tensor, packed_seq_params: PingPangPackedSeqParams):
        packed_seq_params = packed_seq_params.to_device()
        tensor_input = tensor_input.cuda()
        self.layer.train()
        # if not debug, add communication stream here.
        if not packed_seq_params.debug:
            for params in packed_seq_params.seq_params:
                setattr(params, "stream", self.stream)
            setattr(packed_seq_params.seq_params[0], "dispatcher_id", 0)
            setattr(packed_seq_params.seq_params[0], "dispatcher_id", 1)
        else:
            for params in packed_seq_params.seq_params:
                setattr(params, "stream", torch.cuda.current_stream())
        return self.layer.ping_pang_forward(tensor_input, packed_seq_params=packed_seq_params)


def create_one_batch(
    world_size: int, total_seq_len: int, num_seqs: int, max_cp_degree: int,
    hidden_size_q: int, hidden_size_k: int, element_size: int
):
    (
        fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata,
        attention_metadata_attn_layout, intermediates, seq_lens
    ) = create_qkv_dispatch(
        world_size, total_seq_len, num_seqs, max_cp_degree,
        return_intermediate=True, return_mlp_no_shard_seq_lens=True
    )
    # NOTE: this already adds prepended zeros and is sharded to tuples (remove padding seqs)
    (cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv,
     num_local_seqs_recv) = attention_metadata_attn_layout

    (qkv_fwd_fa2a_metadata, qkv_rev_fa2a_metadata,
     attn_out_fwd_fa2a_metadata, attn_out_rev_fa2a_metadata,
    ) = compute_fa2a_metadata_from_logical_metadata(
        fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata,
        intermediates, total_seq_len, hidden_size_q, hidden_size_k,
        element_size,
    )
    logical_metadata = (
        fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata,
    )
    fa2a_metadata = (
        qkv_fwd_fa2a_metadata, qkv_rev_fa2a_metadata,
        attn_out_fwd_fa2a_metadata, attn_out_rev_fa2a_metadata,
    )
    attn_metadata = (
        cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv,
    )
    raw_seq_lens = seq_lens
    return logical_metadata, fa2a_metadata, attn_metadata, raw_seq_lens


def get_single_step_packed_seq_params(
    fa2a_metadata, attn_metadata, rank: int
):
    (
        qkv_fwd_fa2a_metadata, qkv_rev_fa2a_metadata,
        attn_out_fwd_fa2a_metadata, attn_out_rev_fa2a_metadata,
    ) = fa2a_metadata
    (
        cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv,
    ) = attn_metadata
    ping_pang_params = PingPangSingleStepPackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_q[rank],
        cu_seqlens_kv=cu_seqlens_kv[rank],
        max_seqlen_q=max_seqlen_q[rank],
        max_seqlen_kv=max_seqlen_kv[rank],
        qkv_fwd_metadata=qkv_fwd_fa2a_metadata.get_slice(rank),
        qkv_bwd_metadata=qkv_rev_fa2a_metadata.get_slice(rank),
        attn_out_fwd_metadata=attn_out_fwd_fa2a_metadata.get_slice(rank),
        attn_out_bwd_metadata=attn_out_rev_fa2a_metadata.get_slice(rank),
    )
    return ping_pang_params


@torch.no_grad()
def test_forward(
    seed, world_size, total_seq_len, num_seqs, max_cp_degree,
    worker: PingPangLayerWorker, hidden_size_q: int, hidden_size_k: int,
    debug=False, profile=False,
):
    torch.manual_seed(seed)
    dtype = torch.float16
    element_size = dtype.itemsize
    # Create two splits for ping-pong
    _, fa2a_metadata_0, attn_metadata_0, raw_seq_lens_0 = create_one_batch(
        world_size, total_seq_len, num_seqs, max_cp_degree,
        hidden_size_q, hidden_size_k, element_size
    )
    _, fa2a_metadata_1, attn_metadata_1, raw_seq_lens_1 = create_one_batch(
        world_size, total_seq_len, num_seqs, max_cp_degree,
        hidden_size_q, hidden_size_k, element_size
    )

    # Create tensor input
    torch.manual_seed(seed)
    tensors = torch.randn(
        (world_size, total_seq_len * 2, 1, hidden_size_q), dtype=dtype
    )
    rank = worker.rank
    tensor_shard = tensors[rank]
    seq_lens_local = torch.concat(
        (raw_seq_lens_0[rank][:num_seqs], raw_seq_lens_1[rank][:num_seqs])
    )
    packed_seq_params = mlp_layout_packed_params(seq_lens_local)
    normal_forward_out, debug_ref = worker.forward_normal(
        tensor_shard, packed_seq_params
    )

    ping_pang_params_0 = get_single_step_packed_seq_params(
        fa2a_metadata_0, attn_metadata_0, rank
    )
    ping_pang_params_1 = get_single_step_packed_seq_params(
        fa2a_metadata_1, attn_metadata_1, rank
    )
    print(f"{rank=}, pingpong 0 send bytes:{ping_pang_params_0.qkv_fwd_metadata.fa2a_metadata[1]}, pingpong 1 send bytes:{ping_pang_params_1.qkv_fwd_metadata.fa2a_metadata[1]}")
    mlp_layout_seq_params = tuple(
        mlp_layout_packed_params(seq_lens) for seq_lens in
        (raw_seq_lens_0[rank][:num_seqs], raw_seq_lens_1[rank][:num_seqs])
    )
    ping_pang_params = PingPangPackedSeqParams(
        seq_params=[ping_pang_params_0, ping_pang_params_1],
        mlp_layout_seq_params=list(mlp_layout_seq_params),
        max_seqlen_q=mlp_layout_seq_params[0].max_seqlen_q,
        max_seqlen_kv=mlp_layout_seq_params[0].max_seqlen_kv,
        qkv_format="thd",
        do_gather=True,
        debug=debug,
    )
    ans = worker.forward_ping_pang(
        tensor_shard, ping_pang_params
    )
    torch.testing.assert_close(ans, normal_forward_out)
    print(f"Rank {rank} forward ping-pang passed.")
    if profile:
        for _ in range(3):
            worker.forward_ping_pang(
                tensor_shard, ping_pang_params
            )
        torch.cuda.synchronize()
        torch.distributed.barrier()
        print("warmup done")
        for _ in range(20):
            worker.forward_ping_pang(
                tensor_shard, ping_pang_params
            )
        torch.cuda.synchronize()
        torch.distributed.barrier()
        print("ping-pang forward done")


def test(args):
    # only test multi-head-attention. For GQA, update get_gpt_config
    world_size = args.world_size
    max_tokens_query = args.num_tokens * world_size
    max_tokens_key_value = args.num_tokens * world_size
    hidden_size = args.hidden_size
    max_cp_degree = args.max_cp_degree
    seed = args.seed
    tp_size = args.tp_size
    num_heads = args.num_heads
    dtype = torch.float16
    num_query_heads = args.num_query_heads
    hidden_size_kv = (hidden_size * num_query_heads) // num_heads

    worker = init_megatron_test(
        world_size, hidden_size, num_heads, num_query_heads, dtype,
        max_tokens_query, max_tokens_key_value, max_cp_degree, tp_size, seed,
        PingPangLayerWorker
    )
    test_forward(
        args.seed, world_size, args.num_tokens, args.num_seqs,
        max_cp_degree, worker, hidden_size, hidden_size_kv,
        profile=args.profile
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--num-seqs", type=int, default=3)
    parser.add_argument("--max-cp-degree", type=int, default=2)
    # NOTE: when increasing this value, remember to increase num-heads as well
    # because FA2 only supports head_dim_qk <= 256.
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-query-heads", type=int, default=2)
    parser.add_argument("--profile", action="store_true", default=False,)
    args = parser.parse_args()
    test(args)
