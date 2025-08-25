"""
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
torchrun --nnodes 1 --nproc_per_node 2 test_megatron_layer_bwd_resend.py \
    --world-size 2
"""

from megatron.core.packed_seq_params import PackedSeqParams
import torch

from d2.runtime.compute_metadata import (
    from_planner_output, backward_from_planner_output, get_attn_metadata
)
from d2.runtime.megatron_patch.packed_seq_params import PingPangSingleStepPackedSeqParams

from test_util import random_shard_info_linear_layout_dp, simulate_communication
from test_megatron_layer import MegatronLayerWorker, init_megatron_test


def test_forward(
    seed, world_size, total_seq_len, num_docs, max_cp_degree,
    worker: MegatronLayerWorker, hidden_size_q: int, hidden_size_k: int, num_heads: int
):
    torch.manual_seed(seed)
    dtype = torch.float16
    element_size = dtype.itemsize
    lse_size = num_heads * torch.float32.itemsize // element_size
    (scheduler_output, glob_doc_lens, scheduler_output_bwd
     ) = random_shard_info_linear_layout_dp(
        world_size, num_docs, total_seq_len,
        max_num_shard=max_cp_degree, seed=seed,
        create_backward_redispatch=True,
    )
    glob_doc_lens = [torch.tensor(l) for l in glob_doc_lens]
    (qkv_linear_to_attn, _, out_attn_to_linear, _,
     fwd_attn_metadata,) = from_planner_output(
        world_size, scheduler_output, hidden_size_q, hidden_size_k,
        lse_size, element_size, is_pipeline_tick=True
    )
    (qkv_resend_and_out_grad_linear_to_attn, qkv_grad_attn_to_linear,
     bwd_attn_metadata) = backward_from_planner_output(
        world_size, scheduler_output_bwd, hidden_size_q, hidden_size_k,
        lse_size, element_size,
     )

    # thd layout's hidden size input is "t,1,h"
    torch.manual_seed(seed)
    tensors = torch.randn(
        (world_size, total_seq_len, 1, hidden_size_q), dtype=dtype
    )
    rank = worker.rank
    tensor_shard = tensors[rank]

    # NOTE: this already adds prepended zeros and is sharded to tuples (remove padding seqs)
    bwd_packed_seq_params = PackedSeqParams(
        qkv_format="thd", **bwd_attn_metadata[rank],
    )

    # 1. normal forward. Need to provide the PackedSeqParams
    packed_seq_params = get_attn_metadata(
        glob_doc_lens[rank], get_packed_seq_params=True
    )
    normal_forward_out, debug_ref = worker.forward_normal(
        tensor_shard, packed_seq_params, return_grad=True
    )

    ping_pang_params = PingPangSingleStepPackedSeqParams(
        qkv_format="thd",
        **fwd_attn_metadata[rank],
        qkv_fwd_metadata=qkv_linear_to_attn.get_slice(rank),
        qkv_bwd_metadata=qkv_grad_attn_to_linear.get_slice(rank),
        attn_out_fwd_metadata=out_attn_to_linear.get_slice(rank),
        attn_out_bwd_metadata=qkv_resend_and_out_grad_linear_to_attn.get_slice(rank),
        bwd_packed_seq_params=bwd_packed_seq_params,
    )
    ping_pang_out, debug_out = worker.forward_ping_pang_one_stage(
        tensor_shard, ping_pang_params, return_grad=True
    )
    torch.testing.assert_close(normal_forward_out, ping_pang_out, rtol=1e-3, atol=1e-3)


def test(args):
    # only test multi-head-attention. For GQA, update get_gpt_config
    world_size = args.world_size
    max_tokens_query = args.num_tokens * world_size
    max_tokens_key_value = args.num_tokens * world_size
    hidden_size = args.hidden_size
    max_cp_degree = args.max_cp_degree
    seed = args.seed
    tp_size = 1
    num_heads = args.num_heads
    dtype = torch.float16
    num_query_heads = num_heads
    hidden_size_kv = (hidden_size * num_query_heads) // num_heads

    worker = init_megatron_test(
        world_size, hidden_size, num_heads, num_query_heads, dtype,
        max_tokens_query, max_tokens_key_value, max_cp_degree, tp_size, seed,
    )

    test_forward(
        args.seed, world_size, args.num_tokens, args.num_docs,
        max_cp_degree, worker, hidden_size, hidden_size_kv, num_heads,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--num-docs", type=int, default=3)
    parser.add_argument("--max-cp-degree", type=int, default=2)
    # NOTE: when increasing this value, remember to increase num-heads as well
    # because FA2 only supports head_dim_qk <= 256.
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=2)
    args = parser.parse_args()
    test(args)
