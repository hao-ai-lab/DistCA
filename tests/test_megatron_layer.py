"""
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
torchrun --nnodes 1 --nproc_per_node 2 test_megatron_layer.py \
    --world-size 2
"""

from typing import Optional

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
import torch

from d2.runtime.inplace_metadata import mlp_layout_packed_params
from d2.runtime.megatron_patch.model_patch import get_gpt_layer_with_transformer_engine_spec, get_gpt_config
from d2.runtime.megatron_patch.packed_seq_params import PingPangSingleStepPackedSeqParams
from d2.runtime.megatron_patch.transformer_layer import TransformerLayer as PingPangTransformerLayer
from d2.runtime.fast_alltoall_metadata import compute_fa2a_metadata_from_logical_metadata

from test_util import (
    MegatronBaseWorker, ParallelConfig, simulate_communication,
    create_qkv_dispatch, init_worker_torch_distributed,
)


class MegatronLayerWorker(MegatronBaseWorker):
    def __init__(self, rank: int, world_size: int):
        super().__init__(rank, world_size)
        self.layer: Optional[PingPangTransformerLayer] = None

    def init_layer(self, config: TransformerConfig, spec: ModuleSpec,
                   seed: int):
        torch.manual_seed(seed)
        self.layer = build_module(spec, config)

    def forward_normal(self, tensor_input: torch.Tensor, packed_seq_params: PackedSeqParams):
        packed_seq_params = PackedSeqParams(
            qkv_format=packed_seq_params.qkv_format,
            cu_seqlens_q=packed_seq_params.cu_seqlens_q.cuda().to(torch.int32),
            cu_seqlens_kv=packed_seq_params.cu_seqlens_kv.cuda().to(torch.int32),
            max_seqlen_q=packed_seq_params.max_seqlen_q.cuda().to(torch.int32),
            max_seqlen_kv=packed_seq_params.max_seqlen_kv.cuda().to(torch.int32),
        )
        tensor_input = tensor_input.cuda().detach()
        tensor_input.requires_grad = True
        self.layer.train()
        mlp_output, context, debug = self.layer.forward_no_switch(tensor_input, packed_seq_params=packed_seq_params)
        mlp_output.mean().backward()
        torch.cuda.synchronize()
        print(self.rank, "normal forward done")
        return (mlp_output, context, tensor_input.grad), debug

    def forward_ping_pang_one_stage(
        self, tensor_input: torch.Tensor,
        packed_seq_params: PingPangSingleStepPackedSeqParams,
    ):
        packed_seq_params = packed_seq_params.to_device()
        tensor_input = tensor_input.cuda().detach()
        tensor_input.requires_grad = True

        backward_resend_qkv = packed_seq_params.bwd_packed_seq_params is not None

        self.layer.train()
        mlp_output, context, debug_tensors = self.layer.forward_one_stage(
            tensor_input, packed_seq_params=packed_seq_params,
            backward_resend_qkv=backward_resend_qkv
        )
        mlp_output.mean().backward()
        torch.cuda.synchronize()
        print(self.rank, "ping-pong one stage forward done")
        return (mlp_output, context, tensor_input.grad), debug_tensors


def test_forward(
    seed, world_size, total_seq_len, num_seqs, max_cp_degree,
    worker: MegatronLayerWorker, hidden_size_q: int, hidden_size_k: int
):
    torch.manual_seed(seed)
    dtype = torch.float16
    element_size = dtype.itemsize
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

    # thd layout's hidden size input is "t,1,h"
    torch.manual_seed(seed)
    tensors = torch.randn(
        (world_size, total_seq_len, 1, hidden_size_q), dtype=dtype
    )
    rank = worker.rank
    tensor_shard = tensors[rank]
    # 1. normal forward. Need to provide the PackedSeqParams
    seq_lens_local = seq_lens[rank][:num_seqs]
    packed_seq_params = mlp_layout_packed_params(seq_lens_local)
    normal_forward_out, debug_ref = worker.forward_normal(
        tensor_shard, packed_seq_params
    )

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
    ping_pang_out, debug_out = worker.forward_ping_pang_one_stage(
        tensor_shard, ping_pang_params
    )

    ref_debug = [None] * world_size
    ans_debug = [None] * world_size
    pingpong_seq_params = [None] * world_size
    torch.distributed.all_gather_object(ref_debug, debug_ref)
    torch.distributed.all_gather_object(ans_debug, debug_out)
    torch.distributed.all_gather_object(pingpong_seq_params, ping_pang_params)
    print("debug tensors gathered.")
    if rank == 0:
        device = torch.device("cuda", rank)
        def to_device(o):
            if isinstance(o, torch.Tensor):
                return o.to(device)
            elif isinstance(o, tuple):
                return tuple(to_device(x) for x in o)
            elif isinstance(o, list):
                return [to_device(x) for x in o]
            else:
                return o
        ref_debug = [to_device(debug_tensor) for debug_tensor in ref_debug]
        ans_debug = [to_device(debug_tensor) for debug_tensor in ans_debug]

        ans_debug_qkvs_pre_transfer = [ans_debug[0] for ans_debug in ans_debug]
        ans_debug_qkvs_post_transfer = [ans_debug[1] for ans_debug in ans_debug]
        ans_debug_core_attn_out = [ans_debug[2] for ans_debug in ans_debug]
        ans_debug_core_attn_out_post_transfer = [ans_debug[3] for ans_debug in ans_debug]
        ref_qkvs = [debug_tensor[0] for debug_tensor in ref_debug]
        ref_attn_outs = [debug_tensor[1] for debug_tensor in ref_debug]
        torch.testing.assert_close(ref_qkvs, ans_debug_qkvs_pre_transfer)
        print("debug pre-layout-transfer qkv allclose")
        ref_qs = [debug_tensor[0] for debug_tensor in ref_qkvs]
        ref_ks = [debug_tensor[1] for debug_tensor in ref_qkvs]
        ref_vs = [debug_tensor[2] for debug_tensor in ref_qkvs]
        ref_qs_post_comm = simulate_communication(ref_qs, fwd_q_metadata)
        ref_ks_post_comm = simulate_communication(ref_ks, fwd_k_metadata)
        ref_vs_post_comm = simulate_communication(ref_vs, fwd_k_metadata)
        ref_qkvs_post_comm = [
            (ref_qs_post_comm[rank], ref_ks_post_comm[rank], ref_vs_post_comm[rank]) for rank in range(world_size)
        ]
        torch.testing.assert_close(
            ans_debug_qkvs_post_transfer, ref_qkvs_post_comm
        )
        print("post transfer debug qkv allclose")

        from flash_attn import flash_attn_varlen_func
        ref_attn_outs_a_layout = []
        for rank in range(world_size):
            metadata = pingpong_seq_params[rank].to_device()
            ref_attn_out = flash_attn_varlen_func(
                ref_qs_post_comm[rank], ref_ks_post_comm[rank], ref_vs_post_comm[rank],
                cu_seqlens_q = metadata.cu_seqlens_q,
                cu_seqlens_k = metadata.cu_seqlens_kv,
                max_seqlen_q = metadata.max_seqlen_q,
                max_seqlen_k = metadata.max_seqlen_kv,
                causal = True,
                dropout_p = 0.0,
            )
            ref_attn_out = ref_attn_out.reshape(ref_attn_out.shape[0], 1, -1)
            ref_attn_outs_a_layout.append(ref_attn_out)
        ref_attn_outs_post_comm = simulate_communication(
            ref_attn_outs_a_layout, rev_q_metadata
        )
        torch.testing.assert_close(ref_attn_outs, ref_attn_outs_post_comm)
        print("simulated attn out allclose with expected value")
        torch.testing.assert_close(ans_debug_core_attn_out, ref_attn_outs_a_layout)
        print("core attn out allclose")
        torch.testing.assert_close(ans_debug_core_attn_out_post_transfer, ref_attn_outs)
        print("post transfer debug attn out allclose")

        torch.testing.assert_close(normal_forward_out, ping_pang_out)
        print("pass final result")


def init_megatron_test(
    world_size, hidden_size, num_heads, num_query_heads, dtype,
    max_tokens_query, max_tokens_key_value, max_cp_degree, tp_size, seed,
    worker_cls=MegatronLayerWorker
):
    assert hidden_size // num_heads <= 256, "FA requires head_dim <= 256"
    assert dtype == torch.float16
    token_bytes_q = hidden_size * dtype.itemsize
    token_bytes_kv = hidden_size * dtype.itemsize
    buffer_size = (
        token_bytes_q * max_tokens_query * 3 +
        num_heads * torch.float32.itemsize * 2 * max_tokens_query +
        token_bytes_kv * max_tokens_key_value * max_cp_degree * 2
    )
    parallel_config = ParallelConfig(
        tensor_model_parallel_size=tp_size
    )
    worker = init_worker_torch_distributed(
        world_size, buffer_size, worker_cls, parallel_config
    )
    spec = get_gpt_layer_with_transformer_engine_spec()
    config = get_gpt_config(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_query_groups=num_query_heads,
        ffn_hidden_size=hidden_size * 4,
        fp16=True,
        deterministic_mode=True,
        params_dtype=dtype,
    )
    worker.init_layer(config, spec, seed=seed)
    return worker


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
    num_query_heads = num_heads
    hidden_size_kv = (hidden_size * num_query_heads) // num_heads

    worker = init_megatron_test(
        world_size, hidden_size, num_heads, num_query_heads, dtype,
        max_tokens_query, max_tokens_key_value, max_cp_degree, tp_size, seed,
    )

    test_forward(
        args.seed, world_size, args.num_tokens, args.num_seqs,
        max_cp_degree, worker, hidden_size, hidden_size_kv
    )


if __name__ == "__main__":
    import argparse
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
    args = parser.parse_args()
    test(args)
