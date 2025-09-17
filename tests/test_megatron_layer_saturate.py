"""
NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 torchrun --nnodes 1 --nproc_per_node 1 test_megatron_layer_saturate.py --world-size 1
"""

from contextlib import nullcontext
from typing import Optional

import megatron.core.parallel_state as mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
import torch

from d2.runtime.compute_metadata import from_planner_output, get_attn_metadata
from d2.runtime.megatron_patch.create_group import get_attn_server_group
from d2.runtime.megatron_patch.model_patch import get_gpt_layer_with_transformer_engine_spec, get_gpt_config
from d2.runtime.megatron_patch.packed_seq_params import PingPangSingleStepPackedSeqParams
from d2.runtime.megatron_patch.transformer_layer import TransformerLayer as PingPangTransformerLayer

from test_util import (
    MegatronBaseWorker, ParallelConfig,
    init_worker_torch_distributed, random_shard_info_linear_layout_dp
)
from test_shard_info_to_fa2a import simulate_all2all


class MegatronLayerWorker(MegatronBaseWorker):
    def __init__(self, rank: int, world_size: int):
        super().__init__(rank, world_size)
        self.layer: Optional[PingPangTransformerLayer] = None

    def init_layer(self, config: TransformerConfig, spec: ModuleSpec,
                   seed: int):
        torch.manual_seed(seed + mpu.get_tensor_model_parallel_rank())
        self.layer = build_module(spec, config)

    def forward_normal(self, tensor_input: torch.Tensor, packed_seq_params: PackedSeqParams,
                       return_grad: bool = False):
        packed_seq_params = PackedSeqParams(
            qkv_format=packed_seq_params.qkv_format,
            cu_seqlens_q=packed_seq_params.cu_seqlens_q.cuda().to(torch.int32),
            cu_seqlens_kv=packed_seq_params.cu_seqlens_kv.cuda().to(torch.int32),
            max_seqlen_q=packed_seq_params.max_seqlen_q,
            max_seqlen_kv=packed_seq_params.max_seqlen_kv,
        )
        tensor_input = tensor_input.cuda().detach()
        tensor_input.requires_grad = True
        self.layer.train()

        if return_grad:
            ctx = torch.enable_grad()
        else:
            ctx = nullcontext()
        with ctx:
            mlp_output, context, debug = self.layer.forward_orig_impl(
                tensor_input, packed_seq_params=packed_seq_params, return_debug=True,
            )
            if return_grad:
                mlp_output.sum().backward()

        torch.cuda.synchronize()
        print(self.rank, "normal forward done")
        return (mlp_output, context, *((tensor_input.grad,) if return_grad else ())), debug

    def forward_ping_pang_one_stage(
        self, tensor_input: torch.Tensor,
        packed_seq_params: PingPangSingleStepPackedSeqParams,
        return_grad: bool = False,
    ):
        packed_seq_params = packed_seq_params.to_device()
        tensor_input = tensor_input.cuda().detach()
        tensor_input.requires_grad = True

        self.layer.train()

        if return_grad:
            ctx = torch.enable_grad()
        else:
            ctx = nullcontext()
        with ctx:
            mlp_output, context, debug_tensors = self.layer.forward_ping_pong_single_sided(
                tensor_input, packed_seq_params=packed_seq_params,
                return_debug=True,
            )
            if return_grad:
                mlp_output.sum().backward()

        torch.cuda.synchronize()
        print(self.rank, "ping-pong one stage forward done")
        return (mlp_output, context, *((tensor_input.grad,) if return_grad else ())), debug_tensors


def test_forward(
    seed: int, total_seq_len: int, num_docs: int,
    worker: MegatronLayerWorker, hidden_size_q: int, hidden_size_k: int,
    tp_size: int = 1, profile: bool = False,
):
    torch.manual_seed(seed)
    dtype = torch.float16
    element_size = dtype.itemsize
    as_world_size = worker.as_world_size
    as_rank = worker.as_rank
    print(f"as_world_size: {as_world_size}, as_rank: {as_rank}")

    # thd layout's hidden size input is "t,1,h"
    tensors = torch.randn(
        (as_world_size, total_seq_len, 1, hidden_size_q), dtype=dtype
    )
    tensor_shard = tensors[as_rank]
    print(f"tensor_shard: {tensor_shard.shape}")
    # 1. normal forward. Need to provide the PackedSeqParams
    doc_lens_this_rank = torch.tensor([total_seq_len], dtype=torch.int32, device='cuda')
    packed_seq_params = get_attn_metadata(
        doc_lens_this_rank, 
        get_packed_seq_params=True,
    )
    
    for _ in range(3):
        worker.forward_normal(
            tensor_shard, packed_seq_params, return_grad=True
        )
    
    import time
    torch.cuda.synchronize()
    start_time = time.time()
    N = 10
    for _ in range(N):
        with torch.cuda.nvtx.range(f"[N={N}] T={total_seq_len}"):
            worker.forward_normal(
                tensor_shard, packed_seq_params, return_grad=True
            )
    torch.cuda.synchronize()
    end_time = time.time()
    duraiton_ms = (end_time - start_time) * 1000
    duraiton_ms = duraiton_ms / N
    print(f"normal forward: {total_seq_len} tokens in {duraiton_ms:.2f} ms")
    return duraiton_ms
    
        
def init_megatron_test(
    world_size, head_dim, num_query_heads, num_key_value_heads, dtype,
    tp_size, seed, worker_cls=MegatronLayerWorker,
    mlp_factor=3.5
):
    # assert hidden_size // num_heads <= 256, "FA requires head_dim <= 256"
    assert dtype == torch.float16
    buffer_size = 1024**3
    parallel_config = ParallelConfig(
        tensor_model_parallel_size=tp_size,
    )
    worker = init_worker_torch_distributed(
        world_size, buffer_size, worker_cls, parallel_config
    )
    spec = get_gpt_layer_with_transformer_engine_spec()
    config = get_gpt_config(
        num_layers=1,
        hidden_size=head_dim * num_query_heads,
        num_attention_heads=num_query_heads,
        num_query_groups=num_query_heads // num_key_value_heads,
        ffn_hidden_size=int(head_dim * mlp_factor),
        fp16=True,
        deterministic_mode=False,
        params_dtype=dtype,
        tensor_model_parallel_size=tp_size,
    )
    worker.init_layer(config, spec, seed=seed)
    return worker


def test(args):
    # only test multi-head-attention. For GQA, update get_gpt_config
    dtype = torch.float16
    world_size = args.world_size
    max_tokens_query = args.num_tokens
    max_tokens_key_value = args.num_tokens
    head_dim = args.head_dim
    max_cp_degree = args.max_cp_degree
    seed = args.seed
    tp_size = args.tp_size
    num_query_heads = args.num_query_heads
    num_key_value_heads = args.num_key_value_heads
    hidden_size_q = args.head_dim * num_query_heads
    hidden_size_kv = args.head_dim * num_key_value_heads

    worker: MegatronLayerWorker = init_megatron_test(
        world_size, head_dim, num_query_heads, num_key_value_heads, dtype,
        tp_size, seed, 
    )

    all_duration_ms = {}
    for num_tokens in args.num_tokens:
        print(f"testing {num_tokens} tokens")
        duration_ms = test_forward(
            args.seed, num_tokens, 1, worker, hidden_size_q, hidden_size_kv,
            tp_size, profile=args.profile,
        )
        all_duration_ms[num_tokens] = duration_ms
    print(all_duration_ms)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-size", type=int, default=1)

    parser.add_argument("--num-tokens", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    parser.add_argument("--num-docs", type=int, default=1)
    parser.add_argument("--max-cp-degree", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--profile", action="store_true", default=False,)
    
    # Model configs
    parser.add_argument("--num-query-heads", type=int, default=32)
    parser.add_argument("--num-key-value-heads", type=int, default=8)
    parser.add_argument("--mlp-factor", type=float, default=3.5)
    parser.add_argument("--head-dim", type=int, default=128)
    
    args = parser.parse_args()
    test(args)
