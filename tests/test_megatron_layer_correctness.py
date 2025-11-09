"""
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
torchrun --nnodes 1 --nproc_per_node 2 test_megatron_layer.py \
    --world-size 2

NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
torchrun --nnodes 1 --nproc_per_node 4 test_megatron_layer.py \
    --world-size 4 --tp-size 2

# Profile: Node = 2, TP = 8, SeqLen = 16k 
# (testing 8k does not make comp/comm overlap)

mkdir -p nsys-profile

NVSHMEM_DEBUG=DEBUG NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 nsys profile --force-overwrite=true -o nsys-profile/test_megatron_e2e.n0.t16k.nsys-rep -t cuda,nvtx torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=<master_addr> --master_port=29500 test_megatron_e2e.py --num-nodes=2 --num-gpus-per-node=8 --tp-size=8 --num-tokens 16384

NVSHMEM_DEBUG=DEBUG NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 nsys profile --force-overwrite=true -o nsys-profile/test_megatron_e2e.n1.t16k.nsys-rep -t cuda,nvtx torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 --master_addr=<master_addr> --master_port=29500 test_megatron_e2e.py --num-nodes=2 --num-gpus-per-node=8 --tp-size=8 --num-tokens 16384
"""

from contextlib import nullcontext
from typing import Optional

import megatron.core.parallel_state as mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
import torch

from d2.runtime.compute_metadata import from_planner_output, get_attn_metadata
from d2.runtime.megatron.create_group import get_attn_server_group
from d2.runtime.megatron.model_patch import get_gpt_layer_with_transformer_engine_spec, get_gpt_config
from d2.runtime.megatron.packed_seq_params import PingPangSingleStepPackedSeqParams
from d2.runtime.megatron.ping_pong.transformer_layer import TransformerLayer as PingPangTransformerLayer

from test_util import (
    MegatronBaseWorker, ParallelConfig,
    init_worker_torch_distributed, random_shard_info_linear_layout_dp
)
from test_shard_info_to_fa2a import simulate_all2all


def _summarize_pair(a, b, name: str, depth: int = 0, max_items: int = 3):
    indent = "  " * depth
    try:
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            try:
                print(f"{indent}- {name}: Tensor vs Tensor | dtype {a.dtype} vs {b.dtype} | device {a.device} vs {b.device}")
                print(f"{indent}  Shape a: {tuple(a.shape)}, Shape b: {tuple(b.shape)}")
                if a.shape == b.shape:
                    diff = torch.abs(a - b)
                    max_diff = torch.max(diff).item() if diff.numel() > 0 else 0.0
                    mean_diff = torch.mean(diff).item() if diff.numel() > 0 else 0.0
                    a_has_inf = torch.isinf(a).any().item()
                    a_has_nan = torch.isnan(a).any().item()
                    b_has_inf = torch.isinf(b).any().item()
                    b_has_nan = torch.isnan(b).any().item()
                    print(f"{indent}  Max diff: {max_diff}")
                    print(f"{indent}  Mean diff: {mean_diff}")
                    print(f"{indent}  a has inf: {a_has_inf}, a has nan: {a_has_nan}")
                    print(f"{indent}  b has inf: {b_has_inf}, b has nan: {b_has_nan}")
                else:
                    print(f"{indent}  Shapes differ; skipping elementwise diff")
            except Exception as ee:
                print(f"{indent}  (summary error on tensors: {ee})")
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            print(f"{indent}- {name}: {type(a).__name__} vs {type(b).__name__} | len {len(a)} vs {len(b)}")
            n = min(len(a), len(b), max_items)
            for i in range(n):
                _summarize_pair(a[i], b[i], f"{name}[{i}]", depth + 1, max_items)
            if len(a) != len(b):
                print(f"{indent}  Lengths differ; additional items omitted")
            elif len(a) > max_items:
                print(f"{indent}  ... ({len(a) - max_items} more items omitted)")
        elif isinstance(a, dict) and isinstance(b, dict):
            a_keys = set(a.keys())
            b_keys = set(b.keys())
            shared = sorted(a_keys & b_keys)
            print(f"{indent}- {name}: dict vs dict | keys shared {len(shared)}, a-only {len(a_keys - b_keys)}, b-only {len(b_keys - a_keys)}")
            for i, k in enumerate(shared):
                if i >= max_items:
                    print(f"{indent}  ... ({len(shared) - max_items} more keys omitted)")
                    break
                _summarize_pair(a[k], b[k], f"{name}['{k}']", depth + 1, max_items)
            if a_keys - b_keys:
                print(f"{indent}  a-only keys: {sorted(list(a_keys - b_keys))[:max_items]}")
            if b_keys - a_keys:
                print(f"{indent}  b-only keys: {sorted(list(b_keys - a_keys))[:max_items]}")
        else:
            print(f"{indent}- {name}: {type(a).__name__} vs {type(b).__name__}")
    except Exception as e:
        print(f"{indent}(summary failure for {name}: {e})")


def assert_close_report(name, a, b, **kwargs):
    try:
        torch.testing.assert_close(a, b, **kwargs)
        print(f"✅ {name} are close")
    except AssertionError as e:
        print(f"❌ {name} are NOT close")
        _summarize_pair(a, b, name)
        print(f"   Error: {str(e)}")
        raise


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
    seed: int, total_seq_len: int, num_docs: int, max_cp_degree: int,
    worker: MegatronLayerWorker, hidden_size_q: int, hidden_size_k: int,
    tp_size: int = 1, profile: bool = False,
):
    torch.manual_seed(seed)
    dtype = torch.float16
    element_size = dtype.itemsize
    as_world_size = worker.as_world_size
    as_rank = worker.as_rank

    planner_output, doc_lens_per_rank = random_shard_info_linear_layout_dp(
        as_world_size, num_docs, tot_num_token=total_seq_len,
        max_num_shard=max_cp_degree, seed=seed,
    )
    doc_lens_per_rank = [
        torch.tensor(val, dtype=torch.int32, device='cuda') for val in doc_lens_per_rank
    ]

    hidden_size_q_tp = hidden_size_q // tp_size
    hidden_size_k_tp = hidden_size_k // tp_size

    lse_size = 0
    (qkv_fwd_fa2a_metadata, qkv_rev_fa2a_metadata,
     attn_out_fwd_fa2a_metadata, attn_out_rev_fa2a_metadata,
     as_attn_metadata,
    ) = from_planner_output(
        as_world_size, planner_output, hidden_size_q_tp, hidden_size_k_tp,
        lse_size, element_size, is_pipeline_tick=False
    )

    # thd layout's hidden size input is "t,1,h"
    torch.manual_seed(seed)
    tensors = torch.randn(
        (as_world_size, total_seq_len, 1, hidden_size_q), dtype=dtype
    )
    tensor_shard = tensors[as_rank]
    # 1. normal forward. Need to provide the PackedSeqParams
    packed_seq_params = get_attn_metadata(
        doc_lens_per_rank[as_rank], get_packed_seq_params=True
    )
    normal_forward_out, debug_ref = worker.forward_normal(
        tensor_shard, packed_seq_params
    )
    
    torch.cuda.synchronize(); import time; time.sleep(1)

    ping_pong_params = PingPangSingleStepPackedSeqParams(
        qkv_format="thd",
        qkv_fwd_metadata=qkv_fwd_fa2a_metadata.get_slice(as_rank),
        qkv_bwd_metadata=qkv_rev_fa2a_metadata.get_slice(as_rank),
        attn_out_fwd_metadata=attn_out_fwd_fa2a_metadata.get_slice(as_rank),
        attn_out_bwd_metadata=attn_out_rev_fa2a_metadata.get_slice(as_rank),
        **as_attn_metadata[as_rank],
    )
    ping_pang_out, debug_out = worker.forward_ping_pang_one_stage(
        tensor_shard, ping_pong_params
    )

    torch.cuda.synchronize(); import time; time.sleep(1)

    print("Both forward done. Now checking correctness...")

    ref_debug = [None] * as_world_size
    ans_debug = [None] * as_world_size
    pingpong_seq_params = [None] * as_world_size
    as_group = get_attn_server_group()
    try:
        assert_close_report("normal_forward_out vs ping_pang_out", normal_forward_out, ping_pang_out)
    except AssertionError:
        print("Continuing to collect debug tensors for deeper comparison...")

    torch.distributed.all_gather_object(ref_debug, debug_ref, group=as_group)
    torch.distributed.all_gather_object(ans_debug, debug_out, group=as_group)
    torch.distributed.all_gather_object(pingpong_seq_params, ping_pong_params, group=as_group)
    print("debug tensors gathered.")
    if as_rank == 0:
        device = torch.device("cuda", worker.rank)
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
        assert_close_report("ref_qkvs vs ans_debug_qkvs_pre_transfer", ref_qkvs, ans_debug_qkvs_pre_transfer)
        ref_qs = [debug_tensor[0].flatten(start_dim=1) for debug_tensor in ref_qkvs]
        ref_ks = [debug_tensor[1].flatten(start_dim=1) for debug_tensor in ref_qkvs]
        ref_vs = [debug_tensor[2].flatten(start_dim=1) for debug_tensor in ref_qkvs]
        ref_qs_post_comm, ref_ks_post_comm, ref_vs_post_comm = simulate_all2all(
            ref_qs, ref_ks, ref_vs, qkv_fwd_fa2a_metadata,
            element_size, hidden_size_q_tp, hidden_size_k_tp,
            is_from_linear_layout=True,
        )
        ref_qs_post_comm = [
            t.unsqueeze(1) for t in ref_qs_post_comm
        ]
        ref_ks_post_comm = [
            t.unsqueeze(1) for t in ref_ks_post_comm
        ]
        ref_vs_post_comm = [
            t.unsqueeze(1) for t in ref_vs_post_comm
        ]
        ref_qkvs_post_comm = [
            (ref_qs_post_comm[rank], ref_ks_post_comm[rank], ref_vs_post_comm[rank]) for rank in range(as_world_size)
        ]
        assert_close_report("ans_debug_qkvs_post_transfer vs ref_qkvs_post_comm", ans_debug_qkvs_post_transfer, ref_qkvs_post_comm)

        from flash_attn import flash_attn_varlen_func
        ref_attn_outs_a_layout = []
        for rank in range(as_world_size):
            metadata = pingpong_seq_params[rank].to_device()
            print(ref_qs_post_comm[rank].shape)
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
        ref_attn_outs_post_comm, _, _ = simulate_all2all(
            [t.flatten(start_dim=1) for t in ref_attn_outs_a_layout],
            None, None, attn_out_fwd_fa2a_metadata,
            element_size, hidden_size_q_tp, None, is_from_linear_layout=False
        )
        ref_attn_outs_post_comm = [t.unsqueeze(1) for t in ref_attn_outs_post_comm]
        assert_close_report("ref_attn_outs vs ref_attn_outs_post_comm", ref_attn_outs, ref_attn_outs_post_comm)
        assert_close_report("ans_debug_core_attn_out vs ref_attn_outs_a_layout", ans_debug_core_attn_out, ref_attn_outs_a_layout)
        assert_close_report("ans_debug_core_attn_out_post_transfer vs ref_attn_outs", ans_debug_core_attn_out_post_transfer, ref_attn_outs)

        assert_close_report("normal_forward_out vs ping_pang_out (final)", normal_forward_out, ping_pang_out)
        print("pass final result")

    if profile:
        for _ in range(3):
            normal_forward_out, debug_ref = worker.forward_normal(
                tensor_shard, packed_seq_params, return_grad=True
            )
        torch.cuda.synchronize()
        torch.distributed.barrier()
        for _ in range(15):
            normal_forward_out, debug_ref = worker.forward_normal(
                tensor_shard, packed_seq_params, return_grad=True
            )
            torch.cuda.synchronize()
            torch.distributed.barrier()
        print("normal forward profiling done")
        for _ in range(3):
            ping_pang_out, debug_out = worker.forward_ping_pang_one_stage(
                tensor_shard, ping_pong_params, return_grad=True
            )
        torch.cuda.synchronize()
        torch.distributed.barrier()
        for _ in range(20):
            ping_pang_out, debug_out = worker.forward_ping_pang_one_stage(
                tensor_shard, ping_pong_params, return_grad=True
            )
            torch.cuda.synchronize()
            torch.distributed.barrier()
        print("ping-pong one stage forward profiling done")


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
        tensor_model_parallel_size=tp_size,
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
    num_query_heads = args.num_query_heads
    hidden_size_kv = (hidden_size * num_query_heads) // num_heads

    worker: MegatronLayerWorker = init_megatron_test(
        world_size, hidden_size, num_heads, num_query_heads, dtype,
        max_tokens_query, max_tokens_key_value, max_cp_degree, tp_size, seed,
    )

    test_forward(
        args.seed, args.num_tokens, args.num_docs, max_cp_degree,
        worker, hidden_size, hidden_size_kv,
        tp_size, profile=args.profile,
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
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-query-heads", type=int, default=2)
    parser.add_argument("--profile", action="store_true", default=False,)
    args = parser.parse_args()
    test(args)
