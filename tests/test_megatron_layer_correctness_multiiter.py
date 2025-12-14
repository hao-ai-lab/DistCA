"""
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
torchrun --nnodes 1 --nproc_per_node 2 test_megatron_layer_correctness.py \
    --world-size 2

NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
torchrun --nnodes 1 --nproc_per_node 4 test_megatron_layer_correctness.py \
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
from d2.runtime.megatron.model_patch import get_gpt_layer_with_transformer_engine_spec, get_gpt_config
from d2.runtime.megatron.packed_seq_params import PingPangSingleStepPackedSeqParams
from d2.runtime.megatron.ping_pong.transformer_layer import TransformerLayer as PingPangTransformerLayer

from test_util import (
    MegatronBaseWorker, ParallelConfig,
    init_worker_torch_distributed, random_shard_info_linear_layout_dp
)


def check_close(actual, expected, test_name: str):
    """Helper function to check tensor closeness and report with ✅ or ❌"""
    try:
        torch.testing.assert_close(actual, expected)
        print(f"✅ {test_name}")
        return True
    except AssertionError as e:
        print(f"❌ {test_name}")
        print(f"   Error: {str(e)}")
        return False


def get_key_module_grads(layer):
    """Extract gradients/main_grads for key module weights."""
    grads = {}
    modules = [
        ('attn_qkv', layer.self_attention.linear_qkv.weight),
        ('attn_proj', layer.self_attention.linear_proj.weight),
        ('mlp_fc1', layer.mlp.linear_fc1.weight),
        ('mlp_fc2', layer.mlp.linear_fc2.weight),
    ]
    for name, param in modules:
        if hasattr(param, 'main_grad') and param.main_grad is not None:
            grads[name] = param.main_grad
        else:
            grads[name] = param.grad
    return grads


def clone_layer_parameters(layer):
    return [param.detach().clone() for param in layer.parameters()]


def compare_parameter_sets(layer_normal, layer_pingpong, as_rank: int, prefix: str):
    for (name_a, param_a), (name_b, param_b) in zip(
        layer_normal.named_parameters(),
        layer_pingpong.named_parameters(),
    ):
        assert name_a == name_b, "Parameter ordering mismatch between model replicas."
        check_close(
            param_a, param_b,
            f"[Rank {as_rank}] {prefix}{name_a} parameter update match (normal vs ping-pong)"
        )


def run_backward_pair(
    worker: 'MegatronLayerWorker',
    tensor_shard: torch.Tensor,
    packed_seq_params: PackedSeqParams,
    ping_pong_params: PingPangSingleStepPackedSeqParams,
    loss_fn,
    base_seed: int,
    seed_offset: int,
    return_grad: bool,
):
    torch.manual_seed(base_seed + seed_offset)
    normal_backward_out, _ = worker.forward_normal(
        tensor_shard, packed_seq_params, return_grad=return_grad, loss_fn=loss_fn
    )
    torch.manual_seed(base_seed + seed_offset)
    ping_backward_out, _ = worker.forward_ping_pang_one_stage(
        tensor_shard, ping_pong_params, return_grad=return_grad, loss_fn=loss_fn
    )
    return normal_backward_out, ping_backward_out


def compare_backward_results(normal_backward_out, ping_backward_out, as_rank: int, prefix: str = ""):
    _, _, normal_input_grad, normal_loss = normal_backward_out
    _, _, pp_input_grad, pp_loss = ping_backward_out
    check_close(
        normal_loss, pp_loss,
        f"[Rank {as_rank}] {prefix}Loss value comparison (normal vs ping-pong)"
    )
    check_close(
        normal_input_grad, pp_input_grad,
        f"[Rank {as_rank}] {prefix}Input gradient comparison (normal vs ping-pong)"
    )


def compare_module_gradients(worker: 'MegatronLayerWorker', as_rank: int, prefix: str = ""):
    normal_grads = get_key_module_grads(worker.layer_normal)
    pp_grads = get_key_module_grads(worker.layer_pingpong)
    for name in normal_grads:
        check_close(
            normal_grads[name], pp_grads[name],
            f"[Rank {as_rank}] {prefix}{name} weight gradient (normal vs ping-pong)"
        )


class MegatronLayerWorker(MegatronBaseWorker):
    def __init__(self, rank: int, world_size: int):
        super().__init__(rank, world_size)
        self.layer_normal: Optional[PingPangTransformerLayer] = None
        self.layer_pingpong: Optional[PingPangTransformerLayer] = None

    def init_layer(self, config: TransformerConfig, spec: ModuleSpec,
                   seed: int):
        # Initialize both models with the same seed to ensure identical weights
        torch.manual_seed(seed + mpu.get_tensor_model_parallel_rank())
        self.layer_normal = build_module(spec, config)
        
        torch.manual_seed(seed + mpu.get_tensor_model_parallel_rank())
        self.layer_pingpong = build_module(spec, config)

    def forward_normal(self, tensor_input: torch.Tensor, packed_seq_params: PackedSeqParams,
                       return_grad: bool = False, loss_fn=None):
        packed_seq_params = PackedSeqParams(
            qkv_format=packed_seq_params.qkv_format,
            cu_seqlens_q=packed_seq_params.cu_seqlens_q.cuda().to(torch.int32),
            cu_seqlens_kv=packed_seq_params.cu_seqlens_kv.cuda().to(torch.int32),
            max_seqlen_q=packed_seq_params.max_seqlen_q,
            max_seqlen_kv=packed_seq_params.max_seqlen_kv,
        )
        tensor_input = tensor_input.cuda().detach()
        tensor_input.requires_grad = True
        self.layer_normal.train()

        if return_grad:
            ctx = torch.enable_grad()
        else:
            ctx = nullcontext()
        with ctx:
            mlp_output, context, debug = self.layer_normal.forward_orig_impl(
                tensor_input, packed_seq_params=packed_seq_params, return_debug=True,
            )
            if return_grad:
                if loss_fn is not None:
                    loss = loss_fn(mlp_output)
                else:
                    loss = mlp_output.sum()
                loss.backward()
            else:
                loss = None

        torch.cuda.synchronize()
        print(self.rank, "normal forward done")
        if return_grad:
            return (mlp_output, context, tensor_input.grad, loss), debug
        else:
            return (mlp_output, context), debug

    def forward_ping_pang_one_stage(
        self, tensor_input: torch.Tensor,
        packed_seq_params: PingPangSingleStepPackedSeqParams,
        return_grad: bool = False,
        loss_fn=None,
    ):
        packed_seq_params = packed_seq_params.to_device()
        tensor_input = tensor_input.cuda().detach()
        tensor_input.requires_grad = True

        self.layer_pingpong.train()

        if return_grad:
            ctx = torch.enable_grad()
        else:
            ctx = nullcontext()
        with ctx:
            mlp_output, context, debug_tensors = self.layer_pingpong.forward_ping_pong_single_sided(
                tensor_input, packed_seq_params=packed_seq_params,
                return_debug=True,
            )
            if return_grad:
                if loss_fn is not None:
                    loss = loss_fn(mlp_output)
                else:
                    loss = mlp_output.sum()
                loss.backward()
            else:
                loss = None

        torch.cuda.synchronize()
        print(self.rank, "ping-pong one stage forward done")
        if return_grad:
            return (mlp_output, context, tensor_input.grad, loss), debug_tensors
        else:
            return (mlp_output, context), debug_tensors


def test_forward(
    seed: int, total_seq_len: int, num_docs: int, max_cp_degree: int,
    worker: MegatronLayerWorker, hidden_size_q: int, hidden_size_k: int,
    tp_size: int = 1, return_grad: bool = True,
    num_iterations = 3
):
    # ========================================
    # Setup and metadata preparation
    # ========================================
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

    # ========================================
    # Generate input tensors
    # ========================================
    torch.manual_seed(seed)
    tensors = torch.randn(
        (as_world_size, total_seq_len, 1, hidden_size_q), dtype=dtype
    )
    tensor_shard = tensors[as_rank]
    
    # ========================================
    # Normal forward pass
    # ========================================
    packed_seq_params = get_attn_metadata(
        doc_lens_per_rank[as_rank], get_packed_seq_params=True
    )
    normal_forward_out, _ = worker.forward_normal(
        tensor_shard, packed_seq_params
    )

    # ========================================
    # Ping-pong forward pass
    # ========================================
    ping_pong_params = PingPangSingleStepPackedSeqParams(
        qkv_format="thd",
        qkv_fwd_metadata=qkv_fwd_fa2a_metadata.get_slice(as_rank),
        qkv_bwd_metadata=qkv_rev_fa2a_metadata.get_slice(as_rank),
        attn_out_fwd_metadata=attn_out_fwd_fa2a_metadata.get_slice(as_rank),
        attn_out_bwd_metadata=attn_out_rev_fa2a_metadata.get_slice(as_rank),
        **as_attn_metadata[as_rank],
    )
    ping_pang_out, _ = worker.forward_ping_pang_one_stage(
        tensor_shard, ping_pong_params
    )

    # ========================================
    # Forward output comparison
    # ========================================
    check_close(
        normal_forward_out, ping_pang_out,
        f"[Rank {as_rank}] Forward output comparison (normal vs ping-pong)"
    )

    # ========================================
    # Backward pass and gradient comparison
    # ========================================
    if return_grad:
        loss_fn = lambda x: x.mean()
        opt_normal = torch.optim.SGD(
            [param for param in worker.layer_normal.parameters() if param.requires_grad],
            lr=1e-3,
        )
        opt_pingpong = torch.optim.SGD(
            [param for param in worker.layer_pingpong.parameters() if param.requires_grad],
            lr=1e-3,
        )
        
        opt_normal.zero_grad(set_to_none=True)
        opt_pingpong.zero_grad(set_to_none=True)

        for iter_idx in range(num_iterations):
            iter_prefix = f"Iter {iter_idx + 1}/{num_iterations} "
            print(f"[Rank {as_rank}] ===== {iter_prefix}pre-step validation =====")

            # Pre-step backward/gradient comparison
            seed_pre = 1000 + iter_idx * 100
            normal_backward_out, ping_backward_out = run_backward_pair(
                worker, tensor_shard, packed_seq_params, ping_pong_params,
                loss_fn, seed, seed_pre, return_grad
            )
            compare_backward_results(normal_backward_out, ping_backward_out, as_rank)
            compare_module_gradients(worker, as_rank)

            # Optimizer step
            normal_before = clone_layer_parameters(worker.layer_normal)
            ping_before = clone_layer_parameters(worker.layer_pingpong)

            opt_normal.step()
            opt_pingpong.step()

            compare_parameter_sets(
                worker.layer_normal, worker.layer_pingpong, as_rank,
                prefix="Post-optimizer "
            )

            changed_normal = any(
                not torch.allclose(param, before)
                for param, before in zip(worker.layer_normal.parameters(), normal_before)
            )
            changed_ping = any(
                not torch.allclose(param, before)
                for param, before in zip(worker.layer_pingpong.parameters(), ping_before)
            )
            if not (changed_normal or changed_ping):
                print(f"⚠️ [Rank {as_rank}] Optimizer step did not change any parameters.")

            opt_normal.zero_grad(set_to_none=True)
            opt_pingpong.zero_grad(set_to_none=True)
            print(f"[Rank {as_rank}] ----- {iter_prefix}post-step validation -----")

            # Post-step backward/gradient comparison
            seed_post = 2000 + iter_idx * 100
            normal_backward_out_post, ping_backward_out_post = run_backward_pair(
                worker, tensor_shard, packed_seq_params, ping_pong_params,
                loss_fn, seed, seed_post, return_grad
            )
            compare_backward_results(
                normal_backward_out_post, ping_backward_out_post, as_rank, prefix="Post-optimizer "
            )
            compare_module_gradients(worker, as_rank, prefix="Post-optimizer ")

            opt_normal.zero_grad(set_to_none=True)
            opt_pingpong.zero_grad(set_to_none=True)
            print(f"[Rank {as_rank}] ===== Completed {iter_prefix}cycle =====")


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
        tp_size, return_grad=args.return_grad,
        num_iterations=args.num_iterations,
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
    parser.add_argument(
        "--return-grad",
        action="store_true",
        default=False,
        help="Run backward pass, compare grads, perform THREE optimizer steps (with prints), and re-validate each time.",
    )
    parser.add_argument("--num-iterations", type=int, default=3)
    args = parser.parse_args()
    test(args)

