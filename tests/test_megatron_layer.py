"""
Using ray to init processes. This makes multi process logging format cleaner and launching script simpler.
"""

import argparse
from typing import Optional

from megatron.core import parallel_state as mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
import ray
from ray.util.placement_group import placement_group
import torch

from d2.runtime.inplace_metadata import compute_attn_layout_seqlens, orchestrate_simulate, Metadata
from d2.runtime.megatron_patch.model_patch import get_gpt_layer_with_transformer_engine_spec, get_gpt_config
from d2.runtime.megatron_patch.packed_seq_params import PingPangPackedSeqParams, PingPangSingleStepPackedSeqParams
from d2.runtime.megatron_patch.transformer_layer import TransformerLayer as PingPangTransformerLayer

from test_dispatch_qkv import create_testcase_qkv
from test_util import MegatronBaseWorker, ParallelConfig


def create_pg(num_nodes: int, num_gpus_per_node: int, worker_cls):
    gpu_nodes = [node for node in ray.nodes() if node.get("Resources", {}).get("GPU")]
    gpu_nodes.sort(key=lambda node: node["NodeManagerAddress"])

    workers = []
    world_size = num_nodes * num_gpus_per_node
    master_addr = None
    master_port = None
    worker_cls = ray.remote(worker_cls)

    for n_id in range(num_nodes):
        node = gpu_nodes[n_id]
        node_id = node["NodeID"]
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus_per_node)]
        pg = placement_group(bundles, strategy="STRICT_PACK", name=f"node_{n_id}", _soft_target_node_id=node_id)
        ray.get(pg.ready())
        for i in range(num_gpus_per_node):

            rank = n_id * num_gpus_per_node + i
            env_vars = {
                "WORLD_SIZE": str(world_size),
                "RANK": str(rank),
                "LOCAL_RANK": str(i),
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",    # NOTE: this is for deterministic model in debug.
            }
            if rank > 0:
                env_vars["MASTER_ADDR"] = master_addr
                env_vars["MASTER_PORT"] = master_port

            worker = worker_cls.options(
                # Target the placement group and a specific bundle within it
                placement_group=pg,
                placement_group_bundle_index=i,
                num_gpus=1,
                num_cpus=1,
                runtime_env={"env_vars": env_vars},
            ).remote(
                world_size=world_size,
                rank=n_id * num_gpus_per_node + i,
            )
            workers.append(worker)

            if rank == 0:
                master_addr, master_port = ray.get(worker.get_node_ip_port.remote())
                worker.set_master_addr_port.remote(master_addr, master_port)
    return workers


class MegatronLayerWorker(MegatronBaseWorker):
    def __init__(self, rank: int, world_size: int):
        super().__init__(rank, world_size)
        self.layer: Optional[PingPangTransformerLayer] = None

    #### Megatron layer init and running functions
    def init_layer(self, config: TransformerConfig, spec: ModuleSpec,
                   seed: int):
        torch.manual_seed(seed)
        self.layer = build_module(spec, config)
        # FIXME: init layer weights

    def forward_normal(self, tensor_input: torch.Tensor, packed_seq_params: PackedSeqParams):
        packed_seq_params = PackedSeqParams(
            qkv_format=packed_seq_params.qkv_format,
            cu_seqlens_q=packed_seq_params.cu_seqlens_q.cuda().to(torch.int32),
            cu_seqlens_kv=packed_seq_params.cu_seqlens_kv.cuda().to(torch.int32),
            max_seqlen_q=packed_seq_params.max_seqlen_q.cuda().to(torch.int32),
            max_seqlen_kv=packed_seq_params.max_seqlen_kv.cuda().to(torch.int32),
        )
        tensor_input = tensor_input.cuda()
        self.layer.train()
        mlp_output, context, debug = self.layer.forward_no_switch(tensor_input, packed_seq_params=packed_seq_params)
        torch.cuda.synchronize()
        print(self.rank, "normal forward done")
        return (mlp_output, context), debug

    def forward_ping_pang_one_stage(self, tensor_input: torch.Tensor, packed_seq_params: PingPangSingleStepPackedSeqParams):
        packed_seq_params = packed_seq_params.to_device()
        tensor_input = tensor_input.cuda()
        self.layer.train()
        mlp_output, context, debug_tensors = self.layer.forward_one_stage(tensor_input, packed_seq_params=packed_seq_params)
        torch.cuda.synchronize()
        print(self.rank, "ping-pong one stage forward done")
        return (mlp_output, context), debug_tensors


def get_seqlen_shard(cu_seqlens_q: torch.Tensor, cu_seqlens_kv: torch.Tensor,
                     max_seqlen_q: torch.Tensor, max_seqlen_kv: torch.Tensor, num_local_seqs_recv: torch.Tensor, rank: int):
    num_seq = num_local_seqs_recv[rank].item()
    max_seqlen_q = max_seqlen_q[rank]
    max_seqlen_kv = max_seqlen_kv[rank]
    cu_seqlens_q = torch.cat([torch.zeros((1,), dtype=cu_seqlens_q.dtype), cu_seqlens_q[rank][:num_seq]])
    cu_seqlens_kv = torch.cat([torch.zeros((1,), dtype=cu_seqlens_kv.dtype), cu_seqlens_kv[rank][:num_seq]])
    return cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, num_seq


def simulate_communication(tensors: list[torch.Tensor], metadata: Metadata):
    world_size = len(tensors)
    assert world_size == metadata.world_size
    output_seq_len = metadata.num_recv_tokens.max()
    input_pad_len = max(tensor.shape[0] for tensor in tensors)
    pad_tensors = [
        torch.cat([
            tensor,
            torch.zeros(
                (input_pad_len - tensor.shape[0], *tensor.shape[1:]),
                dtype=tensor.dtype, device=tensor.device
            )
        ], dim=0).unsqueeze(0) for tensor in tensors
    ]
    input_tensor = torch.cat(pad_tensors, dim=0)
    output_tensor = torch.zeros((world_size, output_seq_len, *input_tensor.shape[2:]), dtype=input_tensor.dtype, device=input_tensor.device)
    output_tensor = orchestrate_simulate(
        input_tensor.reshape(world_size, input_pad_len, -1),
        output_tensor.reshape(world_size, output_seq_len, -1),
        metadata
    ).reshape(world_size, output_seq_len, *input_tensor.shape[2:])
    output_tensors = torch.split(output_tensor, 1, dim=0)
    output_tensors_split = [
        t[0, :metadata.num_recv_tokens[rank].max()] for rank, t in enumerate(output_tensors)
    ]
    return output_tensors_split


@torch.no_grad()
def test_dp_single_split(workers, seed: int, num_tokens: int, max_cp_degree: int, num_seqs: int, hidden_size: int, seqlen_multiple: int=1):
    world_size = len(workers)
    # Create two splits for ping-pong
    (
        fwd_q_metadata, rev_q_metadata,
        fwd_kv_metadata, rev_kv_metadata,
        sp_kv_dst, sp_seq_lens, sp_query_dst, sp_dst_kv_len, seq_lens,
    ) = create_testcase_qkv(
        seed, world_size, num_tokens, max_cp_degree, num_seqs,
        seqlen_multiple=seqlen_multiple,
    )

    (cu_seqlens_q_pp, cu_seqlens_kv_pp, max_seqlen_q_pp, max_seqlen_kv_pp, num_local_seqs_recv_pp) = compute_attn_layout_seqlens(
        sp_seq_lens, sp_dst_kv_len, sp_query_dst
    )

    # Create tensor input
    tensor_input = torch.randn(world_size, num_tokens, hidden_size, dtype=torch.float16)
    ref_ans_handles = []
    ans_handles = []
    args = []
    for rank in range(world_size):
        tensor_input_local = tensor_input[rank].unsqueeze(1)
        fwd_q_metadata_local = fwd_q_metadata.get_slice(rank)
        rev_q_metadata_local = rev_q_metadata.get_slice(rank)
        fwd_kv_metadata_local = fwd_kv_metadata.get_slice(rank)
        rev_kv_metadata_local = rev_kv_metadata.get_slice(rank)

        # Create packed seq params metadata
        # Normal forward. No layout switch. Running data parallel
        seq_lens_local = seq_lens[rank]
        cu_seqlens_q = torch.cat([torch.zeros((1,), dtype=seq_lens_local.dtype), seq_lens_local.cumsum(dim=0)])
        cu_seqlens_kv = cu_seqlens_q.clone()
        max_seqlen_q = seq_lens_local.max()
        max_seqlen_kv = max_seqlen_q.clone()
        packed_seq_params_normal = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
        )
        # Ping-pong forward. cu_seqlens is in a special layout.
        (cu_seqlens_q_pp_local, cu_seqlens_kv_pp_local, max_seqlen_q_pp_local, max_seqlen_kv_pp_local, num_seq_pp_local) = get_seqlen_shard(
            cu_seqlens_q_pp, cu_seqlens_kv_pp, max_seqlen_q_pp, max_seqlen_kv_pp, num_local_seqs_recv_pp, rank
        )
        packed_seq_params_stage_0 = PingPangSingleStepPackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_q_pp_local,
            cu_seqlens_kv=cu_seqlens_kv_pp_local,
            max_seqlen_q=max_seqlen_q_pp_local,
            max_seqlen_kv=max_seqlen_kv_pp_local,
            mlp_to_attn_metadata=fwd_q_metadata_local,
            attn_to_mlp_metadata=rev_q_metadata_local,
            mlp_to_attn_kv_metadata=fwd_kv_metadata_local,
            mlp_to_attn_kv_grad_metadata=rev_kv_metadata_local,
        )

        # Compute the reference answer and test result
        ref_ans_handle = workers[rank].forward_normal.remote(
            tensor_input_local, packed_seq_params_normal
        )
        args.append([tensor_input_local, packed_seq_params_stage_0])
        ref_ans_handles.append(ref_ans_handle)
    ref_ans = ray.get(ref_ans_handles)
    ref_debug = [ref[1] for ref in ref_ans]
    ref_ans = [ref[0] for ref in ref_ans]

    for rank in range(world_size):
        ans_handle = workers[rank].forward_ping_pang_one_stage.remote(
            *args[rank]
        )
        ans_handles.append(ans_handle)
    ans = ray.get(ans_handles)
    ans_debug = [ans[1] for ans in ans]

    # Debug intermediate tensors
    ans_debug_qkvs_pre_transfer = [ans_debug[0] for ans_debug in ans_debug]
    ans_debug_qkvs_post_transfer = [ans_debug[1] for ans_debug in ans_debug]
    ans_debug_core_attn_out = [ans_debug[2] for ans_debug in ans_debug]
    ans_debug_core_attn_out_post_transfer = [ans_debug[3] for ans_debug in ans_debug]
    ans = [ans[0] for ans in ans]

    ref_qkvs = [debug_tensor[0] for debug_tensor in ref_debug]
    ref_attn_outs = [debug_tensor[1] for debug_tensor in ref_debug]
    torch.testing.assert_close(ref_qkvs, ans_debug_qkvs_pre_transfer)
    print("debug pre-layout-transfer qkv allclose")
    ref_qs = [debug_tensor[0] for debug_tensor in ref_qkvs]
    ref_ks = [debug_tensor[1] for debug_tensor in ref_qkvs]
    ref_vs = [debug_tensor[2] for debug_tensor in ref_qkvs]
    ref_qs_post_comm = simulate_communication(ref_qs, fwd_q_metadata)
    ref_ks_post_comm = simulate_communication(ref_ks, fwd_kv_metadata)
    ref_vs_post_comm = simulate_communication(ref_vs, fwd_kv_metadata)
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
        metadata = args[rank][1].to_device()
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

    torch.testing.assert_close(ref_ans, ans)


def init_test(args, worker_cls=MegatronLayerWorker):
    ray.init()
    workers = create_pg(args.num_nodes, args.num_gpus_per_node, worker_cls)
    print("Workers created")
    stride_q = args.hidden_size * torch.float16.itemsize
    stride_kv = args.hidden_size * torch.float16.itemsize * 2
    world_size = len(workers)
    # NOTE: a reason very likely causing the hanging is that
    # max_tokens_query and max_tokens_key_value are not large enough (nvshmem buffer not enough)
    max_tokens_query = args.num_tokens * world_size
    max_tokens_key_value = args.num_tokens * world_size
    parallel_config = ParallelConfig(
        tensor_model_parallel_size=args.tp_size
    )
    ray.get([worker.init_comm.remote(
        stride_q, stride_kv, max_tokens_query, max_tokens_key_value, parallel_config
    ) for worker in workers])
    print("Communication groups initialized")

    seed = args.seed
    spec = get_gpt_layer_with_transformer_engine_spec()
    config = get_gpt_config(
        num_layers=1,
        hidden_size=args.hidden_size,
        num_attention_heads=2,
        ffn_hidden_size=args.hidden_size * 4,
        fp16=True,
        deterministic_mode=True,
        params_dtype=torch.float16,
    )
    ray.get([
        worker.init_layer.remote(config, spec, seed) for worker in workers
    ])
    return workers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--cp-degree", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-seqs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-gpus-per-node", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=1)
    args = parser.parse_args()
    workers = init_test(args)
    print("Test env initialized.")
    test_dp_single_split(workers, args.seed, args.num_tokens, args.cp_degree, args.num_seqs, args.hidden_size)
    print("test done.")
