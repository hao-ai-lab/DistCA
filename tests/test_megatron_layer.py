"""
Using ray to init processes. This makes multi process logging format cleaner and launching script simpler.
"""

import argparse
from dataclasses import dataclass
import os
import socket
from typing import Optional

from megatron.core import parallel_state as mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
import ray
from ray.util.placement_group import placement_group
import torch

from d2.runtime.attn_kernels.ops import nvshmem_get_unique_id, nvshmem_alloc_empty_unique_id, DispatcherWrapper
from d2.runtime.inplace_metadata import compute_attn_layout_seqlens
from d2.runtime.megatron_patch.model_patch import get_gpt_layer_with_transformer_engine_spec, get_gpt_config
from d2.runtime.megatron_patch.packed_seq_params import PingPangPackedSeqParams, PingPangSingleStepPackedSeqParams
from d2.runtime.megatron_patch.transformer_layer import TransformerLayer as PingPangTransformerLayer

from test_dispatch_qkv import create_testcase_qkv


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


@dataclass
class ParallelConfig:
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: Optional[int] = None
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: Optional[int] = None


class MegatronBaseWorker:
    """Worker base class to init communication groups (megatron and nvshmem)."""
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.nvshmem_initialized = False
        self.nvshmem_pe = None

    #### General init functions
    def get_node_ip_port(self):
        host_ipv4 = os.getenv("MY_HOST_IP", None)
        host_ipv6 = os.getenv("MY_HOST_IPV6", None)
        host_ip_by_env = host_ipv4 or host_ipv6
        host_ip_by_sdk = ray._private.services.get_node_ip_address()

        host_ip = host_ip_by_env or host_ip_by_sdk

        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return host_ip, str(port)

    def set_master_addr_port(self, master_addr: str, master_port: str):
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port

    def init_comm(self, stride: int, max_tokens_query: int, max_tokens_key_value: int,
                  parallel_config: ParallelConfig):
        # Init megatron communication.
        if not torch.distributed.is_initialized():
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl", rank=self.rank, world_size=self.world_size)
        # NOTE: do not set to local_rank here because the cuda visible device is set by ray.
        torch.cuda.set_device(0)

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=parallel_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=parallel_config.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=parallel_config.virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank=None,
            use_sharp=False,
            context_parallel_size=parallel_config.context_parallel_size,
            expert_model_parallel_size=parallel_config.expert_model_parallel_size,
            expert_tensor_parallel_size=parallel_config.expert_tensor_parallel_size,
            nccl_communicator_config_path=None,
        )
        # Init nvshmem.
        if self.rank == 0:
            uid = nvshmem_get_unique_id()
        else:
            uid = nvshmem_alloc_empty_unique_id()
        torch.distributed.broadcast(uid, src=0)

        DispatcherWrapper.init(
            q_stride=stride,
            kv_stride=stride,
            max_tokens_query=max_tokens_query,
            max_tokens_key_value=max_tokens_key_value,
            rank=self.rank,
            world_size=self.world_size,
            uid=uid,
        )


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

    def forward_normal(self, tensor_input: torch.Tensor, packed_seq_params: PingPangPackedSeqParams):
        return self.layer(tensor_input, packed_seq_params=packed_seq_params)

    def forward_ping_pang(self, tensor_input: torch.Tensor, packed_seq_params: PingPangPackedSeqParams):
        return self.layer.ping_pang_forward(tensor_input, packed_seq_params=packed_seq_params)

    def forward_ping_pang_one_stage(self, tensor_input: torch.Tensor, packed_seq_params: PingPangSingleStepPackedSeqParams):
        packed_seq_params = packed_seq_params.to_device()
        return self.layer.forward_one_stage(tensor_input, packed_seq_params=packed_seq_params)

def get_seqlen_shard(cu_seqlens_q: torch.Tensor, cu_seqlens_kv: torch.Tensor,
                     max_seqlen_q: torch.Tensor, max_seqlen_kv: torch.Tensor, num_local_seqs_recv: torch.Tensor, rank: int):
    num_seq = num_local_seqs_recv[rank]
    max_seqlen_q = max_seqlen_q[rank]
    max_seqlen_kv = max_seqlen_kv[rank]
    cu_seqlens_q = cu_seqlens_q[rank][:num_seq]
    cu_seqlens_kv = cu_seqlens_kv[rank][:num_seq]
    return cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, num_seq


@torch.no_grad()
def test_dp(workers, seed, num_tokens, max_cp_degree, num_seqs, hidden_size):
    world_size = len(workers)
    # Create two splits for ping-pong
    (
        fwd_q_metadata_0, rev_q_metadata_0,
        fwd_kv_metadata_0, rev_kv_metadata_0,
        sp_kv_dst_0, sp_seq_lens_0, cp_dst_kv_len_0, seq_lens_0,
    ) = create_testcase_qkv(seed, world_size, num_tokens, max_cp_degree, num_seqs)
    (
        fwd_q_metadata_1, rev_q_metadata_1,
        fwd_kv_metadata_1, rev_kv_metadata_1,
        sp_kv_dst_1, sp_seq_lens_1, cp_dst_kv_len_1, seq_lens_1,
    ) = create_testcase_qkv(seed, world_size, num_tokens, max_cp_degree, num_seqs)
    (cu_seqlens_q_pp_0, cu_seqlens_kv_pp_0, max_seqlen_q_pp_0, max_seqlen_kv_pp_0, num_local_seqs_recv_pp_0) = compute_attn_layout_seqlens(
        sp_seq_lens_0, cp_dst_kv_len_0, sp_kv_dst_0
    )
    (cu_seqlens_q_pp_1, cu_seqlens_kv_pp_1, max_seqlen_q_pp_1, max_seqlen_kv_pp_1, num_local_seqs_recv_pp_1) = compute_attn_layout_seqlens(
        sp_seq_lens_1, cp_dst_kv_len_1, sp_kv_dst_1
    )

    # Create tensor input
    tensor_input = torch.randn(world_size, num_tokens * 2, hidden_size, dtype=torch.float16)
    ref_ans_handles = []
    ans_handles = []
    for rank in range(world_size):
        tensor_input_local = tensor_input[rank]
        fwd_q_metadata_0_local = fwd_q_metadata_0.get_slice(rank)
        rev_q_metadata_0_local = rev_q_metadata_0.get_slice(rank)
        fwd_kv_metadata_0_local = fwd_kv_metadata_0.get_slice(rank)
        rev_kv_metadata_0_local = rev_kv_metadata_0.get_slice(rank)
        fwd_q_metadata_1_local = fwd_q_metadata_1.get_slice(rank)
        rev_q_metadata_1_local = rev_q_metadata_1.get_slice(rank)
        fwd_kv_metadata_1_local = fwd_kv_metadata_1.get_slice(rank)
        rev_kv_metadata_1_local = rev_kv_metadata_1.get_slice(rank)

        # Create packed seq params metadata
        # Normal forward. No layout switch. Batches are in the normal data parallel on each rank.
        seq_lens_local_0 = seq_lens_0[rank]
        seq_lens_local_1 = seq_lens_1[rank]
        seq_lens_local = torch.cat([seq_lens_local_0, seq_lens_local_1], dim=0)
        cu_seqlens_q = seq_lens_local.cuda().cumsum(dim=0)
        cu_seqlens_kv = cu_seqlens_q.clone()
        max_seqlen_q = seq_lens_local.max().cuda()
        max_seqlen_kv = max_seqlen_q.clone()
        packed_seq_params_normal = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
        )
        # Ping-pong forward. cu_seqlens is in a special layout.
        (cu_seqlens_q_0, cu_seqlens_kv_0, max_seqlen_q_0, max_seqlen_kv_0, num_seq_0) = get_seqlen_shard(
            cu_seqlens_q_pp_0, cu_seqlens_kv_pp_0, max_seqlen_q_pp_0, max_seqlen_kv_pp_0, num_local_seqs_recv_pp_0, rank
        )
        (cu_seqlens_q_1, cu_seqlens_kv_1, max_seqlen_q_1, max_seqlen_kv_1, num_seq_1) = get_seqlen_shard(
            cu_seqlens_q_pp_1, cu_seqlens_kv_pp_1, max_seqlen_q_pp_1, max_seqlen_kv_pp_1, num_local_seqs_recv_pp_1, rank
        )
        packed_seq_params_stage_0 = PingPangSingleStepPackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_q_0,
            cu_seqlens_kv=cu_seqlens_kv_0,
            max_seqlen_q=max_seqlen_q_0,
            max_seqlen_kv=max_seqlen_kv_0,
            mlp_to_attn_metadata=fwd_q_metadata_0_local,
            attn_to_mlp_metadata=rev_q_metadata_0_local,
            mlp_to_attn_kv_metadata=fwd_kv_metadata_0_local,
            mlp_to_attn_kv_grad_metadata=rev_kv_metadata_0_local,
        )
        packed_seq_params_stage_1 = PingPangSingleStepPackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_q_1,
            cu_seqlens_kv=cu_seqlens_kv_1,
            max_seqlen_q=max_seqlen_q_1,
            max_seqlen_kv=max_seqlen_kv_1,
            mlp_to_attn_metadata=fwd_q_metadata_1_local,
            attn_to_mlp_metadata=rev_q_metadata_1_local,
            mlp_to_attn_kv_metadata=fwd_kv_metadata_1_local,
            mlp_to_attn_kv_grad_metadata=rev_kv_metadata_1_local,
        )
        packed_seq_params_ping_pang = PingPangPackedSeqParams(
            debug=True,
            seq_params=[packed_seq_params_stage_0, packed_seq_params_stage_1],
        )

        # Compute the reference answer and test result
        ref_ans_handle = workers[rank].forward_normal.remote(
            tensor_input_local, packed_seq_params_normal
        )

        ans_handle = workers[rank].forward_ping_pang.remote(
            tensor_input_local, packed_seq_params_ping_pang
        )
        ref_ans_handles.append(ref_ans_handle)
        ans_handles.append(ans_handle)

    ref_ans = ray.get(ref_ans_handles)
    ans = ray.get(ans_handles)
    for r, a in zip(ref_ans, ans):
        torch.testing.assert_close(ref_ans, ans)


def test_dp_single_split(workers, seed: int, num_tokens: int, max_cp_degree: int, num_seqs: int, hidden_size: int):
    world_size = len(workers)
    # Create two splits for ping-pong
    (
        fwd_q_metadata, rev_q_metadata,
        fwd_kv_metadata, rev_kv_metadata,
        sp_kv_dst, sp_seq_lens, cp_dst_kv_len, seq_lens,
    ) = create_testcase_qkv(seed, world_size, num_tokens, max_cp_degree, num_seqs)

    (cu_seqlens_q_pp, cu_seqlens_kv_pp, max_seqlen_q_pp, max_seqlen_kv_pp, num_local_seqs_recv_pp) = compute_attn_layout_seqlens(
        sp_seq_lens, cp_dst_kv_len, sp_kv_dst
    )

    # Create tensor input
    tensor_input = torch.randn(world_size, num_tokens, hidden_size, dtype=torch.float16)
    ref_ans_handles = []
    ans_handles = []
    for rank in range(world_size):
        tensor_input_local = tensor_input[rank]
        fwd_q_metadata_local = fwd_q_metadata.get_slice(rank)
        rev_q_metadata_local = rev_q_metadata.get_slice(rank)
        fwd_kv_metadata_local = fwd_kv_metadata.get_slice(rank)
        rev_kv_metadata_local = rev_kv_metadata.get_slice(rank)

        # Create packed seq params metadata
        # Normal forward. No layout switch. Running data parallel
        seq_lens_local = seq_lens[rank]
        cu_seqlens_q = seq_lens_local.cuda().cumsum(dim=0)
        cu_seqlens_kv = cu_seqlens_q.clone()
        max_seqlen_q = seq_lens_local.max().cuda()
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

        ans_handle = workers[rank].forward_ping_pang_one_stage.remote(
            tensor_input_local, packed_seq_params_stage_0
        )
        ref_ans_handles.append(ref_ans_handle)
        ans_handles.append(ans_handle)


def init_test(args):
    ray.init()
    workers = create_pg(args.num_nodes, args.num_gpus_per_node, MegatronLayerWorker)
    print("Workers created")
    stride = args.hidden_size * torch.float16.itemsize
    world_size = len(workers)
    max_tokens_query = args.num_tokens * world_size
    max_tokens_key_value = args.num_tokens * world_size
    parallel_config = ParallelConfig(
        tensor_model_parallel_size=args.tp_size
    )
    ray.get([worker.init_comm.remote(
        stride, max_tokens_query, max_tokens_key_value, parallel_config
    ) for worker in workers])
    print("Communication groups initialized")

    seed = args.seed
    spec = get_gpt_layer_with_transformer_engine_spec()
    config = get_gpt_config(
        num_layers=1,
        hidden_size=args.hidden_size,
        num_attention_heads=1,
        ffn_hidden_size=args.hidden_size * 4,
    )
    ray.get([
        worker.init_layer.remote(config, spec, seed) for worker in workers
    ])
    return workers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--cp-degree", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-seqs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-gpus-per-node", type=int, default=4)
    parser.add_argument("--tp-size", type=int, default=1)
    args = parser.parse_args()
    workers = init_test(args)
    print("Test env initialized.")
    test_dp_single_split(workers, args.seed, args.num_tokens, args.cp_degree, args.num_seqs, args.hidden_size)
