# Run the test with:
# torchrun --nproc_per_node=4 --nnodes=1 --rdzv_id=5800 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 test_megatron_layer.py

import os

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
import torch

from d2.runtime.attn_kernels.ops import nvshmem_get_unique_id, nvshmem_alloc_empty_unique_id, DispatcherWrapper
from d2.runtime.inplace_metadata import compute_attn_layout_seqlens
from d2.runtime.megatron_patch.packed_seq_params import PingPangPackedSeqParams, PingPangSingleStepPackedSeqParams
from d2.runtime.megatron_patch.model_patch import get_gpt_layer_with_transformer_engine_spec, get_gpt_config

from test_dispatch_qkv import create_testcase_qkv


class Worker:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.nvshmem_initialized = False
        self.nvshmem_pe = None
        self.layer = None

    def init_comm(self, uid: torch.Tensor, stride: int, max_tokens_query: int, max_tokens_key_value: int):
        DispatcherWrapper.init(
            q_stride=stride,
            kv_stride=stride,
            max_tokens_query=max_tokens_query,
            max_tokens_key_value=max_tokens_key_value,
            rank=self.rank,
            world_size=self.world_size,
            uid=uid,
        )
        # TODO: init megatron distributed environment to test TP, CP, DP backward, etc.

    def init_layer(self, config: TransformerConfig, spec: ModuleSpec,
                   seed: int):
        torch.manual_seed(seed)
        self.layer = build_module(spec, config)
        # FIXME: init layer weights

    def forward_normal(self, tensor_input: torch.Tensor, packed_seq_params: PingPangPackedSeqParams):
        return self.layer(tensor_input, packed_seq_params=packed_seq_params)

    def forward_ping_pang(self, tensor_input: torch.Tensor, packed_seq_params: PingPangPackedSeqParams):
        return self.layer.ping_pang_forward(tensor_input, packed_seq_params=packed_seq_params)


def get_seqlen_shard(cu_seqlens_q: torch.Tensor, cu_seqlens_kv: torch.Tensor,
                     max_seqlen_q: torch.Tensor, max_seqlen_kv: torch.Tensor, num_local_seqs_recv: torch.Tensor, rank: int):
    num_seq = num_local_seqs_recv[rank]
    max_seqlen_q = max_seqlen_q[rank]
    max_seqlen_kv = max_seqlen_kv[rank]
    cu_seqlens_q = cu_seqlens_q[rank][:num_seq]
    cu_seqlens_kv = cu_seqlens_kv[rank][:num_seq]
    return cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, num_seq


@torch.no_grad()
def test(worker: Worker, seed, rank,
         world_size, num_tokens, max_cp_degree, num_seqs,
         hidden_size):
    # Create two splits for ping-pong
    (
        fwd_q_metadata_0, rev_q_metadata_0,
        fwd_kv_metadata_0, rev_kv_metadata_0,
        sp_kv_dst_0, sp_seq_lens_0, seq_lens_0,
    ) = create_testcase_qkv(seed, world_size, num_tokens, max_cp_degree, num_seqs)
    (
        fwd_q_metadata_1, rev_q_metadata_1,
        fwd_kv_metadata_1, rev_kv_metadata_1,
        sp_kv_dst_1, sp_seq_lens_1, seq_lens_1,
    ) = create_testcase_qkv(seed, world_size, num_tokens, max_cp_degree, num_seqs)

    # Create tensor input
    tensor_input = torch.randn(world_size, num_tokens * 2, hidden_size, dtype=torch.float16)
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
    (cu_seqlens_q_0, cu_seqlens_kv_0, max_seqlen_q_0, max_seqlen_kv_0) = get_seqlen_shard(*compute_attn_layout_seqlens(
        seq_lens_0, sp_seq_lens_0, sp_kv_dst_0
    ), rank)
    (cu_seqlens_q_1, cu_seqlens_kv_1, max_seqlen_q_1, max_seqlen_kv_1) = get_seqlen_shard(*compute_attn_layout_seqlens(
        seq_lens_1, sp_seq_lens_1, sp_kv_dst_1
    ), rank)
    packed_seq_params_stage_0 = PingPangSingleStepPackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_q_0,
        cu_seqlens_kv=cu_seqlens_kv_0,
        max_seqlen_q=max_seqlen_q_0,
        max_seqlen_kv=max_seqlen_kv_0,
        mlp_to_attn_metadata=fwd_q_metadata_0_local.normalize_dtype().cuda(),
        attn_to_mlp_metadata=rev_q_metadata_0_local.normalize_dtype().cuda(),
        mlp_to_attn_kv_metadata=fwd_kv_metadata_0_local.normalize_dtype().cuda(),
        mlp_to_attn_kv_grad_metadata=rev_kv_metadata_0_local.normalize_dtype().cuda(),
    )
    packed_seq_params_stage_1 = PingPangSingleStepPackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_q_1,
        cu_seqlens_kv=cu_seqlens_kv_1,
        max_seqlen_q=max_seqlen_q_1,
        max_seqlen_kv=max_seqlen_kv_1,
        mlp_to_attn_metadata=fwd_q_metadata_1_local.normalize_dtype().cuda(),
        attn_to_mlp_metadata=rev_q_metadata_1_local.normalize_dtype().cuda(),
        mlp_to_attn_kv_metadata=fwd_kv_metadata_1_local.normalize_dtype().cuda(),
        mlp_to_attn_kv_grad_metadata=rev_kv_metadata_1_local.normalize_dtype().cuda(),
    )
    packed_seq_params_ping_pang = PingPangPackedSeqParams(
        debug=True,
        seq_params=[packed_seq_params_stage_0, packed_seq_params_stage_1],
    )

    # Compute the reference answer and test result
    ref_ans = worker.forward_normal(
        tensor_input_local, packed_seq_params_normal
    )

    ans = worker.forward_ping_pang(
        tensor_input_local, packed_seq_params_ping_pang
    ) 
    torch.testing.assert_close(ref_ans, ans)


def init_test(args):
    spec = get_gpt_layer_with_transformer_engine_spec()
    config = get_gpt_config(
        num_layers=1,
        hidden_size=args.hidden_size,
        num_attention_heads=1,
        ffn_hidden_size=args.hidden_size * 4,
    )
    num_tokens = args.num_tokens
    hidden_size = args.hidden_size
    seed = args.seed

    torch.distributed.init_process_group(
        backend="cpu:gloo",
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    if rank == 0:
        uid = nvshmem_get_unique_id()
    else:
        uid = nvshmem_alloc_empty_unique_id()
    torch.distributed.broadcast(uid, src=0)
    torch.cuda.set_device(local_rank)
    worker = Worker(rank, world_size)

    world_size = torch.distributed.get_world_size()
    stride = hidden_size * torch.float16.itemsize
    max_tokens_query = num_tokens * world_size
    max_tokens_key_value = num_tokens * world_size
    worker.init_comm(uid, stride, max_tokens_query, max_tokens_key_value)
    worker.init_layer(config, spec, seed)
    return worker


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--cp-degree", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-seqs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    worker = init_test(args)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    test(
        worker, args.seed, rank, world_size, args.num_tokens, args.cp_degree, args.num_seqs,
        args.hidden_size, args.num_heads
    )
