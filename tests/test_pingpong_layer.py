"""
Profiling:

NVSHMEM_IB_ENABLE_IBGDA=true NVSHMEM_DEBUG=DEBUG NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
    nsys profile -o pingpang_layer_%p.nsys-rep -t cuda,nvtx \
torchrun --nnodes 1 --nproc_per_node 2 test_pingpong_layer.py \
    --world-size 2 \
    --profile \
    --num-query-heads 8 --num-heads 32 --hidden-size 4096 --num-tokens 8192

Correctness:

DP:
NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
torchrun --nnodes 1 --nproc_per_node 2 test_pingpong_layer.py \
    --world-size 2 \
    --num-query-heads 8 --num-heads 32 --hidden-size 4096 --num-tokens 8192
DP+TP:
NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
torchrun --nnodes 1 --nproc_per_node 4 test_pingpong_layer.py \
    --world-size 4 --tp-size 2 \
    --num-query-heads 8 --num-heads 32 --hidden-size 4096 --num-tokens 8192
"""

import argparse

import torch

from d2.runtime.megatron_patch.packed_seq_params import PingPangPackedSeqParams, PingPangSingleStepPackedSeqParams
from d2.runtime.compute_metadata import from_planner_output, get_attn_metadata

from test_util import random_shard_info_linear_layout_dp
from test_megatron_layer import MegatronLayerWorker, init_megatron_test


class PingPangLayerWorker(MegatronLayerWorker):
    def __init__(self, rank: int, world_size: int):
        super().__init__(rank, world_size)
        # get a higher priority than the torch default stream
        self.stream = None

    def init_torch_distributed(self):
        # NOTE: worker should only create stream after knowing which device
        # it is on.
        super().init_torch_distributed()
        self.stream = torch.cuda.Stream(device=self.device, priority=-1)

    def forward_ping_pang(self, tensor_input: torch.Tensor, packed_seq_params: PingPangPackedSeqParams,
                          run_backward: bool = False):
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
        torch.distributed.barrier()

        if run_backward:
            tensor_input.requires_grad = True
            ctx = torch.enable_grad()
        else:
            ctx = torch.no_grad()
        with ctx:
            ret = self.layer.ping_pang_forward(tensor_input, packed_seq_params=packed_seq_params)
            if not run_backward:
                return ret
            output, context = ret
            loss = (output**2).mean()
            loss.backward()
            self.layer.zero_grad()


def create_one_batch(
    seed: int, world_size: int, total_token_per_rank: int, num_docs: int,
    max_cp_degree: int, hidden_size_q: int, hidden_size_k: int, element_size: int
):
    planner_output, doc_lens_per_rank = random_shard_info_linear_layout_dp(
        world_size, num_docs, tot_num_token=total_token_per_rank,
        max_num_shard=max_cp_degree, seed=seed,
    )
    doc_lens_per_rank = [
        torch.tensor(val, dtype=torch.int32, device='cuda') for val in doc_lens_per_rank
    ]

    lse_size = 0
    (qkv_fwd_fa2a_metadata, qkv_rev_fa2a_metadata,
     attn_out_fwd_fa2a_metadata, attn_out_rev_fa2a_metadata,
     as_attn_metadata,
    ) = from_planner_output(
        world_size, planner_output, hidden_size_q, hidden_size_k,
        lse_size, element_size, is_pipeline_tick=False
    )
    fa2a_metadata = (
        qkv_fwd_fa2a_metadata, qkv_rev_fa2a_metadata,
        attn_out_fwd_fa2a_metadata, attn_out_rev_fa2a_metadata,
    )
    return planner_output, fa2a_metadata, as_attn_metadata, doc_lens_per_rank


# TODO(yonghao): move this to planner / d2/utils.py because it's not only used for test.
def get_single_step_packed_seq_params(
    fa2a_metadata, attn_metadata, rank: int
):
    (
        qkv_fwd_fa2a_metadata, qkv_rev_fa2a_metadata,
        attn_out_fwd_fa2a_metadata, attn_out_rev_fa2a_metadata,
    ) = fa2a_metadata
    ping_pang_params = PingPangSingleStepPackedSeqParams(
        qkv_format="thd",
        **attn_metadata[rank],
        qkv_fwd_metadata=qkv_fwd_fa2a_metadata.get_slice(rank),
        qkv_bwd_metadata=qkv_rev_fa2a_metadata.get_slice(rank),
        attn_out_fwd_metadata=attn_out_fwd_fa2a_metadata.get_slice(rank),
        attn_out_bwd_metadata=attn_out_rev_fa2a_metadata.get_slice(rank),
    )
    return ping_pang_params


@torch.no_grad()
def test_forward(
    seed, total_token_per_rank, num_docs, max_cp_degree,
    worker: PingPangLayerWorker, hidden_size_q: int, hidden_size_k: int,
    debug=False, profile=False, tp_size: int = 1
):
    torch.manual_seed(seed)
    dtype = torch.float16
    element_size = dtype.itemsize
    as_world_size = worker.as_world_size
    as_rank = worker.as_rank

    hidden_size_q_tp = hidden_size_q // tp_size
    hidden_size_k_tp = hidden_size_k // tp_size

    # Create two splits for ping-pong
    _, fa2a_metadata_0, attn_metadata_0, doc_lens_0 = create_one_batch(
        seed, as_world_size, total_token_per_rank, num_docs, max_cp_degree,
        hidden_size_q_tp, hidden_size_k_tp, element_size
    )
    _, fa2a_metadata_1, attn_metadata_1, doc_lens_1 = create_one_batch(
        seed + 1, as_world_size, total_token_per_rank, num_docs, max_cp_degree,
        hidden_size_q_tp, hidden_size_k_tp, element_size
    )

    # Create tensor input
    torch.manual_seed(seed)
    # NOTE: this input is not sharded.
    tensors = torch.randn(
        (as_world_size, total_token_per_rank * 2, 1, hidden_size_q), dtype=dtype
    )
    as_rank = worker.as_rank
    tensor_shard = tensors[as_rank]

    # packed seq params for orig impl
    doc_lens_local = torch.concat(
        (doc_lens_0[as_rank], doc_lens_1[as_rank])
    )
    packed_seq_params = get_attn_metadata(
        doc_lens_local, get_packed_seq_params=True,
    )
    # orig forward
    normal_forward_out, debug_ref = worker.forward_normal(
        tensor_shard, packed_seq_params
    )

    ping_pang_params_0 = get_single_step_packed_seq_params(
        fa2a_metadata_0, attn_metadata_0, as_rank
    )
    ping_pang_params_1 = get_single_step_packed_seq_params(
        fa2a_metadata_1, attn_metadata_1, as_rank
    )
    print(f"{worker.rank=}, Attention Server Rank {as_rank}, pingpong 0 send bytes:{ping_pang_params_0.qkv_fwd_metadata.fa2a_metadata[1]}, pingpong 1 send bytes:{ping_pang_params_1.qkv_fwd_metadata.fa2a_metadata[1]}")
    mlp_layout_seq_params = get_attn_metadata(
        (doc_lens_0[as_rank], doc_lens_1[as_rank]),
        get_packed_seq_params=True
    )
    max_seqlens = max(psp.max_seqlen_q for psp in mlp_layout_seq_params)
    ping_pang_params = PingPangPackedSeqParams(
        seq_params=[ping_pang_params_0, ping_pang_params_1],
        mlp_layout_seq_params=list(mlp_layout_seq_params),
        max_seqlen_q=max_seqlens,
        max_seqlen_kv=max_seqlens,
        qkv_format="thd",
        do_gather=True,
        debug=debug,
    )
    ans = worker.forward_ping_pang(
        tensor_shard, ping_pang_params
    )
    torch.testing.assert_close(ans, normal_forward_out)
    print(f"Rank {as_rank} forward ping-pang passed.")
    if profile:
        for _ in range(3):
            worker.forward_ping_pang(
                tensor_shard, ping_pang_params,
                run_backward=True
            )
        torch.cuda.synchronize()
        torch.distributed.barrier()
        print("warmup done")
        for _ in range(20):
            worker.forward_ping_pang(
                tensor_shard, ping_pang_params,
                run_backward=True
            )
        torch.cuda.synchronize()
        torch.distributed.barrier()
        print("ping-pang forward done")
    torch.cuda.synchronize()
    torch.distributed.barrier()


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
        args.seed, args.num_tokens, args.num_docs, max_cp_degree,
        worker, hidden_size, hidden_size_kv,
        profile=args.profile, tp_size=tp_size,
    )


if __name__ == "__main__":
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
