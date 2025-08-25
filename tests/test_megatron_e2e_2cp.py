"""
# 🟢 Passed
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
nsys profile -o pingpang_layer.nsys-rep -t cuda,nvtx \
torchrun --nnodes 1 --nproc_per_node 2 test_megatron_e2e_2cp.py \
    --num-nodes=1 --num-gpus-per-node=2 --cp-degree=4

# 🟢 Passed
SYNC_ALL=1 \
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
torchrun --nnodes 1 --nproc_per_node 4 test_megatron_e2e_2cp.py \
    --num-nodes=1 --num-gpus-per-node=4 --cp-degree=8

# ⚠️ Stuck
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
torchrun --nnodes 1 --nproc_per_node 4 test_megatron_e2e_2cp.py \
    --num-nodes=1 --num-gpus-per-node=4 --cp-degree=8

# 🟢 Passed
SYNC_ALL=1 \
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
torchrun --nnodes 1 --nproc_per_node 8 test_megatron_e2e_2cp.py \
    --num-nodes=1 --num-gpus-per-node=8 --cp-degree=16

# ⚠️ Stuck
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
torchrun --nnodes 1 --nproc_per_node 8 test_megatron_e2e_2cp.py \
    --num-nodes=1 --num-gpus-per-node=8 --cp-degree=16
"""
import argparse
import rich
import os

import torch
from transformers import AutoConfig

from d2.runtime.megatron_patch.packed_seq_params import PingPangPackedSeqParams
from d2.runtime.inplace_metadata import mlp_layout_packed_params
from d2.runtime.fast_alltoall_metadata import compute_fa2a_metadata_from_logical_metadata

from test_util import create_qkv_dispatch_2cp, set_random_seed
from test_pingpang_layer import get_single_step_packed_seq_params
from test_megatron_e2e import (
    MegatronE2eWorker as MegatronE2eBaseWorker, init_megatron_e2e_test
)

SYNC_ALL = os.environ.get("SYNC_ALL", "0") == "1"


class MegatronE2eWorker(MegatronE2eBaseWorker):
    pass


def create_one_batch_2cp(
    world_size: int, total_seq_len: int, num_seqs: int, max_cp_degree: int,
    hidden_size_q: int, hidden_size_k: int, element_size: int
):
    (
        fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata,
        attention_metadata_attn_layout, intermediates, seq_lens
    ) = create_qkv_dispatch_2cp(
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


def test(args):
    seed = args.seed
    num_tokens = args.num_tokens
    max_cp_degree = args.cp_degree
    num_seqs = args.num_seqs
    tp_size = args.tp_size
    world_size = args.num_nodes * args.num_gpus_per_node
    total_seq_len = args.num_tokens

    dtype = torch.bfloat16
    element_size = dtype.itemsize

    model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    hf_config = AutoConfig.from_pretrained(model_path)
    hidden_size_q = hf_config.hidden_size

    hidden_size_kv = hidden_size_q
    if hasattr(hf_config, "num_key_value_heads"):
        hidden_size_kv = (hidden_size_kv * hf_config.num_key_value_heads //
                          hf_config.num_attention_heads)

    worker: MegatronE2eWorker = init_megatron_e2e_test(
        hidden_size_q, hidden_size_kv, num_tokens,
        world_size, max_cp_degree, tp_size,
        dtype, MegatronE2eWorker
    )
    worker.set_config(dtype=dtype)
    worker.init(model_path, seed=seed)
    # set again to potentially adapt to the ray launch case.
    set_random_seed(seed, set_megatron=False)

    rank = worker.rank

    _, fa2a_metadata_0, attn_metadata_0, raw_seq_lens_0 = create_one_batch_2cp(
        world_size, total_seq_len, num_seqs, max_cp_degree,
        hidden_size_q, hidden_size_kv, element_size
    )
    _, fa2a_metadata_1, attn_metadata_1, raw_seq_lens_1 = create_one_batch_2cp(
        world_size, total_seq_len, num_seqs, max_cp_degree,
        hidden_size_q, hidden_size_kv, element_size
    )

    set_random_seed(seed, set_megatron=False)
    input_ids = torch.randint(0, 100, (world_size, total_seq_len * 2))
    position_ids = torch.arange(total_seq_len).repeat(world_size, 2)
    input_ids_local = input_ids[rank]
    position_ids_local = position_ids[rank]
    ping_pang_params_0 = get_single_step_packed_seq_params(
        fa2a_metadata_0, attn_metadata_0, rank
    )
    ping_pang_params_1 = get_single_step_packed_seq_params(
        fa2a_metadata_1, attn_metadata_1, rank
    )

    # NOTE: we don't consider that seq_lens var has padding because our data generation
    # guarantees so. However, in practice, this is not true.
    mlp_seq_params_0 = mlp_layout_packed_params(raw_seq_lens_0[rank])
    mlp_seq_params_1 = mlp_layout_packed_params(raw_seq_lens_1[rank])
    ping_pang_params = PingPangPackedSeqParams(
        seq_params=[ping_pang_params_0, ping_pang_params_1],
        mlp_layout_seq_params=[mlp_seq_params_0, mlp_seq_params_1],
        max_seqlen_q=torch.tensor([total_seq_len * 2], dtype=torch.int32)[0],
        max_seqlen_kv=torch.tensor([total_seq_len * 2], dtype=torch.int32)[0],
        qkv_format="thd",
    )
    microbatch = {
        "input_ids": input_ids_local,
        "position_ids": position_ids_local,
        "packed_seq_params": ping_pang_params,
    }
    # print(rank, microbatch["packed_seq_params"])
    microbatches = [microbatch]
    import time
    # for _ in range(3):
    for _ in range(1):
        print(f"Rank {rank} forward_backward_batch[{_}]: starting")
        ref = worker.forward_backward_batch(
            microbatches=microbatches,
            normal_forward_fn=False,
            forward_only=False,
        )
        print(f"Rank {rank} forward_backward_batch[{_}]: returned")
        if SYNC_ALL:
            torch.cuda.synchronize()
            print(f"Rank {rank} forward_backward_batch[{_}]: synchronize done")
            torch.distributed.barrier()
            print(f"Rank {rank} forward_backward_batch[{_}]: barrier done")
    time.sleep(1)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    if rank == 0:
        print("=" * 20 + "warmup done")
    for _ in range(1):
        print(f"Rank {rank} forward_backward_batch[{_}]: starting")
        ref = worker.forward_backward_batch(
            microbatches=microbatches,
            normal_forward_fn=False,
            forward_only=False,
        )
        print(f"Rank {rank} forward_backward_batch[{_}]: returned")
        if SYNC_ALL:
            torch.cuda.synchronize()
            print(f"Rank {rank} forward_backward_batch[{_}]: synchronize done")
            torch.distributed.barrier()
            print(f"Rank {rank} forward_backward_batch[{_}]: barrier done")

    torch.cuda.synchronize()
    torch.distributed.barrier()
    print("=" * 20 + "forward_backward_batch attention server, done")

    if rank == 0:
        rich.print(f"🟢 Test {__file__} passed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--num-seqs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-gpus-per-node", type=int, default=2)
    parser.add_argument("--cp-degree", type=int, default=4)
    parser.add_argument("--tp-size", type=int, default=1)
    args = parser.parse_args()
    test(args)
