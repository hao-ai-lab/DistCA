import json
import os
import sys
import random
from itertools import accumulate
import argparse
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rich import print
from pathlib import Path

parser = argparse.ArgumentParser(description="per-sequence CP test arguments")
parser.add_argument("--context_length", type=int,  default=128)   # ×1024
parser.add_argument("--batch_size",    type=int,  default=1)
parser.add_argument("--num_heads",     type=int,  default=32)
parser.add_argument("--head_dim",      type=int,  default=128)
parser.add_argument("--avg_doc_len",   type=float,default=0.5) 
parser.add_argument("--std_doc_len",   type=float,default=0.5)
parser.add_argument("--cp_size",       type=int,  default=4)
parser.add_argument("--fix_seed",      type=int,  default=1)
parser.add_argument("--doc_lens",      type=str,  default=None)
parser.add_argument("--output_path",   type=str,  default=None)

default_output_path = Path(__file__).parent.parent / "results" / "docshard_result.jsonl"



from flash_attn.flash_attn_interface import (
    flash_attn_varlen_func, 
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)

from utils import(
    compute_per_seq_metadate_combined,
    compute_per_doc_metadate_combined,
    compute_per_doc_cp_shard_doc_len,
    per_seq_correctness_evaluate,
    per_doc_correctness_evaluate,
    generate_doc_lens
)

from per_seq_cp_attn import PerSequenceCPAttention
from per_doc_cp_attn import PerDocumentCPAttention


def print_on_main(rank, content):
    if rank == 0:
        print(content)

def random_tensor_generation(batch_size, context_length, num_heads, head_dim, device):
    q_global = torch.randn(batch_size * context_length, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_global = torch.randn(batch_size * context_length, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_global = torch.randn(batch_size * context_length, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    d_out_global = torch.randn(batch_size * context_length, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    q_global.requires_grad_(True)
    k_global.requires_grad_(True)
    v_global.requires_grad_(True)
    return q_global, k_global, v_global, d_out_global

# distributed run
def run(rank: int, world_size: int, args):
    # nccl
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    cp_size = world_size
    cp_group = dist.group.WORLD
    dist.barrier(device_ids=[rank])
    print_on_main(rank, "CP group initialization finished")

    # args
    batch_size      = args.batch_size              # 1
    num_heads       = args.num_heads
    head_dim        = args.head_dim
    context_length  = args.context_length * 1024   # tokens
    softmax_scale   = head_dim ** -0.5
    doc_lens        = args.doc_lens
    device = torch.device("cuda", rank)
    if args.fix_seed:
        random.seed(42)
        torch.manual_seed(42)
    n_warmup = 10
    n_iter = 20

    # ======= Generate random input sequence consists of multiple docs =======
    if doc_lens is None:
        if rank == 0:
            doc_lens = generate_doc_lens(args.avg_doc_len, args.std_doc_len, context_length)
            doc_lens_tensor = torch.tensor(doc_lens, dtype=torch.int32, device=torch.device(rank))
            n_doc_tensor = torch.tensor([len(doc_lens)], dtype=torch.int32, device=device)
        else:
            n_doc_tensor = torch.empty(1, dtype=torch.int32, device=device)
        dist.broadcast(n_doc_tensor, src=0, group=cp_group)

        if rank != 0:
            doc_lens_tensor = torch.empty(n_doc_tensor[0].item(), dtype=torch.int32, device=device)
        dist.broadcast(doc_lens_tensor, src=0, group=cp_group)
        doc_lens = doc_lens_tensor.tolist()
    else:
        doc_lens = eval(doc_lens)
        context_length = sum(doc_lens)
    
    print_on_main(rank, f"doc_lens: {doc_lens}, context_length: {context_length}")

    dist.barrier(device_ids=[rank])
    print_on_main(rank, "Random input generation finished")
    print_on_main(rank, f"Generated document lengths: {doc_lens}")

    # ======= Profile Per-Seq Latency =======
    print_on_main(rank, "Start profiling per-seq CP latency")
    q_global, k_global, v_global, d_out_global = random_tensor_generation(batch_size, context_length, num_heads, head_dim, device)
    local_q, local_k, local_v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, k_offsets, local_d_out = compute_per_seq_metadate_combined(
        context_length, 
        q_global, 
        k_global, 
        v_global, 
        doc_lens, 
        cp_size, 
        rank, 
        d_out=d_out_global
    )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # warmup:
    for _ in range(n_warmup):
        out = PerSequenceCPAttention.apply(
            local_q, local_k, local_v,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            k_offsets, 
            0.0, # dropout_p
            softmax_scale, 
            "causal",
            cp_group,
            torch.cuda.current_stream(device) 
        )
        out.backward(local_d_out)
    
    iter_range = tqdm(range(n_iter)) if rank == 0 else range(n_iter)
    start.record()
    for _ in iter_range:
        out = PerSequenceCPAttention.apply(
            local_q, local_k, local_v,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            k_offsets, 
            0.0, # dropout_p
            softmax_scale, 
            "causal",
            cp_group,
            torch.cuda.current_stream(device) 
        )
        out.backward(local_d_out)
    end.record()
    torch.cuda.synchronize()
    per_seq_latency = start.elapsed_time(end) / n_iter

    # ======= Profile Per-Doc Latency =======
    print_on_main(rank, "Start profiling per-doc CP latency")
    q_global, k_global, v_global, d_out_global = random_tensor_generation(batch_size, context_length, num_heads, head_dim, device)
    doc_shards = compute_per_doc_cp_shard_doc_len(doc_lens, context_length, cp_size)
    local_q_doc, local_k_doc, local_v_doc, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, kv_idx_list, local_d_out = compute_per_doc_metadate_combined(    
        context_length, 
        q_global, 
        k_global, 
        v_global, 
        doc_lens, 
        doc_shards,
        cp_size, 
        rank, 
        d_out=d_out_global
    )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # warmup:
    for _ in range(n_warmup):
        out = PerDocumentCPAttention.apply(
            local_q_doc, local_k_doc, local_v_doc,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            doc_lens, doc_shards, kv_idx_list, 
            0.0, # dropout_p
            softmax_scale, 
            "causal",
            cp_group,
            torch.cuda.current_stream(device) 
        )
        out.backward(local_d_out)
    
    iter_range = tqdm(range(n_iter)) if rank == 0 else range(n_iter)
    start.record()
    for _ in iter_range:
        out = PerDocumentCPAttention.apply(
            local_q_doc, local_k_doc, local_v_doc,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            doc_lens, doc_shards, kv_idx_list, 
            0.0, # dropout_p
            softmax_scale, 
            "causal",
            cp_group,
            torch.cuda.current_stream(device) 
        )
        out.backward(local_d_out)

    end.record()
    torch.cuda.synchronize()
    per_doc_latency = start.elapsed_time(end) / n_iter

    speedup = per_seq_latency / per_doc_latency
    print("rank:{}, per_seq_latency:{:.3f}ms, per_doc_latency:{:.3f}ms, speedup:{:.3f}x".format(rank, per_seq_latency, per_doc_latency, speedup))
    dist.barrier(device_ids=[rank])
    
    # Convert float latencies to tensors
    per_seq_latency_tensor = torch.tensor([per_seq_latency], dtype=torch.float32, device=device)
    per_doc_latency_tensor = torch.tensor([per_doc_latency], dtype=torch.float32, device=device)

    if rank == 0:
        # Only create gather lists on rank 0
        per_seq_latency_list = [torch.zeros(1, dtype=torch.float32, device=device) for _ in range(world_size)]
        per_doc_latency_list = [torch.zeros(1, dtype=torch.float32, device=device) for _ in range(world_size)]
        dist.gather(per_seq_latency_tensor, per_seq_latency_list, dst=0)
        dist.gather(per_doc_latency_tensor, per_doc_latency_list, dst=0)
    else:
        # On other ranks, don't specify gather_list
        dist.gather(per_seq_latency_tensor, dst=0)
        dist.gather(per_doc_latency_tensor, dst=0)

    dist.destroy_process_group()

    if rank == 0:
        per_seq_latency_list = [
            per_seq_latency_list[i].tolist() for i in range(world_size)
        ]
        per_doc_latency_list = [
            per_doc_latency_list[i].tolist() for i in range(world_size)
        ]

        item = dict(
            per_seq_latency=per_seq_latency_list,
            per_doc_latency=per_doc_latency_list,
            speedup=speedup,
            context_length=context_length,
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            cp_size=cp_size,
            doc_lens=doc_lens,
        )
        
        with open(args.output_path, "a") as f:
            f.write(json.dumps(item) + "\n")

        print_on_main(rank, item)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    if args.output_path is None:
        args.output_path = str(default_output_path)

    world_size = args.cp_size
    mp.spawn(
        run,
        nprocs=world_size,
        args=(world_size, args),
        join=True,
    )


"""
python main_docshard.py --batch_size 1 --num_heads 32 --head_dim 128 --cp_size 4 --doc_lens '[i * 1024 for i in [16] + [1] * 16 ]'
python main_docshard.py --batch_size 1 --num_heads 32 --head_dim 128 --cp_size 4 --doc_lens '[i * 1024 for i in [2] * 16 ]'
"""