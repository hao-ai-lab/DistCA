import os
import sys
sys.path.append("./playground")

import json
import torch
import torch.nn as nn

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

    # Run experiment
    num_heads = 32
    head_dim = 128
    batch_size = 1
    softmax_scale = head_dim ** -0.5
    device = torch.device("cuda", rank)
    context_lengths = [2 ** i for i in range(10, 20)]
    n_warmup = 3
    n_iter = 5
    
    for context_length in context_lengths:
        # --- Profile Per-Seq Latency ---
        dist.barrier(device_ids=[rank])
        print_on_main(rank, "Start profiling per-seq CP latency")
        doc_lens = [context_length]
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
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        forward_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        forward_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        backward_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        backward_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        per_seq_forward_times = []
        per_seq_backward_times = []

        start.record()
        for i in iter_range:
            forward_start_events[i].record()
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
            forward_end_events[i].record()

            backward_start_events[i].record()
            out.backward(local_d_out)
            backward_end_events[i].record()

        end.record()
        torch.cuda.synchronize()
        per_seq_latency = start.elapsed_time(end) / n_iter
        print_on_main(rank, f"Per-seq CP latency: {per_seq_latency} ms")
        for i in range(n_iter):
            per_seq_forward_times.append(forward_start_events[i].elapsed_time(forward_end_events[i]))
            per_seq_backward_times.append(backward_start_events[i].elapsed_time(backward_end_events[i]))
        per_seq_forward_time = sum(per_seq_forward_times) / n_iter
        per_seq_backward_time = sum(per_seq_backward_times) / n_iter
        # print_on_main(rank, f"Per-seq forward time: {per_seq_forward_time} ms")
        # print_on_main(rank, f"Per-seq backward time: {per_seq_backward_time} ms")
        
        # --- Profile Per-Doc Latency ---
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
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        forward_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        forward_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        backward_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        backward_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        per_doc_forward_times = []
        per_doc_backward_times = []

        start.record()
        for i in iter_range:
            forward_start_events[i].record()
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
            forward_end_events[i].record()

            backward_start_events[i].record()
            out.backward(local_d_out)
            backward_end_events[i].record()
        end.record()
        torch.cuda.synchronize()
        per_doc_latency = start.elapsed_time(end) / n_iter
        print_on_main(rank, f"Per-doc CP latency: {per_doc_latency} ms")
        for i in range(n_iter):
            per_doc_forward_times.append(forward_start_events[i].elapsed_time(forward_end_events[i]))
            per_doc_backward_times.append(backward_start_events[i].elapsed_time(backward_end_events[i]))
        per_doc_forward_time = sum(per_doc_forward_times) / n_iter
        per_doc_backward_time = sum(per_doc_backward_times) / n_iter
        # print_on_main(rank, f"Per-doc forward time: {per_doc_forward_time} ms")
        # print_on_main(rank, f"Per-doc backward time: {per_doc_backward_time} ms")

        if rank == 0:
            # print(f"context_length: {context_length}, per_seq_latency: {per_seq_latency} ms, per_doc_latency: {per_doc_latency} ms, per_seq_forward_time: {per_seq_forward_time} ms, per_seq_backward_time: {per_seq_backward_time} ms, per_doc_forward_time: {per_doc_forward_time} ms, per_doc_backward_time: {per_doc_backward_time} ms")


            item = dict(
                context_length=context_length,
                per_seq_latency=per_seq_latency,
                per_doc_latency=per_doc_latency,
                per_seq_forward_time=per_seq_forward_time,
                per_seq_backward_time=per_seq_backward_time,
                per_doc_forward_time=per_doc_forward_time,
                per_doc_backward_time=per_doc_backward_time,
                cp_degree = world_size,
                num_heads = num_heads,
                head_dim = head_dim,
                batch_size = batch_size,
            )
            print(item)

            with open("attn_profile.jsonl", "a") as f:
                f.write(json.dumps(item) + "\n")
    
    
    

if __name__ == "__main__":
    world_size = 1
    mp.spawn(
        run,
        nprocs=world_size,
        args=(world_size, None),
        join=True,
    )