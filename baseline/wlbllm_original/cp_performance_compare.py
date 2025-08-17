import os
import sys
import random
from itertools import accumulate
import argparse
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

parser = argparse.ArgumentParser(description="per-sequence CP test arguments")
parser.add_argument("--context_length", type=int,  default=128)   # ×1024
parser.add_argument("--batch_size",    type=int,  default=1)
parser.add_argument("--num_heads",     type=int,  default=32)
parser.add_argument("--head_dim",      type=int,  default=128)
parser.add_argument("--avg_doc_len",   type=float,default=0.5) 
parser.add_argument("--std_doc_len",   type=float,default=0.5)
parser.add_argument("--cp_size",       type=int,  default=4)
parser.add_argument("--fix_seed",      type=int,  default=1)
parser.add_argument("--include_backward",   type=bool,  default=True)
parser.add_argument("--gqa",           type=int,  default=1)

from attn_module import (
    flash_attn_varlen_func, 
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)

from utils import(
    compute_per_doc_cp_shard_doc_len,
    generate_doc_lens
)

from per_seq_cp_attn import PerSequenceCPAttention
from per_doc_cp_attn import PerDocumentCPAttention


def print_on_main(rank, content):
    if rank == 0:
        print(content)

def random_tensor_generation(batch_size, context_length, num_heads, head_dim, device, gqa=1):
    num_kv_heads = max(1, num_heads // gqa)
    q_global = torch.randn(batch_size * context_length, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_global = torch.randn(batch_size * context_length, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_global = torch.randn(batch_size * context_length, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    d_out_global = torch.randn(batch_size * context_length, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    q_global.requires_grad_(True)
    k_global.requires_grad_(True)
    v_global.requires_grad_(True)
    return q_global, k_global, v_global, d_out_global


def compute_per_seq_metadate_chunk(context_length, q_tensor, k_tensor, v_tensor, doc_lens, cp_size, rank, d_out=None):
    """
    Compute the cumulative sequence lengths for per-sequence CP.
    """
    # ============== Split doc lens for sequence sharding =================
    chunk_size = context_length // (2 * cp_size)

    split_doc_lens = []
    prefix_lens = []
    cur_length = 0
    for i, doc_len in enumerate(doc_lens):
        if cur_length + doc_len <= chunk_size: 
            split_doc_lens.append(doc_len)
            prefix_lens.append(0)
            cur_length += doc_len
        else: # split the document
            split_doc_lens.append(chunk_size - cur_length)
            prefix_lens.append(0)
            cu_prefix = chunk_size - cur_length
            remained_length = doc_len - (chunk_size - cur_length)
            while remained_length > chunk_size:
                split_doc_lens.append(chunk_size)
                prefix_lens.append(cu_prefix)
                cu_prefix += chunk_size
                remained_length -= chunk_size
            if remained_length > 0:
                split_doc_lens.append(remained_length)
                prefix_lens.append(cu_prefix)
                cur_length = remained_length
            else:
                cur_length = 0
        
        if cur_length == chunk_size:
            cur_length = 0
    assert sum(split_doc_lens) == context_length, f"Total length {sum(split_doc_lens)} must equals context length {context_length}."
    
    cur_offset = 0
    doc_idx_list = [0] # to record the document index for each chunk
    for i, doc_len in enumerate(split_doc_lens):
        cur_length += doc_len
        if cur_length == chunk_size:
            doc_idx_list.append(i + 1)
            cur_length = 0
        elif cur_length > chunk_size:
            assert False, "cur_length > chunk_size, this should not happen."
        
    for i in range(len(doc_idx_list)-1):
        assert sum(split_doc_lens[doc_idx_list[i]:doc_idx_list[i+1]]) == chunk_size, f"error doc per chunk"
    
    # ============== Compute metadata =================
    cu_seqlens_q_list = []
    max_seqlen_q_list = []
    cu_seqlens_k_list = []
    max_seqlen_k_list = []
    k_offset_list = []
    for chunk_id in range(2):
        if chunk_id == 0:
            chunk_index = rank
        else:
            chunk_index = 2 * cp_size - 1 - rank
    
        this_chunk_docs = split_doc_lens[doc_idx_list[chunk_index]:doc_idx_list[chunk_index+1]]
        k_offset = chunk_index * chunk_size
        doc_id_split = doc_idx_list[chunk_index]

        cu_seqlens_q_list.append(torch.tensor([0] + list(accumulate(this_chunk_docs)), dtype=torch.int32).to(q_tensor.device))
        max_seqlen_q_list.append(torch.tensor([max(this_chunk_docs)], dtype=torch.int32).to(q_tensor.device))

        # check if the first doc is splitted
        if prefix_lens[doc_id_split] > 0:
            k_offset -= prefix_lens[doc_id_split]
            this_chunk_docs[0] += prefix_lens[doc_id_split]
            assert k_offset >= 0, f"error k_offset {k_offset} < 0"

        cu_seqlens_k_list.append(torch.tensor([0] + list(accumulate(this_chunk_docs)), dtype=torch.int32).to(q_tensor.device))
        max_seqlen_k_list.append(torch.tensor([max(this_chunk_docs)], dtype=torch.int32).to(q_tensor.device))
        k_offset_list.append(k_offset)

    local_q = q_tensor.chunk(cp_size, dim=0)[rank]
    local_k = k_tensor.chunk(cp_size, dim=0)[rank]
    local_v = v_tensor.chunk(cp_size, dim=0)[rank]
    local_q.requires_grad_(True)
    local_k.requires_grad_(True)
    local_v.requires_grad_(True)
    if d_out is not None:
        local_d_out = d_out.chunk(cp_size, dim=0)[rank]
        local_d_out.requires_grad_(True)
    else:
        local_d_out = None

    return local_q, local_k, local_v, cu_seqlens_q_list, cu_seqlens_k_list, max_seqlen_q_list, max_seqlen_k_list, k_offset_list, local_d_out


def compute_per_doc_metadate_chunk(context_length, q, k, v, doc_lens, doc_shards, cp_size, rank, d_out=None):
    """
    Compute the metadata (e.g., cumulative sequence lengths) for per-document CP.
    """
    # ============== Compute metadata =================
    chunk_size = context_length // (2 * cp_size)
    global_cu_lens =  [0] + list(accumulate(doc_lens))

    cu_seqlens_q_list = []
    max_seqlen_q_list = []
    cu_seqlens_k_list = []
    max_seqlen_k_list = []
    kv_idx_list = []
    for chunk_id in range(2):
        if chunk_id == 0:
            chunk_index = rank
        else:
            chunk_index = 2 * cp_size - 1 - rank

        this_doc_shards = doc_shards[chunk_index]
        this_chunk_docs = []

        kv_len_list = []
        kv_idx = []

        for doc_shard_i in this_doc_shards:
            if doc_shard_i is None:
                continue
            else:
                this_chunk_docs.append(doc_shard_i.shard_len)

                k_chunk_start = global_cu_lens[doc_shard_i.doc_id]
                k_chunk_end = k_chunk_start + doc_shard_i.prefix_len + doc_shard_i.shard_len
                kv_idx.append((k_chunk_start, k_chunk_end))
                kv_len_list.append(doc_shard_i.prefix_len + doc_shard_i.shard_len)
    
        assert sum(this_chunk_docs) == chunk_size, f"Total length {sum(this_chunk_docs)} must equals chunk_size {chunk_size}."

        cu_seqlens_q_list.append(torch.tensor([0] + list(accumulate(this_chunk_docs)), dtype=torch.int32).to(q.device))
        max_seqlen_q_list.append(torch.tensor([max(this_chunk_docs)], dtype=torch.int32).to(q.device))
        cu_seqlens_k_list.append(torch.tensor([0] + list(accumulate(kv_len_list)), dtype=torch.int32).to(q.device))
        max_seqlen_k_list.append(torch.tensor([max(kv_len_list)], dtype=torch.int32).to(q.device))
        kv_idx_list.append(kv_idx)

    local_q = q.chunk(cp_size, dim=0)[rank]
    local_k = k.chunk(cp_size, dim=0)[rank]
    local_v = v.chunk(cp_size, dim=0)[rank]
    local_q.requires_grad_(True)
    local_k.requires_grad_(True)
    local_v.requires_grad_(True)
    if d_out is not None:
        local_d_out = d_out.chunk(cp_size, dim=0)[rank]
        local_d_out.requires_grad_(True)
    else:
        local_d_out = None

    return local_q, local_k, local_v, cu_seqlens_q_list, cu_seqlens_k_list, max_seqlen_q_list, max_seqlen_k_list, kv_idx_list, local_d_out


# distributed run
def run(rank: int, world_size: int, args, return_dict):
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
    device = torch.device("cuda", rank)
    if args.fix_seed:
        random.seed(42)
        torch.manual_seed(42)
    n_warmup = 10
    n_iter = 20

    # ======= Generate random input sequence consists of multiple docs =======
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

    dist.barrier(device_ids=[rank])
    print_on_main(rank, "Random input generation finished")
    print_on_main(rank, f"Generated document lengths: {doc_lens}")

    # ======= Profile Per-Seq Latency =======
    print_on_main(rank, "Start profiling per-seq CP latency")
    q_global, k_global, v_global, d_out_global = random_tensor_generation(
        batch_size, context_length, num_heads, head_dim, device,
        gqa=args.gqa
    )
    local_q, local_k, local_v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, k_offsets, local_d_out = compute_per_seq_metadate_chunk(
        context_length, 
        q_global, 
        k_global, 
        v_global, 
        doc_lens, 
        cp_size, 
        rank, 
        d_out=d_out_global
    )


    # Initialize events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    perseq_allgather_events = []
    perseq_allreduce_events = []
    perseq_attn_events = []
    for _ in range(n_iter):
        perseq_allgather_events.append([torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)])
        perseq_allreduce_events.append([torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)])
        perseq_attn_events.append([torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)])
    
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
            torch.cuda.current_stream(device),
            None,
            None,
            None,
        )
        if args.include_backward:
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
            torch.cuda.current_stream(device),
            # allgather_events = perseq_allgather_events[_],
            perseq_allgather_events[_],
            # allreduce_events = perseq_allreduce_events[_],
            None,
            # attn_events = perseq_attn_events[_],
            perseq_attn_events[_],
        )
        if args.include_backward:
            out.backward(local_d_out)

    end.record()
    torch.cuda.synchronize()
    per_seq_latency = start.elapsed_time(end) / n_iter
    

    # ======= Profile Per-Doc Latency =======
    print_on_main(rank, "Start profiling per-doc CP latency")
    q_global, k_global, v_global, d_out_global = random_tensor_generation(batch_size, context_length, num_heads, head_dim, device)
    doc_shards = compute_per_doc_cp_shard_doc_len(doc_lens, context_length, cp_size)
    local_q_doc, local_k_doc, local_v_doc, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, kv_idx_list, local_d_out = compute_per_doc_metadate_chunk(    
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
    perdoc_allgather_events = []
    perdoc_allreduce_events = []
    perdoc_attn_events = []
    for _ in range(n_iter):
        perdoc_allgather_events.append([torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)])
        perdoc_allreduce_events.append([torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)])
        perdoc_attn_events.append([torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)])
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
            torch.cuda.current_stream(device),
            None,
            None,
            None,
        )
        # out.backward(local_d_out)
    
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
            torch.cuda.current_stream(device),
            # allgather_events = perdoc_allgather_events[_],
            perdoc_allgather_events[_],
            # allreduce_events = perdoc_allreduce_events[_],
            None,
            # attn_events = perdoc_attn_events[_],
            perdoc_attn_events[_],
        )
        # out.backward(local_d_out)

    end.record()
    torch.cuda.synchronize()
    per_doc_latency = start.elapsed_time(end) / n_iter

    speedup = per_seq_latency / per_doc_latency
    print("rank:{}, per_seq_latency:{:.3f}ms, per_doc_latency:{:.3f}ms, speedup:{:.3f}x".format(rank, per_seq_latency, per_doc_latency, speedup))

    dist.barrier(device_ids=[rank])
    
    # Handle communication latency
    perseq_allgather_times = []
    perseq_attn_times = []
    for allgather_event in perseq_allgather_events:
        allgather_event[0].wait(); allgather_event[1].wait()
        perseq_allgather_times.append(allgather_event[0].elapsed_time(allgather_event[1]))
    for attn_event in perseq_attn_events:
        attn_event[0].wait(); attn_event[1].wait()
        perseq_attn_times.append(attn_event[0].elapsed_time(attn_event[1]))
    
    perdoc_allgather_times = []
    perdoc_attn_times = []
    for allgather_event in perdoc_allgather_events:
        allgather_event[0].wait(); allgather_event[1].wait();
        perdoc_allgather_times.append(allgather_event[0].elapsed_time(allgather_event[1]))
    for attn_event in perdoc_attn_events:
        attn_event[0].wait(); attn_event[1].wait()
        perdoc_attn_times.append(attn_event[0].elapsed_time(attn_event[1]))
    
    dist.destroy_process_group()

    return_dict[rank] = dict(
        per_seq_latency=per_seq_latency, # ms
        per_doc_latency=per_doc_latency, # ms
        perseq_allgather_time=sum(perseq_allgather_times) / n_iter, # ms
        perseq_attn_time=sum(perseq_attn_times) / n_iter, # ms
        perdoc_allgather_time=sum(perdoc_allgather_times) / n_iter, # ms
        perdoc_attn_time=sum(perdoc_attn_times) / n_iter, # ms
        output_shape=list(out.shape),
    )

import torch.multiprocessing as mp

if __name__ == "__main__":
    args = parser.parse_args()
    return_dict = mp.Manager().dict()

    world_size = args.cp_size
    mp.spawn(
        run,
        nprocs=world_size,
        args=(world_size, args, return_dict),
        join=True,
    )