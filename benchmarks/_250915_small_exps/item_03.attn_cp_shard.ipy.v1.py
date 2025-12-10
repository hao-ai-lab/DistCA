# %%

from flash_attn.flash_attn_interface import (
    flash_attn_varlen_func
)
import torch
import time

def get_flops(batch, nheads, seqlen_q, cp_degree, headdim, causal=False, mode='fwd'):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen_q ** 2 // cp_degree * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

# cp_shard_lens = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
# cp_shard_lens = [16384, 32768]
cp_shard_lens = [262144, 131072, 65536, 32768, 16384,  ]
total_seq_len = max(cp_shard_lens)
batch_size = 8

for cp_shard_len in cp_shard_lens:
    num_seq = total_seq_len // cp_shard_len * batch_size

    L = total_seq_len
    q_lens = [cp_shard_len] * (num_seq)
    if num_seq // 2 > 0:
        kv_lens = [L // 2, L // 2 + cp_shard_len] * (num_seq // 2)
    else:
        kv_lens = [L]

    print(f"cp_shard_len: {cp_shard_len}, num_seq: {num_seq}")
    print(f"q_lens: {q_lens}, kv_lens: {kv_lens}, L: {L}")
    # continue


    num_heads = 32
    head_dim = 128

    torch.cuda.set_device(0)

    q = torch.ones(sum(q_lens), num_heads, head_dim, dtype=torch.bfloat16).cuda()
    k = torch.ones(sum(kv_lens), num_heads, head_dim, dtype=torch.bfloat16).cuda()
    v = torch.ones(sum(kv_lens), num_heads, head_dim, dtype=torch.bfloat16).cuda()

    cu_seqlens_q = torch.cumsum(torch.tensor(q_lens), dim=0, dtype=torch.int32).cuda()
    cu_seqlens_k = torch.cumsum(torch.tensor(kv_lens), dim=0, dtype=torch.int32).cuda()
    max_seqlen_q = torch.max(torch.tensor(q_lens)).item()
    max_seqlen_k = torch.max(torch.tensor(kv_lens)).item()


    print(f"cu_seqlens_q: {cu_seqlens_q}")
    print(f"cu_seqlens_k: {cu_seqlens_k}")
    print(f"max_seqlen_q: {max_seqlen_q}")
    print(f"max_seqlen_k: {max_seqlen_k}")


    N_warmup = 10
    N_iters = 50
    
    with torch.cuda.nvtx.range(f"cp_shard={cp_shard_len}, num_seq={num_seq}"):
        torch.cuda.synchronize(); start_time = time.time()
        for _ in range(N_iters + N_warmup):
            out = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True)
        torch.cuda.synchronize(); end_time = time.time()
        duration = end_time - start_time

    cp_degree = total_seq_len // cp_shard_len
    flops = get_flops(num_seq, num_heads, max_seqlen_q, cp_degree, head_dim, causal=True, mode='fwd')

    avg_duration = duration / N_iters
    avg_duration_ms = avg_duration * 1000
    print(out.shape)

    peak_tflops = 989 # TFLOPS
    achieved_tflops = (flops / (avg_duration_ms / 1e3)) / 1e12  # TFLOPs/s
    flops_utilization = achieved_tflops / peak_tflops

    print(f"cp_shard_len: {cp_shard_len}, num_seq: {num_seq}, L: {L}, num_heads: {num_heads}, head_dim: {head_dim}, causal: True, mode: fwd, duration_ms: {avg_duration_ms}, achieved_tflops: {achieved_tflops}, flops_utilization: {flops_utilization}")

# %%

"""
nsys profile -t nvtx,cuda -o cp_shard.nsys python item_03.attn_cp_shard.ipy.py
"""

# %%
"""
3. Attention Divisibility (Priority: High. EST: 3 hr?)
   1. With xx K total tokens (32K? 64K? we don't need to make this number very high), different combination of CP shards have the same throughput
   2. x axis: size of each CP shards (KV context size is randomly sampled); y axis: throughput(FLOPs per GPU)

(Template) Text and graph

To verify this, we profile the Core Attention MFU for document shards with different lengths and context sizes. The Core Attention is computed by FlashAttention 2's latest stable release, and for each shard length, we batch requests to \todo{} tokens to avoid GPU SMs not fully utilized. 
The result is shown in \todo{fig}. When the shard length is above 128 tokens, the MFU is almost stable.  This is because FlashAttention tunes the its tile size to 128 tokens.  Shards fewer than that unit are padded to 128 tokens, and thus waste some of their thread blocks' compute. 

"""