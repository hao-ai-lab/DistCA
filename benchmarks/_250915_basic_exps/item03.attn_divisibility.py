# %%

from flash_attn.flash_attn_interface import (
    flash_attn_varlen_func
)
import torch
import time

# cp_shard_lens = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
# cp_shard_lens = [16384, 32768]
# cp_shard_lens = [262144, 131072, 65536, 32768, 16384]
# cp_shard_lens = [65536, 32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16]
cp_shard_lens = [16384, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16]
total_seq_len = max(cp_shard_lens)
batch_size = 1

num_heads = 32
head_dim = 128
causal = True
flops = 4 * batch_size * total_seq_len ** 2 * num_heads * head_dim // (2 if causal else 1)
peak_tflops = 989 # TFLOPS

for cp_shard_len in cp_shard_lens:
    num_seq = total_seq_len // cp_shard_len * batch_size

    L = total_seq_len
    if num_seq == 1:
        q_lens = [cp_shard_len]
        kv_lens = [L]
        pass
    else:
        q_lens = [cp_shard_len * 2] * (num_seq // 2)
        kv_lens = [L // 2 + cp_shard_len] * (num_seq // 2)
        pass

    print(f"cp_shard_len: {cp_shard_len}, num_seq: {num_seq}")
    print(f"q_lens: {q_lens}, kv_lens: {kv_lens}, L: {L}")

    torch.cuda.set_device(0)

    q = torch.ones(sum(q_lens), num_heads, head_dim, dtype=torch.bfloat16).cuda()
    k = torch.ones(sum(kv_lens), num_heads, head_dim, dtype=torch.bfloat16).cuda()
    v = torch.ones(sum(kv_lens), num_heads, head_dim, dtype=torch.bfloat16).cuda()

    cu_seqlens_q = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(torch.tensor(q_lens), dim=0, dtype=torch.int32)]).cuda()
    cu_seqlens_k = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(torch.tensor(kv_lens), dim=0, dtype=torch.int32)]).cuda()
    max_seqlen_q = torch.max(torch.tensor(q_lens)).item()
    max_seqlen_k = torch.max(torch.tensor(kv_lens)).item()


    print(f"cu_seqlens_q: {cu_seqlens_q}")
    print(f"cu_seqlens_k: {cu_seqlens_k}")
    print(f"max_seqlen_q: {max_seqlen_q}")
    print(f"max_seqlen_k: {max_seqlen_k}")


    N_warmup = 10
    N_iters = 50
    
    from tqdm import tqdm
    
    with torch.cuda.nvtx.range(f"cp_shard={cp_shard_len}, num_seq={num_seq}"):
        torch.cuda.synchronize(); start_time = time.time()
        for _ in tqdm(range(N_iters + N_warmup)):
            out = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True)
        torch.cuda.synchronize(); end_time = time.time()
        duration = end_time - start_time

    cp_degree = total_seq_len // cp_shard_len

    avg_duration = duration / N_iters
    avg_duration_ms = avg_duration * 1000
    print(out.shape)

    result = dict(cp_shard_len=cp_shard_len, num_seq=num_seq, L=L, num_heads=num_heads, head_dim=head_dim, causal=True, mode='fwd', duration_ms=avg_duration_ms)
    print(f"🟡 {result = }")

    import json
    with open(f"item_03.result.nh{num_heads}.hdim{head_dim}.L{L}.jsonl", "a") as f:
        f.write(json.dumps(result))
        f.write("\n")

    del q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, out
    torch.cuda.empty_cache()

# %%
"""
nsys profile -t nvtx,cuda --force-overwrite true -o cp_shard.nsys python item_03.attn_cp_shard.ipy.py
"""
