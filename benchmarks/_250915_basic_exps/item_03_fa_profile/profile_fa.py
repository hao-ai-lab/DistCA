import time

from flash_attn.flash_attn_interface import (
    _wrapped_flash_attn_varlen_forward, _wrapped_flash_attn_varlen_backward
)
import torch
import tqdm


def run_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            cu_seqlens_q: torch.Tensor, cu_seqlens_kv: torch.Tensor,
            max_seqlen_q: int, max_seqlen_kv: int,):
    causal = True
    block_table = None
    dropout_p = 0.0
    softcap = 0.0
    softmax_scale = q.shape[-1] ** -0.5
    window_size = (-1, -1)
    alibi_slopes = None
    return_attn_probs = False
    out = _wrapped_flash_attn_varlen_forward(
        q, k, v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        dropout_p,
        softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        return_softmax=return_attn_probs and dropout_p > 0,
        block_table=block_table,
    )
    return out


def create_tensor(tot_num_tokens_q: int, tot_num_tokens_k: int, num_heads_q: int, num_heads_k: int,
                  head_dim: int, dtype: torch.dtype = torch.float16, requires_grad: bool=False):
    q = torch.randn(tot_num_tokens_q, num_heads_q, head_dim, dtype=dtype).cuda()
    k = torch.randn(tot_num_tokens_k, num_heads_k, head_dim, dtype=dtype).cuda()
    v = torch.randn(tot_num_tokens_k, num_heads_k, head_dim, dtype=dtype).cuda()
    q.requires_grad = requires_grad
    k.requires_grad = requires_grad
    v.requires_grad = requires_grad
    return q, k, v


def create_testcase(tot_num_tokens: int, shard_size: int, context_size_min: int, context_size_max: int, context_size_round: int):
    assert tot_num_tokens % shard_size == 0
    num_seqs = tot_num_tokens // shard_size
    context_sizes = torch.randint(context_size_min, context_size_max + 1, (num_seqs,))
    if context_size_round > 1:
        context_sizes = (context_sizes.float() / context_size_round).ceil().long() * context_size_round
    cu_seqlens_q = torch.arange(0, num_seqs + 1, dtype=torch.long) * shard_size
    cu_seqlens_kv = torch.concat((torch.zeros(1, dtype=torch.long), context_sizes.cumsum(dim=0)))
    max_seqlen_q = shard_size
    max_seqlen_kv = context_sizes.max().item()
    return cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, context_sizes


def compute_flops(
    shard_size: int, num_heads_q: int, head_dim: int, context_sizes: torch.Tensor
):
    flops = 2 * 2 * num_heads_q * head_dim * shard_size * (2 * context_sizes - shard_size) / 2
    return torch.sum(flops).item()


def test_one_case(
    tot_num_tokens: int, shard_size: int,
    context_size_min: int, context_size_max: int, context_size_round: int,
    num_heads_q: int, num_heads_k: int, head_dim: int, dtype: torch.dtype = torch.float16,
    warmup: int = 10, num_runs: int = 100
):
    cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, context_sizes = create_testcase(
        tot_num_tokens, shard_size, context_size_min, context_size_max, context_size_round)
    q, k, v = create_tensor(
        tot_num_tokens, cu_seqlens_kv[-1].item(),
        num_heads_q, num_heads_k, head_dim,
        dtype=dtype
    )
    print(f"test case created: {shard_size=}, {max_seqlen_kv=}, {k.shape=}")
    flops = compute_flops(shard_size, num_heads_q, head_dim, context_sizes)
    cu_seqlens_q = cu_seqlens_q.cuda().to(torch.int32)
    cu_seqlens_kv = cu_seqlens_kv.cuda().to(torch.int32)
    with torch.no_grad():
        print("start profiling...")
        for _ in range(warmup):
            out = run_fwd(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
        torch.cuda.synchronize()
        print("warmup done.")
        tik = time.time()
        for _ in range(num_runs):
            out = run_fwd(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
        torch.cuda.synchronize()
        tok = time.time()
        t = tok - tik
        tms = t * 1000 / num_runs
        throughput = flops * num_runs / t / 1e12
    print(f"avg latency: {tms:.2f} ms, throughput: {throughput:.2f} TFLOPS")
    return tms, throughput


def test():
    # llama 8B, TP8
    num_heads_q = 32
    num_heads_k = 8
    tp_size = 8
    num_heads_q //= tp_size
    num_heads_k //= tp_size
    head_dim = 128
    dtype = torch.float16

    tot_num_tokens = 32 * 1024
    results = {}
    torch.set_default_device('cuda')
    for shard_size in [16, 32, 64, 128, 256, 512, 1024]:
        throughputs = []
        print("testing shard size:", shard_size)
        for sample_id in tqdm.tqdm(range(10)):
            context_size_min = 128
            context_size_max = max(shard_size, 16384)
            context_size_round = 8
            tms, throughput = test_one_case(
                tot_num_tokens, shard_size,
                context_size_min, context_size_max, context_size_round,
                num_heads_q, num_heads_k, head_dim, dtype=dtype,
                warmup=10, num_runs=50
            )
            throughputs.append(throughput)
        results[shard_size] = throughputs
    print("Results:")
    print(f"Config: {num_heads_q=}, {num_heads_k=}, {head_dim=}, {dtype=}")
    for shard_size in results:
        throughputs = results[shard_size]
        avg_tp = sum(throughputs) / len(throughputs)
        print(f"shard size: {shard_size}, avg throughput: {avg_tp:.2f} TFLOPS, samples: {', '.join([f'{tp:.2f}' for tp in throughputs])}")
    # dump result:
    output_name = f"fa_profile_h{num_heads_q}_d{head_dim}_tok{tot_num_tokens}_ss{shard_size}.pkl"
    with open(output_name, "wb") as f:
        import pickle
        pickle.dump(results, f)
    print(f"results dumped to {output_name}")


if __name__ == "__main__":
    test()
