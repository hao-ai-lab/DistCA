"""
Profiling the attention time of the attention server.
Use Modal to deploy the attention server.
"""

import modal

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm-flash-attn")
)


image = vllm_image

K = 1024

app = modal.App("profiling-attn", image=image)

@app.function(gpu="H100:1")
def attn_flash_attn(
    batch,
    num_qo_heads = 64,
    num_kv_heads = 4,
    head_dim = 128,
    cp = 1,
    tp = 1,
):
    import time

    import torch
    device = torch.device("cuda")
    # print(torch)

    # # Test basic torch matmul
    # a = torch.randn(1024, 1024, device=device)
    # b = torch.randn(1024, 1024, device=device)
    # c = torch.matmul(a, b)
    # torch.cuda.synchronize()
    # print("Successfully finished matmul")

    import vllm_flash_attn
    # print(vllm_flash_attn)
    # print(dir(vllm_flash_attn))

    import vllm_flash_attn.flash_attn_interface
    # print(dir(vllm_flash_attn.flash_attn_interface))

    from vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

    # Qwen3 253B activate attention data
    num_qo_heads = num_qo_heads // tp
    num_kv_heads = max(num_kv_heads // tp, 1)

    kv_lens = [(1/2 + 1/(2 * cp)) * i for i in batch]
    
    batch = [int(i // cp) for i in batch]
    kv_lens = [int(i) for i in kv_lens]

    total_tokens = sum(batch)
    total_kv_tokens = sum(kv_lens)

    q = torch.randn(total_tokens, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(total_kv_tokens, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(total_kv_tokens, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    max_seqlen_q = max(batch)
    max_seqlen_k = max(kv_lens)

    cu_seqlens_q = [0,]
    cu_seqlens_k = [0,]
    for idx, _ in enumerate(batch):
        cu_seqlens_q.append(sum(batch[:idx+1]))
        cu_seqlens_k.append(sum(kv_lens[:idx+1]))
    cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=device)
    max_seqlen_q = torch.tensor(max_seqlen_q, dtype=torch.int32, device=device)
    max_seqlen_k = torch.tensor(max_seqlen_k, dtype=torch.int32, device=device)
    

    def test_flash_attn():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        output = flash_attn_varlen_func(
            q, k, v, 
            cu_seqlens_q, cu_seqlens_k, 
            max_seqlen_q, max_seqlen_k,
            dropout_p=0.0, causal=True,
        )
        end_event.record()
        torch.cuda.synchronize()
        duration = start_event.elapsed_time(end_event)
        return duration
    
    # warmup
    for _ in range(5):
        test_flash_attn()
    
    # benchmark
    num_iters = 10
    durations = []
    for _ in range(num_iters):
        duration = test_flash_attn()
        durations.append(duration)

    avg_duration = sum(durations) / len(durations)

    # print(f"TP: {tp}, CP: {cp}, Result: {avg_duration:.2f} ms")

    return avg_duration


@app.function(gpu="H100:1")
def mlp_gemm(
    batch,
    num_qo_heads = 64,
    num_kv_heads = 4,
    head_dim = 128,
    mlp_dim = 4096,
    tp = 1,
    cp = 1,
):
    import torch
    import time

    num_qo_heads = num_qo_heads // tp
    num_kv_heads = max(num_kv_heads // tp, 1)

    total_tokens = sum(batch)
    total_kv_tokens = sum(kv_lens)

    device = torch.device("cuda")
    a = torch.randn(m, k, device=device)
    b = torch.randn(k, n, device=device)
    for _ in range(10):
        _ = torch.matmul(a, b)
    
    start = time.time()
    for _ in range(10):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    end = time.time()

    duration = end - start
    duration *= 1000
    return duration

def total_gemm_time():
    pass


@app.local_entrypoint()
def main():
    model_config = dict(
        num_qo_heads = 64,
        num_kv_heads = 4,
        head_dim = 128,
    )
    batch = " [i * K for i in [16] + [2] * 8 ] "
    print(f"Batch: {batch}")
    print("-" * 10)
    for tp in [1, 2, 4, 8]:
        for cp in [1, 2, 4, 8]:
            avg_duration = attn_flash_attn.remote(
                batch = eval(batch),
                cp = cp,
                tp = tp,
                **model_config,
            )
            print(f"TP: {tp}, CP: {cp}, Result: {avg_duration:.2f} ms")
    print("-" * 10)