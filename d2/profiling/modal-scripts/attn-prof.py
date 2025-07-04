"""
Profiling the attention time of the attention server.
Use Modal to deploy the attention server.
"""
K = 1024

import os
import modal

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm-flash-attn")
)

image = vllm_image


app = modal.App("attn-prof", image=image)

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

    import vllm_flash_attn
    import vllm_flash_attn.flash_attn_interface
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
        torch.cuda.synchronize()
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
    for _ in range(10):
        test_flash_attn()
    
    # benchmark
    num_iters = 10
    durations = []
    for _ in range(num_iters):
        duration = test_flash_attn()
        durations.append(duration)

    avg_duration = sum(durations) / len(durations)
    return avg_duration



def run_single_sequence():
    model_configs = [
        # Qwen3 235B activate attention data
        dict(
            num_qo_heads = 64,
            num_kv_heads = 4,
            head_dim = 128,
        ),
    ]

    batches = [
        "[128]", "[256]", "[512]",
        "[1*K]", "[2*K]", "[4*K]", "[8*K]",  "[16*K]", 
        "[32*K]", "[48*K]", "[64*K]", 
        "[96*K]", "[128*K]",
    ]

    gpu_type = "H100"
    
    
    if not os.path.exists(f"compute-attn-{gpu_type}.psv"):
        with open(f"compute-attn-{gpu_type}.psv", "w") as f:
            pass
    
    with open(f"compute-attn-{gpu_type}.psv", "a") as f:
        def print_dual(*args):
            print(*args, file=f, flush=True)
            print(*args, flush=True)
            return
        
        print_dual(f"gpu_type|op|total_len|batch_size|tp|cp|latency(ms)|hqo|hkv|d|batch")

        for model_config in model_configs:
            hqo, hkv, d = model_config['num_qo_heads'], model_config['num_kv_heads'], model_config['head_dim']
            for batch in batches:
                total_len = sum(eval(batch))
                batch_size = len(eval(batch))
                for tp in [1, 2, 4, 8]:
                    for cp in [1, 2, 4, 8]:
                        try:
                            avg_duration = attn_flash_attn.remote(
                                batch = eval(batch),
                                cp = cp, tp = tp,
                                **model_config,
                            )
                            print_dual(f"{gpu_type}|attn|{total_len}|{batch_size}|{tp}|{cp}|{avg_duration:.2f}|{hqo}|{hkv}|{d}|{batch}")
                        except Exception as e:
                            print(f"Batch: {batch}, TP: {tp}, CP: {cp}, Model Config: {model_config} encountered error: {e}")
                            continue


def run_amortized_batch_sequence():
    model_configs = [
        # Qwen3 235B activate attention data
        dict(
            num_qo_heads = 64,
            num_kv_heads = 4,
            head_dim = 128,
        ),
    ]

    seq_lens = [
        32, 64,
        128, 256, 512, 
        1*K, 2*K, 4*K,
        8*K, 16*K, 32*K,
        # 64*K, 128*K,
    ]
    max_seq_len = max(seq_lens)
    # max_seq_len = 128*K
    batches = [
        f"[{s}] * ({max_seq_len // s})"
        for s in seq_lens
    ]

    gpu_type = "H100"
    
    
    if not os.path.exists(f"compute-attn-{gpu_type}.psv"):
        with open(f"compute-attn-{gpu_type}.psv", "w") as f:
            pass
    
    with open(f"compute-attn-{gpu_type}.psv", "a") as f:
        def print_dual(*args):
            print(*args, file=f, flush=True)
            print(*args, flush=True)
            return
        
        print_dual(f"gpu_type|op|total_len|batch_size|tp|cp|latency(ms)|total_latency(ms)|hqo|hkv|d|batch")

        for model_config in model_configs:
            hqo, hkv, d = model_config['num_qo_heads'], model_config['num_kv_heads'], model_config['head_dim']
            for batch in batches:
                total_len = sum(eval(batch))
                batch_size = len(eval(batch))
                for tp in [1, 2, 4, 8]:
                    for cp in [1, 2, 4, 8]:
                        try:
                            total_avg_duration = attn_flash_attn.remote(
                                batch = eval(batch),
                                cp = cp, tp = tp,
                                **model_config,
                            )
                            per_doc_avg_duration = total_avg_duration / batch_size
                            print_dual(f"{gpu_type}|attn|{total_len}|{batch_size}|{tp}|{cp}|{per_doc_avg_duration}|{total_avg_duration}|{hqo}|{hkv}|{d}|{batch}")
                        except Exception as e:
                            print(f"Batch: {batch}, TP: {tp}, CP: {cp}, Model Config: {model_config} encountered error: {e}")
                            continue



@app.local_entrypoint()
def main():
    # run_single_sequence()
    run_amortized_batch_sequence()