import time
from flash_attn.flash_attn_interface import (
    flash_attn_func, 
    flash_attn_varlen_func,
)
import torch

def get_qkv(
    batch: list[int] = None,
    head_dim: int = 128,
    n_qo_head: int = 32,
    n_kv_head: int = 8,
    device: str = 'cuda',
):
    assert batch is not None, "batch is required"

    q = []
    k = []
    v = []
    cu_seqlens_q = [0,]
    cu_seqlens_k = [0,]
    max_seqlen_q = max(batch)
    max_seqlen_k = max(batch)
    for idx, i in enumerate(batch):
        # q.append(torch.randn(i, n_qo_head, head_dim, dtype=torch.float16))
        # k.append(torch.randn(i, n_kv_head, head_dim, dtype=torch.float16))
        # v.append(torch.randn(i, n_kv_head, head_dim, dtype=torch.float16))
        cu_seqlens_q.append(sum(batch[:idx+1]))
        cu_seqlens_k.append(sum(batch[:idx+1]))
    
    sum_tokens = sum(i for i in batch)
    q = torch.randn(sum_tokens, n_qo_head, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(sum_tokens, n_kv_head, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(sum_tokens, n_kv_head, head_dim, dtype=torch.float16, device=device)
    cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=device)
    max_seqlen_q = torch.tensor(max_seqlen_q, dtype=torch.int32, device=device)
    max_seqlen_k = torch.tensor(max_seqlen_k, dtype=torch.int32, device=device)
    return q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k


def calculate_memory_requirement(
    batch: list[int], 
    head_dim: int, n_qo_head: int, n_kv_head: int,
    dtype: torch.dtype = torch.float16
):
    """
    Calculate the memory requirement for the given batch size and model configuration.
    """
    # Calculate the number of tokens in the batch
    num_tokens = sum(batch)

    # Calculate the memory requirement for q, k, and v
    q_memory = num_tokens * n_qo_head * head_dim * dtype.itemsize
    k_memory = num_tokens * n_kv_head * head_dim * dtype.itemsize
    v_memory = num_tokens * n_kv_head * head_dim * dtype.itemsize

    # Total memory requirement
    total_memory = q_memory + k_memory + v_memory

    return total_memory


def measure_flash_attn_duration_single_batch(
    llama8b_config, batch, warmup=10, n_iter=30,
) -> float: 
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = get_qkv(
        **llama8b_config, batch=batch
    )

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for idx in range(n_iter + warmup):
        if idx == warmup:
            start_event.record()
        flash_attn_varlen_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            dropout_p=0.0, causal=True,
        )
    end_event.record()
    torch.cuda.synchronize()
    duration = start_event.elapsed_time(end_event) / n_iter
    return duration

llama8b_config = dict(
    head_dim=128,
    n_qo_head=32,
    n_kv_head=8,
)

def test_batch():
    K = 2 ** 10
    batches = [
        [2 ** 10, 2 ** 10],
        [2 ** 10],

        [2 ** 10, 2 ** 5],
        [2 ** 10],
        [2 ** 5],


        [2 ** 10, 2 ** 10, 2 ** 10, 2 ** 4],
        [2 ** 10,  2 ** 10,],
        [2 ** 10, 2 ** 4],
    ]

    for batch in batches:
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = get_qkv(
            **llama8b_config,
            batch=batch,
        )

        warmup = 10
        n_iter = 30
        for idx in range(n_iter + warmup):
            if idx == warmup:
                start = time.time()
            flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
                window_size=(-1, -1),  # -1 means infinite context window
                softcap=0.0, # 0.0 means deactivated
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=False,
                block_table=None,
            )
        torch.cuda.synchronize()
        end = time.time()
        duration = (end - start) * 1000 * 1000 / n_iter
        print(f"{batch = }, duration = {duration:.2f} us")
        