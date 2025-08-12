import torch
import time
import rich
from flash_attn import flash_attn_varlen_func


"""
args[0] =
torch.Size([4096, 32, 128])
args[1] =
torch.Size([4096, 8, 128])
args[2] =
torch.Size([4096, 8, 128])
args[3:]:
(
    tensor([   0, 2048, 4096], device='cuda:0', dtype=torch.int32),
    tensor([   0, 2048, 4096], device='cuda:0', dtype=torch.int32),
    tensor(2048, device='cuda:0', dtype=torch.int32),
    tensor(2048, device='cuda:0', dtype=torch.int32),
    0.0
)
kwargs:
"""

def test_flash_attn_varlen(batch_size=2, tp_size=8, seq_len=32768):
    # Set device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    # Parameters
    num_heads = 32 // tp_size
    head_dim = 128
    
    # Create input tensors
    total_tokens = batch_size * seq_len
    q = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
    
    # Create cumulative sequence lengths (for varlen format)
    cu_seqlens = torch.tensor([i * seq_len for i in range(batch_size + 1)], dtype=torch.int32, device=device)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=seq_len,
                max_seqlen_k=seq_len,
                dropout_p=0.0,
                causal=True,
            )
    
    torch.cuda.synchronize()
    forward_times = []
    backward_times = []
    
    for _ in range(10):
        torch.cuda.empty_cache()

        # Forward pass timing
        torch.cuda.synchronize()
        start_time = time.time()
        
        output = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            dropout_p=0.0,
            causal=True
        )
        
        torch.cuda.synchronize()
        forward_time = (time.time() - start_time) * 1000  # Convert to ms
        forward_times.append(forward_time)
        
        # Create gradient for backward pass
        grad_output = torch.randn_like(output)
        
        # Backward pass timing
        torch.cuda.synchronize()
        start_time = time.time()
        
        output.backward(grad_output)
        
        torch.cuda.synchronize()
        backward_time = (time.time() - start_time) * 1000  # Convert to ms
        backward_times.append(backward_time)
    
        torch.cuda.empty_cache()

        rich.print(f"🟢 Forward: {forward_time:.2f} ms, Backward: {backward_time:.2f} ms, Backward / Forward: {backward_time / forward_time:.2f}")

    # Calculate statistics
    import numpy as np
    forward_avg = np.mean(forward_times)
    forward_std = np.std(forward_times)
    backward_avg = np.mean(backward_times)
    backward_std = np.std(backward_times)
    total_avg = forward_avg + backward_avg
    
    rich.print(f"Flash Attention Varlen - {seq_len = } x {batch_size = } (causal):")
    rich.print(f"- Forward: {forward_avg:.2f} ± {forward_std:.2f} ms, Backward: {backward_avg:.2f} ± {backward_std:.2f} ms, Backward / Forward: {backward_avg / forward_avg:.2f}. Total: {total_avg:.2f} ms. Output shape: {output.shape}")



def test_flash_attn_varlen_sweep():
    K = 1024
    
    # Sweep parameters
    seq_lengths = [1 * K, 2 * K, 4 * K, 8 * K, 16 * K, 32 * K, 64 * K]
    batch_sizes = [1, 2, 4, 8]
    tp_sizes = [1, 2, 4, 8]
    
    for seq_len in seq_lengths:
        for batch_size in batch_sizes:
            for tp_size in tp_sizes:
                rich.print(f"\n{'='*60}")
                rich.print(f"Testing: seq_len={seq_len//K}K, batch_size={batch_size}, tp_size={tp_size}")
                rich.print(f"{'='*60}")
                try:
                    test_flash_attn_varlen(batch_size=batch_size, tp_size=tp_size, seq_len=seq_len)
                except Exception as e:
                    rich.print(f"[red]Error: {e}[/red]")
                    continue
        

if __name__ == "__main__":
    test_flash_attn_varlen() # warm up
    
    test_flash_attn_varlen_sweep()
