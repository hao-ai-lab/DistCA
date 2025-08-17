import torch
import torch.cuda
import torch.profiler
from flash_attn import flash_attn_func
import os
import torch.cuda.nvtx


def benchmark_flash_attention():
    # Set device
    device = torch.cuda.current_device()
    
    # Example parameters
    total_seq_len = 1024 * 64
    seq_len = 1024 * 64
    batch_size = total_seq_len // seq_len
    num_heads = 4
    head_dim = 128
    
    # Create random input tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    
    # Warmup
    for _ in range(5):
        _ = flash_attn_func(q, k, v, causal=True)
    
    torch.cuda.synchronize()
    
    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Create a CUDA stream
    stream = torch.cuda.Stream(device=device)
    
    print("Starting Flash Attention benchmark with CUDA events...")
    
    # Record timing with CUDA events
    with torch.cuda.nvtx.range("flash_attention_benchmark"):
        with torch.cuda.stream(stream):
            start_event.record(stream)
            
            # Run Flash Attention
            output = flash_attn_func(q, k, v, causal=True)
            
            end_event.record(stream)
            # Wait for completion
            end_event.synchronize()
    
    # Calculate elapsed time
    elapsed_time = start_event.elapsed_time(end_event)
    print(f"Flash Attention elapsed time: {elapsed_time:.4f} ms")
    print(f"Output shape: {output.shape}")
    
    return output

def benchmark_flash_attention_with_profiler():
    """
    Benchmark Flash Attention using torch.profiler to capture CUDA streams
    and export profiling data for analysis.
    """
    # Set device
    device = torch.cuda.current_device()
    
    # Example parameters (same as original benchmark)
    batch_size = 2
    seq_len = 8 * 1024
    num_heads = 16
    head_dim = 64
    
    # Create random input tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    
    # Warmup
    for _ in range(5):
        _ = flash_attn_func(q, k, v, causal=True)
    
    torch.cuda.synchronize()
    
    # Create a CUDA stream
    stream = torch.cuda.Stream(device=device)
    
    print("Starting Flash Attention benchmark with torch.profiler...")
    
    # Set up profiler with CUDA and CPU activities
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        # record_shapes=True,
        # profile_memory=True,
        # with_stack=True,
        # with_flops=True,
        # Export to multiple formats for analysis
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs")
    ) as prof:
        with torch.cuda.nvtx.range("flash_attention_benchmark"):
            with torch.cuda.stream(stream):
                # Run Flash Attention
                output = flash_attn_func(q, k, v, causal=True)
                torch.cuda.synchronize()
    
    # Export profiler results to different formats
    print("Exporting profiler results...")
    
    # Export to Chrome trace format (JSON) - searchable
    prof.export_chrome_trace("flash_attn_profile.json")
    print("Chrome trace exported to: flash_attn_profile.json")
    
    # Export to stacks format for detailed analysis
    prof.export_stacks("flash_attn_stacks.txt", "self_cuda_time_total")
    print("Stack trace exported to: flash_attn_stacks.txt")
    
    # Print summary table
    print("\n" + "="*80)
    print("PROFILER SUMMARY - Key CUDA operations:")
    print("="*80)
    
    # Print key CUDA operations
    key_events_table = prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20,
        max_name_column_width=60
    )
    print(key_events_table)
    
    # Search for flash attention related operations
    print("\n" + "="*80)
    print("FLASH ATTENTION SPECIFIC OPERATIONS:")
    print("="*80)
    
    flash_attn_events = []
    for event in prof.key_averages():
        if "flash" in event.key.lower() or "attention" in event.key.lower():
            flash_attn_events.append(event)
    
    if flash_attn_events:
        for event in flash_attn_events:
            print(f"Name: {event.key}")
            print(f"  CUDA Time: {event.cuda_time_total:.2f} ms")
            print(f"  CPU Time: {event.cpu_time_total:.2f} ms")
            print(f"  Count: {event.count}")
            print("-" * 60)
    else:
        print("No flash attention specific events found in top-level keys.")
        print("Check the exported files for detailed NVTX ranges.")
    
    print(f"\nOutput shape: {output.shape}")
    print("\nProfile data exported to:")
    print("  - flash_attn_profile.json (Chrome trace format - open in chrome://tracing)")
    print("  - flash_attn_stacks.txt (Stack traces)")
    print("  - ./profiler_logs/ (TensorBoard format)")
    print("\nTo analyze in TensorBoard: tensorboard --logdir=./profiler_logs")
    
    return output, prof

if __name__ == "__main__":
    print("Flash Attention Benchmark with CUDA Events")
    print("=" * 50)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        exit(1)
    
    print(f"Using device: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")
    
    # Run original benchmark
    result = benchmark_flash_attention()
    
    # print("\n" + "="*80)
    # print("RUNNING TORCH.PROFILER BENCHMARK")
    # print("="*80)
    
    # # Run profiler benchmark
    # result_prof, profiler = benchmark_flash_attention_with_profiler()
    
    # print("\nBenchmark completed successfully!")
    # print("To profile with nsys, run:")
    # print("nsys profile --trace=cuda,nvtx --output=flash_attn_profile python main.py")
