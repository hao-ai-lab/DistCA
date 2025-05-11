import torch
import torch.nn as nn
import json

from rich import print

# Function to benchmark the Linear layer
def benchmark_linear_layer(context_lengths, input_dim, output_dim, device='cuda'):
    results = []
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    linear_layer = nn.Linear(input_dim, output_dim, device=device, dtype=dtype)
    
    for context_length in context_lengths:
        # Generate random input tensor
        input_tensor = torch.randn(1, context_length, input_dim, device=device, dtype=dtype)
        input_tensor.requires_grad_(True)

        output_target = torch.randn(1, context_length, output_dim, device=device, dtype=dtype)
        output_target.requires_grad_(False)
        
        # Warm-up phase
        for _ in range(5):  # Warm up 5 times
            out = linear_layer(input_tensor)
            out.backward(output_target, retain_graph=True)
        
        # Forward and Backward pass sampling
        forward_times = []
        backward_times = []
        for _ in range(10):  # Sample 10 times
            start_event_forward = torch.cuda.Event(enable_timing=True)
            end_event_forward = torch.cuda.Event(enable_timing=True)

            start_event_backward = torch.cuda.Event(enable_timing=True)
            end_event_backward = torch.cuda.Event(enable_timing=True)
            
            # Forward pass
            start_event_forward.record()
            output = linear_layer(input_tensor)
            end_event_forward.record()
                        
            # Backward pass
            start_event_backward.record()
            output.backward(output_target, retain_graph=True)
            end_event_backward.record()

            torch.cuda.synchronize()
            forward_times.append(start_event_forward.elapsed_time(end_event_forward))
            backward_times.append(start_event_backward.elapsed_time(end_event_backward))
        
        # Average forward and backward times
        forward_time = sum(forward_times) / len(forward_times)
        backward_time = sum(backward_times) / len(backward_times)
        
        # Store the results
        item = {
            'context_length': context_length,
            'input_dim': input_dim,
            'output_dim': output_dim,
            'forward_time': forward_time,
            'backward_time': backward_time,
            'backward_forward_ratio': backward_time / forward_time
        }
        results.append(item)
        
        with open('mlp_profile.jsonl', 'a') as f:
            f.write(json.dumps(item) + '\n')

    return results

# Example usage
context_lengths = [64, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]  # Different context lengths to test

# Run the benchmark
# TP = 1
results = benchmark_linear_layer(context_lengths, input_dim = 4096, output_dim = 4096, device='cuda')
results = benchmark_linear_layer(context_lengths, input_dim = 1024, output_dim = 4096, device='cuda')
results2 = benchmark_linear_layer(context_lengths, input_dim = 4096, output_dim = 14336, device='cuda')
results2 = benchmark_linear_layer(context_lengths, input_dim = 14336, output_dim = 4096, device='cuda')

# TP = 2
