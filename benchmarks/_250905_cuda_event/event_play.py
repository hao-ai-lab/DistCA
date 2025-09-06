"""

"""

import torch
# Create some test tensors on GPU
a = torch.randn(1024, 1024, device='cuda')
b = torch.randn(1024, 1024, device='cuda')

# Create CUDA events to measure time
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Record start event
start_event.record()

# Do matmul
c = torch.matmul(a, b)

# Record end event
end_event.record()

# Wait for completion
torch.cuda.synchronize()

# Get elapsed time in milliseconds
elapsed_time = start_event.elapsed_time(end_event)

print(f"Matrix multiplication took {elapsed_time:.2f} ms")
