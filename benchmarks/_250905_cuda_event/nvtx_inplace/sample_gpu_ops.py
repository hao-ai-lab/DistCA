import torch
import torch.cuda.nvtx as nvtx
import time

def main():
    # Start NVTX range for profiling
    nvtx.range_push("full_program")
    
    # Create some tensors on GPU
    nvtx.range_push("data_preparation")
    size = (2048, 2048)
    a = torch.randn(size, device='cuda')
    b = torch.randn(size, device='cuda')
    torch.cuda.synchronize()
    nvtx.range_pop()  # end data_preparation
    
    # Do some matrix operations
    nvtx.range_push("matrix_operations")
    
    # Matrix multiplication
    nvtx.range_push("matmul")
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    nvtx.range_pop()  # end matmul
    
    # Element-wise operations
    nvtx.range_push("element_wise")
    d = torch.sin(c)
    e = torch.exp(d)
    torch.cuda.synchronize()
    nvtx.range_pop()  # end element_wise
    
    nvtx.range_pop()  # end matrix_operations
    
    # Simulate some CPU work
    nvtx.range_push("cpu_work")
    time.sleep(0.5)  # Simulate CPU processing
    nvtx.range_pop()  # end cpu_work
    
    nvtx.range_pop()  # end full_program
    
    print("Operations completed successfully!")

if __name__ == "__main__":
    main()
