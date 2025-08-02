# CUDA Dispatch Kernel

## Purpose
The CUDA dispatch kernel (`dispatch_kernel` in `in_place_attn_switch.cu`) is the GPU implementation that performs the actual distributed communication for attention computation. It uses NVSHMEM for high-performance inter-GPU communication and executes the routing plans computed by the Python metadata system.

## Kernel Signature
```cuda
template <bool KEY_VALUE, bool IS_KV_BACKWARD>
__global__ void dispatch_kernel(
  // Input and output tensors
  const std::byte *send_tensor,        // Data to send (MLP layout)
  std::byte *recv_tensor,             // Where to receive data (attention layout)
  const std::byte *kv_send_tensor,    // KV data to send
  std::byte *kv_recv_tensor,          // Where to receive KV data
  
  // Metadata tensors (from Python compute_metadata)
  const int32_t *dst_ranks,           // Destination ranks for each sequence
  const uint32_t *dst_offsets,        // Offsets in destination buffers
  const uint64_t *num_recv_tokens,    // Token counts to receive from each rank
  const uint32_t *seq_lens,           // Length of each sequence
  
  // KV-specific metadata
  const int32_t *kv_dst_ranks,        // KV destination ranks (3D: seq × cp_degree)
  const uint32_t *kv_dst_offsets,     // KV destination offsets
  const uint64_t *kv_num_recv_tokens, // KV token counts
  
  // Execution parameters
  const size_t num_tokens,            // Total tokens to process
  const size_t num_sequence,          // Number of sequences
  const size_t max_cp_degree,         // Maximum context parallelism degree
  const size_t stride,                // Bytes per token
  const size_t kv_stride,             // Bytes per KV token
  const unsigned rank,                // This GPU's rank
  const unsigned world_size,          // Total number of GPUs
  
  // NVSHMEM communication buffers
  std::byte *q_send_buffer,           // Query send buffer
  std::byte *q_recv_buffer,           // Query receive buffer
  std::byte *kv_send_buffer,          // KV send buffer
  std::byte *kv_recv_buffer,          // KV receive buffer
  uint64_t *q_signal_buffer,          // Query synchronization signals
  uint64_t *kv_signal_buffer,         // KV synchronization signals
  
  // Backward pass metadata
  const uint32_t *seq_recv_mask,      // Which KV sequences are active
  const uint32_t *recv_seq_lens,      // Original sequence lengths
  const size_t kv_backward_num_tokens // Total tokens in backward pass
)
```

## Kernel Architecture

### Execution Model
- **Grid-stride loop**: Each thread block processes multiple tokens
- **Warp-level parallelism**: 32-thread warps handle token communication
- **Cooperative kernel**: Synchronization across all thread blocks
- **NVSHMEM**: Zero-copy inter-GPU communication

### Thread Organization
```cuda
const unsigned WARP_SIZE = 32;
const unsigned NUM_WARPS = blockDim.x / WARP_SIZE;  // 10 warps per block
const unsigned warp_id = threadIdx.x / WARP_SIZE;
const unsigned warp_group_id = blockIdx.x;          // Block processes one token at a time
```

## Virtual Execution Walkthrough

Let's trace through the kernel execution with a concrete example to understand exactly how it works.

### Example Setup
```cuda
// Input parameters for our walkthrough
world_size = 3;
num_tokens = 12;  // 4 tokens per rank
num_sequence = 3; // 1 sequence per rank for simplicity
max_cp_degree = 2;
stride = 128;     // 128 bytes per token
rank = 1;         // We're executing on rank 1

// Sequence lengths (from Python metadata)
seq_lens = [4, 4, 4];  // Each rank has 4 tokens

// Query dispatch metadata (from Python compute_metadata)
dst_ranks = [2, 0, 1];    // R0→R2, R1→R0, R2→R1
dst_offsets = [0, 4, 8];  // Offsets in destination buffers

// KV dispatch metadata (2D: seq × cp_degree)
kv_dst_ranks = [
  [2, 0],  // R0 KV goes to R2 and R0
  [0, -1], // R1 KV goes to R0 only
  [1, 2]   // R2 KV goes to R1 and R2
];
kv_dst_offsets = [
  [0, 4],   // Offsets for R0's KV
  [8, 0],   // Offsets for R1's KV  
  [12, 16]  // Offsets for R2's KV
];

// Launch configuration
numBlocks = 4;      // Process 4 tokens in parallel
numWarps = 10;      // 10 warps per block
```

Now let's trace through the kernel execution:

### Phase 1: Kernel Launch and Setup

```cuda
// Kernel launched with:
// Grid: (4 blocks, 1, 1)
// Block: (320 threads = 10 warps × 32 threads, 1, 1)

// Each block gets a warp_group_id = blockIdx.x
// Block 0: processes tokens 0, 4, 8, ... (grid-stride)
// Block 1: processes tokens 1, 5, 9, ...
// Block 2: processes tokens 2, 6, 10, ...  
// Block 3: processes tokens 3, 7, 11, ...

// For our example, let's follow Block 1 (token_idx starts at 1)
warp_group_id = 1;  // blockIdx.x
num_warp_groups = 4; // gridDim.x
```

### Phase 2: Sender-Side Logic (Grid-Stride Loop)

```cuda
// Block 1 processes tokens: 1, 5, 9 (within num_tokens=12)
for (int token_idx = 1; token_idx < 12; token_idx += 4) {
  
  // ITERATION 1: token_idx = 1
  // This token belongs to sequence 0 (tokens 0-3 are seq 0)
  
  // STEP 1: Copy token to send buffer
  const int4* send_token = (int4*)(send_tensor + 1 * 128);  // Token 1 from input
  std::byte* send_buffer_token = q_send_buffer + 1 * 128;   // Slot in send buffer
  
  // All 320 threads cooperatively copy 128 bytes (32 int4s)
  for (int i = threadIdx.x; i * 16 < 128; i += 320) {
    ((int4*)send_buffer_token)[i] = send_token[i];
  }
  
  // If KEY_VALUE template parameter is true, also copy KV data
  if constexpr (KEY_VALUE) {
    const int4* kv_send_token = (int4*)(kv_send_tensor + 1 * kv_stride);
    std::byte* kv_send_buffer_token = kv_send_buffer + 1 * kv_stride;
    
    for (int i = threadIdx.x; i * 16 < kv_stride; i += 320) {
      ((int4*)kv_send_buffer_token)[i] = kv_send_token[i];
    }
  }
  
  // Synchronize all warps in this block before communication
  asm volatile("bar.sync 1, %0;" ::"r"(320));
  
  // STEP 2: Query dispatch (warp 0 handles this)
  if (warp_id == 0) {
    // Update sequence tracking for token 1
    while (token_idx >= sequence_end) {
      sequence_id += 1;  // sequence_id becomes 0
      sequence_len = seq_lens[0] = 4;
      recv_sequence_begin_token_id = 0;
      sequence_end = 4;
      recv_rank = dst_ranks[0] = 2;      // Send to rank 2
      recv_offset = dst_offsets[0] = 0;  // At offset 0
    }
    
    // Calculate where this token goes in destination
    recv_token_offset = 0 + (1 - 0) = 1;  // offset + (token_idx - seq_begin)
    std::byte* recv_buffer_token = q_recv_buffer + 1 * 128;
    
    // NVSHMEM communication: send token 1 to rank 2
    nvshmemx_putmem_signal_nbi_warp(
      recv_buffer_token,      // Destination buffer on rank 2
      send_buffer_token,      // Source buffer (local)
      128,                    // Size
      &q_signal_buffer[1],    // Signal counter (increment for rank 1)
      1,                      // Increment amount
      NVSHMEM_SIGNAL_ADD,     // Signal operation
      2                       // Target rank
    );
  }
  
  // STEP 3: KV dispatch (warps 1, 2, ... handle different CP degrees)
  else if constexpr (KEY_VALUE) {
    if (warp_id <= max_cp_degree) {  // warps 1 and 2 for cp_degree=2
      
      // Warp 1 handles first KV destination
      if (warp_id == 1) {
        // Update sequence tracking
        while (token_idx >= sequence_end) {
          sequence_id += 1;  // 0
          sequence_len = 4;
          kv_recv_sequence_begin_token_id = 0;
          sequence_end = 4;
          kv_recv_rank = kv_dst_ranks[0 * 2 + 0] = 2;     // First destination
          kv_recv_offset = kv_dst_offsets[0 * 2 + 0] = 0; // First offset
        }
        
        // Send KV token 1 to first destination (rank 2)
        if (kv_recv_rank != -1) {
          kv_recv_token_offset = 0 + (1 - 0) = 1;
          std::byte* kv_recv_buffer_token = kv_recv_buffer + 1 * kv_stride;
          
          nvshmemx_putmem_signal_nbi_warp(
            kv_recv_buffer_token,
            kv_send_buffer_token,
            kv_stride,
            &kv_signal_buffer[1],  // Our rank's signal
            1,
            NVSHMEM_SIGNAL_ADD,
            2                      // Send to rank 2
          );
        }
      }
      
      // Warp 2 handles second KV destination
      if (warp_id == 2) {
        // Similar logic for second destination
        kv_recv_rank = kv_dst_ranks[0 * 2 + 1] = 0;     // Second destination
        kv_recv_offset = kv_dst_offsets[0 * 2 + 1] = 4; // Second offset
        
        // Send KV token 1 to second destination (rank 0)
        if (kv_recv_rank != -1) {
          kv_recv_token_offset = 4 + (1 - 0) = 5;
          std::byte* kv_recv_buffer_token = kv_recv_buffer + 5 * kv_stride;
          
          nvshmemx_putmem_signal_nbi_warp(
            kv_recv_buffer_token,
            kv_send_buffer_token,
            kv_stride,
            &kv_signal_buffer[1],
            1,
            NVSHMEM_SIGNAL_ADD,
            0                      // Send to rank 0
          );
        }
      }
    }
  }
  
  // ITERATION 2: token_idx = 5
  // This token belongs to sequence 1 (tokens 4-7)
  // Similar process but with different dst_ranks[1] and dst_offsets[1]
  
  // ITERATION 3: token_idx = 9  
  // This token belongs to sequence 2 (tokens 8-11)
  // Uses dst_ranks[2] and dst_offsets[2]
}
```

### Phase 3: Grid-Wide Synchronization

```cuda
// All blocks must finish sending before any can start receiving
cooperative_groups::this_grid().sync();
```

### Phase 4: Receiver-Side Synchronization

```cuda
// Wait for all expected data to arrive
for (size_t i = threadIdx.x; i < world_size; i += 32) {
  // Wait for expected number of tokens from each rank
  const uint64_t expected_from_rank_i = num_recv_tokens[i];
  nvshmem_uint64_wait_until(&q_signal_buffer[i], NVSHMEM_CMP_EQ, expected_from_rank_i);
  
  if constexpr (KEY_VALUE) {
    const uint64_t expected_kv_from_rank_i = kv_num_recv_tokens[i];
    nvshmem_uint64_wait_until(&kv_signal_buffer[i], NVSHMEM_CMP_EQ, expected_kv_from_rank_i);
  }
}
__syncthreads();
```

### Phase 5: Receiver-Side Memory Copy

```cuda
// Copy received data from NVSHMEM buffers to output tensors
if constexpr (IS_KV_BACKWARD) {
  // Special handling for KV backward pass
  _dispatch_recv_impl_kv_backward(
    recv_tensor,
    recv_seq_lens,
    stride,
    world_size,
    Q_BUFFER_STRIDE,
    q_recv_buffer,
    seq_recv_mask,
    max_cp_degree,
    kv_backward_num_tokens
  );
} else {
  // Standard forward pass reception
  uint64_t tot_num_recv_tokens = num_recv_tokens[world_size];  // Total tokens to receive
  
  // Grid-stride loop over all received tokens
  for (size_t token_idx = blockIdx.x; token_idx < tot_num_recv_tokens; token_idx += gridDim.x) {
    std::byte* recv_buffer_token = q_recv_buffer + token_idx * Q_BUFFER_STRIDE;
    int4* recv_token = (int4*)(recv_tensor + token_idx * stride);
    
    // Copy from NVSHMEM buffer to output tensor
    for (int i = threadIdx.x; i * 16 < stride; i += blockDim.x) {
      recv_token[i] = ((int4*)recv_buffer_token)[i];
    }
  }
  
  // Similar process for KV data if KEY_VALUE is true
}
```

## Key Architectural Insights

### NVSHMEM Communication Pattern
```cuda
// The kernel uses one-sided RDMA-style communication:
// 1. Sender writes directly to receiver's memory
// 2. Signal counters track completion
// 3. No explicit receiver participation needed

nvshmemx_putmem_signal_nbi_warp(
  remote_addr,     // Address in target GPU's memory
  local_addr,      // Address in this GPU's memory  
  size,            // Number of bytes
  signal_addr,     // Counter to increment on completion
  signal_value,    // Amount to increment
  NVSHMEM_SIGNAL_ADD,
  target_pe        // Target GPU rank
);
```

### Warp-Level Parallelism
- **Warp 0**: Handles query dispatch for all tokens
- **Warps 1-N**: Handle KV dispatch to different CP destinations
- **All warps**: Cooperatively copy data and synchronize

### Memory Layout Optimization
```cuda
// Buffer strides are aligned to int4 (16 bytes) for efficient copying
const unsigned Q_BUFFER_STRIDE = round_up(stride, sizeof(int4));
const unsigned KV_BUFFER_STRIDE = round_up(kv_stride, sizeof(int4));

// Copy operations use vectorized int4 loads/stores
for (int i = threadIdx.x; i * sizeof(int4) < stride; i += blockDim.x) {
  ((int4*)dst)[i] = ((int4*)src)[i];
}
```

## Execution Flow Summary

Our example execution on rank 1:

1. **Token 1**: 
   - Query → sent to rank 2 at offset 1
   - KV → sent to rank 2 at offset 1 AND rank 0 at offset 5

2. **Token 5**:
   - Query → sent to rank 0 at offset 5 (different sequence)
   - KV → sent according to sequence 1's dispatch plan

3. **Token 9**:
   - Query → sent to rank 1 at offset 9 (self-communication)
   - KV → sent according to sequence 2's dispatch plan

4. **Synchronization**: Wait for expected incoming data from ranks 0 and 2

5. **Reception**: Copy received data from NVSHMEM buffers to final output tensors

## Template Specializations

The kernel supports three modes via template parameters:

```cuda
// Forward pass, query only
dispatch_kernel<false, false>

// Forward pass, query + key-value  
dispatch_kernel<true, false>

// Backward pass, KV gradients
dispatch_kernel<false, true>
```

Each specialization optimizes for its specific communication pattern while sharing the core dispatch logic.

## Performance Characteristics

### Strengths
- **Zero-copy communication**: Direct GPU-to-GPU transfers
- **Overlap computation/communication**: Async operations
- **Vectorized memory access**: int4 operations for bandwidth efficiency
- **Warp-level parallelism**: Exploits GPU architecture

### Considerations
- **NVSHMEM dependency**: Requires specialized hardware/software stack
- **Memory pressure**: Multiple buffers (send, receive, signal) per GPU
- **Synchronization overhead**: Grid-wide barriers for correctness
- **Load balancing**: Grid-stride helps but may not be perfectly balanced

---

*This CUDA kernel is the high-performance implementation that executes the communication plans computed by the Python metadata system, bridging the gap between algorithmic planning and hardware execution.*