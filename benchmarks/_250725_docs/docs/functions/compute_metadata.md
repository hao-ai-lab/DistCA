# compute_metadata Function

## Purpose
Computes forward and reverse communication metadata for query tensors given sequence lengths and dispatch decisions. This is the core function that translates high-level dispatch plans into low-level communication instructions.

## Function Signature
```python
@torch.no_grad()
def compute_metadata(
    seq_len: torch.Tensor,      # (world_size, max_num_local_seqs)
    global_dispatch: torch.Tensor,  # (world_size, max_num_local_seqs) or (..., max_cp_degree)
    return_intermediate: bool = False,
) -> Tuple[Metadata, Metadata]
```

## Parameters

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `seq_len` | `torch.Tensor` | `(world_size, max_num_local_seqs)` | Length of each sequence shard |
| `global_dispatch` | `torch.Tensor` | `(world_size, max_num_local_seqs[, max_cp_degree])` | Destination rank for each sequence (-1 for padding) |
| `return_intermediate` | `bool` | - | Whether to return intermediate computation results |

## Returns
- `fwd_metadata`: Forward communication metadata
- `rev_metadata`: Reverse communication metadata  
- `intermediates`: (optional) Intermediate values for debugging/optimization

## Algorithm Breakdown

Let's walk through the algorithm with a concrete example to see exactly how it works.

### Example Setup
```python
# Input values for our walkthrough
world_size = 3
seq_len = torch.tensor([
    [10, 5, 0],    # Rank 0: sequences of length 10, 5, and 0 (padding)
    [8, 12, 4],    # Rank 1: sequences of length 8, 12, 4
    [6, 0, 9]      # Rank 2: sequences of length 6, 0 (padding), 9
])
global_dispatch = torch.tensor([
    [1, 2, -1],    # Rank 0: seq0→rank1, seq1→rank2, seq2→padding
    [2, 0, 1],     # Rank 1: seq0→rank2, seq1→rank0, seq2→rank1
    [0, -1, 2]     # Rank 2: seq0→rank0, seq1→padding, seq2→rank2
])
```

Now let's trace through each phase:

### Phase 1: Input Validation and Setup
```python
world_size = global_dispatch.shape[0]  # = 3
max_num_local_seqs = global_dispatch.shape[1]  # = 3

# Handle both 2D (query) and 3D (kv) dispatch tensors
if global_dispatch.dim() == 2:  # True for our example
    global_dispatch = global_dispatch.unsqueeze(-1)
    
# After unsqueeze, global_dispatch becomes shape (3, 3, 1):
# [[[1], [2], [-1]],     # Rank 0
#  [[2], [0], [1]],      # Rank 1  
#  [[0], [-1], [2]]]     # Rank 2
```

### Phase 2: Forward Metadata Computation

#### Step 2.1: One-Hot Encoding
```python
# Convert destination ranks to one-hot encoding (+1 to handle -1 padding)
# global_dispatch + 1 converts -1→0, 0→1, 1→2, 2→3
dispatch_plus_one = global_dispatch + 1  # Shape (3, 3, 1)
# [[[2], [3], [0]],     # Rank 0: 1→2, 2→3, -1→0
#  [[3], [1], [2]],     # Rank 1: 2→3, 0→1, 1→2
#  [[1], [0], [3]]]     # Rank 2: 0→1, -1→0, 2→3

# One-hot with 4 classes (0,1,2,3), then remove class 0 (padding)
seq_to_dst_one_hot = F.one_hot(dispatch_plus_one, num_classes=4)
# seq_to_dst_one_hot.shape = [3, 3, 1, 4]
seq_to_dst = seq_to_dst_one_hot[:, :, :, 1:]
# Shape: (3, 3, 1, 3) - the last dimension represents [rank0, rank1, rank2]
# seq_to_dst = tensor([
# [
#  [[0, 1, 0]], # - Rank 0: Sequence 0 is dispatched to rank 1, represented as [0, 1, 0]
#  [[0, 0, 1]], # - Rank 0: Sequence 1 is dispatched to rank 2, represented as [0, 0, 1]
#  [[0, 0, 0]] # - Rank 0: Sequence 2 is padding, represented as [0, 0, 0]
# ],
# 
# [
#  [[0, 0, 1]], # - Rank 1: Sequence 0 is dispatched to rank 2, represented as [0, 0, 1]
#  [[1, 0, 0]], # - Rank 1: Sequence 1 is dispatched to rank 0, represented as [1, 0, 0]
#  [[0, 1, 0]] # - Rank 1: Sequence 2 is dispatched to rank 1, represented as [0, 1, 0]
# ],
# 
# [
#  [[1, 0, 0]], # - Rank 2: Sequence 0 is dispatched to rank 0, represented as [1, 0, 0]
#  [[0, 0, 0]], # - Rank 2: Sequence 1 is padding, represented as [0, 0, 0]
#  [[0, 0, 1]] # - Rank 2: Sequence 2 is dispatched to rank 2, represented as [0, 0, 1]
# ]
# ])


#### Step 2.2: Token Routing
```python
# Multiply by sequence lengths to get actual token counts
seq_len_expanded = seq_len.unsqueeze(-1).unsqueeze(-1)  # Shape: (3, 3) -> (3, 3, 1, 1)
tokens_to_dst_per_dispatch = seq_to_dst * seq_len_expanded
# Shape: (3, 3, 1, 3) - tokens sent from each sequence to each rank

# Example: Rank 0, seq 0 (length 10) goes to rank 1:
# seq_to_dst[0,0,0,:] = [0, 1, 0] 
# seq_len[0,0] = 10
# Result: [0, 10, 0] - sends 10 tokens to rank 1

# Complete tokens_to_dst_per_dispatch for our example:
# Rank 0: [[[0,10,0]], [[0,0,5]], [[0,0,0]]]   # 10→rank1, 5→rank2, 0 (padding)
# Rank 1: [[[0,0,8]], [[12,0,0]], [[0,4,0]]]   # 8→rank2, 12→rank0, 4→rank1  
# Rank 2: [[[6,0,0]], [[0,0,0]], [[0,0,9]]]    # 6→rank0, 0 (padding), 9→rank2

# Flatten to (9, 3) and compute cumulative offsets
tokens_to_dst = tokens_to_dst_per_dispatch.reshape(-1, 3) # shape of (9, 3)
seq_begin_offset = tokens_to_dst.cumsum(dim=0) - tokens_to_dst # shape of (9, 3)
# 
# seq_begin_offset[rank][dispatch_id] = for the dispatch_id-th dispatch, how many tokens have been sent to rank rank, and therefore the offset that the new sequence should be placed at (as starting point in the mem buffer)
#
# seq_begin_offset = tensor([
# [ 0,  0,  0],
# [ 0, 10,  0],
# [ 0, 10,  5],
# [ 0, 10,  5],
# [ 0, 10, 13],
# [12, 10, 13],
# [12, 14, 13],
# [18, 14, 13],
# [18, 14, 13]
# ])

# This gives the offset where each sequence will be placed in destination buffers

```

#### Step 2.3: Destination Buffer Offsets
```python
# STEP 1: Apply masking and reshape back to original tensor structure
# seq_begin_offset currently has shape (9, 3) - flattened version
# We need to reshape it back to (3, 3, 1, 3) and apply the dispatch mask

seq_begin_offset = seq_begin_offset.reshape(*seq_to_dst.shape) * seq_to_dst
# seq_begin_offset.shape = (3, 3, 1, 3)
# This multiplication masks out offsets for non-dispatched sequences (padding)

# Let's see what this produces:
# tensor([[[[ 0,  0,  0]],   # R0,seq0: no offset to rank0, offset 0 to rank1, no offset to rank2
#          [[ 0,  0,  5]],   # R0,seq1: no offset to rank0, no offset to rank1, offset 5 to rank2
#          [[ 0,  0,  0]]],  # R0,seq2: padding - all zeros

#         [[[ 0,  0,  5]],   # R1,seq0: no offset to rank0, no offset to rank1, offset 5 to rank2  
#          [[12,  0,  0]],   # R1,seq1: offset 12 to rank0, no offset to rank1, no offset to rank2
#          [[ 0, 10,  0]]],  # R1,seq2: no offset to rank0, offset 10 to rank1, no offset to rank2

#         [[[12,  0,  0]],   # R2,seq0: offset 12 to rank0, no offset to rank1, no offset to rank2
#          [[ 0,  0,  0]],   # R2,seq1: padding - all zeros
#          [[ 0,  0, 13]]]])  # R2,seq2: no offset to rank0, no offset to rank1, offset 13 to rank2

# STEP 2: Collapse the destination dimension to get final offsets
seq_begin_offset = seq_begin_offset.sum(dim=-1)
# Shape changes from (3, 3, 1, 3) to (3, 3, 1)
# This gives us the actual offset where each sequence will be placed in its destination buffer

# Final result:
# tensor([[[ 0],   # R0,seq0 → rank1 at offset 0
#          [ 5],   # R0,seq1 → rank2 at offset 5  
#          [ 0]],  # R0,seq2 → padding (unused)

#         [[ 5],   # R1,seq0 → rank2 at offset 5
#          [12],   # R1,seq1 → rank0 at offset 12
#          [10]],  # R1,seq2 → rank1 at offset 10

#         [[12],   # R2,seq0 → rank0 at offset 12
#          [ 0],   # R2,seq1 → padding (unused)
#          [13]]])  # R2,seq2 → rank2 at offset 13

# WHAT THIS MEANS:
# seq_begin_offset[i][j][0] = the starting position in the destination buffer
#                           where sequence j from rank i should be placed
#
# Example: R1,seq1 (12 tokens) goes to rank0 at offset 12
#          This means it occupies positions [12:24] in rank0's receive buffer
#
# Example: R1,seq2 (4 tokens) goes to rank1 at offset 10  
#          This means it occupies positions [10:14] in rank1's receive buffer

# Why these specific offsets?
# - They ensure no overlap in destination buffers
# - They follow the global ordering established by flattening all dispatches
# - Each destination rank gets a contiguous layout of received sequences
```

#### Step 2.4: Receive Token Counts
```python
# Count total tokens each rank receives from each other rank
# Reshape to (3, 3, 3) and sum over middle dimension (sequences)
recv_matrix = tokens_to_dst_per_dispatch.reshape(3, -1, 3).sum(dim=1)
# recv_matrix[i,j] = tokens that rank j receives from rank i

# For our example:
# recv_matrix = [[0, 10, 5],   # Rank 0 sends: 0→rank0, 10→rank1, 5→rank2
#                [12, 4, 8],   # Rank 1 sends: 12→rank0, 4→rank1, 8→rank2  
#                [6, 0, 9]]    # Rank 2 sends: 6→rank0, 0→rank1, 9→rank2

# Transpose to get receive perspective
num_recv_tokens = recv_matrix.transpose(0, 1)
# num_recv_tokens[i,j] = tokens that rank i receives from rank j

# Result: [[0, 12, 6],    # Rank 0 receives: 0 from R0, 12 from R1, 6 from R2
#          [10, 4, 0],    # Rank 1 receives: 10 from R0, 4 from R1, 0 from R2
#          [5, 8, 9]]     # Rank 2 receives: 5 from R0, 8 from R1, 9 from R2

# Add total column
totals = num_recv_tokens.sum(dim=1, keepdim=True)  # [18, 14, 22]
num_recv_tokens = torch.cat([num_recv_tokens, totals], dim=1)
# Final: [[0, 12, 6, 18],   # Rank 0 total: 18 tokens
#         [10, 4, 0, 14],   # Rank 1 total: 14 tokens  
#         [5, 8, 9, 22]]    # Rank 2 total: 22 tokens
```

### Phase 3: Reverse Metadata Computation

#### Step 3.1: Source Rank Tracking
```python
# STEP 1: Track which rank each sequence comes from
seq_rank_expanded = torch.arange(3).reshape(3, 1, 1).expand_as(global_dispatch)
# Result: [[[0], [0], [0]],    # All sequences from rank 0
#          [[1], [1], [1]],    # All sequences from rank 1
#          [[2], [2], [2]]]    # All sequences from rank 2

# STEP 2: Compute source offsets (where data was in original MLP layout)
# This is for the REVERSE direction - we need to know where to send data back to

# For each rank, sequences are placed sequentially in MLP layout: seq0, seq1, seq2, ...
seq_len_expanded = seq_len.unsqueeze(2).expand_as(global_dispatch).transpose(1, 2)
# After transpose: shape (3, 1, 3) 
# [[[10, 5, 0]],     # Rank 0 sequence lengths  
#  [[8, 12, 4]],     # Rank 1 sequence lengths
#  [[6, 0, 9]]]      # Rank 2 sequence lengths

# STEP 3: Calculate where each sequence starts in its original rank's buffer
seq_offset_expanded = seq_len_expanded.reshape(3, -1).cumsum(dim=1).reshape(seq_len_expanded.shape) - seq_len_expanded

# What this computes:
# seq_offset_expanded[rank][0][seq] = starting position of sequence 'seq' in rank 'rank's original buffer
#
# [[[0, 10, 15]],    # Rank 0: seq0 starts@0, seq1 starts@10, seq2 starts@15
#  [[0, 8, 20]],     # Rank 1: seq0 starts@0, seq1 starts@8, seq2 starts@20  
#  [[0, 6, 6]]]      # Rank 2: seq0 starts@0, seq1 starts@6, seq2 starts@6 (seq1 has length 0)

# WHAT THIS MEANS FOR REVERSE COMMUNICATION:
# When we receive data in attention layout and want to send it back to MLP layout:
# - Data from R1,seq1 (which was 12 tokens) needs to go back to rank 1 at offset 8
# - Data from R2,seq0 (which was 6 tokens) needs to go back to rank 2 at offset 0
# - etc.

seq_offset_expanded = seq_offset_expanded.transpose(1, 2)  # Back to (3, 3, 1)

# DIFFERENCE BETWEEN seq_begin_offset AND seq_offset_expanded:
# - seq_begin_offset: Where to place incoming data in attention layout (forward direction)
# - seq_offset_expanded: Where to place outgoing data back in MLP layout (reverse direction)
# They serve opposite purposes in the bidirectional communication!
```

#### Step 3.2: Reverse Routing Table
```python
# Create reverse destination arrays
# Count how many sequences each rank will receive
num_received_seqs = seq_to_dst.reshape(-1, 3).sum(0)
# From our seq_to_dst one-hot vectors:
# Rank 0 receives: from R1,seq1 (12 tokens) + from R2,seq0 (6 tokens) = 2 sequences
# Rank 1 receives: from R0,seq0 (10 tokens) + from R1,seq2 (4 tokens) = 2 sequences
# Rank 2 receives: from R0,seq1 (5 tokens) + from R1,seq0 (8 tokens) + from R2,seq2 (9 tokens) = 3 sequences
# num_received_seqs = [2, 2, 3]

max_rev_seqs = 3  # Maximum sequences any rank receives

# Initialize reverse metadata arrays (will be filled by scatter operation)
rev_dst_rank = torch.zeros(3, 3, dtype=torch.int64)    # Which rank to send back to
rev_dst_offset = torch.zeros(3, 3, dtype=torch.int64)  # Offset in original rank's buffer
rev_seq_len = torch.zeros(3, 3, dtype=torch.int64)     # Length of each sequence
```

#### Step 3.3: Vectorized Scatter Operation
```python
# Use scatter to efficiently populate reverse arrays
valid_mask_flat = (global_dispatch != -1).flatten()
# global_dispatch.flatten() = [1, 2, -1, 2, 0, 1, 0, -1, 2]
# valid_mask_flat = [True, True, False, True, True, True, True, False, True]
valid_indices = torch.where(valid_mask_flat)[0]  # [0, 1, 3, 4, 5, 6, 8]

# Get valid source ranks, destinations, offsets, and lengths
valid_src_ranks = seq_rank_expanded.flatten()[valid_indices]        # [0, 0, 1, 1, 1, 2, 2]
valid_dst_ranks = global_dispatch.flatten()[valid_indices]          # [1, 2, 2, 0, 1, 0, 2]
valid_src_offsets = seq_offset_expanded.flatten()[valid_indices]    # [0, 10, 0, 8, 20, 0, 6]
valid_src_seq_lens = seq_len_expanded.flatten()[valid_indices]      # [10, 5, 8, 12, 4, 6, 9]

# Compute sequence-level offsets (not token-level)
# For each destination rank, determine which sequence slot this should occupy
dst_seq_local_offset = [0, 0, 1, 0, 1, 1, 2]  # Sequence order within each destination rank

# Scatter source information to destination positions
global_dst_indices = valid_dst_ranks * 3 + dst_seq_local_offset
# = [1*3+0, 2*3+0, 2*3+1, 0*3+0, 1*3+1, 0*3+1, 2*3+2]
# = [3, 6, 7, 0, 4, 1, 8]

# Fill reverse metadata using scatter
rev_dst_rank.view(-1).scatter_(0, global_dst_indices, valid_src_ranks)
rev_dst_offset.view(-1).scatter_(0, global_dst_indices, valid_src_offsets)  
rev_seq_len.view(-1).scatter_(0, global_dst_indices, valid_src_seq_lens)

# Final reverse metadata:
# rev_dst_rank = [[1, 2, 0],    # Rank 0 will send back to: R1, R2, (unused)
#                 [0, 1, 0],    # Rank 1 will send back to: R0, R1, (unused)  
#                 [0, 1, 2]]    # Rank 2 will send back to: R0, R1, R2
#
# rev_dst_offset = [[8, 0, 0],   # Send to: R1@offset8, R2@offset0, (unused)
#                   [0, 20, 0],  # Send to: R0@offset0, R1@offset20, (unused)
#                   [10, 0, 6]]  # Send to: R0@offset10, R1@offset0, R2@offset6
#
# rev_seq_len = [[12, 6, 0],     # Lengths: 12, 6, (unused)
#                [10, 4, 0],     # Lengths: 10, 4, (unused)
#                [5, 8, 9]]      # Lengths: 5, 8, 9
```

### Phase 4: Final Results Summary

Our concrete example produces the following metadata:

**Forward Metadata:**
- `dst_rank`: Where each sequence goes for attention computation
  ```
  [[[1], [2], [-1]],    # R0: seq0→R1, seq1→R2, seq2→padding
   [[2], [0], [1]],     # R1: seq0→R2, seq1→R0, seq2→R1
   [[0], [-1], [2]]]    # R2: seq0→R0, seq1→padding, seq2→R2
  ```
- `dst_offset`: Where in attention layout destination buffers (seq_begin_offset)
  ```
  [[[0], [5], [0]],     # R0 sequences: seq0@offset0 in R1, seq1@offset5 in R2
   [[5], [12], [10]],   # R1 sequences: seq0@offset5 in R2, seq1@offset12 in R0, seq2@offset10 in R1
   [[12], [0], [13]]]   # R2 sequences: seq0@offset12 in R0, seq1@unused, seq2@offset13 in R2
  ```
- `num_recv_tokens`: Token counts received by each rank in attention layout
  ```
  [[0, 12, 6, 18],      # R0 receives 18 total (12 from R1, 6 from R2)
   [10, 4, 0, 14],      # R1 receives 14 total (10 from R0, 4 from R1)
   [5, 8, 9, 22]]       # R2 receives 22 total (5 from R0, 8 from R1, 9 from R2)
  ```

**Reverse Metadata:**
- `dst_rank`: Where to send data back to original MLP layout ranks
- `dst_offset`: Where in original MLP layout buffers (seq_offset_expanded values)
  ```
  # These offsets target the original sequential layout in each rank:
  # R0 had: [seq0: 0-9, seq1: 10-14, seq2: 15-14 (empty)]
  # R1 had: [seq0: 0-7, seq1: 8-19, seq2: 20-23]  
  # R2 had: [seq0: 0-5, seq1: 6-5 (empty), seq2: 6-14]
  ```
- `seq_len`: Lengths of sequences to send back

**KEY INSIGHT - OFFSET PURPOSES:**
- **Forward offsets** (seq_begin_offset): "Where should I place this incoming sequence in my attention buffer?"
- **Reverse offsets** (seq_offset_expanded): "Where should I place this outgoing sequence back in the original MLP buffer?"

The reverse metadata exactly undoes the forward operation, ensuring perfect reconstruction.

## Key Algorithmic Insights

### One-Hot Transformation
- **Purpose**: Convert sparse dispatch decisions to dense communication matrix
- **Benefit**: Enables vectorized computation of token flows
- **Padding**: +1 offset handles -1 padding values elegantly

### Cumulative Offset Computation
- **Global ordering**: All sequences get globally consistent placement
- **Efficient packing**: No gaps in destination buffers
- **Deterministic**: Same input always produces same layout

### Scatter-Based Reverse
- **Efficiency**: Single scatter operation instead of nested loops
- **Correctness**: Handles arbitrary dispatch patterns correctly
- **Padding-aware**: -1 values automatically excluded

## Memory Layout Implications

### Forward Layout (MLP → Attention)
```
Rank 0: [seq0_part0][seq1_part0][seq2_part0]...
Rank 1: [seq0_part1][seq1_part1][seq2_part1]...
...
```

### Attention Layout (After Forward)  
```
Rank 0: [seqs_for_query0][seqs_for_query1]...
Rank 1: [seqs_for_query0][seqs_for_query1]...
...
```

### Reverse Layout (Attention → MLP)
```
Back to original MLP layout with perfect reconstruction
```

## Validation Properties

### ✅ Mathematical Guarantees
- **Conservation**: Total tokens in = total tokens out
- **Bijection**: Forward + reverse = identity (for queries)
- **Consistency**: Same global ordering across all ranks
- **Completeness**: All non-padding sequences properly routed

### ✅ Performance Characteristics  
- **Vectorized**: Minimal loops, maximum tensor operations
- **Memory efficient**: Single pass through data
- **Deterministic**: Same inputs → same outputs
- **Scalable**: O(total_sequences) complexity

### ❌ Does NOT Handle
- **Dynamic sequences**: All lengths must be known upfront
- **Load balancing**: No attempt to equalize work
- **Communication optimization**: No topology awareness
- **Error recovery**: No fault tolerance

## Usage Patterns

### Query Processing (Simple)
```python
# 2D dispatch tensor (no context parallelism)
seq_len = torch.tensor([[128, 256, 0], [64, 192, 128]])  # 2 ranks, 3 seqs max
dispatch = torch.tensor([[1, 0, -1], [0, 1, 0]])         # destination ranks

fwd_meta, rev_meta = compute_metadata(seq_len, dispatch)
```

### Key-Value Processing (Complex)
```python  
# 3D dispatch tensor (with context parallelism)
seq_len = torch.tensor([[128, 256], [64, 192]])          # 2 ranks, 2 seqs
dispatch = torch.tensor([[[1, 0], [-1, -1]], [[0, 1], [1, 0]]])  # CP degree 2

fwd_meta, rev_meta = compute_metadata(seq_len, dispatch)
```

## Debug Information

When `return_intermediate=True`, returns:
- `tokens_to_dst_per_dispatch`: Token flow matrix
- `seq_to_dst`: One-hot dispatch matrix  
- `num_received_seqs`: Sequence count per rank

---

*This function is the mathematical heart of the distributed attention system, converting abstract dispatch plans into concrete communication instructions.*

------------


```python
fwd_metadata = Metadata(
    # dst_rank: Tensor(world_size, max_num_local_seqs, max_cp_degree)
    #   - dst_rank[src_rank][seq_id] = dst_rank that the sequence [src_rank][seq_id] is dispatched to.
    #   - if the sequence is padding, then dst_rank[src_rank][seq_id] = -1.
    dst_rank=global_dispatch.reshape(dispatch_shape),
    # dst_offset: Tensor(world_size, max_num_local_seqs, max_cp_degree)
    #   - dst_offset[src_rank][seq_id] = the destination buffer offset that the sequence [src_rank][seq_id] is sent to.
    #   - if the sequence is padding, then dst_offset[src_rank][seq_id] = 0.
    dst_offset=seq_begin_offset.reshape(dispatch_shape),
    # seq_len: Tensor(world_size, max_num_local_seqs)
    #   - seq_len[src_rank][seq_id] = the length of the sequence [src_rank][seq_id].
    #   - if the sequence is padding, then seq_len[src_rank][seq_id] = 0.
    seq_len=seq_len,
    # num_recv_tokens: Tensor(world_size, max_num_local_seqs + 1)
    #   - num_recv_tokens[src_rank][seq_id] = the number of tokens that the sequence [src_rank][seq_id] is sent to.
    #   - num_recv_tokens[src_rank][-1] = the total number of tokens that the rank [src_rank] receives.
    #   - seq_id < max_num_local_seqs
    #   - if the sequence is padding, then num_recv_tokens[src_rank][seq_id] = 0.
    num_recv_tokens=num_recv_tokens,
    # num_seqs: Tensor(world_size)
    #   - num_seqs[src_rank] = the number of (valid) sequences that the rank [src_rank] is sending.
    #   - if the sequence is padding, then num_seqs[src_rank] = 0.
    num_seqs=num_seqs,
    # world_size: int
    #   - world_size = the number of ranks.
    world_size=world_size,
    # num_total_recv_tokens: list[int]
    #   - num_total_recv_tokens[dst_rank] = the total number of tokens that the rank [dst_rank] receives.
    num_total_recv_tokens=num_recv_tokens[:, -1].tolist(),
)
```