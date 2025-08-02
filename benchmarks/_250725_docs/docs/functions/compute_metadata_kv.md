# compute_metadata_kv Function

## Purpose
Computes forward and reverse communication metadata specifically for key-value tensors in context parallel attention. Unlike queries which have simple 1:1 dispatch patterns, key-value tensors require complex many-to-many mappings to support causal attention and context parallelism.

## Function Signature
```python
def compute_metadata_kv(
    kv_to_q_mapping: torch.Tensor,     # (world_size, max_num_local_seqs, max_cp_degree, 2)
    kv_to_q_rank: torch.Tensor,        # (world_size, max_num_local_seqs, max_cp_degree)  
    kv_context_size: torch.Tensor,     # (world_size, max_num_local_seqs)
    q_to_num_kv_seq: torch.Tensor,     # (world_size, max_num_local_seqs)
    q_to_num_kv_token: torch.Tensor,   # (world_size, max_num_local_seqs)
    seq_len: torch.Tensor,             # (world_size, max_num_local_seqs)
    num_seqs: torch.Tensor,            # (world_size,)
    # Query metadata (from compute_metadata)
    q_dispatch: torch.Tensor,          # (world_size, max_num_local_seqs)
    q_seq_to_dst: torch.Tensor,        # (world_size, max_num_local_seqs, world_size)
    max_num_local_seqs: int
) -> Tuple[Metadata, Metadata]
```

## Parameters

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `kv_to_q_mapping` | `torch.Tensor` | `(world_size, max_seqs, max_cp, 2)` | Maps each KV shard to query (rank, seq_id) |
| `kv_to_q_rank` | `torch.Tensor` | `(world_size, max_seqs, max_cp)` | Rank of this KV among all KVs mapping to same query |
| `kv_context_size` | `torch.Tensor` | `(world_size, max_seqs)` | Tokens before this KV shard in sequence |
| `q_to_num_kv_seq` | `torch.Tensor` | `(world_size, max_seqs)` | Number of KV shards mapping to each query |
| `q_to_num_kv_token` | `torch.Tensor` | `(world_size, max_seqs)` | Total KV tokens mapping to each query |
| `seq_len` | `torch.Tensor` | `(world_size, max_seqs)` | Length of each sequence shard |
| `num_seqs` | `torch.Tensor` | `(world_size,)` | Number of sequences per rank |
| `q_dispatch` | `torch.Tensor` | `(world_size, max_seqs)` | Query dispatch plan (from compute_metadata) |
| `q_seq_to_dst` | `torch.Tensor` | `(world_size, max_seqs, world_size)` | Query one-hot dispatch matrix |
| `max_num_local_seqs` | `int` | - | Maximum sequences per rank |

## Returns
- `fwd_metadata`: Forward KV communication metadata
- `bwd_metadata`: Backward KV communication metadata with gradient accumulation support

## Key Concepts

### Context Parallelism for KV
Unlike queries which go to a single destination, KV tensors are **broadcasted** to multiple destinations:
- Each KV shard can attend to multiple query shards (causal attention)
- Query shard i can attend to KV shards [0, 1, ..., i] (causal constraint)
- KV must be sent to all ranks that have queries needing it

### KV-to-Query Mapping
The `kv_to_q_mapping` tensor encodes which query each KV shard serves:
- `kv_to_q_mapping[rank][seq][cp][0]` = query rank
- `kv_to_q_mapping[rank][seq][cp][1]` = query sequence index
- `kv_to_q_rank[rank][seq][cp]` = position among KVs serving the same query

## Algorithm Breakdown

Let's walk through the algorithm with a concrete example to understand the complex KV routing logic.

### Example Setup
```python
# Input values for our walkthrough
world_size = 3
max_num_local_seqs = 2
max_cp_degree = 3

# Sequence lengths
seq_len = torch.tensor([
    [6, 4],    # Rank 0: seq0=6 tokens, seq1=4 tokens
    [8, 6],    # Rank 1: seq0=8 tokens, seq1=6 tokens  
    [4, 8]     # Rank 2: seq0=4 tokens, seq1=8 tokens
])

# Query dispatch (from compute_metadata)
q_dispatch = torch.tensor([
    [1, 2],    # R0: seq0→R1, seq1→R2
    [2, 0],    # R1: seq0→R2, seq1→R0
    [0, 1]     # R2: seq0→R0, seq1→R1
])

# KV-to-Query mapping (complex causal relationships)
kv_to_q_mapping = torch.tensor([
    [[[1, 0], [0, 0], [-1, -1]],    # R0,seq0 KV serves: R1,seq0 and R0,seq0
     [[2, 1], [-1, -1], [-1, -1]]],  # R0,seq1 KV serves: R2,seq1 only
    
    [[[2, 0], [1, 0], [0, 0]],       # R1,seq0 KV serves: R2,seq0, R1,seq0, R0,seq0
     [[0, 1], [1, 1], [-1, -1]]],    # R1,seq1 KV serves: R0,seq1, R1,seq1
     
    [[[0, 0], [-1, -1], [-1, -1]],   # R2,seq0 KV serves: R0,seq0 only
     [[1, 1], [2, 1], [-1, -1]]]     # R2,seq1 KV serves: R1,seq1, R2,seq1
])

# KV rank among all KVs serving the same query (for ordering)
kv_to_q_rank = torch.tensor([
    [[1, 0, -1],     # R0,seq0: rank 1 for R1,seq0, rank 0 for R0,seq0
     [0, -1, -1]],   # R0,seq1: rank 0 for R2,seq1
     
    [[2, 1, 0],      # R1,seq0: rank 2 for R2,seq0, rank 1 for R1,seq0, rank 0 for R0,seq0
     [0, 1, -1]],    # R1,seq1: rank 0 for R0,seq1, rank 1 for R1,seq1
     
    [[0, -1, -1],    # R2,seq0: rank 0 for R0,seq0
     [0, 1, -1]]     # R2,seq1: rank 0 for R1,seq1, rank 1 for R2,seq1
])

# Context sizes (tokens before each KV shard)
kv_context_size = torch.tensor([
    [0, 6],    # R0: seq0 starts at 0, seq1 starts at 6
    [0, 8],    # R1: seq0 starts at 0, seq1 starts at 8
    [0, 4]     # R2: seq0 starts at 0, seq1 starts at 4
])
```

Now let's trace through the algorithm:

### Phase 1: Forward KV Destination Computation

#### Step 1.1: Compute KV Destination Ranks
```python
# STEP 1: Create validity mask for non-padding entries
kv_valid_mask = kv_to_q_mapping[..., 0] >= 0
# Shape: (3, 2, 3) - True where KV mapping is valid (not -1)

# STEP 2: Flatten the mapping for efficient indexing
kv_to_q_mapping_flatten = kv_to_q_mapping[..., 0] * max_num_local_seqs + kv_to_q_mapping[..., 1]
# Convert (rank, seq_idx) pairs to flat indices for lookup

# Example for R0,seq0:
# kv_to_q_mapping[0,0,:,0] = [1, 0, -1] (ranks)
# kv_to_q_mapping[0,0,:,1] = [0, 0, -1] (seq indices)
# Flattened: [1*2+0, 0*2+0, -1*2+-1] = [2, 0, 3] (but 3 will be masked)

# STEP 3: Look up where each query goes (from q_dispatch)
kv_dst_rank = (q_dispatch.flatten()[kv_to_q_mapping_flatten].reshape(kv_valid_mask.shape) * kv_valid_mask +
               (kv_valid_mask.int() - 1))

# This computes: for each KV shard, where should it go based on where its target query goes?
# Example: R0,seq0 KV serves R1,seq0 query, and R1,seq0 goes to rank 2
# So R0,seq0 KV should go to rank 2

# Result kv_dst_rank:
# [[[2, 1, -1],     # R0,seq0 KV goes to: R2 (for R1,seq0), R1 (for R0,seq0)
#   [1, -1, -1]],   # R0,seq1 KV goes to: R1 (for R2,seq1)
#  
#  [[0, 2, 1],      # R1,seq0 KV goes to: R0 (for R2,seq0), R2 (for R1,seq0), R1 (for R0,seq0)  
#   [0, 2, -1]],    # R1,seq1 KV goes to: R0 (for R0,seq1), R2 (for R1,seq1)
#   
#  [[1, -1, -1],    # R2,seq0 KV goes to: R1 (for R0,seq0)
#   [2, 1, -1]]]    # R2,seq1 KV goes to: R2 (for R1,seq1), R1 (for R2,seq1)
```

#### Step 1.2: Compute KV Destination Sequence IDs
```python
# STEP 1: Calculate how many KV sequences each query sends to each destination
num_kv_seq_to_dst = (q_seq_to_dst * q_to_num_kv_seq.unsqueeze(-1)).reshape(-1, world_size)
# This gives us the distribution of KV sequences across destination ranks

# STEP 2: Compute sequence-level offsets for KV placement
query_dst_kv_seq_id = exclusive_cumsum(num_kv_seq_to_dst, dim=0) * q_seq_to_dst.bool().reshape(-1, world_size)
query_dst_kv_seq_id = query_dst_kv_seq_id.sum(dim=-1).reshape(world_size, max_num_local_seqs)

# STEP 3: Map KV shards to their destination sequence IDs
kv_dst_seq_id = query_dst_kv_seq_id.flatten()[kv_to_q_mapping_flatten].reshape(kv_valid_mask.shape) * kv_valid_mask
kv_dst_seq_id = kv_dst_seq_id + kv_to_q_rank

# This computes where each KV shard will be placed in the destination rank's sequence order
```

#### Step 1.3: Compute KV Destination Token Offsets
```python
# STEP 1: Calculate token-level flow to destinations
num_token_to_dst = (q_seq_to_dst * q_to_num_kv_token.unsqueeze(-1)).reshape(-1, world_size)

# STEP 2: Compute inter-query-group offsets
query_dst_kv_token_id = exclusive_cumsum(num_token_to_dst, dim=0) * q_seq_to_dst.bool().reshape(-1, world_size)
query_dst_kv_token_id = query_dst_kv_token_id.sum(dim=-1).reshape(world_size, max_num_local_seqs)

# STEP 3: Get base offset for each KV shard
kv_dst_token_offset = query_dst_kv_token_id.flatten()[kv_to_q_mapping_flatten].reshape(kv_valid_mask.shape)

# STEP 4: Add intra-query-group offset (context position)
kv_dst_token_offset = (kv_dst_token_offset + kv_context_size.unsqueeze(-1)) * kv_valid_mask

# This gives the exact token position where each KV shard should be placed
# Example: If a KV shard has context_size=10 and the base offset is 50,
# it will be placed at token position 60 in the destination buffer
```

#### Step 1.4: Compute Receive Token Counts
```python
# STEP 1: Calculate tokens sent to each destination
num_send_tokens = num_token_to_dst.reshape(world_size, -1, world_size).sum(dim=1)

# STEP 2: Transpose to get receive perspective
num_recv_tokens = num_send_tokens.transpose(0, 1)
num_total_recv_tokens = num_recv_tokens.sum(dim=1)

# STEP 3: Add total column
num_recv_tokens = torch.concat([num_recv_tokens, num_total_recv_tokens.unsqueeze(1)], dim=1)

# Create forward metadata
fwd_metadata = Metadata(
    dst_rank=kv_dst_rank,
    dst_offset=kv_dst_token_offset,
    seq_len=seq_len,
    num_recv_tokens=num_recv_tokens,
    num_seqs=num_seqs,
    world_size=world_size,
    num_total_recv_tokens=num_total_recv_tokens.tolist()
)
```

### Phase 2: Backward KV Metadata Computation

#### Step 2.1: Reverse Routing Table Setup
```python
# STEP 1: Count sequences each rank will receive in backward pass
num_seq_bwd = num_kv_seq_to_dst.sum(dim=0)
max_num_local_seqs_rev = int(num_seq_bwd.max().item())

# STEP 2: Create validity mask for reverse sequences
rev_seq_valid_mask = (torch.arange(max_num_local_seqs_rev).view(1, -1).repeat(world_size, 1)
                      < num_seq_bwd.unsqueeze(1))

# STEP 3: Compute global sequence IDs for scatter operation
kv_dst_global_seq_id = kv_dst_seq_id + kv_dst_rank * max_num_local_seqs_rev
src_rank_expand = torch.arange(world_size).view(-1, 1, 1).expand_as(kv_dst_global_seq_id)
```

#### Step 2.2: Gradient Buffer Layout
```python
# SPECIAL LAYOUT: (cp_degree, num_tokens, hidden_dim) for efficient gradient accumulation
# This layout makes copying CP replicas consecutive in memory

# STEP 1: Compute inter-replica offsets (between different CP degrees)
inter_replica_offset = (
    torch.arange(max_cp_degree).reshape(1, -1) * seq_len.sum(1).reshape(-1, 1)
).unsqueeze(1)

# STEP 2: Compute intra-replica offsets (within same CP degree)  
intra_replica_offset = exclusive_cumsum(seq_len, dim=1).unsqueeze(-1)

# STEP 3: Combine offsets
src_kv_offset = inter_replica_offset + intra_replica_offset

# Example layout for rank with sequences [6, 4] and max_cp_degree=3:
# CP0: [seq0: 0-5, seq1: 6-9]
# CP1: [seq0: 10-15, seq1: 16-19]  
# CP2: [seq0: 20-25, seq1: 26-29]
```

#### Step 2.3: Populate Reverse Metadata
```python
# STEP 1: Create reverse destination rank mapping
rev_kv_dst_rank = torch.empty((world_size, max_num_local_seqs_rev), dtype=torch.int64)
rev_kv_dst_rank = index_put_with_neg_padding_1d(
    rev_kv_dst_rank, src_rank_expand, kv_dst_global_seq_id
)

# STEP 2: Create gradient buffer offset mapping
src_kv_grad_buffer_offset = torch.zeros_like(rev_kv_dst_rank)
src_kv_grad_buffer_offset = index_put_with_neg_padding_1d(
    src_kv_grad_buffer_offset, src_kv_offset, kv_dst_global_seq_id
)

# STEP 3: Create sequence length mapping
rev_kv_seqlen = torch.zeros_like(rev_kv_dst_rank)
src_kv_seqlen = fwd_metadata.seq_len.unsqueeze(-1).repeat(1, 1, max_cp_degree)
rev_kv_seqlen = index_put_with_neg_padding_1d(
    rev_kv_seqlen, src_kv_seqlen, kv_dst_global_seq_id
)

# Create backward metadata
bwd_metadata = Metadata(
    dst_rank=rev_kv_dst_rank,
    dst_offset=src_kv_grad_buffer_offset,
    seq_len=rev_kv_seqlen,
    num_recv_tokens=num_send_tokens,  # Forward send = backward receive
    seq_recv_mask=kv_valid_mask,      # Which sequences are active
    recv_seq_lens=fwd_metadata.seq_len,
    num_seqs=num_seq_bwd,
    world_size=world_size
)
```

### Phase 3: Final Results Summary

Our example produces the following KV metadata:

**Forward KV Metadata:**
- `dst_rank`: Where each KV shard goes (based on query destinations)
  ```python
  # Each KV can go to multiple destinations due to context parallelism
  [[[2, 1, -1],     # R0,seq0 → R2 and R1  
    [1, -1, -1]],   # R0,seq1 → R1 only
   [[0, 2, 1],      # R1,seq0 → R0, R2, and R1
    [0, 2, -1]],    # R1,seq1 → R0 and R2
   [[1, -1, -1],    # R2,seq0 → R1 only
    [2, 1, -1]]]    # R2,seq1 → R2 and R1
  ```

- `dst_offset`: Token positions in destination buffers
  ```python
  # Computed based on context sizes and query group offsets
  # Each offset ensures proper causal ordering at destination
  ```

**Backward KV Metadata:**
- Special gradient accumulation layout: `(cp_degree, num_tokens, hidden_dim)`
- Ensures efficient summation of overlapping KV gradients
- Maps attention layout back to MLP layout with proper deduplication

## Key Differences from Query Metadata

### Complexity Factors
1. **Many-to-Many Mapping**: Each KV can serve multiple queries (causal attention)
2. **Context Parallelism**: KV shards are broadcast to multiple destinations
3. **Causal Constraints**: KV shard i can only serve query shards [0, i]
4. **Gradient Accumulation**: Backward pass must sum overlapping gradients

### Special Considerations
1. **Memory Layout**: Backward pass uses special layout for efficient gradient accumulation
2. **Sequence Ordering**: KV placement must respect causal attention patterns
3. **Duplicate Handling**: Forward broadcast → backward deduplication
4. **Context Tracking**: Each KV knows its position in the overall sequence

## Performance Implications

### Forward Pass
- **Memory overhead**: KV data replicated across multiple destinations
- **Communication volume**: Higher than queries due to broadcasting
- **Load balancing**: May be uneven due to causal constraints

### Backward Pass  
- **Gradient accumulation**: Requires summation of multiple sources
- **Memory layout optimization**: Special layout reduces gather/scatter overhead
- **Synchronization**: Must coordinate gradient collection from multiple sources

---

*This function handles the most complex routing logic in distributed attention, managing the intricate relationships between KV shards and query shards while respecting causal attention constraints and enabling efficient gradient accumulation.*