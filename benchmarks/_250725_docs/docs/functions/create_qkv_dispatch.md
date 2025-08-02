# create_qkv_dispatch Function

## Purpose
Generates complete QKV (Query, Key, Value) dispatch plans for distributed context parallel attention. This function creates randomized but realistic test scenarios that simulate complex distributed attention patterns with context parallelism, causal constraints, and variable sequence lengths.

## Function Signature
```python
def create_qkv_dispatch(
    world_size: int,           # Number of GPUs/ranks
    total_seq_len: int,        # Total sequence length across all ranks
    num_seqs: int,             # Number of sequences per rank
    max_cp_degree: int         # Maximum context parallelism degree
) -> Tuple[Metadata, Metadata, Metadata, Metadata, Tuple]
```

## Parameters

| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| `world_size` | `int` | Number of GPU ranks in the system | ≥ 1 |
| `total_seq_len` | `int` | Total tokens across all sequences | Must be divisible by `max_cp_degree` |
| `num_seqs` | `int` | Number of sequences per rank | ≥ 1 |
| `max_cp_degree` | `int` | Maximum context parallelism degree | Must be power of 2 |

## Returns
A tuple of 5 metadata objects:
1. `fwd_q_metadata`: Forward query communication metadata
2. `rev_q_metadata`: Reverse query communication metadata  
3. `fwd_k_metadata`: Forward key-value communication metadata
4. `rev_k_metadata`: Reverse key-value communication metadata
5. `attention_metadata`: Attention layout sequence parameters

## Key Concepts

### Context Parallelism (CP)
Context parallelism splits long sequences across multiple GPUs to handle sequences longer than what fits on a single GPU:
- Each sequence can be split into 1, 2, 4, 8, ... shards (powers of 2)
- Each shard is processed on a different GPU
- Causal attention requires KV data from earlier shards

### Causal Attention Constraints
For causal attention, query shard `i` can only attend to KV shards `[0, 1, ..., i]`:
- This creates asymmetric communication patterns
- KV data must be broadcast to multiple query shards
- Earlier KV shards are needed by more query shards

### Random Test Generation
The function generates realistic but randomized scenarios:
- Random sequence lengths (respecting total length constraint)
- Random CP degrees for each sequence (powers of 2)
- Random destination assignments for query shards

## Algorithm Breakdown

Let's walk through the algorithm with a concrete example to understand the complex dispatch generation.

### Example Setup
```python
# Input parameters for our walkthrough
world_size = 3
total_seq_len = 96  # Must be divisible by max_cp_degree
num_seqs = 2        # 2 sequences per rank
max_cp_degree = 4   # Maximum CP degree of 4

# This gives us _num_tokens_shard = 96 // 4 = 24 tokens per shard
```

### Phase 1: Generate Base Sequence Lengths

```python
# STEP 1: Generate random sequence lengths per rank
_num_tokens_shard = total_seq_len // max_cp_degree  # 24
seq_lens = gen_seq_lens(world_size, num_seqs, _num_tokens_shard).long()
# Example result:
# seq_lens = tensor([
#     [8, 16],   # Rank 0: seq0=8, seq1=16 (total=24)  
#     [12, 12],  # Rank 1: seq0=12, seq1=12 (total=24)
#     [6, 18]    # Rank 2: seq0=6, seq1=18 (total=24)
# ])

# STEP 2: Scale by max_cp_degree to ensure divisibility
seq_lens *= max_cp_degree  # Each length now divisible by any CP degree ≤ 4
# seq_lens = tensor([
#     [32, 64],   # Rank 0: seq0=32, seq1=64
#     [48, 48],   # Rank 1: seq0=48, seq1=48  
#     [24, 72]    # Rank 2: seq0=24, seq1=72
# ])
```

### Phase 2: Generate Context Parallelism Configuration

```python
# STEP 1: Generate random CP degrees (powers of 2)
log_cp_num = torch.randint(0, int(math.log2(max_cp_degree)) + 1, (world_size, num_seqs))
# Example: log_cp_num = [[1, 2], [0, 2], [2, 1]]  # log2 values
cp_num = torch.pow(2, log_cp_num)
# cp_num = tensor([
#     [2, 4],  # Rank 0: seq0 has CP=2, seq1 has CP=4
#     [1, 4],  # Rank 1: seq0 has CP=1, seq1 has CP=4
#     [4, 2]   # Rank 2: seq0 has CP=4, seq1 has CP=2
# ])

# STEP 2: Generate random destination ranks for each CP shard
cp_dst_helper = torch.rand((world_size, num_seqs, world_size)).argsort(dim=2)
cp_dst = cp_dst_helper[:, :, :max_cp_degree]  # Take first max_cp_degree ranks

# STEP 3: Mask unused CP destinations
mask = torch.arange(max_cp_degree).expand(world_size, num_seqs, max_cp_degree)
cp_num_expanded = cp_num.unsqueeze(-1)
mask = mask >= cp_num_expanded
cp_dst[mask] = -1  # Mark unused slots as padding

# Example result:
# cp_dst = tensor([
#     [[1, 2, -1, -1],   # R0,seq0: 2 shards go to ranks 1,2
#      [0, 1, 2, -1]],   # R0,seq1: 4 shards go to ranks 0,1,2,-1
#     [[0, -1, -1, -1],  # R1,seq0: 1 shard goes to rank 0
#      [2, 0, 1, -1]],   # R1,seq1: 4 shards go to ranks 2,0,1,-1
#     [[1, 0, 2, -1],    # R2,seq0: 4 shards go to ranks 1,0,2,-1
#      [0, 1, -1, -1]]   # R2,seq1: 2 shards go to ranks 0,1
# ])
```

### Phase 3: Build Global Dispatch Data Structures

This phase creates tensors that map from the original sequence-based organization to a **CP shard-based organization**. Instead of indexing by `[rank][sequence_id]`, we now index by `[rank][cp_shard_id]` where CP shards are flattened across all sequences on a rank.

```python
# STEP 1: Calculate total CP shards per rank
num_cp_shards = cp_num.sum(dim=1)  # [6, 5, 6] shards per rank
pad_len = torch.max(num_cp_shards)  # 6 (maximum shards any rank has)

# STEP 2: Initialize data structures with detailed tensor indexing explanation

# cp_seq_lens[rank_id][cp_shard_id] = number of tokens in this CP shard
# - rank_id: which GPU rank (0 to world_size-1)
# - cp_shard_id: flattened CP shard index within this rank (0 to pad_len-1)
# - value: number of tokens in this specific CP shard
cp_seq_lens = torch.zeros(world_size, pad_len, dtype=torch.int64)

# cp_query_dst[rank_id][cp_shard_id] = destination rank for this query CP shard
# - rank_id: source rank where this query shard originates
# - cp_shard_id: flattened CP shard index within source rank
# - value: destination rank where this query shard will be sent (-1 for padding)
cp_query_dst = torch.ones(world_size, pad_len, dtype=torch.int64) * -1

# kv_to_q_mapping[rank_id][cp_shard_id][cp_index][0/1] = query location served by this KV
# - rank_id: source rank where this KV shard originates  
# - cp_shard_id: flattened KV shard index within source rank
# - cp_index: which query this KV serves (0 to max_cp_degree-1, due to causal attention)
# - 0/1: [query_rank, query_shard_id] - the (rank, local_shard_id) of served query
# - value: -1 for padding, otherwise valid query coordinates
kv_to_q_mapping = torch.ones((world_size, pad_len, max_cp_degree, 2), dtype=torch.int64) * -1

# kv_to_q_rank[rank_id][cp_shard_id][cp_index] = ordering rank among KVs serving same query
# - rank_id: source rank where this KV shard originates
# - cp_shard_id: flattened KV shard index within source rank  
# - cp_index: which query this KV serves (same as above)
# - value: 0, 1, 2, ... indicating this KV's position among all KVs serving the same query
#         (used for proper causal ordering at destination)
kv_to_q_rank = torch.ones((world_size, pad_len, max_cp_degree), dtype=torch.int64) * -1

# kv_context_size[rank_id][cp_shard_id] = tokens before this KV shard in its original sequence
# - rank_id: source rank where this KV shard originates
# - cp_shard_id: flattened KV shard index within source rank
# - value: number of tokens that appear before this shard in the complete sequence
#         (0 for first shard, seq_shard_len for second shard, etc.)
kv_context_size = torch.zeros((world_size, pad_len), dtype=torch.int64)

# num_kv_to_q[rank_id][cp_shard_id] = number of KV shards (including self) available to this query
# - rank_id: source rank (interpreting as query rank in this context)
# - cp_shard_id: flattened query shard index within source rank
# - value: number of KV shards this query can attend to (1, 2, 3, ... due to causal constraint)
#         (query shard i can attend to KV shards [0, 1, ..., i])
num_kv_to_q = torch.zeros((world_size, pad_len), dtype=torch.int64)

# STEP 3: Compute cumulative CP shard offsets for sequence-to-shard mapping
num_cul_cp_shards = exclusive_cumsum(cp_num, dim=1)
# Example: For cp_num = [[2,4], [1,4], [4,2]]
# num_cul_cp_shards = [[0,2], [0,1], [0,4]]  # Starting shard index for each sequence
# 
# MEANING: num_cul_cp_shards[rank][seq] = starting CP shard index for sequence seq on rank
# - This maps from sequence-based indexing to flattened CP shard indexing
# - Sequence 0 starts at shard 0, sequence 1 starts at shard cp_num[rank][0], etc.
```

### Critical Index Transformation

**From Sequence Space to CP Shard Space:**
```python
# Original organization: [rank][sequence_id] 
# New organization: [rank][cp_shard_id] where cp_shard_id is flattened

# For a sequence with CP degree N, it creates N consecutive CP shards:
# sequence (rank=i, seq=j, cp_degree=N) → cp_shards at indices [start, start+1, ..., start+N-1]
# where start = num_cul_cp_shards[i][j]

# Example transformation:
# Rank 0: seq0(CP=2) + seq1(CP=4) → 6 CP shards total
# - seq0 shards: cp_shard_id = [0, 1] 
# - seq1 shards: cp_shard_id = [2, 3, 4, 5]
# - Remaining slots [6, 7, ...] are padding (-1)
```

### Padding and Masking Rules

**Padding Values (-1):**
- `cp_query_dst[rank][shard] = -1` → this CP shard slot is unused (padding)
- `kv_to_q_mapping[rank][shard][cp][0/1] = -1` → this KV doesn't serve this query slot
- `kv_to_q_rank[rank][shard][cp] = -1` → this ranking slot is unused

**Valid Value Ranges:**
- `cp_query_dst`: 0 to world_size-1 (destination ranks)
- `kv_to_q_mapping[..., 0]`: 0 to world_size-1 (query ranks)  
- `kv_to_q_mapping[..., 1]`: 0 to pad_len-1 (query CP shard indices)
- `kv_to_q_rank`: 0 to max_cp_degree-1 (ordering among KVs)
- `kv_context_size`: 0 to sequence_length-seq_shard_len (token offsets)
- `num_kv_to_q`: 1 to max_cp_degree (number of available KV shards)

### Phase 4: Generate Detailed KV-to-Query Mappings

This is the most complex part - for each rank and sequence, we build the causal attention mappings. Let's trace through a concrete example:

```python
# CONCRETE EXAMPLE: Rank 0, Sequence 0 (CP=2, seq_len=32)
# This sequence gets split into 2 CP shards of 16 tokens each

i, j = 0, 0  # rank=0, sequence=0
num_cp = 2   # CP degree for this sequence
seq_len = 32
seq_shard_len = 16  # 32 / 2 tokens per shard

# Starting CP shard index for this sequence on this rank
start_shard_idx = num_cul_cp_shards[0][0] = 0  # First sequence starts at shard 0

# STEP 1: Generate sequence lengths for each CP shard
cp_seq_lens_local.append([16, 16])  # 2 shards of 16 tokens each
# These will be stored at: cp_seq_lens[0][0] = 16, cp_seq_lens[0][1] = 16

# STEP 2: Generate query destinations from the random assignment
cp_query_dst_local.append([1, 2])  # From cp_dst[0,0,:2] = [1,2]
# Meaning: cp_query_dst[0][0] = 1 (shard 0 goes to rank 1)
#         cp_query_dst[0][1] = 2 (shard 1 goes to rank 2)

# STEP 3: Build KV-to-Query mapping (CAUSAL ATTENTION LOGIC)
row_indices = torch.arange(2).view(-1, 1)  # [[0], [1]] - KV shard indices
col_indices = torch.arange(4).view(1, -1)  # [0,1,2,3] - potential query positions

# CAUSAL CONSTRAINT: KV shard i serves query shards [i, i+1, i+2, ...]
mask = col_indices < (num_cp - row_indices)
# mask = [[True, True, False, False],   # KV shard 0 serves queries at positions 0,1
#         [True, False, False, False]]  # KV shard 1 serves query at position 1 only

# Build the mapping tensor: kv_to_q_mapping[kv_shard][query_position][rank_or_shard_id]
kv_to_q_mapping_seq = torch.empty((2, 4, 2), dtype=torch.int64)

# Channel 0: Query rank (all queries in this example are on rank 0)
kv_to_q_mapping_seq[..., 0] = torch.where(mask, 0, -1)

# Channel 1: Global query shard index 
vals_ch1 = row_indices + col_indices + num_cul_cp_shards[0][0]  # Global indices
kv_to_q_mapping_seq[..., 1] = torch.where(mask, vals_ch1, -1)

# RESULT: kv_to_q_mapping_seq shape (2, 4, 2)
# kv_to_q_mapping_seq[0] = [[0,0], [0,1], [-1,-1], [-1,-1]]  # KV shard 0 mappings
# kv_to_q_mapping_seq[1] = [[0,1], [-1,-1], [-1,-1], [-1,-1]] # KV shard 1 mappings
#
# INTERPRETATION:
# - kv_to_q_mapping[0][0][0] = [0,0] → KV shard 0 serves query at (rank=0, shard=0)
# - kv_to_q_mapping[0][0][1] = [0,1] → KV shard 0 serves query at (rank=0, shard=1) 
# - kv_to_q_mapping[0][1][0] = [0,1] → KV shard 1 serves query at (rank=0, shard=1)
# - All [-1,-1] entries are padding (KV doesn't serve those query positions)

# STEP 4: Generate KV-to-Query rank (ordering among multiple KVs serving same query)
kv_to_q_rank_seq = torch.arange(2).view(-1, 1).repeat(1, 4) * mask + (mask.int() - 1)
# Result shape (2, 4):
# kv_to_q_rank_seq = [[0, 1, -1, -1],   # KV shard 0: rank 0 for query 0, rank 1 for query 1
#                     [0, -1, -1, -1]]  # KV shard 1: rank 0 for query 1
#
# INTERPRETATION:
# - When query 1 needs KV data, it gets KV shard 0 (rank=1) and KV shard 1 (rank=0)
# - This ordering ensures proper causal attention assembly at the destination

# STEP 5: Generate context sizes (tokens before each KV shard within the sequence)
kv_context_size_seq = torch.arange(2) * 16  # [0, 16]
# INTERPRETATION:
# - kv_context_size[0][0] = 0  → KV shard 0 has 0 tokens before it (sequence start)
# - kv_context_size[0][1] = 16 → KV shard 1 has 16 tokens before it (after shard 0)

# STEP 6: Generate KV count per query (total KV shards available to each query due to causality)
num_kv_to_q_seq = torch.arange(2) + 1  # [1, 2]  
# INTERPRETATION:
# - Query shard 0 can access 1 KV shard (shard 0 only - causal constraint)
# - Query shard 1 can access 2 KV shards (shards 0,1 - can see previous + current)

# FINAL STORAGE: These local arrays are concatenated and stored in global tensors:
# cp_seq_lens[0][0:2] = [16, 16]
# cp_query_dst[0][0:2] = [1, 2] 
# kv_to_q_mapping[0][0:2] = kv_to_q_mapping_seq
# kv_to_q_rank[0][0:2] = kv_to_q_rank_seq
# kv_context_size[0][0:2] = [0, 16]
# num_kv_to_q[0][0:2] = [1, 2]
```

### Phase 5: Consolidate and Generate Final Metadata

```python
# STEP 1: Concatenate all local data for each rank and store in global tensors
# After processing all ranks and sequences, our global tensors contain:

# EXAMPLE FINAL TENSORS (for our 3-rank, 2-sequence example):
# cp_seq_lens[rank][cp_shard_id] = tokens in each CP shard
# cp_seq_lens = tensor([
#     [16, 16, 16, 16, 16, 16],  # Rank 0: 6 CP shards (seq0: 2 shards, seq1: 4 shards)
#     [48, 12, 12, 12, 12, -1],  # Rank 1: 5 CP shards (seq0: 1 shard, seq1: 4 shards) 
#     [6, 6, 6, 6, 36, 36]       # Rank 2: 6 CP shards (seq0: 4 shards, seq1: 2 shards)
# ])

# cp_query_dst[rank][cp_shard_id] = destination rank for each query CP shard
# cp_query_dst = tensor([
#     [1, 2, 0, 1, 2, -1],       # Rank 0 query destinations
#     [0, 2, 0, 1, -1, -1],      # Rank 1 query destinations
#     [1, 0, 2, -1, 0, 1]        # Rank 2 query destinations  
# ])

# kv_to_q_mapping[rank][cp_shard_id][cp_idx][0/1] = (query_rank, query_shard_id)
# Complex 4D tensor showing which queries each KV shard serves

# kv_context_size[rank][cp_shard_id] = tokens before each KV shard
# kv_context_size = tensor([
#     [0, 16, 0, 16, 32, 48],    # Rank 0: context offsets
#     [0, 0, 8, 16, 24, 0],      # Rank 1: context offsets
#     [0, 6, 12, 18, 0, 36]      # Rank 2: context offsets
# ])

# STEP 2: Calculate total KV tokens available to each query
num_total_kv_to_q = kv_context_size + cp_seq_lens
# For each query shard: context_before + current_shard = total_available_tokens
# This is used by attention kernels to know the total KV context size

# STEP 3: Generate query metadata using compute_metadata
fwd_q_metadata, rev_q_metadata, intermediates = compute_metadata(
    cp_seq_lens,     # Shard lengths: (world_size, pad_len) 
    cp_query_dst,    # Query destinations: (world_size, pad_len)
    return_intermediate=True
)
# Returns forward/reverse Metadata objects for query communication
# Intermediate results include q_seq_to_dst (one-hot dispatch matrix)

_, q_seq_to_dst, _ = intermediates

# STEP 4: Generate KV metadata using compute_metadata_kv (the complex function!)
fwd_k_metadata, rev_k_metadata = compute_metadata_kv(
    kv_to_q_mapping,    # KV→Query mappings: (world_size, pad_len, max_cp_degree, 2)  
    kv_to_q_rank,       # KV ordering: (world_size, pad_len, max_cp_degree)
    kv_context_size,    # Context sizes: (world_size, pad_len)
    num_kv_to_q,        # KV counts per query: (world_size, pad_len)
    num_total_kv_to_q,  # Total KV tokens per query: (world_size, pad_len) 
    cp_seq_lens,        # Shard lengths: (world_size, pad_len)
    num_cp_shards,      # CP shards per rank: (world_size,)
    cp_query_dst,       # Query destinations: (world_size, pad_len)
    q_seq_to_dst.squeeze(2),  # Query one-hot matrix: (world_size, pad_len, world_size)
    pad_len             # Maximum shards per rank: int
)
# Returns forward/reverse Metadata objects for KV communication with context parallelism

# STEP 5: Generate attention layout metadata for GPU kernels
attention_metadata = compute_attn_layout_seqlens(
    cp_seq_lens,        # Shard lengths for attention layout
    num_total_kv_to_q,  # Total KV context per query
    cp_query_dst        # Query dispatch destinations
)
# Returns PackedSeqParams-like structure for attention kernel execution

# FINAL RETURN: 5 metadata objects describing complete distributed attention setup
return (
    fwd_q_metadata,      # Forward query communication
    rev_q_metadata,      # Reverse query communication  
    fwd_k_metadata,      # Forward KV communication
    rev_k_metadata,      # Reverse KV communication
    attention_metadata   # Attention kernel parameters
)
```

### Complete Tensor State Summary

After Phase 5 completes, we have transformed from simple sequence-based inputs to complex CP shard-based communication plans:

**Input Transformation:**
```python
# INPUT: Simple sequence organization
seq_lens = [[32, 64], [48, 48], [24, 72]]    # (world_size, num_seqs)
cp_num = [[2, 4], [1, 4], [4, 2]]           # (world_size, num_seqs)

# OUTPUT: Complex CP shard organization  
cp_seq_lens = [[16,16,16,16,16,16], ...]     # (world_size, pad_len)
cp_query_dst = [[1,2,0,1,2,-1], ...]        # (world_size, pad_len)
kv_to_q_mapping = [[[...]], ...]            # (world_size, pad_len, max_cp_degree, 2)
# + 3 more complex tensors describing causal attention relationships
```

**Metadata Object Contents:**
- **fwd_q_metadata**: Where each query shard goes, token counts, sequence organization
- **rev_q_metadata**: How to reassemble query gradients back to MLP layout  
- **fwd_k_metadata**: Complex KV broadcasting to multiple query destinations
- **rev_k_metadata**: Gradient accumulation for overlapping KV contributions
- **attention_metadata**: GPU kernel parameters for packed attention computation

## Virtual Execution Example

Let's trace through a complete small example:

### Setup
```python
world_size = 2
total_seq_len = 16  # Small example
num_seqs = 1
max_cp_degree = 2
```

### Execution Trace
```python
# Phase 1: Generate sequences
_num_tokens_shard = 16 // 2 = 8
seq_lens = gen_seq_lens(2, 1, 8) * 2
# Possible result: seq_lens = [[16], [16]]  # 1 sequence per rank, 16 tokens each

# Phase 2: Generate CP configuration  
# Assume: cp_num = [[2], [2]]  # Both sequences use CP=2
# Assume: cp_dst = [[[1, 0]], [[0, 1]]]  # Random destinations

# Phase 3: Build structures
num_cp_shards = [2, 2]  # Both ranks have 2 CP shards
pad_len = 2

# Phase 4: Generate mappings for each rank
# Rank 0, Sequence 0:
#   - 2 CP shards of 8 tokens each
#   - Shard 0 goes to rank 1, shard 1 goes to rank 0
#   - KV shard 0 serves query shards [0,1], KV shard 1 serves query shard [1]
#   - Context sizes: [0, 8]

# Rank 1, Sequence 0: 
#   - 2 CP shards of 8 tokens each
#   - Shard 0 goes to rank 0, shard 1 goes to rank 1  
#   - Similar causal mappings

# Phase 5: Final metadata generation
# Results in 5 metadata objects describing complete communication plan
```

Allocating the global tensors:


```python
num_cp_shards = cp_num.sum(dim=1)
pad_len = torch.max(num_cp_shards)
cp_seq_lens = torch.zeros(world_size, pad_len, dtype=torch.int64)
cp_query_dst = torch.ones(world_size, pad_len, dtype=torch.int64) * -1
kv_to_q_mapping = torch.ones((world_size, pad_len, max_cp_degree, 2), dtype=torch.int64) * -1
kv_to_q_rank = torch.ones((world_size, pad_len, max_cp_degree), dtype=torch.int64) * -1
kv_context_size = torch.zeros((world_size, pad_len), dtype=torch.int64)
num_kv_to_q = torch.zeros((world_size, pad_len), dtype=torch.int64)

# cumulative number of cp shards before this one.
num_cul_cp_shards = exclusive_cumsum(cp_num, dim=1)
```

```python
> /workspace/d2/tests/test_util.py(279)create_qkv_dispatch()
-> fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata, attention_metadata
(Pdb) world_size
4
(Pdb) total_seq_len
1024
(Pdb) num_seqs
2
(Pdb) max_cp_degree
4
(Pdb) seq_lens
tensor([[424, 600],
        [624, 400],
        [556, 468],
        [324, 700]])
(Pdb) cp_num
tensor([[1, 1],
        [1, 2],
        [2, 4],
        [4, 1]])
(Pdb) cp_dst
tensor([[[ 1, -1, -1, -1],
         [ 3, -1, -1, -1]],

        [[ 2, -1, -1, -1],
         [ 3,  1, -1, -1]],

        [[ 3,  0, -1, -1],
         [ 3,  2,  0,  1]],

        [[ 3,  1,  0,  2],
         [ 1, -1, -1, -1]]])
(Pdb) cp_seq_lens
tensor([[424, 600,   0,   0,   0,   0],
        [624, 200, 200,   0,   0,   0],
        [278, 278, 117, 117, 117, 117],
        [ 81,  81,  81,  81, 700,   0]])
(Pdb) cp_seq_lens.shape
torch.Size([4, 6])
(Pdb) num_cp_shards
tensor([2, 3, 6, 5])
(Pdb) pad_len
tensor(6)
(Pdb) cp_query_dst
tensor([[ 1,  3, -1, -1, -1, -1],
        [ 2,  3,  1, -1, -1, -1],
        [ 3,  0,  3,  2,  0,  1],
        [ 3,  1,  0,  2,  1, -1]])
(Pdb) cp_query_dst.shape
torch.Size([4, 6])
(Pdb) kv_to_q_mapping
tensor([[[[ 0,  0],
          [-1, -1],
          [-1, -1],
          [-1, -1]],

         [[ 0,  1],
          [-1, -1],
          [-1, -1],
          [-1, -1]],

         [[-1, -1],
          [-1, -1],
          [-1, -1],
          [-1, -1]],

         [[-1, -1],
          [-1, -1],
          [-1, -1],
          [-1, -1]],

         [[-1, -1],
          [-1, -1],
          [-1, -1],
          [-1, -1]],

         [[-1, -1],
          [-1, -1],
          [-1, -1],
          [-1, -1]]],


        [[[ 1,  0],
          [-1, -1],
          [-1, -1],
          [-1, -1]],

         [[ 1,  1],
          [ 1,  2],
          [-1, -1],
          [-1, -1]],

         [[ 1,  2],
          [-1, -1],
          [-1, -1],
          [-1, -1]],

         [[-1, -1],
          [-1, -1],
          [-1, -1],
          [-1, -1]],

         [[-1, -1],
          [-1, -1],
          [-1, -1],
          [-1, -1]],

         [[-1, -1],
          [-1, -1],
          [-1, -1],
          [-1, -1]]],


        [[[ 2,  0],
          [ 2,  1],
          [-1, -1],
          [-1, -1]],

         [[ 2,  1],
          [-1, -1],
          [-1, -1],
          [-1, -1]],

         [[ 2,  2],
          [ 2,  3],
          [ 2,  4],
          [ 2,  5]],

         [[ 2,  3],
          [ 2,  4],
          [ 2,  5],
          [-1, -1]],

         [[ 2,  4],
          [ 2,  5],
          [-1, -1],
          [-1, -1]],

         [[ 2,  5],
          [-1, -1],
          [-1, -1],
          [-1, -1]]],


        [[[ 3,  0],
          [ 3,  1],
          [ 3,  2],
          [ 3,  3]],

         [[ 3,  1],
          [ 3,  2],
          [ 3,  3],
          [-1, -1]],

         [[ 3,  2],
          [ 3,  3],
          [-1, -1],
          [-1, -1]],

         [[ 3,  3],
          [-1, -1],
          [-1, -1],
          [-1, -1]],

         [[ 3,  4],
          [-1, -1],
          [-1, -1],
          [-1, -1]],

         [[-1, -1],
          [-1, -1],
          [-1, -1],
          [-1, -1]]]])
(Pdb) kv_to_q_mapping.shape
torch.Size([4, 6, 4, 2])
(Pdb) kv_to_q_rank
tensor([[[ 0, -1, -1, -1],
         [ 0, -1, -1, -1],
         [-1, -1, -1, -1],
         [-1, -1, -1, -1],
         [-1, -1, -1, -1],
         [-1, -1, -1, -1]],

        [[ 0, -1, -1, -1],
         [ 0,  0, -1, -1],
         [ 1, -1, -1, -1],
         [-1, -1, -1, -1],
         [-1, -1, -1, -1],
         [-1, -1, -1, -1]],

        [[ 0,  0, -1, -1],
         [ 1, -1, -1, -1],
         [ 0,  0,  0,  0],
         [ 1,  1,  1, -1],
         [ 2,  2, -1, -1],
         [ 3, -1, -1, -1]],

        [[ 0,  0,  0,  0],
         [ 1,  1,  1, -1],
         [ 2,  2, -1, -1],
         [ 3, -1, -1, -1],
         [ 0, -1, -1, -1],
         [-1, -1, -1, -1]]])
(Pdb) kv_to_q_rank.shape
torch.Size([4, 6, 4])
(Pdb) kv_context_size
tensor([[  0,   0,   0,   0,   0,   0],
        [  0,   0, 200,   0,   0,   0],
        [  0, 278,   0, 117, 234, 351],
        [  0,  81, 162, 243,   0,   0]])
(Pdb) kv_context_size.shape
torch.Size([4, 6])
(Pdb) num_kv_to_q
tensor([[1, 1, 0, 0, 0, 0],
        [1, 1, 2, 0, 0, 0],
        [1, 2, 1, 2, 3, 4],
        [1, 2, 3, 4, 1, 0]])
(Pdb) num_kv_to_q.shape
torch.Size([4, 6])
(Pdb) num_total_kv_to_q
tensor([[424, 600,   0,   0,   0,   0],
        [624, 200, 400,   0,   0,   0],
        [278, 556, 117, 234, 351, 468],
        [ 81, 162, 243, 324, 700,   0]])
(Pdb) num_total_kv_to_q.shape
torch.Size([4, 6])
(Pdb) cp_seq_lens
tensor([[424, 600,   0,   0,   0,   0],
        [624, 200, 200,   0,   0,   0],
        [278, 278, 117, 117, 117, 117],
        [ 81,  81,  81,  81, 700,   0]])
(Pdb) cp_seq_lens.shape
torch.Size([4, 6])
(Pdb) num_cp_shards
tensor([2, 3, 6, 5])
(Pdb) num_cp_shards.shape
torch.Size([4])
(Pdb) cp_query_dst
tensor([[ 1,  3, -1, -1, -1, -1],
        [ 2,  3,  1, -1, -1, -1],
        [ 3,  0,  3,  2,  0,  1],
        [ 3,  1,  0,  2,  1, -1]])
(Pdb) cp_query_dst.shape
torch.Size([4, 6])
(Pdb) q_seq_to_dst
tensor([[[[0, 1, 0, 0]],
         [[0, 0, 0, 1]],
         [[0, 0, 0, 0]],
         [[0, 0, 0, 0]],
         [[0, 0, 0, 0]],
         [[0, 0, 0, 0]]],


        [[[0, 0, 1, 0]],
         [[0, 0, 0, 1]],
         [[0, 1, 0, 0]],
         [[0, 0, 0, 0]],
         [[0, 0, 0, 0]],
         [[0, 0, 0, 0]]],


        [[[0, 0, 0, 1]],
         [[1, 0, 0, 0]],
         [[0, 0, 0, 1]],
         [[0, 0, 1, 0]],
         [[1, 0, 0, 0]],
         [[0, 1, 0, 0]]],

        [[[0, 0, 0, 1]],
         [[0, 1, 0, 0]],
         [[1, 0, 0, 0]],
         [[0, 0, 1, 0]],
         [[0, 1, 0, 0]],
         [[0, 0, 0, 0]]]])
         
(Pdb) q_seq_to_dst.shape
torch.Size([4, 6, 1, 4])
(Pdb) q_seq_to_dst.squeeze(2)
tensor([[[0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],

        [[0, 0, 1, 0],
         [0, 0, 0, 1],
         [0, 1, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],

        [[0, 0, 0, 1],
         [1, 0, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0],
         [1, 0, 0, 0],
         [0, 1, 0, 0]],

        [[0, 0, 0, 1],
         [0, 1, 0, 0],
         [1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 0]]])
(Pdb) pad_len
tensor(6)
(Pdb) pad_len
tensor(6)
(Pdb) q_seq_to_dst.squeeze(2).shape
torch.Size([4, 6, 4])
(Pdb) fwd_q_metadata
Metadata(dst_rank=tensor([
        [ 1,  3, -1, -1, -1, -1],
        [ 2,  3,  1, -1, -1, -1],
        [ 3,  0,  3,  2,  0,  1],
        [ 3,  1,  0,  2,  1, -1]]), 
        dst_offset=tensor([
        [   0,    0,    0,    0,    0,    0],
        [   0,  600,  424,    0,    0,    0],
        [ 800,    0, 1078,  624,  278,  624],
        [1195,  741,  395,  741,  822,    0]]), 
        seq_len=tensor([
        [424, 600,   0,   0,   0,   0],
        [624, 200, 200,   0,   0,   0],
        [278, 278, 117, 117, 117, 117],
        [ 81,  81,  81,  81, 700,   0]]), 
        num_recv_tokens=tensor([
        [   0,    0,  395,   81,  476],
        [ 424,  200,  117,  781, 1522],
        [   0,  624,  117,   81,  822],
        [ 600,  200,  395,   81, 1276]]), 
        seq_recv_mask=None, 
        recv_seq_lens=None, 
        num_seqs=tensor([2, 3, 6, 5]), 
        world_size=4, 
        normalized=False, 
        num_total_recv_tokens=[476, 1522, 822, 1276])
(Pdb) rev_q_metadata
Metadata(dst_rank=tensor([
        [ 2,  2,  3, -1, -1],
        [ 0,  1,  2,  3,  3],
        [ 1,  2,  3, -1, -1],
        [ 0,  1,  2,  2,  3]]), 
        dst_offset=tensor([
        [278, 790, 162,   0,   0],
        [  0, 824, 907,  81, 324],
        [  0, 673, 243,   0,   0],
        [424, 624,   0, 556,   0]]), 
        seq_len=tensor([
        [278, 117,  81,   0,   0],
        [424, 200, 117,  81, 700],
        [624, 117,  81,   0,   0],
        [600, 200, 278, 117,  81]]), 
        num_recv_tokens=tensor([
        [   0,  424,    0,  600, 1024],
        [   0,  200,  624,  200, 1024],
        [ 395,  117,  117,  395, 1024],
        [  81,  781,   81,   81, 1024]]), 
        seq_recv_mask=None, 
        recv_seq_lens=None, 
        num_seqs=tensor([3, 5, 3, 5]), 
        world_size=4, 
        normalized=False, 
        num_total_recv_tokens=[1024, 1024, 1024, 1024])
(Pdb) q_seq_to_dst
tensor([[[[0, 1, 0, 0]],

         [[0, 0, 0, 1]],

         [[0, 0, 0, 0]],

         [[0, 0, 0, 0]],

         [[0, 0, 0, 0]],

         [[0, 0, 0, 0]]],


        [[[0, 0, 1, 0]],

         [[0, 0, 0, 1]],

         [[0, 1, 0, 0]],

         [[0, 0, 0, 0]],

         [[0, 0, 0, 0]],

         [[0, 0, 0, 0]]],


        [[[0, 0, 0, 1]],

         [[1, 0, 0, 0]],

         [[0, 0, 0, 1]],

         [[0, 0, 1, 0]],

         [[1, 0, 0, 0]],

         [[0, 1, 0, 0]]],


        [[[0, 0, 0, 1]],

         [[0, 1, 0, 0]],

         [[1, 0, 0, 0]],

         [[0, 0, 1, 0]],

         [[0, 1, 0, 0]],

         [[0, 0, 0, 0]]]])
(Pdb) intermediates[0]
tensor([[[[  0, 424,   0,   0]],

         [[  0,   0,   0, 600]],

         [[  0,   0,   0,   0]],

         [[  0,   0,   0,   0]],

         [[  0,   0,   0,   0]],

         [[  0,   0,   0,   0]]],


        [[[  0,   0, 624,   0]],

         [[  0,   0,   0, 200]],

         [[  0, 200,   0,   0]],

         [[  0,   0,   0,   0]],

         [[  0,   0,   0,   0]],

         [[  0,   0,   0,   0]]],


        [[[  0,   0,   0, 278]],

         [[278,   0,   0,   0]],

         [[  0,   0,   0, 117]],

         [[  0,   0, 117,   0]],

         [[117,   0,   0,   0]],

         [[  0, 117,   0,   0]]],


        [[[  0,   0,   0,  81]],

         [[  0,  81,   0,   0]],

         [[ 81,   0,   0,   0]],

         [[  0,   0,  81,   0]],

         [[  0, 700,   0,   0]],

         [[  0,   0,   0,   0]]]])
(Pdb) intermediates[1]
tensor([[[[0, 1, 0, 0]],

         [[0, 0, 0, 1]],

         [[0, 0, 0, 0]],

         [[0, 0, 0, 0]],

         [[0, 0, 0, 0]],

         [[0, 0, 0, 0]]],


        [[[0, 0, 1, 0]],

         [[0, 0, 0, 1]],

         [[0, 1, 0, 0]],

         [[0, 0, 0, 0]],

         [[0, 0, 0, 0]],

         [[0, 0, 0, 0]]],


        [[[0, 0, 0, 1]],

         [[1, 0, 0, 0]],

         [[0, 0, 0, 1]],

         [[0, 0, 1, 0]],

         [[1, 0, 0, 0]],

         [[0, 1, 0, 0]]],


        [[[0, 0, 0, 1]],

         [[0, 1, 0, 0]],

         [[1, 0, 0, 0]],

         [[0, 0, 1, 0]],

         [[0, 1, 0, 0]],

         [[0, 0, 0, 0]]]])
(Pdb) intermediates[2]
tensor([3, 5, 3, 5])
(Pdb) fwd_k_metadata
Metadata(dst_rank=tensor([[
         [ 1, -1, -1, -1],
         [ 3, -1, -1, -1],
         [-1, -1, -1, -1],
         [-1, -1, -1, -1],
         [-1, -1, -1, -1],
         [-1, -1, -1, -1]],

        [[ 2, -1, -1, -1],
         [ 3,  1, -1, -1],
         [ 1, -1, -1, -1],
         [-1, -1, -1, -1],
         [-1, -1, -1, -1],
         [-1, -1, -1, -1]],

        [[ 3,  0, -1, -1],
         [ 0, -1, -1, -1],
         [ 3,  2,  0,  1],
         [ 2,  0,  1, -1],
         [ 0,  1, -1, -1],
         [ 1, -1, -1, -1]],

        [[ 3,  1,  0,  2],
         [ 1,  0,  2, -1],
         [ 0,  2, -1, -1],
         [ 2, -1, -1, -1],
         [ 1, -1, -1, -1],
         [-1, -1, -1, -1]]]), 
         dst_offset=tensor([[
         [   0,    0,    0,    0],
         [   0,    0,    0,    0],
         [   0,    0,    0,    0],
         [   0,    0,    0,    0],
         [   0,    0,    0,    0],
         [   0,    0,    0,    0]],

        [[   0,    0,    0,    0],
         [ 600,  424,    0,    0],
         [ 624,    0,    0,    0],
         [   0,    0,    0,    0],
         [   0,    0,    0,    0],
         [   0,    0,    0,    0]],

        [[ 800,    0,    0,    0],
         [ 278,    0,    0,    0],
         [1078,  624,  556,  824],
         [ 741,  673,  941,    0],
         [ 790, 1058,    0,    0],
         [1175,    0,    0,    0]],

        [[1195, 1292,  907,  858],
         [1373,  988,  939,    0],
         [1069, 1020,    0,    0],
         [1101,    0,    0,    0],
         [1454,    0,    0,    0],
         [   0,    0,    0,    0]]]), 
         seq_len=tensor([
        [424, 600,   0,   0,   0,   0],
        [624, 200, 200,   0,   0,   0],
        [278, 278, 117, 117, 117, 117],
        [ 81,  81,  81,  81, 700,   0]]), 
        num_recv_tokens=tensor([
        [   0,    0,  907,  243, 1150],
        [ 424,  400,  468,  862, 2154],
        [   0,  624,  234,  324, 1182],
        [ 600,  200,  395,   81, 1276]]), 
        seq_recv_mask=None, 
        recv_seq_lens=None, 
        num_seqs=tensor([2, 3, 6, 5]), 
        world_size=4, 
        normalized=False, 
        num_total_recv_tokens=[1150, 2154, 1182, 1276])
(Pdb) rev_k_metadata
Metadata(dst_rank=tensor([
        [ 2,  2,  2,  2,  2,  3,  3,  3, -1, -1],
        [ 0,  1,  1,  2,  2,  2,  2,  3,  3,  3],
        [ 1,  2,  2,  3,  3,  3,  3, -1, -1, -1],
        [ 0,  1,  2,  2,  3, -1, -1, -1, -1, -1]]), 
        dst_offset=tensor([
        [1024,  278, 2604, 1697,  790, 2048, 1105,  162,    0,    0],
        [   0, 1648,  824, 3628, 2721, 1814,  907, 1024,   81,  324],
        [   0, 1580,  673, 3072, 2129, 1186,  243,    0,    0,    0],
        [ 424,  624,    0,  556,    0,    0,    0,    0,    0,    0]]), seq_len=tensor([[278, 278, 117, 117, 117,  81,  81,  81,   0,   0],
        [424, 200, 200, 117, 117, 117, 117,  81,  81, 700],
        [624, 117, 117,  81,  81,  81,  81,   0,   0,   0],
        [600, 200, 278, 117,  81,   0,   0,   0,   0,   0]]), num_recv_tokens=tensor([[  0, 424,   0, 600],
        [  0, 400, 624, 200],
        [907, 468, 234, 395],
        [243, 862, 324,  81]]), seq_recv_mask=tensor([[[ True, False, False, False],
         [ True, False, False, False],
         [False, False, False, False],
         [False, False, False, False],
         [False, False, False, False],
         [False, False, False, False]],

        [[ True, False, False, False],
         [ True,  True, False, False],
         [ True, False, False, False],
         [False, False, False, False],
         [False, False, False, False],
         [False, False, False, False]],

        [[ True,  True, False, False],
         [ True, False, False, False],
         [ True,  True,  True,  True],
         [ True,  True,  True, False],
         [ True,  True, False, False],
         [ True, False, False, False]],

        [[ True,  True,  True,  True],
         [ True,  True,  True, False],
         [ True,  True, False, False],
         [ True, False, False, False],
         [ True, False, False, False],
         [False, False, False, False]]]), recv_seq_lens=tensor([[424, 600,   0,   0,   0,   0],
        [624, 200, 200,   0,   0,   0],
        [278, 278, 117, 117, 117, 117],
        [ 81,  81,  81,  81, 700,   0]]), 
        num_seqs=tensor([ 8, 10,  7,  5]), world_size=4, normalized=False, num_total_recv_tokens=None)
(Pdb) attention_metadata
(tensor([[   0,  117,  198,  198,  198],
        [ 424,  624,  741,  822, 1522],
        [ 624,  741,  822,  822,  822],
        [ 600,  800, 1078, 1195, 1276]]), 
        tensor([[   0,  351,  594,  594,  594],
        [ 424,  824, 1292, 1454, 2154],
        [ 624,  858, 1182, 1182, 1182],
        [ 600,  800, 1078, 1195, 1276]]), 
        tensor([117, 700, 624, 600]), 
        tensor([351, 700, 624, 600]), 
        tensor([3, 3, 5, 4]))
```

## Data Structure Summary

The function builds several complex data structures:

### Core Tensors
- `cp_seq_lens[rank][shard_id]` → tokens in this CP shard
- `cp_query_dst[rank][shard_id]` → destination rank for query shard  
- `kv_to_q_mapping[rank][shard_id][cp_idx][0/1]` → (query_rank, query_shard_id)
- `kv_to_q_rank[rank][shard_id][cp_idx]` → rank among KVs serving same query
- `kv_context_size[rank][shard_id]` → tokens before this KV shard
- `num_kv_to_q[rank][shard_id]` → number of KV shards serving this query

### Causal Relationships
The mappings encode causal attention constraints where:
- Query shard `i` can attend to KV shards `[0, 1, ..., i]`
- Each KV shard knows which queries it serves
- Each query knows which KV shards it can access

## Key Features

### Realistic Test Scenarios
1. **Variable Sequence Lengths**: Random but balanced distribution
2. **Mixed CP Degrees**: Some sequences use more parallelism than others
3. **Random Destinations**: Simulates real load balancing scenarios
4. **Causal Constraints**: Respects attention causality requirements

### Comprehensive Coverage
1. **Forward Communication**: Query and KV data dispatch
2. **Backward Communication**: Gradient accumulation and routing
3. **Attention Layout**: Packed sequence parameters for kernels
4. **Edge Cases**: Handles padding, variable lengths, different CP degrees

### Integration Points
1. **With compute_metadata**: Generates query routing metadata
2. **With compute_metadata_kv**: Generates complex KV routing metadata  
3. **With attention kernels**: Provides layout parameters
4. **With test framework**: Creates realistic test scenarios

## Performance Considerations

### Complexity Factors
- **Memory overhead**: O(world_size × max_seqs × max_cp_degree²) for mappings
- **Computation complexity**: O(num_seqs × max_cp_degree²) per rank
- **Communication patterns**: Can create hotspots with uneven CP degrees

### Optimization Opportunities
- **Load balancing**: Random destinations help distribute load
- **Memory layout**: Contiguous allocation for better cache behavior
- **Batch processing**: Multiple sequences processed together

## Testing Integration

This function serves as the **primary test data generator** for the distributed attention system:

### What It Tests
1. **Complex CP scenarios**: Variable parallelism degrees
2. **Causal attention**: Proper KV-to-Query mappings
3. **Communication patterns**: Realistic dispatch scenarios
4. **Edge cases**: Padding, variable lengths, boundary conditions

### How It's Used
```python
# Generate complete test scenario
fwd_q, rev_q, fwd_kv, rev_kv, attn_params = create_qkv_dispatch(4, 128, 3, 8)

# Use in end-to-end testing
test_qkv_dispatch(args)  # Uses this function internally

# Validate mathematical properties
assert_conservation_laws(fwd_q, rev_q)
assert_causal_constraints(fwd_kv, kv_to_q_mapping)
```

---

*This function is the cornerstone of distributed attention testing, generating complex but realistic scenarios that thoroughly exercise all aspects of the communication and computation pipeline.*