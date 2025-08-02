# Metadata Class

## Purpose
The `Metadata` class is the central data structure that encapsulates all information needed for distributed communication between GPU ranks. It describes how tensor data should be routed during forward and backward passes in distributed attention computation.

## Class Definition
```python
@dataclass
class Metadata:
    dst_rank: torch.Tensor
    dst_offset: torch.Tensor
    seq_len: torch.Tensor
    num_recv_tokens: torch.Tensor
    seq_recv_mask: Optional[torch.Tensor] = None
    recv_seq_lens: Optional[torch.Tensor] = None
    num_seqs: Optional[torch.Tensor] = None
    world_size: int = None
    normalized: bool = False
    num_total_recv_tokens: Union[int, tuple[int]] = None
```

## Field Descriptions

### Core Communication Fields

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `dst_rank` | `torch.Tensor` | `(world_size, max_seqs[, max_cp])` | Destination rank for each sequence shard |
| `dst_offset` | `torch.Tensor` | `(world_size, max_seqs[, max_cp])` | Byte offset in destination buffer |
| `seq_len` | `torch.Tensor` | `(world_size, max_seqs)` | Length of each sequence shard |
| `num_recv_tokens` | `torch.Tensor` | `(world_size, world_size + 1)` | Tokens received from each rank (last col = total) |

### Optional Fields for Advanced Use Cases

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `seq_recv_mask` | `torch.Tensor` | `(world_size, max_seqs, max_cp)` | Which KV shards are active (not padding) |
| `recv_seq_lens` | `torch.Tensor` | `(world_size, max_seqs)` | Original sequence lengths for received data |
| `num_seqs` | `torch.Tensor` | `(world_size,)` | Number of actual sequences per rank (rest is padding) |

### Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `world_size` | `int` | Total number of ranks in the system |
| `normalized` | `bool` | Whether tensors use optimized dtypes |
| `num_total_recv_tokens` | `Union[int, tuple[int]]` | Total tokens each rank will receive |

## Key Methods

### `get_slice(rank: int) -> Metadata`
Extracts metadata for a specific rank, removing padding and global information.

```python
def get_slice(self, rank: int):
    num_seqs = self.num_seqs[rank]
    return Metadata(
        dst_rank=self.dst_rank[rank][:num_seqs],        # Remove padding
        dst_offset=self.dst_offset[rank][:num_seqs],    # Remove padding  
        seq_len=self.seq_len[rank][:num_seqs],          # Remove padding
        num_recv_tokens=self.num_recv_tokens[rank],     # Keep full (world_size,) shape
        # ... other fields
        num_total_recv_tokens=self.num_total_recv_tokens[rank],
    )
```

### `normalize_dtype() -> Metadata`
Converts tensors to optimized data types for GPU kernels.

```python
def normalize_dtype(self):
    return Metadata(
        dst_rank=self.dst_rank.to(torch.int32),         # 32-bit for rank IDs
        dst_offset=self.dst_offset.to(torch.uint32),    # 32-bit unsigned for offsets
        seq_len=self.seq_len.to(torch.uint32),          # 32-bit unsigned for lengths
        num_recv_tokens=self.num_recv_tokens.to(torch.uint64),  # 64-bit for large counts
        # ... other normalizations
        normalized=True,
    )
```

### `cuda() -> Metadata`
Moves all tensors to GPU memory with contiguous layout.

```python
def cuda(self):
    return Metadata(
        dst_rank=self.dst_rank.cuda().contiguous(),
        dst_offset=self.dst_offset.cuda().contiguous(),
        seq_len=self.seq_len.cuda().contiguous(),
        # ... all tensors moved to GPU
    )
```

## Usage Patterns

### Forward Pass Communication
```python
# Use forward metadata to route data from MLP layout to attention layout
for src_rank in range(world_size):
    dst_ranks = fwd_metadata.dst_rank[src_rank]
    dst_offsets = fwd_metadata.dst_offset[src_rank]  
    seq_lengths = fwd_metadata.seq_len[src_rank]
    
    # Send each sequence shard to its destination
    for seq_id, (dst_rank, dst_offset, seq_len) in enumerate(zip(dst_ranks, dst_offsets, seq_lengths)):
        if dst_rank >= 0:  # -1 indicates padding
            send_data(src_rank, dst_rank, dst_offset, seq_len)
```

### Reverse Pass Communication  
```python
# Use reverse metadata to route gradients back to MLP layout
for dst_rank in range(world_size):
    src_ranks = rev_metadata.dst_rank[dst_rank]     # "dst" from forward becomes "src" for reverse
    src_offsets = rev_metadata.dst_offset[dst_rank]
    seq_lengths = rev_metadata.seq_len[dst_rank]
    
    # Receive gradients from each source
    for src_rank, src_offset, seq_len in zip(src_ranks, src_offsets, seq_lengths):
        if src_rank >= 0:
            receive_data(src_rank, dst_rank, src_offset, seq_len)
```

## Data Layout Examples

### Simple Query Metadata (2D)
```python
# 2 ranks, 3 sequences max per rank
dst_rank = torch.tensor([
    [1, 0, -1],    # Rank 0: seq0→rank1, seq1→rank0, seq2→padding
    [0, 1,  0]     # Rank 1: seq0→rank0, seq1→rank1, seq2→rank0
])

dst_offset = torch.tensor([
    [0, 128, 0],   # Offsets in destination buffers
    [0,   0, 256]
])

seq_len = torch.tensor([
    [128, 64, 0],  # Sequence lengths (0 = padding)
    [96, 128, 32]
])
```

### Complex KV Metadata (3D with Context Parallelism)
```python
# Same setup but with CP degree 2
dst_rank = torch.tensor([
    [[1, 0], [0, -1], [-1, -1]],    # Rank 0: seq0→[rank1,rank0], seq1→[rank0], padding
    [[0, 1], [1, 0],  [0, -1]]      # Rank 1: seq0→[rank0,rank1], seq1→[rank1,rank0], seq2→[rank0]
])
# Shape: (world_size=2, max_seqs=3, max_cp_degree=2)
```

## Padding Convention

### Padding Values
- **dst_rank = -1**: No communication (padding entry)
- **dst_offset = 0**: Default offset for padding (ignored)
- **seq_len = 0**: No tokens in this sequence (padding)

### Padding Semantics
```python
# Example: Rank has 2 actual sequences, but metadata padded to 4
dst_rank = torch.tensor([
    [1, 0, -1, -1],    # Only first 2 are real
    [0, 1, -1, -1]     # Only first 2 are real  
])

num_seqs = torch.tensor([2, 2])  # Tracks actual sequence count
```

## Validation Properties

### ✅ Internal Consistency
- **Shape compatibility**: All tensors have compatible shapes
- **Padding consistency**: -1 ranks correspond to 0 lengths
- **Sum conservation**: Total sent tokens = total received tokens
- **Offset validity**: All offsets within destination buffer bounds

### ✅ Communication Correctness
- **Bijective mapping**: Forward + reverse = identity (for queries)
- **Causal constraints**: KV shards respect attention causality  
- **No overwrites**: Destination regions don't overlap
- **Complete coverage**: All data has valid destination

### ❌ Does NOT Validate
- **Network topology**: No awareness of physical connections
- **Bandwidth limits**: No consideration of communication costs
- **Memory pressure**: No bounds checking on buffer sizes
- **Fault tolerance**: No handling of failed communications

## Memory Optimization

### Data Type Optimization
```python
# Original: All int64 (8 bytes per element)
original_size = dst_rank.numel() * 8

# Normalized: Mixed types based on value ranges
normalized_meta = meta.normalize_dtype()
# dst_rank: int32 (4 bytes) - rank IDs fit in 32 bits
# dst_offset: uint32 (4 bytes) - offsets typically < 4GB  
# seq_len: uint32 (4 bytes) - sequence lengths < 4B tokens
# num_recv_tokens: uint64 (8 bytes) - may need large counts

optimized_size = dst_rank.numel() * 4  # ~50% reduction
```

### GPU Memory Layout
```python
# Ensure contiguous memory for efficient GPU kernels
gpu_meta = meta.cuda()  # All tensors become contiguous on GPU
```

---

*The Metadata class is the interface between high-level dispatch decisions and low-level communication operations - understanding its structure is essential for working with distributed attention systems.*