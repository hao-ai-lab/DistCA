# orchestrate_simulate Function

## Purpose
Simulates inter-rank communication by copying tensor data according to metadata specifications. This function mimics what would happen in a real distributed system where ranks exchange data via NCCL or NVSHMEM.

## Function Signature
```python
@torch.no_grad()
def orchestrate_simulate(tensor: torch.Tensor, output_tensor: torch.Tensor, metadata: Metadata) -> torch.Tensor
```

## Parameters

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `tensor` | `torch.Tensor` | `(world_size, num_tokens, hidden_dim)` | Source tensor with data from all ranks |
| `output_tensor` | `torch.Tensor` | `(world_size, max_recv_tokens, hidden_dim)` | Destination tensor (initially zeros) |
| `metadata` | `Metadata` | - | Communication metadata specifying how to route data |

## Returns
- `torch.Tensor`: The modified `output_tensor` with data copied according to metadata

## Detailed Behavior

### Step-by-Step Process

1. **Iterate over source ranks**: For each rank that has data to send
2. **Extract destination info**: Get where this rank's data should go
3. **Process sequences**: Handle each sequence shard individually
4. **Handle CP dimensions**: Support both 1D and 2D destination ranks (for context parallelism)
5. **Copy data**: Move tensor slices to correct positions

### Key Operations

```python
# For each source rank
for src_rank in range(world_size):
    dst_rank = metadata.dst_rank[src_rank]      # Where to send data
    dst_offset = metadata.dst_offset[src_rank]  # Offset in destination buffer
    seq_lens = metadata.seq_len[src_rank]       # Length of each sequence
    
    # Process each sequence from this rank
    acu_tokens = 0
    for j, rs in enumerate(dst_rank):
        seq_len = seq_lens[j]
        seq = tensor[src_rank][acu_tokens:acu_tokens + seq_len]
        
        # Handle single destination rank
        if dst_rank.dim() == 1:
            rank = rs
            if rank >= 0:  # -1 means padding/no-op
                output_tensor[rank][dst_offset[j]: dst_offset[j] + seq_len] = seq
        
        # Handle multiple destination ranks (context parallelism)
        else:
            for k, rank in enumerate(rs):
                if rank >= 0:
                    output_tensor[rank][dst_offset[j][k]: dst_offset[j][k] + seq_len] = seq
        
        acu_tokens += seq_len
```

## Important Details

### Padding Handling
- **Value -1** in `dst_rank` indicates padding (no operation)
- **Positive values** indicate valid destination ranks
- **No error** if padding values are encountered

### Context Parallelism Support
- **1D dst_rank**: Simple rank-to-rank communication
- **2D dst_rank**: One sequence shard sent to multiple ranks
- **Broadcasts data** when multiple destinations exist

### Error Handling
- **Shape mismatches** will raise `RuntimeError`
- **Debug information** printed on errors (source rank, destination rank, offsets, lengths)

## What This Function Tests

### ✅ Validates
- **Correct addressing**: Data goes to right rank and offset
- **Proper slicing**: Sequence boundaries respected  
- **Multi-destination handling**: Context parallelism works
- **Padding behavior**: -1 values properly ignored

### ❌ Does NOT Test
- **Real network communication**: Pure memory copy simulation
- **Concurrent access**: No race conditions tested
- **Network failures**: No error recovery
- **Memory pressure**: No OOM scenarios

## Common Usage Patterns

### Forward Pass Simulation
```python
# Create output buffer sized for attention layout
max_recv_tokens = fwd_metadata.num_recv_tokens.max()
output_tensor = torch.zeros((world_size, max_recv_tokens, hidden_size), device=device)

# Simulate forward communication
output_tensor = orchestrate_simulate(input_tensor, output_tensor, fwd_metadata)
```

### Reverse Pass Simulation  
```python
# Create output buffer sized for MLP layout
rev_tensor = torch.zeros((world_size, total_seq_len, hidden_size), device=device)

# Simulate reverse communication
rev_tensor = orchestrate_simulate(attention_output, rev_tensor, rev_metadata)
```

## Debugging Tips

### Common Issues
1. **IndexError**: Check tensor shapes match metadata expectations
2. **Shape mismatch**: Verify `dst_offset + seq_len <= output_tensor.shape[1]`
3. **Wrong results**: Ensure metadata was computed correctly

### Debug Information
The function prints detailed info on RuntimeError:
- Source and destination ranks
- Offset and length values  
- Tensor shapes
- Accumulated token count

---

*This function is the core simulation engine that validates metadata correctness without requiring actual distributed hardware.*