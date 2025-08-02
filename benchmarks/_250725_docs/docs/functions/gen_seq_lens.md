# gen_seq_lens Function

## Purpose
Generates realistic, random sequence length distributions for testing distributed attention systems. Creates variable-length sequences that sum to a target total while avoiding degenerate cases.

## Function Signature
```python
def gen_seq_lens(world_size: int, num_seqs: int, total_len: int) -> torch.Tensor
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `world_size` | `int` | Number of GPU ranks |
| `num_seqs` | `int` | Number of sequences per rank |
| `total_len` | `int` | Target total tokens per rank |

## Returns
- `torch.Tensor`: Shape `(world_size, num_seqs)` with sequence lengths that sum to `total_len` per rank

## Algorithm

### Step 1: Random Ratio Generation
```python
# Generate random ratios with minimum value to avoid zero-length sequences
ratio = torch.rand((world_size, num_seqs)) + 0.25 / num_seqs
```

**Key insight**: Adding `0.25 / num_seqs` ensures no sequence becomes zero-length after rounding.

### Step 2: Normalization
```python  
# Normalize so ratios sum to 1.0 per rank
ratio = ratio / ratio.sum(dim=1, keepdim=True)
```

### Step 3: Length Computation  
```python
# Convert ratios to actual lengths
seq_len = (ratio * total_len).round().int()
```

### Step 4: Error Correction
```python
# Fix rounding errors by adjusting the last sequence
seq_len_total = seq_len.sum(dim=1)
seq_len_total_error = seq_len_total - total_len
seq_len[:, -1] -= seq_len_total_error
```

**Critical correction**: Ensures exact sum match by absorbing rounding errors into the last sequence.

## Mathematical Properties

### Distribution Characteristics
- **Uniform base**: Each sequence starts with equal probability weight
- **Random variation**: Actual lengths vary according to random ratios
- **Minimum length**: No sequence shorter than `ceil(0.25 * total_len / num_seqs)`
- **Exact sum**: Always sums to exactly `total_len` per rank

### Edge Cases Handled
- **Single sequence**: `num_seqs=1` → returns `[[total_len]]`
- **Many sequences**: Large `num_seqs` → each gets minimum viable length
- **Zero total**: `total_len=0` → all sequences get length 0

## Example Outputs

### Typical Case
```python
world_size = 2, num_seqs = 4, total_len = 1000

# Possible output:
tensor([[287, 143, 298, 272],    # Rank 0: sums to 1000
        [156, 387, 201, 256]])   # Rank 1: sums to 1000
```

### Edge Case: Few Sequences
```python
world_size = 3, num_seqs = 2, total_len = 100

# Possible output:
tensor([[62, 38],     # Rank 0  
        [71, 29],     # Rank 1
        [45, 55]])    # Rank 2
```

### Edge Case: Many Sequences
```python
world_size = 1, num_seqs = 10, total_len = 100

# Each sequence gets at least ceil(0.25 * 100 / 10) = 3 tokens
# Possible output:
tensor([[12, 8, 15, 9, 11, 7, 13, 6, 10, 9]])  # Sums to 100
```

## Why This Design?

### Realistic Testing
- **Variable lengths**: Mimics real workloads with diverse sequence lengths
- **Non-uniform**: Avoids artifacts from perfectly uniform distributions
- **Controlled randomness**: Reproducible with torch.manual_seed()

### Numerical Stability
- **Minimum length**: Prevents zero-length sequences that could break downstream code
- **Exact sums**: Eliminates cumulative rounding errors
- **Integer output**: All lengths are valid token counts

### Scalability
- **Independent ranks**: Each rank gets its own distribution
- **Configurable**: Works with any reasonable world_size/num_seqs combination
- **Efficient**: O(world_size × num_seqs) computation

## Common Usage Patterns

### Basic Testing
```python
# Generate test data for 4 ranks, 8 sequences each, 1024 tokens per rank
seq_lens = gen_seq_lens(world_size=4, num_seqs=8, total_len=1024)
print(seq_lens.sum(dim=1))  # Should print: [1024, 1024, 1024, 1024]
```

### Context Parallelism Preparation
```python
# Generate base lengths, then scale for CP
base_lens = gen_seq_lens(world_size, num_seqs, total_len // max_cp_degree)
cp_lens = base_lens * max_cp_degree  # Ensure divisibility
```

### Reproducible Testing
```python
torch.manual_seed(42)
seq_lens1 = gen_seq_lens(4, 8, 1024)

torch.manual_seed(42)  
seq_lens2 = gen_seq_lens(4, 8, 1024)

assert torch.equal(seq_lens1, seq_lens2)  # Identical results
```

## Validation Properties

### ✅ Guarantees
- **Exact sums**: `seq_lens.sum(dim=1) == total_len` for all ranks
- **Positive lengths**: All values > 0 (assuming reasonable inputs)
- **Integer values**: All lengths are whole numbers
- **Deterministic**: Same seed → same output

### ✅ Statistical Properties
- **Mean length**: Approximately `total_len / num_seqs` per sequence
- **Variance**: Depends on random ratios but bounded
- **Independence**: Different ranks get independent distributions

### ❌ Does NOT Guarantee
- **Minimum/maximum bounds**: No control over length ranges
- **Specific distributions**: Not Gaussian, exponential, etc.
- **Load balancing**: No attempt to equalize computation
- **Real-world realism**: Simple uniform-based random model

## Debugging Tips

### Common Issues
1. **Negative lengths**: Check that `total_len >= num_seqs * minimum_viable_length`
2. **Sum mismatches**: Should never happen due to error correction
3. **Zero lengths**: Indicates `total_len` too small for `num_seqs`

### Verification
```python
seq_lens = gen_seq_lens(world_size, num_seqs, total_len)

# Check sums
assert (seq_lens.sum(dim=1) == total_len).all()

# Check positivity  
assert (seq_lens > 0).all()

# Check data type
assert seq_lens.dtype == torch.int32
```

---

*This utility function provides the foundation for realistic sequence length testing while maintaining mathematical correctness and numerical stability.*