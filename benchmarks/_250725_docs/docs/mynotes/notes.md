# Docs

## `inplace_metadata.py`

### `exclusive_cumsum`

`exclusive_cumsum(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor`

Cumsum but excluding the last element.
```python
> exclusive_cumsum(torch.tensor([0,1,2,3,4,5]))
tensor([ 0, 0, 1, 3, 6, 10])
```

### `mask_by_neg`

`mask_by_neg(tensor: torch.Tensor, mask: torch.Tensor)`

Mask tensor with -1 where mask is False.

```python
> mask_by_neg(torch.tensor([1,1,1]), torch.tensor([True, True, False,]))
tensor([ 1, 1, -1])
```

### `index_put_with_neg_padding_1d`

`index_put_with_neg_padding_1d(tensor: torch.Tensor, src: torch.Tensor, index: torch.Tensor)`

Index put with -1 padding.

```python
# Mask with `-1` will not write to any tensor.

index_put_with_neg_padding_1d(torch.tensor([1,2,3,4]), torch.tensor([10, 11, 12, 13]), torch.tensor([1,2,3,-1]))
tensor([ 1, 10, 11, 12])
```

- Note: [torch.Tensor.index_put_](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.index_put_.html): `tensor.index_put_(indices, values)` is equivalent to `tensor[indices] = values`. 
- Our function `index_put_with_neg_padding_1d` handles the case that when `-1` is in `indices`, it putting value to the "last element" (by we artificially extending the tensor by length 1 and then remove it before we return the tensor). 


### `compute_metadata`

```python
compute_metadata(
    seq_len: torch.Tensor, 
    global_dispatch: torch.Tensor, 
    return_intermediate: bool = False
) -> Tuple[Metadata, Metadata]
```

Compute metadata for query and key-value.

#### Input

- `seq_len`: Tensor shape of (world_size, max_num_local_seqs). 
    - For each `seq_len[rank][seq_id]`, it is the length of that (source) sequence length in the `rank`.
- `global_dispatch`: Tensor shape of (world_size, max_num_local_seqs, max_cp_degree)
    - For each `global_dispatch[src_rank][seq_id][dst_rank]`, it send the sequence `seq_id` from `src_rank` to `dst_rank`. 
- `return_intermediate`: whether to return intermediate computation results to validate the correctness of the metadata.

#### Output
- `Metadata` for forward communication.
- `Metadata` for reverse communication.

#### Calls
- No other dependencies exist.

Construction of the metadata:
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

with the log
```python
In [170]: global_dispatch.reshape(dispatch_shape)
Out[170]:
tensor([[[ 1],
         [ 2],
         [-1]],

        [[ 2],
         [ 0],
         [ 1]],

        [[ 0],
         [-1],
         [ 2]]])

In [171]: seq_begin_offset.reshape(dispatch_shape)
Out[171]:
tensor([[[ 0],
         [ 0],
         [ 0]],

        [[ 5],
         [ 0],
         [10]],

        [[12],
         [ 0],
         [13]]])

In [172]: seq_len
Out[172]:
tensor([[10,  5,  0],
        [ 8, 12,  4],
        [ 6,  0,  9]])

In [173]: num_recv_tokens
Out[173]:
tensor([[ 0, 12,  6, 18],
        [10,  4,  0, 14],
        [ 5,  8,  9, 22]])

In [174]: num_seqs
Out[174]: tensor([2, 3, 2])

In [175]: world_size
Out[175]: 3

In [176]: num_recv_tokens[:, -1].tolist()
Out[176]: [18, 14, 22]
```


### `compute_attn_layout_seqlens`


### `compute_metadata_kv`


### `mlp_layout_packed_params`

## `test_util.py`

### `create_qkv_dispatch`

```python
create_qkv_dispatch(world_size: int, total_seq_len: int, num_seqs: int, max_cp_degree: int)
```

Create a qkv dispatch tensor.


