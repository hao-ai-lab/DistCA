import torch


def get_cp_index_slice(t: int, cp_size: int, cp_rank: int) -> list[int]:
    """
    Returns the list of token indices handled by the `cp_rank`-th
    context-parallel worker when a sequence of length `t` is split
    symmetrically.

    Strategy
    --------
    1.  Let main_part = ⌊t / cp_size⌋ · cp_size.
        Split that prefix into 2·cp_size equal chunks and give worker r
        chunks r and −(r+1).

    2.  Any leftover tail of length (t − main_part) is handed out
        round-robin: worker r gets element main_part + r if it exists.
    """
    if cp_size <= 0:
        raise ValueError("cp_size must be > 0")
    if not (0 <= cp_rank < cp_size):
        raise ValueError("cp_rank must be in [0, cp_size)")

    main_part       = t // (2 * cp_size) * (2 * cp_size)
    remaining_part  = t - main_part
    indices: list[int] = []

    # Handle the symmetric main part
    if main_part:
        chunk = main_part // (2 * cp_size)          # size of each chunk
        if chunk:                                   # only if big enough
            # front chunk
            start_f = cp_rank * chunk
            indices.append(slice(start_f, start_f + chunk))
            # mirrored chunk at the tail of main_part
            start_b = main_part - (cp_rank + 1) * chunk
            indices.append(slice(start_b, start_b + chunk))

    # Hand out leftover tail tokens round-robin
    if cp_rank < remaining_part:
        potential_indices = [
            cp_rank, 
            2 * cp_size - cp_rank - 1
        ]
        for i in potential_indices:
            if main_part + i < t:
                indices.append(slice(main_part + i, main_part + i + 1))

    return indices


def get_batch_on_this_cp_rank__balanced_seq(
    batch, token_lens, 
    cp_size = None, cp_rank = None
):
    """
    Slice the batch with balanced sequence length.
    --------------
    batch = {
        'tokens': tokens, # (B, L)
        'labels': labels, # (B, L)
        'loss_mask': loss_mask, # (B, L)
        'attention_mask': attention_mask, # (B, 1, L, L)
        'position_ids': position_ids, # (B, L)
    }
    where
    - B: micro-batch size
    - L: sequence length
    """

    L = batch['attention_mask'].shape[-1]
    
    if cp_size is None:
        from megatron.core.utils import parallel_state
        cp_size = parallel_state.get_context_parallel_world_size()
    if cp_rank is None:
        from megatron.core.utils import parallel_state
        cp_rank = parallel_state.get_context_parallel_rank()
    
    if cp_size > 1:

        tokens = batch['tokens']
        labels = batch['labels']
        loss_mask = batch['loss_mask']
        attention_mask = batch['attention_mask']
        position_ids = batch['position_ids']

        cp_slices = [
            get_cp_index_slice(token_len, cp_size, cp_rank)
            for token_len in token_lens
        ]
        new_tokens = []
        new_labels = []
        new_loss_mask = []
        new_attention_mask = []
        new_position_ids = []

        for i in range(len(token_lens)):
            index_slices = cp_slices[i]
            _new_tokens = []
            _new_labels = []
            _new_loss_mask = []
            _new_attention_mask = []
            _new_position_ids = []

            for cp_slice in index_slices:
                _new_tokens.append(
                    tokens[i, cp_slice]
                )
                _new_labels.append(labels[i, cp_slice])
                _new_loss_mask.append(loss_mask[i, cp_slice])
                _new_attention_mask.append(attention_mask[i, 0, cp_slice, :])
                _new_position_ids.append(position_ids[i, cp_slice])
            
            _token = torch.cat(_new_tokens, dim=0).reshape(1, -1)
            _label = torch.cat(_new_labels, dim=0).reshape(1, -1)
            _loss_mask = torch.cat(_new_loss_mask, dim=0).reshape(1, -1)
            _attention_mask = torch.cat(_new_attention_mask, dim=0).reshape(1, 1, -1, L)
            _position_ids = torch.cat(_new_position_ids, dim=0).reshape(1, -1)

            new_tokens.append(_token)
            new_labels.append(_label)
            new_loss_mask.append(_loss_mask)
            new_attention_mask.append(_attention_mask)
            new_position_ids.append(_position_ids)
        
        new_tokens = torch.cat(new_tokens, dim=0)
        new_labels = torch.cat(new_labels, dim=0)
        new_loss_mask = torch.cat(new_loss_mask, dim=0)
        new_attention_mask = torch.cat(new_attention_mask, dim=0)
        new_position_ids = torch.cat(new_position_ids, dim=0)
        
        batch = dict(
            tokens = new_tokens,
            labels = new_labels,
            loss_mask = new_loss_mask,
            attention_mask = new_attention_mask,
            position_ids = new_position_ids,
        )
    return batch


def get_approx_kv_len(t: int, cp_size: int, cp_rank: int) -> int:
    """
    Assume the token length for the token is `t`. 
    Then we assume the 

    """
    if cp_size <= 0:
        raise ValueError("cp_size must be > 0")
    if not (0 <= cp_rank < cp_size):
        raise ValueError("cp_rank must be in [0, cp_size)")

    t = t // (2 * cp_size) * (2 * cp_size)
    mid = t // 2
    chunk_size = t // (2 * cp_size)
    end = mid + chunk_size
    return end


def slice_length(slice_obj: slice) -> int:
    start, stop, step = slice_obj.start, slice_obj.stop, slice_obj.step
    step = step or 1
    return max(0, (stop - start + (step - 1 if step > 0 else step + 1)) // step)


def prepare_packed_seq_params(
    token_lens, 
    max_seq_len,
    cp_size, cp_rank,
    device,
):
    # Calculate the q-lengths
    slice_lengths = [
        sum(
            slice_length(s)
            for s in get_cp_index_slice(t, cp_size, cp_rank)
        ) 
        for t in token_lens
    ]

    seqlens_q = [0] + [
        sum(slice_lengths[:i+1])
        for i in range(len(slice_lengths))
    ]

    # Calculate the kv-lengths
    seqlens_kv = [
        get_approx_kv_len(t, cp_size, cp_rank)
        for t in token_lens
    ]

    seqlens_kv = [0] + [
        sum(seqlens_kv[:i+1])
        for i in range(len(seqlens_kv))
    ]

    # Calculate the number of non-padding elements for each sequence
    cu_seqlens_q = torch.tensor(seqlens_q, dtype=torch.int32, device=device)
    cu_seqlens_kv = torch.tensor(seqlens_kv, dtype=torch.int32, device=device)
    max_seqlen_q = max_seq_len
    max_seqlen_kv = max_seq_len

    try:
        from megatron.core.packed_seq_params import PackedSeqParams
    except ImportError:
        from dataclasses import dataclass

        @dataclass
        class PackedSeqParams:
            '''
            parameters to TEDotProductAttention and fused rope kernels for the
            `thd` (packed) sequence format
            '''

            qkv_format: str = None
            cu_seqlens_q: 'Tensor' = None
            cu_seqlens_kv: 'Tensor' = None
            cu_seqlens_q_padded: 'Tensor' = None
            cu_seqlens_kv_padded: 'Tensor' = None
            max_seqlen_q: 'Tensor' = None
            max_seqlen_kv: 'Tensor' = None


    result = PackedSeqParams(
        qkv_format='thd',
        # qkv_format='sbhd',
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
    )
    return result


def test_get_batch_on_this_cp_rank__balanced_seq():
    pass