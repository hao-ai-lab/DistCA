"""
Wrapper torch function of the n2n communication to orchestrate the ping-pang parallel.
"""
from typing import Optional

import torch

def dispatch(
    query_out: torch.Tensor,
    key_value_out: Optional[torch.Tensor],
    query_in: torch.Tensor,
    key_value_in: Optional[torch.Tensor],
    query_dst_id: torch.Tensor,
    query_dst_offset: torch.Tensor,
    key_value_dst_id: Optional[torch.Tensor],
    key_value_dst_offset: Optional[torch.Tensor],
    token: int,
    hidden_q: int,
    hidden_kv: int,
    cp_degree: int
):
    """Template"""

def dispatch_reverse(
    query_out: torch.Tensor,
    key_value_out: Optional[torch.Tensor],
    query_in: torch.Tensor,
    key_value_in: Optional[torch.Tensor],
    query_dst_id: torch.Tensor,
    query_dst_offset: torch.Tensor,
    key_value_dst_id: Optional[torch.Tensor],
    key_value_dst_offset: Optional[torch.Tensor],
    token_query: int,
    token_kv: int,
    hidden_q: int,
    hidden_kv: int,
):
    """Template"""

def both_none_or_neither(a, b):
    return (a is None and b is None) or (a is not None and b is not None)

def both_none_or_same_shape(a, b):
    return (a is None and b is None) or (a is not None and b is not None and a.shape == b.shape)

class n_to_n_dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query_in: torch.Tensor,
                query_dst_id: torch.Tensor,
                query_dst_offset: torch.Tensor,
                out_query_shape: torch.Size,
                key_value_in: Optional[torch.Tensor]=None,
                key_value_dst_id: Optional[torch.Tensor]=None,
                key_value_dst_offset: Optional[torch.Tensor]=None,
                out_key_value_shape: Optional[torch.Size]=None,
                rev_query_dst_id: Optional[torch.Tensor]=None,
                rev_query_dst_offset: Optional[torch.Tensor]=None,
                rev_key_value_dst_id: Optional[torch.Tensor]=None,
                rev_key_value_dst_offset: Optional[torch.Tensor]=None,
                stream: Optional[torch.cuda.Stream]=None,
                event: Optional[torch.cuda.Event]=None,
               ):
        # check pairing values (same shape)
        assert both_none_or_same_shape(rev_query_dst_id, rev_query_dst_offset)
        assert both_none_or_same_shape(rev_key_value_dst_id, rev_key_value_dst_offset)
        assert both_none_or_same_shape(key_value_dst_id, key_value_dst_offset)

        # check key_value related tensors
        assert both_none_or_neither(key_value_in, key_value_dst_id)
        assert both_none_or_neither(key_value_in, out_key_value_shape)
        assert both_none_or_neither(event, stream)

        out_query = torch.empty(out_query_shape, device=query_in.device, dtype=query_in.dtype)
        hidden_q = out_query_shape[-1]
        assert hidden_q == query_in.shape[-1]

        num_token = query_dst_id.shape[0]

        if key_value_in is not None:
            assert num_token == key_value_dst_id.shape[0] == key_value_in.shape[0]
            out_key_value = torch.empty(out_key_value_shape, device=key_value_in.device, dtype=key_value_in.dtype)
            key_value_dst_mask = (key_value_dst_id != -1).to(torch.bool)
            hidden_kv = out_key_value_shape[-1]
            max_cp_degree = key_value_dst_id.shape[-1]
        else:
            out_key_value = None
            key_value_dst_mask = None
            hidden_kv = 0
            max_cp_degree = 0

        if stream is not None:
            with torch.cuda.stream(stream):
                dispatch(
                    out_query, out_key_value, query_in, key_value_in,
                    query_dst_id, query_dst_offset,
                    key_value_dst_id, key_value_dst_offset,
                    num_token, hidden_q, hidden_kv, max_cp_degree
                )
                event.record(stream)
        else:
            dispatch(
                out_query, out_key_value, query_in, key_value_in,
                query_dst_id, query_dst_offset,
                key_value_dst_id, key_value_dst_offset,
                num_token, hidden_q, hidden_kv, max_cp_degree
            )

        ctx.save_for_backward(query_in.shape)
        ctx.save_for_backward(key_value_dst_mask)
        ctx.save_for_backward(query_dst_id.shape)
        ctx.save_for_backward(key_value_dst_id.shape if key_value_dst_id is not None else None)
        ctx.save_for_backward(rev_query_dst_id)
        ctx.save_for_backward(rev_query_dst_offset)
        ctx.save_for_backward(rev_key_value_dst_id)
        ctx.save_for_backward(rev_key_value_dst_offset)
        ctx.save_for_backward(hidden_q)
        ctx.save_for_backward(hidden_kv)
        ctx.save_for_backward(stream)
        ctx.save_for_backward(event)

        return out_query, out_key_value

    @staticmethod
    def backward(ctx, out_query_grad, out_key_value_grad):
        (query_in_shape, key_value_dst_mask, query_dst_id_shape, key_value_dst_id_shape,
         rev_query_dst_id, rev_query_dst_offset, rev_key_value_dst_id,
         rev_key_value_dst_offset, hidden_q, hidden_kv, stream, event) = ctx.saved_tensors

        query_in_grad = torch.empty(query_in_shape, device=out_query_grad.device, dtype=out_query_grad.dtype)

        if out_key_value_grad is not None:
            assert hidden_kv == out_key_value_grad.shape[-1]
            assert key_value_dst_mask is not None
            key_value_in_grad = torch.empty(key_value_dst_id_shape,
                                            device=out_key_value_grad.device,
                                            dtype=out_key_value_grad.dtype)
        else:
            assert key_value_dst_mask is None
            assert hidden_kv == 0
            key_value_in_grad = None

        num_token_query = rev_query_dst_id.shape[0]
        num_token_kv = rev_key_value_dst_id.shape[0]

        if stream is not None:
            with torch.cuda.stream(stream):
                dispatch_reverse(
                    query_in_grad, key_value_in_grad, out_query_grad, out_key_value_grad,
                    rev_query_dst_id, rev_query_dst_offset,
                    rev_key_value_dst_id, rev_key_value_dst_offset,
                    num_token_query, num_token_kv, hidden_q, hidden_kv
                )
                event.record(stream)
        else:
            dispatch_reverse(
                query_in_grad, key_value_in_grad, out_query_grad, out_key_value_grad,
                rev_query_dst_id, rev_query_dst_offset,
                rev_key_value_dst_id, rev_key_value_dst_offset,
                num_token_query, num_token_kv, hidden_q, hidden_kv
            )

        if key_value_in_grad is not None:
            # gather gradients from all copies
            key_value_in_grad = (key_value_in_grad * key_value_dst_mask).sum(dim=-1)

        return (query_in_grad, None, None, None,    # 3 Nones: rank, offset, shape
                key_value_in_grad, None, None, None,# same as above
                None, None, None, None, # reverse indices and offsets
                None, None, # stream and event
                )
