"""
Wrapper torch function of the n2n communication to orchestrate the ping-pang parallel.
"""
from contextlib import nullcontext
from typing import Optional

import torch

from d2.runtime.attn_kernels.ops import DispatcherWrapper, dispatch
from d2.runtime.inplace_metadata import Metadata


def dispatch_reverse(
    q_input_grad: torch.Tensor,
    kv_input_grad: Optional[torch.Tensor],
    q_output_grad: torch.Tensor,
    kv_output_grad: Optional[torch.Tensor],
    query_metadata: Metadata,
    key_value_metadata: Optional[Metadata],
):
    dispatcher = DispatcherWrapper.get_instance()
    # the input has two shapes, so we dispatch them separately.
    dispatch(
        dispatcher, q_output_grad, q_input_grad, query_metadata, None, None, None
    )
    # NOTE: In the reversed pass, each token of kv grad is only sent to one copy, so its behavior
    # is the same as query in forward.
    if kv_input_grad is not None:
        dispatch(dispatcher, kv_output_grad, kv_input_grad, key_value_metadata, None, None, None)


def _both_none_or_neither(a, b):
    return (a is None and b is None) or (a is not None and b is not None)


class n_to_n_dispatch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, query_in: torch.Tensor,
        query_metadata: Metadata,
        key_value_in: Optional[torch.Tensor]=None,
        key_value_metadata: Optional[Metadata]=None,
        rev_query_metadata: Optional[Metadata]=None,
        rev_key_value_metadata: Optional[Metadata]=None,
        stream: Optional[torch.cuda.Stream]=None,
        event: Optional[torch.cuda.Event]=None,
    ):

        # check key_value related tensors
        assert query_metadata.normalized

        assert _both_none_or_neither(key_value_in, key_value_metadata)
        assert _both_none_or_neither(event, stream)

        # The num_head is folded into num_token.
        assert query_in.ndim == 2, "query_in is of shape (num_token, hidden_q)."

        hidden_q = query_in.shape[-1]
        out_query_shape = (query_metadata.num_recv_tokens, hidden_q)
        out_query = torch.empty(out_query_shape, device=query_in.device, dtype=query_in.dtype)

        num_token = query_in.shape[0]

        if key_value_in is not None:
            assert num_token == key_value_metadata.seq_len
            hidden_kv = out_key_value_shape[-1]
            out_key_value_shape = (key_value_metadata.num_recv_tokens, hidden_kv)
            out_key_value = torch.empty(out_key_value_shape, device=key_value_in.device, dtype=key_value_in.dtype)
            key_value_dst_mask = (key_value_metadata.dst_rank != -1).to(torch.bool)
        else:
            out_key_value = None
            key_value_dst_mask = None
            hidden_kv = 0

        if stream is not None:
            ctx = torch.cuda.stream(stream)
        else:
            ctx = nullcontext()
        with ctx:
            dispatch(
                dispatcher=DispatcherWrapper.get_instance(),
                tensor=query_in,
                dst_tensor=out_query,
                metadata=query_metadata,
                kv_tensor=key_value_in,
                kv_dst_tensor=out_key_value,
                kv_metadata=key_value_metadata,
            )
            if event is not None:
                event.record(stream)

        ctx.save_for_backward(query_in.shape)
        ctx.save_for_backward(key_value_in.shape if key_value_in is not None else None)
        ctx.save_for_backward(key_value_dst_mask)
        ctx.save_for_backward(rev_query_metadata)
        ctx.save_for_backward(rev_key_value_metadata)
        ctx.save_for_backward(hidden_q)
        ctx.save_for_backward(hidden_kv)
        ctx.save_for_backward(stream)
        ctx.save_for_backward(event)

        return out_query, out_key_value

    @staticmethod
    def backward(ctx, out_query_grad, out_key_value_grad):
        # NOTE(yonghao): in PP, the backward pass may not do the same thing
        # as the forward. In this case, the whole layer is wrapped with
        # a torch function and we do not run this part.
        (query_in_shape, key_value_in_shape,
         key_value_dst_mask, rev_query_metadata, rev_key_value_metadata,
         hidden_q, hidden_kv, stream, event) = ctx.saved_tensors

        query_in_grad = torch.empty(query_in_shape, device=out_query_grad.device, dtype=out_query_grad.dtype)

        if out_key_value_grad is not None:
            assert hidden_kv == out_key_value_grad.shape[-1]
            assert key_value_dst_mask is not None
            key_value_in_grad = torch.empty(key_value_dst_mask.shape + (hidden_kv,),
                                            device=out_key_value_grad.device,
                                            dtype=out_key_value_grad.dtype)
        else:
            assert key_value_dst_mask is None
            assert hidden_kv == 0
            key_value_in_grad = None

        if stream is not None:
            ctx = torch.cuda.stream(stream)
        else:
            ctx = nullcontext()
        with ctx:
            dispatch_reverse(
                q_input_grad=query_in_grad,
                kv_input_grad=key_value_in_grad,
                q_output_grad=out_query_grad,
                kv_output_grad=out_key_value_grad,
                query_metadata=rev_query_metadata,
                key_value_metadata=rev_key_value_metadata,
            )
            if event is not None:
                event.record(stream)

        if key_value_in_grad is not None:
            # gather gradients from all copies along the cp_degree dimension
            key_value_in_grad = (key_value_in_grad * key_value_dst_mask).sum(dim=-1)
            assert key_value_in_grad.shape == key_value_in_shape

        return (
            query_in_grad, None, key_value_in_grad, None,
            None, None, # metadata
            None, None, # stream and event
        )
