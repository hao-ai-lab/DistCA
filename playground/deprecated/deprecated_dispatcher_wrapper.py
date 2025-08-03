"""
Wrapper torch function of the n2n communication to orchestrate the ping-pang parallel.
"""
from contextlib import nullcontext
from typing import Optional

import torch

from d2.runtime.attn_kernels.ops import DispatcherWrapper, dispatch_kv_backward, dispatch_no_cp_tensor, dispatch_qkv, nvshmem_barrier_all_on_current_stream
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

    # NOTE: a barrier on stream operation to ensure all previous operations are done, and buffers are reset.
    # TODO: check if there is a better way to do this.
    nvshmem_barrier_all_on_current_stream()
    dispatch_no_cp_tensor(
        dispatcher, q_output_grad, q_input_grad, query_metadata,
    )
    # print("dispatch q_grad layout done")
    # NOTE: In the reversed pass, each token of kv grad is only sent to one copy, so its behavior
    # is the same as query in forward.
    if kv_input_grad is not None:
        # NOTE: a barrier on stream operation to ensure all previous operations are done, and buffers are reset.
        # TODO: check if there is a better way to do this.
        nvshmem_barrier_all_on_current_stream()
        dispatch_kv_backward(dispatcher, kv_output_grad, kv_input_grad, key_value_metadata)


def _both_none_or_neither(a, b):
    return (a is None and b is None) or (a is not None and b is not None)


class n_to_n_dispatch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, query_in: torch.Tensor,
        query_metadata: Metadata,
        rev_query_metadata: Metadata,
        key_value_in: Optional[torch.Tensor]=None,
        key_value_metadata: Optional[Metadata]=None,
        rev_key_value_metadata: Optional[Metadata]=None,
        stream: Optional[torch.cuda.Stream]=None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        # check key_value related tensors
        assert query_metadata.normalized

        assert _both_none_or_neither(key_value_in, key_value_metadata)
        assert _both_none_or_neither(key_value_in, rev_key_value_metadata)

        # The num_head is folded into num_token.
        assert query_in.ndim == 2, "query_in is of shape (num_token, hidden_q)."

        num_token = query_in.shape[0]
        hidden_q = query_in.shape[-1]
        out_query_shape = (query_metadata.num_total_recv_tokens, hidden_q)

        if key_value_in is not None:
            assert num_token == key_value_in.shape[0]
            hidden_kv = key_value_in.shape[-1]
            out_key_value_shape = (key_value_metadata.num_total_recv_tokens, hidden_kv)
        else:
            hidden_kv = 0

        if stream is not None:
            stream_ctx = torch.cuda.stream(stream)
        else:
            stream_ctx = nullcontext()

        with stream_ctx:
            out_query = torch.empty(out_query_shape, device=query_in.device, dtype=query_in.dtype)

            if key_value_in is not None:
                out_key_value = torch.empty(out_key_value_shape, device=key_value_in.device, dtype=key_value_in.dtype)
                key_value_dst_mask = (key_value_metadata.dst_rank != -1).to(torch.bool)

                # NOTE: a barrier on stream operation to ensure all previous operations are done, and buffers are reset.
                # TODO: check if there is a better way to do this.
                nvshmem_barrier_all_on_current_stream()

                dispatch_qkv(
                    dispatcher=DispatcherWrapper.get_instance(),
                    tensor=query_in,
                    dst_tensor=out_query,
                    metadata=query_metadata,
                    kv_tensor=key_value_in,
                    kv_dst_tensor=out_key_value,
                    kv_metadata=key_value_metadata,
                )
            else:
                out_key_value = None
                key_value_dst_mask = None

                # NOTE: a barrier on stream operation to ensure all previous operations are done, and buffers are reset.
                # TODO: check if there is a better way to do this.
                nvshmem_barrier_all_on_current_stream()
                dispatch_no_cp_tensor(
                    dispatcher=DispatcherWrapper.get_instance(),
                    tensor=query_in,
                    dst_tensor=out_query,
                    metadata=query_metadata,
                )

        ctx.query_in_shape = query_in.shape
        ctx.key_value_in_shape = key_value_in.shape if key_value_in is not None else None
        ctx.hidden_q = hidden_q
        ctx.hidden_kv = hidden_kv
        ctx.stream = stream
        ctx.bwd_has_kv = key_value_in is not None
        backward_tensors = []
        backward_tensors.append(key_value_dst_mask)
        # Unpack rev query metadata
        backward_tensors.append(rev_query_metadata.dst_rank)
        backward_tensors.append(rev_query_metadata.dst_offset)
        backward_tensors.append(rev_query_metadata.seq_len)
        backward_tensors.append(rev_query_metadata.num_recv_tokens)
        # Unpack rev key value metadata
        if rev_key_value_metadata is not None:
            backward_tensors.append(rev_key_value_metadata.dst_rank)
            backward_tensors.append(rev_key_value_metadata.dst_offset)
            backward_tensors.append(rev_key_value_metadata.seq_len)
            backward_tensors.append(rev_key_value_metadata.num_recv_tokens)
            backward_tensors.append(rev_key_value_metadata.seq_recv_mask)
            backward_tensors.append(rev_key_value_metadata.recv_seq_lens)
        ctx.save_for_backward(*backward_tensors)

        return out_query, out_key_value

    @staticmethod
    def backward(ctx, out_query_grad, out_key_value_grad):
        # NOTE(yonghao): in PP, the backward pass may not do the same thing
        # as the forward. In this case, the whole layer is wrapped with
        # a torch function and we do not run this part.
        saved_tensors = ctx.saved_tensors
        key_value_dst_mask = saved_tensors[0]
        rev_query_metadata = Metadata(
            dst_rank=saved_tensors[1],
            dst_offset=saved_tensors[2],
            seq_len=saved_tensors[3],
            num_recv_tokens=saved_tensors[4],
            normalized=True,
        )

        if out_key_value_grad is not None:
            assert ctx.bwd_has_kv
            rev_key_value_metadata = Metadata(
                dst_rank=saved_tensors[5],
                dst_offset=saved_tensors[6],
                seq_len=saved_tensors[7],
                num_recv_tokens=saved_tensors[8],
                seq_recv_mask=saved_tensors[9],
                recv_seq_lens=saved_tensors[10],
                normalized=True,
            )
        else:
            assert not ctx.bwd_has_kv
            rev_key_value_metadata = None

        query_in_shape = ctx.query_in_shape
        key_value_in_shape = ctx.key_value_in_shape
        hidden_q = ctx.hidden_q
        hidden_kv = ctx.hidden_kv
        stream = ctx.stream

        query_in_grad = torch.empty(query_in_shape, device=out_query_grad.device, dtype=out_query_grad.dtype)

        if out_key_value_grad is not None:
            assert hidden_kv == out_key_value_grad.shape[-1]
            assert key_value_dst_mask is not None
            # (num_repeats, seq_len, hidden_kv)
            num_seqs = rev_key_value_metadata.recv_seq_lens.shape[0]
            assert rev_key_value_metadata.recv_seq_lens.ndim == 1
            assert rev_key_value_metadata.seq_recv_mask.shape[0] == num_seqs
            assert rev_key_value_metadata.seq_recv_mask.ndim == 2
            cp_degree = rev_key_value_metadata.seq_recv_mask.shape[1]
            key_value_grad_in_shape = (
                key_value_dst_mask.shape[1:] + key_value_in_shape
            )
            assert len(key_value_grad_in_shape) == 3
            assert key_value_grad_in_shape == (cp_degree, query_in_shape[0], hidden_kv)
        else:
            assert key_value_dst_mask is None
            assert hidden_kv == 0

        if stream is not None:
            stream_ctx = torch.cuda.stream(stream)
        else:
            stream_ctx = nullcontext()

        with stream_ctx:
            if out_key_value_grad is not None:
                # NOTE: unlike the key grad, we have make sure it's zeros
                # because some padding parts exists but are not written.
                # TODO(yonghao): modify the communication receive kernel:
                # do the summation before writing to the dst address.
                # as it knows the mask, it can skip redundant summation. (although small)
                key_value_in_grad = torch.zeros(key_value_grad_in_shape,
                                                device=out_key_value_grad.device,
                                                dtype=out_key_value_grad.dtype)
                key_value_in_grad = key_value_in_grad.reshape(-1, hidden_kv)
                dispatch_reverse(
                    q_input_grad=query_in_grad,
                    kv_input_grad=key_value_in_grad,
                    q_output_grad=out_query_grad,
                    kv_output_grad=out_key_value_grad,
                    query_metadata=rev_query_metadata,
                    key_value_metadata=rev_key_value_metadata,
                )

                # gather gradients from all copies along the cp_degree dimension
                key_value_in_grad = (key_value_in_grad.reshape(key_value_grad_in_shape)).sum(dim=0)
                assert key_value_in_grad.shape == key_value_in_shape
            else:
                key_value_in_grad = None
                # NOTE: a barrier on stream operation to ensure all previous operations are done, and buffers are reset.
                # TODO: check if there is a better way to do this.
                nvshmem_barrier_all_on_current_stream()

                dispatch_no_cp_tensor(
                    dispatcher=DispatcherWrapper.get_instance(),
                    tensor=out_query_grad,
                    dst_tensor=query_in_grad,
                    metadata=rev_query_metadata,
                )

        return (
            query_in_grad, None, None,
            key_value_in_grad, None, None,
            None, # stream
        )
