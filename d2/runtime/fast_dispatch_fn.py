import torch

from d2.runtime.attn_kernels.dispatch import (
    fast_a2a_qkv, fast_a2a_attn_out, pre_fast_a2a_qkv, post_fast_a2a_qkv,
    pre_fast_a2a_attn_out, post_fast_a2a_attn_out
)
from d2.runtime.attn_kernels.ops import fast_a2a
from d2.runtime.fast_alltoall_metadata import FastAlltoAllMetadata


"""
We split dispatch into 3 autograd functions to only place all2all on the
communication stream, as the memcpy operations takes a lot of threads,
and if allocated on the communication stream, it has a race condition with
the computation.

When splitting dispatch into 3 autograd functions, our calling order is:
Forward: signal = pre_all2all_layout_transfer(qkv_MLP) ->
         signal = all_to_all(signal) ->
         qkv_ATTN = post_all2all_layout_transfer(signal).
During backward, it turns to be:
    signal_grad = pre_all2all_layout_transfer.backward(grad_qkv_ATTN) ->
    signal_grad = all_to_all.backward(signal_grad) ->
    grad_qkv_ATTN = post_all2all_layout_transfer.backward(signal_grad).
The same applies for attn_out dispatch, but _MLP and _ATTN are swapped.
"""


# No stream because this should always run on the compute stream.
class pre_all2all_layout_transfer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        metadata: FastAlltoAllMetadata, bwd_metadata: FastAlltoAllMetadata,
        dispatcher_id: int, is_qkv: bool,
    ):
        # a signal tensor output to maintain the autograd graph dependency.
        signal = torch.empty((1,), dtype=q.dtype, device=q.device)
        save_tensors = []

        if is_qkv:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            q_seq_lens = metadata.seq_lens[0].send_seqlens
            k_seq_lens = metadata.seq_lens[1].send_seqlens
            pre_fast_a2a_qkv(
                q, k, v, metadata.kv_replica_mask, q_seq_lens, k_seq_lens,
                *metadata.send_memcpy_metadata,
                is_fwd=True, instance_id=dispatcher_id,
            )
            save_tensors.append(bwd_metadata.kv_replica_mask)
        else:
            q = q.contiguous()
            assert k is None and v is None
            pre_fast_a2a_attn_out(
                q, metadata.seq_lens[0].send_seqlens,
                *metadata.send_memcpy_metadata, instance_id=dispatcher_id
            )
        save_tensors.extend([
            # q seq_lens and k seq_lens for the backward receiver.
            *(sl.recv_seqlens for sl in bwd_metadata.seq_lens),
            # TODO: send memcpy metadata at fwd is the recv memcpy metadata at bwd
            # we should directly reuse it and can thus merge the two metadata together.
            *bwd_metadata.recv_memcpy_metadata,
        ])
        ctx.dispatcher_id = dispatcher_id
        ctx.is_qkv = is_qkv
        ctx.bwd_recv_shapes = tuple(ts.recv_shape for ts in bwd_metadata.tensor_shape)
        ctx.save_for_backward(*save_tensors)
        return signal

    @staticmethod
    def backward(ctx, signal_grad):
        switch_buffer = ctx.dispatcher_id is None
        if ctx.is_qkv:
            grad_q_shape = ctx.bwd_recv_shapes[0]
            grad_k_shape = ctx.bwd_recv_shapes[1]
            grad_q = torch.empty(grad_q_shape, dtype=signal_grad.dtype, device=signal_grad.device)
            grad_k = torch.zeros(grad_k_shape, dtype=signal_grad.dtype, device=signal_grad.device)
            grad_v = torch.zeros_like(grad_k)
            post_fast_a2a_qkv(
                grad_q, grad_k, grad_v,
                # qkv dispatch, q_seq_tokens, k_seq_tokens, v_seq_tokens, q_offset, k_offset, v_offset
                *ctx.saved_tensors,
                is_fwd=False,
                switch_buffer=switch_buffer,
                instance_id=ctx.dispatcher_id,
            )
            grad_k = grad_k.sum(dim=0)
            grad_v = grad_v.sum(dim=0)
            return (grad_q, grad_k, grad_v) + (None,) * 4
        else:
            grad_attn_out_shape = ctx.bwd_recv_shapes[0]
            grad_attn_out = torch.empty(
                grad_attn_out_shape, dtype=signal_grad.dtype, device=signal_grad.device
            )
            post_fast_a2a_attn_out(
                grad_attn_out,
                *ctx.saved_tensors,
                switch_buffer=switch_buffer,
                instance_id=ctx.dispatcher_id,
            )
            # the dummy position k,v should have a None gradient.
            return (grad_attn_out,) + (None,) * 6


# we have to add the arg stream here because backward cannot be assigned
# a stream outside this function.
class all_to_all(torch.autograd.Function):
    @staticmethod
    def forward(ctx, signal: torch.Tensor, metadata: FastAlltoAllMetadata,
                bwd_metadata: FastAlltoAllMetadata, dispatcher_id: int,
                stream: torch.cuda.Stream = None):
        if stream is None or metadata.single_stream:
            stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            fast_a2a(
                *metadata.fa2a_metadata,
                metadata.my_rank_send_offset, metadata.my_rank_recv_offset, metadata.my_rank_send_sz,
                instance_id=dispatcher_id,
            )
        ctx.dispatcher_id = dispatcher_id
        ctx.stream = stream
        ctx.my_rank_send_offset = bwd_metadata.my_rank_send_offset
        ctx.my_rank_recv_offset = bwd_metadata.my_rank_recv_offset
        ctx.my_rank_send_sz = bwd_metadata.my_rank_send_sz
        ctx.save_for_backward(*bwd_metadata.fa2a_metadata)
        return signal
    @staticmethod
    def backward(ctx, grad_signal: torch.Tensor):
        stream = ctx.stream
        with torch.cuda.stream(stream):
            fast_a2a(
                *ctx.saved_tensors,
                ctx.my_rank_send_offset, ctx.my_rank_recv_offset, ctx.my_rank_send_sz,
                instance_id=ctx.dispatcher_id,
            )
        return (grad_signal,) + (None,) * 4


class post_all2all_layout_transfer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, signal: torch.Tensor, metadata: FastAlltoAllMetadata,
                bwd_metadata: FastAlltoAllMetadata, dispatcher_id: int,
                is_qkv: bool,):
        # NOTE: no stream because this should always run on the compute stream.
        saved_tensors = []
        switch_buffer = dispatcher_id is None
        if is_qkv:
            recv_q_shape = metadata.tensor_shape[0].recv_shape
            recv_k_shape = metadata.tensor_shape[1].recv_shape
            recv_q = torch.empty(recv_q_shape, dtype=signal.dtype, device=signal.device)
            recv_k = torch.empty(recv_k_shape, dtype=signal.dtype, device=signal.device)
            recv_v = torch.empty_like(recv_k)
            post_fast_a2a_qkv(
                recv_q, recv_k, recv_v, None,
                metadata.seq_lens[0].recv_seqlens, metadata.seq_lens[1].recv_seqlens,
                *metadata.recv_memcpy_metadata,
                is_fwd=True,
                switch_buffer=switch_buffer,
                instance_id=dispatcher_id,
            )
            saved_tensors.append(metadata.kv_replica_mask)
        else:
            recv_shape = metadata.tensor_shape[0].recv_shape
            recv_attn_out = torch.empty(
                recv_shape, dtype=signal.dtype, device=signal.device)
            post_fast_a2a_attn_out(
                recv_attn_out,
                metadata.seq_lens[0].recv_seqlens,
                *metadata.recv_memcpy_metadata,
                switch_buffer=switch_buffer,
                instance_id=dispatcher_id,
            )
        ctx.dispatcher_id = dispatcher_id
        ctx.is_qkv = is_qkv
        saved_tensors.extend([
            *(sl.send_seqlens for sl in bwd_metadata.seq_lens),
            *bwd_metadata.send_memcpy_metadata,
        ])
        ctx.save_for_backward(*saved_tensors)

        if is_qkv:
            return recv_q, recv_k, recv_v
        else:
            return recv_attn_out

    @staticmethod
    def backward(ctx, *grads):
        if ctx.is_qkv:
            grad_q, grad_k, grad_v = grads
            pre_fast_a2a_qkv(
                grad_q, grad_k, grad_v, *ctx.saved_tensors,
                is_fwd=False, instance_id=ctx.dispatcher_id,
            )
        else:
            grad_q, = grads
            pre_fast_a2a_attn_out(grad_q, *ctx.saved_tensors,
                                  instance_id=ctx.dispatcher_id,)
        signal_grad = torch.empty((1,), dtype=grad_q.dtype, device=grad_q.device)
        return (signal_grad,) + (None,) * 4


################ QKV
class qkv_dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                metadata: FastAlltoAllMetadata, bwd_metadata: FastAlltoAllMetadata,
                stream: torch.cuda.Stream):
        if metadata.single_stream:
            stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            recv_q_shape = metadata.tensor_shape[0].recv_shape
            recv_k_shape = metadata.tensor_shape[1].recv_shape
            recv_q = torch.empty(
                recv_q_shape, dtype=q.dtype, device=q.device)
            recv_k = torch.empty(
                recv_k_shape, dtype=k.dtype, device=k.device)
            recv_v = torch.empty(
                recv_k_shape, dtype=v.dtype, device=v.device)

            fast_a2a_qkv(
                q, k, v, metadata.kv_replica_mask,
                recv_q, recv_k, recv_v,
                # q seq_num_tokens, k seq_num_tokens,
                metadata.seq_lens[0].send_seqlens, metadata.seq_lens[1].send_seqlens,
                # q recv_seq_num_tokens, k recv_seq_num_tokens,
                metadata.seq_lens[0].recv_seqlens, metadata.seq_lens[1].recv_seqlens,
                # q send_buffer_offset, k send_buffer_offset, v send_buffer_offset,
                *metadata.send_memcpy_metadata,
                # q recv_buffer_offset, k recv_buffer_offset, v recv_buffer_offset,
                *metadata.recv_memcpy_metadata,
                # sender_send_disp, sender_transfer_sz,
                # sender_recv_disp, recver_transfer_sz,
                *metadata.fa2a_metadata,
                metadata.my_rank_send_offset, metadata.my_rank_recv_offset,
                metadata.my_rank_send_sz,
                is_fwd=True,
                # NOTE: in what case can we guarantee that
                # the buffer is always already switched?
                switch_buffer=True,
            )

        ctx.stream = stream
        ctx.q_grad_shape = bwd_metadata.tensor_shape[0].recv_shape
        ctx.k_grad_shape = bwd_metadata.tensor_shape[1].recv_shape
        ctx.grad_my_rank_send_offset = bwd_metadata.my_rank_send_offset
        ctx.grad_my_rank_recv_offset = bwd_metadata.my_rank_recv_offset
        ctx.grad_my_rank_send_sz = bwd_metadata.my_rank_send_sz

        kv_replica_mask = bwd_metadata.kv_replica_mask
        grad_seq_len_q_send = bwd_metadata.seq_lens[0].send_seqlens
        grad_seq_len_k_send = bwd_metadata.seq_lens[1].send_seqlens
        grad_seq_len_q_recv = bwd_metadata.seq_lens[0].recv_seqlens
        grad_seq_len_k_recv = bwd_metadata.seq_lens[1].recv_seqlens
        grad_qkv_send_offset = bwd_metadata.send_memcpy_metadata
        grad_qkv_recv_offset = bwd_metadata.recv_memcpy_metadata
        grad_fa2a_metadata = bwd_metadata.fa2a_metadata
        grad_metadata = [
            kv_replica_mask,
            grad_seq_len_q_send, grad_seq_len_k_send, grad_seq_len_q_recv, grad_seq_len_k_recv,
            *grad_qkv_send_offset, *grad_qkv_recv_offset, *grad_fa2a_metadata
        ]
        ctx.save_for_backward(*grad_metadata)
        return recv_q, recv_k, recv_v

    @staticmethod
    def backward(ctx, grad_recv_q: torch.Tensor, grad_recv_k: torch.Tensor, grad_recv_v: torch.Tensor):
        stream = ctx.stream

        with torch.cuda.stream(stream):
            grad_q_shape = ctx.q_grad_shape
            grad_k_shape = ctx.k_grad_shape
            grad_q = torch.empty(grad_q_shape, dtype=grad_recv_q.dtype, device=grad_recv_q.device)
            grad_k = torch.empty(grad_k_shape, dtype=grad_recv_k.dtype, device=grad_recv_k.device)
            grad_v = torch.empty_like(grad_k)
            fast_a2a_qkv(
                grad_recv_q, grad_recv_k, grad_recv_v, ctx.saved_tensors[0],
                grad_q, grad_k, grad_v,
                # q seq_num_tokens, k seq_num_tokens,
                *ctx.saved_tensors[1:3],
                # q recv_seq_num_tokens, k recv_seq_num_tokens,
                *ctx.saved_tensors[3:5],
                # q send_buffer_offset, k send_buffer_offset, v send_buffer_offset,
                *ctx.saved_tensors[5:8],
                # q recv_buffer_offset, k recv_buffer_offset, v recv_buffer_offset,
                *ctx.saved_tensors[8:11],
                # sender_send_disp, sender_transfer_sz,
                # sender_recv_disp, recver_transfer_sz,
                *ctx.saved_tensors[11:15],
                ctx.grad_my_rank_send_offset, ctx.grad_my_rank_recv_offset,
                ctx.grad_my_rank_send_sz,
                is_fwd=False,
            )
            grad_k = grad_k.sum(dim=0)
            grad_v = grad_v.sum(dim=0)

        return (grad_q, grad_k, grad_v) + (None,) * 3


################ Attention out
class attn_out_dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn_out: torch.Tensor,
                metadata: FastAlltoAllMetadata, bwd_metadata: FastAlltoAllMetadata,
                stream: torch.cuda.Stream = None):
        if stream is None or metadata.single_stream:
            stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            recv_shape = metadata.tensor_shape[0].recv_shape
            recv_attn_out = torch.empty(
                recv_shape, dtype=attn_out.dtype, device=attn_out.device)
            fast_a2a_attn_out(
                attn_out,
                recv_attn_out,
                # seq_num_tokens,
                metadata.seq_lens[0].send_seqlens,
                # recv_seq_num_tokens,
                metadata.seq_lens[0].recv_seqlens,
                # send_buffer_offset,
                *metadata.send_memcpy_metadata,
                # recv_buffer_offset,
                *metadata.recv_memcpy_metadata,
                # sender_send_disp, sender_transfer_sz,
                # sender_recv_disp, recver_transfer_sz,
                *metadata.fa2a_metadata,
                metadata.my_rank_send_offset, metadata.my_rank_recv_offset,
                metadata.my_rank_send_sz,
            )
        ctx.stream = stream
        ctx.rev_recv_shape = bwd_metadata.tensor_shape[0].recv_shape
        ctx.grad_my_rank_send_offset = bwd_metadata.my_rank_send_offset
        ctx.grad_my_rank_recv_offset = bwd_metadata.my_rank_recv_offset
        ctx.grad_my_rank_send_sz = bwd_metadata.my_rank_send_sz
        grad_seq_len_send = bwd_metadata.seq_lens[0].send_seqlens
        grad_seq_len_recv = bwd_metadata.seq_lens[0].recv_seqlens
        grad_qkv_send_offset = bwd_metadata.send_memcpy_metadata[0]
        grad_qkv_recv_offset = bwd_metadata.recv_memcpy_metadata[0]
        grad_fa2a_metadata = bwd_metadata.fa2a_metadata
        metadata = [
            grad_seq_len_send, grad_seq_len_recv, grad_qkv_send_offset, grad_qkv_recv_offset,
            *grad_fa2a_metadata
        ]
        ctx.save_for_backward(*metadata)
        return recv_attn_out

    @staticmethod
    def backward(ctx, grad_recv_attn_out: torch.Tensor):
        stream = ctx.stream
        with torch.cuda.stream(stream):
            grad_attn_out = torch.empty(
                ctx.rev_recv_shape, dtype=grad_recv_attn_out.dtype,
                device=grad_recv_attn_out.device
            )
            fast_a2a_attn_out(
                grad_recv_attn_out, grad_attn_out,
                # seq_num_tokens,
                ctx.saved_tensors[0],
                # recv_seq_num_tokens,
                ctx.saved_tensors[1],
                # send_buffer_offset,
                ctx.saved_tensors[2],
                # recv_buffer_offset,
                ctx.saved_tensors[3],
                # sender_send_disp, sender_transfer_sz,
                # sender_recv_disp, recver_transfer_sz,
                *ctx.saved_tensors[4:8],
                ctx.grad_my_rank_send_offset, ctx.grad_my_rank_recv_offset,
                ctx.grad_my_rank_send_sz,
            )
        return (grad_attn_out,) + (None,) * 3

