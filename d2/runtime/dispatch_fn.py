from typing import Iterable
import torch

from d2.runtime.attn_kernels.dispatch import (
    pre_a2a_qkv, post_a2a_qkv, pre_a2a_attn_out, post_a2a_attn_out
)
from d2.runtime.attn_kernels.ops import fast_a2a
from d2.runtime.metadata import AlltoAllMetadata


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
        metadata: AlltoAllMetadata, bwd_metadata: AlltoAllMetadata,
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
            pre_a2a_qkv(
                q, k, v, metadata.kv_replica_mask, q_seq_lens, k_seq_lens,
                *metadata.send_memcpy_metadata,
                is_fwd=True, instance_id=dispatcher_id,
            )
            save_tensors.append(bwd_metadata.kv_replica_mask)
        else:
            q = q.contiguous()
            assert k is None and v is None
            pre_a2a_attn_out(
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
            post_a2a_qkv(
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
            post_a2a_attn_out(
                grad_attn_out,
                *ctx.saved_tensors,
                switch_buffer=switch_buffer,
                instance_id=ctx.dispatcher_id,
            )
            # the dummy position k,v should have a None gradient.
            return (grad_attn_out,) + (None,) * 6


# No stream because this should always run on the compute stream.
# FIXME: currently duplicating above for easy debugging
class pre_all2all_layout_transfer_for_cuda_graph_fwd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        send_seqlens: tuple[torch.Tensor, torch.Tensor],
        kv_replica_mask: torch.Tensor,
        send_memcpy_metadata: Iterable[torch.Tensor],
        # metadata: AlltoAllMetadata, bwd_metadata: AlltoAllMetadata,
        dispatcher_id: int, is_qkv: bool,
    ):
        # a signal tensor output to maintain the autograd graph dependency.
        signal = torch.empty((1,), dtype=q.dtype, device=q.device)
        save_tensors = []

        if is_qkv:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            q_seq_lens, k_seq_lens = send_seqlens
            pre_a2a_qkv(
                q, k, v, kv_replica_mask, q_seq_lens, k_seq_lens,
                *send_memcpy_metadata,
                is_fwd=True, instance_id=dispatcher_id,
            )
        else:
            q = q.contiguous()
            assert k is None and v is None
            pre_a2a_attn_out(
                q, metadata.seq_lens[0].send_seqlens,
                *metadata.send_memcpy_metadata, instance_id=dispatcher_id
            )
        return q, k, v
    
    @staticmethod
    def backward(ctx, q_grad, k_grad, v_grad):
        return (q_grad, k_grad, v_grad) + (None,) * 5


class pre_all2all_layout_transfer_for_cuda_graph_bwd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        recv_seqlens: tuple[torch.Tensor, torch.Tensor],
        kv_replica_mask: torch.Tensor,
        recv_memcpy_metadata: Iterable[torch.Tensor],
        tensor_shapes: Iterable[tuple[int, ...]],
        dispatcher_id: int, is_qkv: bool,
    ):
        signal = torch.empty((1,), dtype=q.dtype, device=q.device)
        ctx.bwd_recv_shapes = tuple(tensor_shapes)
        ctx.dispatcher_id = dispatcher_id
        ctx.is_qkv = is_qkv
        ctx.save_for_backward(
            kv_replica_mask,
            *recv_seqlens,
            *recv_memcpy_metadata,
        )
        return signal

    @staticmethod
    def backward(ctx, signal_grad):
        switch_buffer = ctx.dispatcher_id is None
        if ctx.is_qkv:
            grad_q_shape = ctx.bwd_recv_shapes[0]
            grad_k_shape = ctx.bwd_recv_shapes[1]
            grad_q = torch.empty(grad_q_shape, dtype=signal_grad.dtype, device=signal_grad.device)
            grad_k = torch.zeros(grad_k_shape, dtype=signal_grad.dtype, device=signal_grad.device)  # must be zero not empty
            grad_v = torch.zeros_like(grad_k)
            post_a2a_qkv(
                grad_q, grad_k, grad_v,
                # qkv dispatch, q_seq_tokens, k_seq_tokens, v_seq_tokens, q_offset, k_offset, v_offset
                *ctx.saved_tensors,
                is_fwd=False,
                switch_buffer=switch_buffer,
                instance_id=ctx.dispatcher_id,
            )
            grad_k = grad_k.sum(dim=0)
            grad_v = grad_v.sum(dim=0)
            return (grad_q, grad_k, grad_v) + (None,) * 6
        else:
            grad_attn_out_shape = ctx.bwd_recv_shapes[0]
            grad_attn_out = torch.empty(
                grad_attn_out_shape, dtype=signal_grad.dtype, device=signal_grad.device
            )
            post_a2a_attn_out(
                grad_attn_out,
                *ctx.saved_tensors,
                switch_buffer=switch_buffer,
                instance_id=ctx.dispatcher_id,
            )
            # the dummy position k,v should have a None gradient.
            return (grad_attn_out,) + (None,) * 8


# we have to add the arg stream here because backward cannot be assigned
# a stream outside this function.
class all_to_all(torch.autograd.Function):
    @staticmethod
    def forward(ctx, signal: torch.Tensor, metadata: AlltoAllMetadata,
                bwd_metadata: AlltoAllMetadata, dispatcher_id: int,
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
    def forward(ctx, signal: torch.Tensor, metadata: AlltoAllMetadata,
                bwd_metadata: AlltoAllMetadata, dispatcher_id: int,
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
            post_a2a_qkv(
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
            post_a2a_attn_out(
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
        if is_qkv:
            assert bwd_metadata.kv_grad_send_dedup is not None
            saved_tensors.extend([
                bwd_metadata.kv_grad_send_dedup.main_copy_mask,
                bwd_metadata.kv_grad_send_dedup.num_copies,
                bwd_metadata.kv_grad_send_dedup.copy_start_id,
            ])
        ctx.save_for_backward(*saved_tensors)

        if is_qkv:
            return recv_q, recv_k, recv_v
        else:
            return recv_attn_out

    @staticmethod
    def backward(ctx, *grads):
        if ctx.is_qkv:
            (kv_replica_mask, q_shard_lens, kv_shard_lens,
             q_memcpy, k_memcpy, v_memcpy,
             main_copy_mask, num_copies, copy_start_id) = ctx.saved_tensors
            grad_q, grad_k, grad_v = grads
            memcpy_dedup_args = (num_copies, copy_start_id, kv_shard_lens)
            pre_a2a_qkv(
                grad_q, grad_k, grad_v, kv_replica_mask,
                q_shard_lens, kv_shard_lens,
                q_memcpy, k_memcpy, v_memcpy,
                is_fwd=False, instance_id=ctx.dispatcher_id,
                kv_grad_copy_shard_mask=main_copy_mask,
                pre_a2a_grad_acc_args=memcpy_dedup_args
            )
        else:
            grad_q, = grads
            pre_a2a_attn_out(grad_q, *ctx.saved_tensors,
                             instance_id=ctx.dispatcher_id,)
        signal_grad = torch.empty((1,), dtype=grad_q.dtype, device=grad_q.device)
        return (signal_grad,) + (None,) * 4
