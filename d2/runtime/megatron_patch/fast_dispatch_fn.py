import torch

from d2.runtime.attn_kernels.dispatch import (
    fast_a2a_qkv, fast_a2a_attn_out
)
from d2.runtime.fast_alltoall_metadata import FastAlltoAllMetadata


# NOTE: we have to add the arg stream here because backward cannot be assigned
# a stream outside this function.
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
            grad_v = torch.empty_like(grad_recv_v)
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

