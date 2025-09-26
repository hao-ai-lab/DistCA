from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch

_Tensor_Or_Tensor_List = Union[torch.Tensor, Sequence[torch.Tensor]]


@dataclass
class SeqLens:
    send_seqlens: _Tensor_Or_Tensor_List
    recv_seqlens: _Tensor_Or_Tensor_List

    def get_slice(self, rank):
        return SeqLens(
            self.send_seqlens[rank], self.recv_seqlens[rank],
        )

    def normalize(self):
        if isinstance(self.send_seqlens, torch.Tensor):
            return SeqLens(
                self.send_seqlens.cuda().to(torch.int64).contiguous(),
                self.recv_seqlens.cuda().to(torch.int64).contiguous()
            )
        return SeqLens(
            tuple(t.cuda().to(torch.int64).contiguous() for t in self.send_seqlens),
            tuple(t.cuda().to(torch.int64).contiguous() for t in self.recv_seqlens)
        )


@dataclass
class LogicalShape:
    """
    Logical shape for input and output tensors. By logical,
    KV send shape is (cp_degree, num_tokens, hidden_size);
    Other tensors have the same shape as physical shape.
    This shape is used to construct empty buffers for recv.
    """
    send_shape: Union[torch.Size, Sequence[torch.Size]]
    recv_shape: Union[torch.Size, Sequence[torch.Size]]
    def get_slice(self, rank):
        return LogicalShape(
            self.send_shape[rank], self.recv_shape[rank],
        )


@dataclass
class AlltoAllMetadata:
    # sender_send_offset, sender_transfer_sz, sender_recv_offset, recver_transfer_sz
    fa2a_metadata: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    # List of (world_size,) tensors, each of shape (num_sequences,). If a slice, no world_size dimension.
    # metadata on the sender side:
    #   the offset to copy each sequence to the buffer, ordered by the sender's sequence idx
    #       NOTE: for kv sender's sequence idx, to make it directly available to the reverse operation,
    #         we use the (cp_id, seq_idx) order instead of the (seq_idx, cp_id) order
    #   the offset to copy each sequence from the buffer, ordered by the recver's sequence idx
    send_memcpy_metadata: Sequence[_Tensor_Or_Tensor_List]
    recv_memcpy_metadata: Sequence[_Tensor_Or_Tensor_List]
    my_rank_send_offset: Union[int, List[int]]
    my_rank_recv_offset: Union[int, List[int]]
    my_rank_send_sz: Union[int, List[int]]
    seq_lens: Sequence[SeqLens]
    # Num received / send tokens for each tensor (i.e. (q,k) or (attn,))/
    # This is to construct the recv buffer size.
    # NOTE: for Q/KV backward, this is just a placeholder and we don't use it.
    tensor_shape: Sequence[LogicalShape]
    # List of kv replica mask for each rank. (or this rank)
    # shape is (num_local_seqs, cp_degree).
    kv_replica_mask: Optional[_Tensor_Or_Tensor_List] = None
    # Debug setting
    single_stream: bool = False
    # For cuda graph, the number of sequences and max cp degree are padded.
    # Below records the original values and should be passed to memcpy kernels.
    send_num_seqs: Optional[Sequence[int]] = None
    recv_num_seqs: Optional[Sequence[int]] = None
    max_cp_degree: Optional[int] = None

    def __better_print__(self):
        """Convert the tensor size to MB. This is just for debugging.
        
        Usage:
        ```
        fa2a_metadata: AlltoAllMetadata = ...
        d = fa2a_metadata.__better_print__()
        print(d)
        # or even fancier
        import rich
        rich.print(d)
        ```
        
        """
        # Print in the order of MB for tensor
        (
            __sender_send_offset,
            __sender_transfer_sz,
            __sender_recv_offset,
            __recver_transfer_sz,
        ) = self.fa2a_metadata

        def convert_to_mb(x):
            y = x // (1024 ** 2)
            return y.to('cpu')
        
        sender_send_offset = convert_to_mb(__sender_send_offset)
        sender_transfer_sz = convert_to_mb(__sender_transfer_sz)
        sender_recv_offset = convert_to_mb(__sender_recv_offset)
        recver_transfer_sz = convert_to_mb(__recver_transfer_sz)

        __send_memcpy_metadata = self.send_memcpy_metadata
        _send_memcpy_metadata = []
        for i, t in enumerate(__send_memcpy_metadata):
            if isinstance(t, torch.Tensor):
                _send_memcpy_metadata.append(convert_to_mb(t))
            else:
                _send_memcpy_metadata.append(t)
        send_memcpy_metadata = tuple(_send_memcpy_metadata)

        __recv_memcpy_metadata = self.recv_memcpy_metadata
        _recv_memcpy_metadata = []
        for i, t in enumerate(__recv_memcpy_metadata):
            if isinstance(t, torch.Tensor):
                _recv_memcpy_metadata.append(convert_to_mb(t))
            else:
                _recv_memcpy_metadata.append(t)
        recv_memcpy_metadata = tuple(_recv_memcpy_metadata)

        my_rank_send_offset = [i // (1024 ** 2) for i in self.my_rank_send_offset] if isinstance(self.my_rank_send_offset, list) else self.my_rank_send_offset // (1024 ** 2)
        my_rank_recv_offset = [i // (1024 ** 2) for i in self.my_rank_recv_offset] if isinstance(self.my_rank_recv_offset, list) else self.my_rank_recv_offset // (1024 ** 2)
        my_rank_send_sz = [i // (1024 ** 2) for i in self.my_rank_send_sz] if isinstance(self.my_rank_send_sz, list) else self.my_rank_send_sz // (1024 ** 2)
        seq_lens = self.seq_lens
        tensor_shape = self.tensor_shape
        kv_replica_mask = self.kv_replica_mask
        single_stream = self.single_stream

        return dict(
            sender_send_offset_mb=sender_send_offset,
            sender_transfer_sz_mb=sender_transfer_sz,
            sender_recv_offset_mb=sender_recv_offset,
            recver_transfer_sz_mb=recver_transfer_sz,
            send_memcpy_metadata='omit',
            recv_memcpy_metadata='omit',
            my_rank_send_offset_mb=my_rank_send_offset,
            my_rank_recv_offset_mb=my_rank_recv_offset,   
            my_rank_send_sz_mb=my_rank_send_sz,
            seq_lens=seq_lens,
            # tensor_shape=tensor_shape,
            # kv_replica_mask=kv_replica_mask,
            # single_stream=single_stream,
        )

    def get_slice(self, rank):
        """
        Returns the metadata for the given rank.
        """
        fa2a_metadata = tuple(t[rank] for t in self.fa2a_metadata)
        send_memcpy_metadata = tuple(t[rank] for t in self.send_memcpy_metadata)
        recv_memcpy_metadata = tuple(t[rank] for t in self.recv_memcpy_metadata)
        seq_lens = tuple(
            sl.get_slice(rank) for sl in self.seq_lens
        )
        tensor_shape = tuple(
            ts.get_slice(rank) for ts in self.tensor_shape
        )
        return AlltoAllMetadata(
            fa2a_metadata, send_memcpy_metadata, recv_memcpy_metadata,
            self.my_rank_send_offset[rank],
            self.my_rank_recv_offset[rank],
            self.my_rank_send_sz[rank],
            seq_lens,
            tensor_shape,
            kv_replica_mask=(
                self.kv_replica_mask[rank] if self.kv_replica_mask is not None else None
            ),
            single_stream=self.single_stream,
        )

    def normalize(self):
        """To device and transfer dtype."""
        return AlltoAllMetadata(
            tuple(t.cuda().to(torch.uint64).contiguous() for t in self.fa2a_metadata),
            tuple(t.cuda().to(torch.int64).contiguous() for t in self.send_memcpy_metadata),
            tuple(t.cuda().to(torch.int64).contiguous() for t in self.recv_memcpy_metadata),
            self.my_rank_send_offset,
            self.my_rank_recv_offset,
            self.my_rank_send_sz,
            tuple(t.normalize() for t in self.seq_lens),
            self.tensor_shape,
            kv_replica_mask=(
                self.kv_replica_mask.cuda().to(torch.int8).contiguous()
                if self.kv_replica_mask is not None else None
            ),
            single_stream=self.single_stream,
        )


def _get_diag(tensor: torch.Tensor, world_size: int):
    assert tensor.shape == (world_size, world_size)
    return torch.diagonal(tensor).flatten().cpu().tolist()


def _get_my_rank_from_metadata(fa2a_metadata: Sequence[torch.Tensor]):
    (sender_send_disp, sender_transfer_sz,
        sender_recv_disp, recver_transfer_sz) = fa2a_metadata
    world_size = sender_send_disp.shape[0]
    return {
        "my_rank_send_offset": _get_diag(sender_send_disp, world_size),
        "my_rank_recv_offset": _get_diag(sender_recv_disp, world_size),
        "my_rank_send_sz": _get_diag(sender_transfer_sz, world_size)
    }


def compute_reverse_a2a_layout_metadata(
    fwd_metadata: AlltoAllMetadata
):
    # TODO: as bwd values are mainly the same as fwd values
    # we should only store those that are different.

    # during backward, the tensor is copied back to the original location.
    send_memcpy_metadata = fwd_metadata.recv_memcpy_metadata
    recv_memcpy_metadata = fwd_metadata.send_memcpy_metadata
    # the tensor is sent back to the original location
    fwd_sender_send_disp, fwd_sender_transfer_sz, fwd_sender_recv_disp, fwd_recver_transfer_sz = fwd_metadata.fa2a_metadata

    # fwd bytes received from each rank -> bwd bytes sent to each rank
    bwd_sender_transfer_sz = fwd_recver_transfer_sz
    bwd_recver_transfer_sz = fwd_sender_transfer_sz
    # fwd_sender_send_disp: [i][j]: rank i send to rank j, offset at rank i
    # fwd_sender_recv_disp: [i][j]: rank i receive from rank j, offset at rank j
    # bwd_sender_send_disp: [j][i]: rank j send to rank i, offset at rank j
    #  == fwd offset that rank j recv from rank i, offset at rank j == fwd_sender_recv_disp.T
    bwd_sender_send_disp = fwd_sender_recv_disp.transpose(0, 1)
    bwd_sender_recv_disp = fwd_sender_send_disp.transpose(0, 1)
    bwd_fa2a_metadata = (
        bwd_sender_send_disp, bwd_sender_transfer_sz, bwd_sender_recv_disp,
        bwd_recver_transfer_sz
    )

    bwd_seqlens = tuple(
        SeqLens(seq_len.recv_seqlens, seq_len.send_seqlens)
        for seq_len in fwd_metadata.seq_lens
    )
    bwd_tensor_shape = tuple(
        LogicalShape(ts.recv_shape, ts.send_shape)
        for ts in fwd_metadata.tensor_shape
    )

    my_rank_vals = _get_my_rank_from_metadata(bwd_fa2a_metadata)
    return AlltoAllMetadata(
        bwd_fa2a_metadata, send_memcpy_metadata, recv_memcpy_metadata,
        **my_rank_vals, seq_lens=bwd_seqlens, tensor_shape=bwd_tensor_shape,
        kv_replica_mask=fwd_metadata.kv_replica_mask,
    )
