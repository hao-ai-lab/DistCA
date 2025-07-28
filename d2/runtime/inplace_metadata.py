from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from megatron.core.packed_seq_params import PackedSeqParams


@dataclass
class Metadata:
    dst_rank: torch.Tensor
    dst_offset: torch.Tensor
    seq_len: torch.Tensor
    num_recv_tokens: torch.Tensor
    # NOTE: used for kv gradient communication.
    # This is a sequence-scale mask, recording the number of sequences received
    # from each rank.
    seq_recv_mask: Optional[torch.Tensor] = None
    # NOTE: used for kv gradient communication.
    # This records the length of all sequences received to this rank.
    recv_seq_lens: Optional[torch.Tensor] = None
    # Number of sequences on each rank. This is used when this Metadata
    # describes all ranks in the world, and thus is padded to the one with
    # the most sequences. Otherwise, this should be None.
    num_seqs: Optional[torch.Tensor] = None
    world_size: int = None
    normalized: bool = False
    num_total_recv_tokens: Union[int, list[int]] = None

    def get_slice(self, rank: int):
        assert self.world_size is not None
        assert self.num_seqs is not None

        num_seqs = self.num_seqs[rank]
        return Metadata(
            dst_rank=self.dst_rank[rank][:num_seqs],
            dst_offset=self.dst_offset[rank][:num_seqs],
            seq_len=self.seq_len[rank][:num_seqs],
            num_recv_tokens=self.num_recv_tokens[rank], # this is of shape (world_size,)
            seq_recv_mask=self.seq_recv_mask[rank] if self.seq_recv_mask is not None else None,
            recv_seq_lens=self.recv_seq_lens[rank] if self.recv_seq_lens is not None else None,
            normalized=self.normalized,
            num_total_recv_tokens=self.num_total_recv_tokens[rank],
        )

    def normalize_dtype(self):
        return Metadata(
            dst_rank=self.dst_rank.to(torch.int32),
            dst_offset=self.dst_offset.to(torch.uint32),
            seq_len=self.seq_len.to(torch.uint32),
            num_recv_tokens=self.num_recv_tokens.to(torch.uint64),
            seq_recv_mask=self.seq_recv_mask.to(torch.uint32) if self.seq_recv_mask is not None else None,
            recv_seq_lens=self.recv_seq_lens.to(torch.uint32) if self.recv_seq_lens is not None else None,
            world_size=self.world_size,
            normalized=True,
            num_total_recv_tokens=self.num_total_recv_tokens,
        )

    def cuda(self):
        return Metadata(
            dst_rank=self.dst_rank.cuda().contiguous(),
            dst_offset=self.dst_offset.cuda().contiguous(),
            seq_len=self.seq_len.cuda().contiguous(),
            num_recv_tokens=self.num_recv_tokens.cuda().contiguous(),
            seq_recv_mask=self.seq_recv_mask.cuda().contiguous() if self.seq_recv_mask is not None else None,
            recv_seq_lens=self.recv_seq_lens.cuda().contiguous() if self.recv_seq_lens is not None else None,
            world_size=self.world_size,
            normalized=self.normalized,
            num_total_recv_tokens=self.num_total_recv_tokens,
        )


@torch.no_grad()
def compute_metadata(
    seq_len: torch.Tensor,  # shape of (world_size, max_num_local_seqs)
    global_dispatch: torch.Tensor,  # shape of (world_size, max_num_local_seqs, max_cp_degree)
) -> Tuple[Metadata, Metadata]:
    """
    Args:
        seq_len (Tensor): shape (world_size, max_num_local_seq_shards). The length of each sequence.
        global_dispatch (Tensor): shape (world_size, max_num_local_seq_shards, max_cp_degree). value -1 is padding.
        The decision tensor. Recording the ranks that each sequence is dispatched to.
    Returns:
        fwd_metadata, rev_metadata: Metadata for forward and reverse communication.
    """
    world_size = global_dispatch.shape[0]
    max_num_local_seqs = global_dispatch.shape[1]
    assert seq_len.shape == (world_size, max_num_local_seqs)
    assert global_dispatch.dtype == torch.int64
    assert seq_len.dtype == torch.int64

    # Query. No max cp degree dimension.
    dispatch_shape = global_dispatch.shape
    if global_dispatch.dim() == 2:
        global_dispatch = global_dispatch.unsqueeze(-1)
    # max_cp_degree = global_dispatch.shape[2]

    ######## Forward side metadata
    seq_to_dst = F.one_hot(global_dispatch + 1, num_classes=world_size + 1)[:, :, :, 1:]
    # (world_size, max_num_local_seqs, max_cp_degree, world_size) [i][j][k][l]:
    # for each sequence ([i][j]), for each dispatch ([k]), the number of sequence sent to rank [l] (either 0 or seq_len[i][j])
    tokens_to_dst_per_dispatch = seq_to_dst * seq_len.unsqueeze(-1).unsqueeze(-1)
    # all tokens by global index i, sending to rank j
    tokens_to_dst = tokens_to_dst_per_dispatch.reshape(-1, world_size)
    seq_begin_offset = tokens_to_dst.cumsum(dim=0) - tokens_to_dst
    # masking and back to the original shape
    seq_begin_offset = seq_begin_offset.reshape(*seq_to_dst.shape) * seq_to_dst
    seq_begin_offset = seq_begin_offset.sum(dim=-1)
    # compute the sequence level offset
    # (do not consider number of tokens. This is only used for reverse communication's indexing in scatter_)
    seq_to_dst_flatten = seq_to_dst.reshape(-1, world_size)
    seq_offset = seq_to_dst_flatten.cumsum(dim=0) - seq_to_dst_flatten
    seq_offset = seq_offset.reshape(*seq_to_dst.shape) * seq_to_dst
    seq_offset = seq_offset.sum(dim=-1)

    # number of tokens received from each rank ([i][j] means tokens received at rank i from rank j), last column is the total number.
    num_recv_tokens = tokens_to_dst_per_dispatch.reshape(world_size, -1, world_size).sum(dim=1).transpose(0, 1)
    num_recv_tokens = torch.cat([num_recv_tokens, num_recv_tokens.sum(dim=1, keepdim=True)], dim=1)

    ######## reverse side metadata:
    seq_rank_expanded = torch.arange(world_size, device=global_dispatch.device).reshape(world_size, 1, 1).expand_as(global_dispatch)
    # seq_offset = seq_len.cumsum(dim=1) - seq_len # NOTE: this is the final offset after deduplication.
    # expanding by max_cp_degree to allow concurrently receiving from multiple ranks
    seq_len_expanded = seq_len.unsqueeze(2).expand_as(global_dispatch).transpose(1, 2)
    seq_offset_expanded = seq_len_expanded.reshape(world_size, -1).cumsum(dim=1).reshape(seq_len_expanded.shape) - seq_len_expanded
    seq_offset_expanded = seq_offset_expanded.transpose(1, 2)
    seq_len_expanded = seq_len_expanded.transpose(1, 2)
    # reverse side communication
    num_received_seqs = seq_to_dst.reshape(-1, world_size).sum(0)
    max_rev_seqs = num_received_seqs.max()
    rev_dst_rank = torch.zeros(world_size, max_rev_seqs, dtype=torch.int64, device=global_dispatch.device)
    rev_dst_offset = torch.zeros(world_size, max_rev_seqs, dtype=torch.int64, device=global_dispatch.device)
    rev_seq_len = torch.zeros(world_size, max_rev_seqs, dtype=torch.int64, device=global_dispatch.device)

    # Create valid mask for non-padding entries
    valid_mask = global_dispatch != -1
    # Flatten all tensors for vectorized processing
    # flatten to [world_size * max_num_local_seqs * max_cp_degree]
    valid_mask_flat = valid_mask.flatten()
    global_dispatch_flat = global_dispatch.flatten()
    seq_rank_expanded_flat = seq_rank_expanded.flatten()
    seq_offset_expanded_flat = seq_offset_expanded.flatten()
    seq_len_expanded_flat = seq_len_expanded.flatten()

    # Use flatten valid indices to choose the valid sequence dispatch entries
    valid_indices = torch.where(valid_mask_flat)[0] # ranging from 0 to world_size * max_num_local_seqs * max_cp_degree - 1
    # from global dispatch id to the dst rank
    valid_dst_ranks = global_dispatch_flat[valid_indices]
    # from global dispatch id to the src rank
    valid_src_ranks = seq_rank_expanded_flat[valid_indices]
    valid_src_offsets = seq_offset_expanded_flat[valid_indices]
    valid_src_seq_lens = seq_len_expanded_flat[valid_indices]
    dst_seq_local_offset = seq_offset.flatten()[valid_indices]

    # Compute global destination (flatten) indices for scatter operation
    # For each valid entry, compute where it should go in the reverse tensors
    global_dst_indices = valid_dst_ranks * max_rev_seqs + dst_seq_local_offset

    rev_dst_rank.view(-1).scatter_(0, global_dst_indices, valid_src_ranks)
    rev_dst_offset.view(-1).scatter_(0, global_dst_indices, valid_src_offsets)
    rev_seq_len.view(-1).scatter_(0, global_dst_indices, valid_src_seq_lens)

    # masking
    rev_dst_rank = rev_dst_rank.masked_fill(
        num_received_seqs.unsqueeze(1) <= torch.arange(max_rev_seqs, device=rev_dst_rank.device), -1)
    rev_dst_offset = rev_dst_offset.masked_fill(
        num_received_seqs.unsqueeze(1) <= torch.arange(max_rev_seqs, device=rev_dst_offset.device), 0)
    rev_seq_len = rev_seq_len.masked_fill(
        num_received_seqs.unsqueeze(1) <= torch.arange(max_rev_seqs, device=rev_seq_len.device), 0)
    # number of tokens received in the reverse communication. This equals the number of tokens sent in the forward communication.
    rev_num_received_tokens = tokens_to_dst_per_dispatch.reshape(world_size, -1, world_size).sum(dim=1)
    rev_num_received_tokens = torch.cat([rev_num_received_tokens, rev_num_received_tokens.sum(dim=1, keepdim=True)], dim=1)

    # If this is kv (has cp degree), we add the sequence-cp mask to the reverse communication metadata.
    seq_recv_mask = None
    if len(dispatch_shape) == 3:
        seq_recv_mask = global_dispatch.reshape(dispatch_shape) >= 0
    num_seqs = (seq_len > 0).sum(dim=1)
    num_seqs_rev = (rev_seq_len > 0).sum(dim=1)

    fwd_metadata = Metadata(
        dst_rank=global_dispatch.reshape(dispatch_shape),
        dst_offset=seq_begin_offset.reshape(dispatch_shape),
        seq_len=seq_len,
        num_recv_tokens=num_recv_tokens,
        num_seqs=num_seqs,
        world_size=world_size,
        num_total_recv_tokens=num_recv_tokens[:, -1].tolist(),
    )
    rev_metadata = Metadata(
        dst_rank=rev_dst_rank,
        dst_offset=rev_dst_offset,
        seq_len=rev_seq_len,
        num_recv_tokens=rev_num_received_tokens,
        seq_recv_mask=seq_recv_mask,
        recv_seq_lens=seq_len if seq_recv_mask is not None else None,
        num_seqs=num_seqs_rev,
        world_size=world_size,
        num_total_recv_tokens=rev_num_received_tokens[:, -1].tolist(),
    )
    return fwd_metadata, rev_metadata


@torch.no_grad()
def compute_attn_layout_seqlens(
    seq_shard_len: torch.Tensor, seq_shard_cumsum: torch.Tensor,
    dispatch: torch.Tensor,
):
    """
    Compute the cu_seqlens_q and cu_seqlens_kv for the attention layout.
    NOTE: both inputs and outputs are the global value, so it has `world_size` dimension.
    Args:
        seq_shard_len: shape (world_size, max_num_local_seqs), length of each sequence shard.
          A sequence shard is the unit sending to a rank.
        seq_shard_cumsum: shape (world_size, max_num_local_seqs). Cumulative number of tokens
          for the sequence shard's context. This is to compute the KV seqlens of the shard.
        dispatch: shape (world_size, max_num_local_seqs). The rank that each sequence
          shard is dispatched to. The value is -1 for padding.
    """
    world_size = dispatch.shape[0]
    max_num_local_seqs = dispatch.shape[1]
    assert dispatch.dim() == 2
    assert seq_shard_len.shape == (world_size, max_num_local_seqs)
    assert seq_shard_cumsum.shape == (world_size, max_num_local_seqs)
    assert dispatch.dtype == torch.int64
    assert seq_shard_len.dtype == torch.int64
    assert seq_shard_cumsum.dtype == torch.int64
    # dispatch[i, j] = the rank that sequence [i,j] is dispatched to.
    flatten_dispatch = dispatch.flatten()

    flatten_dispatch_one_hot = F.one_hot(flatten_dispatch + 1, num_classes=world_size + 1)[:, 1:]
    # shape: (world_size, seq_len, world_size)
    local_indices_flat = (
        # cumsum: the id of this sequence at the dst rank.
        (flatten_dispatch_one_hot.cumsum(dim=0) - 1) * flatten_dispatch_one_hot
    ).sum(dim=1).reshape(-1)
    # if dispatch[i, j] = k, then local_indices_flat[i, j, k] = l means sequence [i,j] is at out_sequence [k,l]
    # out_seqlens_q[k, l] = seq_shard_len[i, j]
    max_num_seq = int(local_indices_flat.max().item() + 1)
    scatter_index = (flatten_dispatch * (flatten_dispatch >= 0)) * max_num_seq + local_indices_flat

    src_seqlens = seq_shard_len.flatten()
    src_seq_lens_kv = seq_shard_cumsum.flatten()

    out_seqlens_q = torch.zeros(world_size * max_num_seq, dtype=torch.int64, device=dispatch.device)
    out_seqlens_kv = torch.zeros(world_size * max_num_seq, dtype=torch.int64, device=dispatch.device)
    out_seqlens_q.scatter_(0, scatter_index, src_seqlens)
    out_seqlens_kv.scatter_(0, scatter_index, src_seq_lens_kv)
    out_seqlens_q = out_seqlens_q.reshape(world_size, max_num_local_seqs)
    out_seqlens_kv = out_seqlens_kv.reshape(world_size, max_num_local_seqs)

    num_local_seqs_recv = local_indices_flat.reshape(-1, world_size).max(dim=0)[0] + 1

    cu_seqlens_q = out_seqlens_q.cumsum(dim=1)
    cu_seqlens_kv = out_seqlens_kv.cumsum(dim=1)
    max_seqlen_q = out_seqlens_q.max(dim=1)[0]
    max_seqlen_kv = out_seqlens_kv.max(dim=1)[0]

    return cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, num_local_seqs_recv


def mlp_layout_packed_params(seq_lens: torch.Tensor):
    """
    Compute the MLP layout packed_seq_params. MLP layout guarantees seqlens_q == seqlens_kv.
    This is mainly for RoPE.
    """
    cu_seqlens = torch.cat([
        torch.zeros((1,), dtype=seq_lens.dtype, device=seq_lens.device),
        seq_lens.cumsum(dim=0)
    ])
    max_seqlen = seq_lens.max()
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )
    return packed_seq_params


######## Correctness testing tools ########
def test_local_proj_metadata(world_size: int, seq_len: int, offset: int):
    """
    Test tool generating the easiest case: Each rank holds a sequence,
    and is sent to a rank with an offset. (0 means sending to itself)
    """
    num_recv_tokens = torch.zeros((world_size, world_size + 1), dtype=torch.int64)
    num_recv_tokens[:, -1] = seq_len
    i_diag = torch.arange(world_size, dtype=torch.int64)
    j_diag = (i_diag + offset) % world_size
    # the diagonal of num_recv_tokens is the seq_len
    fwd_recv_tokens = num_recv_tokens.clone()
    fwd_recv_tokens[j_diag, i_diag] = seq_len

    bwd_j_diag = (i_diag - offset) % world_size
    bwd_recv_tokens = num_recv_tokens.clone()
    bwd_recv_tokens[bwd_j_diag, i_diag] = seq_len

    mlp_to_attn_metadata = Metadata(
        dst_rank=torch.tensor(
            [(i + offset) % world_size for i in range(world_size)]
        ).reshape(world_size, 1),
        dst_offset=torch.tensor(
            [0] * world_size
        ).reshape(world_size, 1),
        seq_len=torch.tensor(
            [seq_len] * world_size
        ).reshape(world_size, 1),
        num_recv_tokens=fwd_recv_tokens,
        num_seqs=torch.tensor([1] * world_size).reshape(world_size, 1),
        world_size=world_size,
        num_total_recv_tokens=[seq_len] * world_size,
    )
    attn_to_mlp_metadata = Metadata(
        dst_rank=torch.tensor(
            [(i - offset) % world_size for i in range(world_size)]
        ).reshape(world_size, 1),
        dst_offset=mlp_to_attn_metadata.dst_offset.clone(),
        seq_len=mlp_to_attn_metadata.seq_len.clone(),
        num_recv_tokens=bwd_recv_tokens,
        num_seqs=mlp_to_attn_metadata.num_seqs.clone(),
        world_size=world_size,
        num_total_recv_tokens=[seq_len] * world_size,
    )
    mlp_to_attn_kv_metadata = Metadata(
        dst_rank=mlp_to_attn_metadata.dst_rank.clone().unsqueeze(-1),
        dst_offset=mlp_to_attn_metadata.dst_offset.clone().unsqueeze(-1),
        seq_len=mlp_to_attn_metadata.seq_len.clone(),
        num_recv_tokens=fwd_recv_tokens.clone(),
        num_seqs=mlp_to_attn_metadata.num_seqs.clone(),
        world_size=world_size,
        num_total_recv_tokens=[seq_len] * world_size,
    )
    mlp_to_attn_kv_grad_metadata = Metadata(
        dst_rank=attn_to_mlp_metadata.dst_rank.clone(),
        dst_offset=attn_to_mlp_metadata.dst_offset.clone(),
        seq_len=attn_to_mlp_metadata.seq_len.clone(),
        num_recv_tokens=bwd_recv_tokens.clone(),
        num_seqs=attn_to_mlp_metadata.num_seqs.clone(),
        world_size=world_size,
        seq_recv_mask=torch.ones(world_size, 1, 1),
        recv_seq_lens=mlp_to_attn_metadata.seq_len.clone(),
        num_total_recv_tokens=[seq_len] * world_size,
    )
    return mlp_to_attn_metadata, attn_to_mlp_metadata, mlp_to_attn_kv_metadata, mlp_to_attn_kv_grad_metadata


@torch.no_grad()
def orchestrate_simulate(tensor: torch.Tensor, output_tensor: torch.Tensor, metadata: Metadata):
    assert tensor.dim() == 3    # (world_size, num_tokens, hidden_dim)
    world_size = tensor.shape[0]
    # handle sending rank-by-rank:
    for src_rank in range(world_size):
        dst_rank = metadata.dst_rank[src_rank]
        dst_offset = metadata.dst_offset[src_rank]
        seq_lens = metadata.seq_len[src_rank]
        acu_tokens = 0
        for j, rs in enumerate(dst_rank):
            seq_len = seq_lens[j]
            seq = tensor[src_rank][acu_tokens:acu_tokens + seq_len]
            if dst_rank.dim() == 1:
                rank = rs
                if rank >= 0:
                    try:
                        output_tensor[rank][dst_offset[j]: dst_offset[j] + seq_len] = seq
                    except RuntimeError as e:
                        print(f"{src_rank=}, {rank=}, {dst_offset[j]=}, {dst_offset[j] + seq_len=}, {seq_len=}, {output_tensor.shape, seq.shape, acu_tokens, tensor.shape}")
                        raise e
            else:
                for k, rank in enumerate(rs):
                    if rank >= 0:
                        output_tensor[rank][dst_offset[j][k]: dst_offset[j][k] + seq_len] = seq
            acu_tokens += seq_len
    return output_tensor


def gen_seq_lens(world_size: int, num_seqs: int, total_len: int) -> torch.Tensor:
    ratio = torch.rand((world_size, num_seqs)) + 0.25 / num_seqs   # Use a min value to guarantee that the sequence is not too short (0 after rounding)
    ratio = ratio / ratio.sum(dim=1, keepdim=True)
    seq_len = (ratio * total_len).round().int()
    seq_len_total = seq_len.sum(dim=1)
    seq_len_total_error = seq_len_total - total_len
    seq_len[:, -1] -= seq_len_total_error
    return seq_len


def test():
    torch.manual_seed(0)
    WORLD_SIZE = 2
    NUM_SEQS = 3
    CP_DEGREE = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    HIDDEN_SIZE = 4
    TOTAL_SEQ_LEN = 1024

    seq_len = gen_seq_lens(WORLD_SIZE, NUM_SEQS, TOTAL_SEQ_LEN).long()
    seq_len = seq_len.to(DEVICE)
    global_dispatch = torch.randint(-1, WORLD_SIZE, (WORLD_SIZE, NUM_SEQS, CP_DEGREE), device=DEVICE)
    global_dispatch_err = torch.max(global_dispatch, dim=2, keepdim=True)[0] < 0    # error means there is no receiver. We need to at least assign it one.
    global_dispatch[:, :, -1:] += global_dispatch_err.int()

    tensor = torch.rand((WORLD_SIZE, TOTAL_SEQ_LEN, HIDDEN_SIZE), device=DEVICE) + 0.1
    fwd_metadata, rev_metadata = compute_metadata(seq_len, global_dispatch)

    # forward
    max_recv_tokens = fwd_metadata.num_recv_tokens.max()
    output_tensor = torch.zeros((WORLD_SIZE, max_recv_tokens, HIDDEN_SIZE), device=DEVICE, dtype=tensor.dtype)
    output_tensor = orchestrate_simulate(tensor, output_tensor, fwd_metadata)

    # reverse
    rev_metadata.num_recv_tokens.max()
    rev_tensor = torch.zeros((WORLD_SIZE, TOTAL_SEQ_LEN * CP_DEGREE, HIDDEN_SIZE), device=DEVICE, dtype=output_tensor.dtype)
    rev_tensor = orchestrate_simulate(output_tensor, rev_tensor, rev_metadata)
    rev_tensor = rev_tensor.reshape(WORLD_SIZE, CP_DEGREE, TOTAL_SEQ_LEN, HIDDEN_SIZE)
    rev_tensor_dedup = rev_tensor.sum(dim=1) / (rev_tensor != 0).sum(dim=1)

    torch.testing.assert_close(rev_tensor_dedup.unsqueeze(1).repeat(1, CP_DEGREE, 1, 1) * (rev_tensor != 0), rev_tensor)
    torch.testing.assert_close(tensor, rev_tensor_dedup)
    print("test metadata passed")


if __name__ == '__main__':
    test()
