from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class Metadata:
    dst_rank: torch.Tensor
    dst_offset: torch.Tensor
    seq_len: torch.Tensor
    num_recv_tokens: torch.Tensor
    world_size: int = None
    normalized: bool = False

    def get_slice(self, rank: int):
        assert self.world_size is not None
        return Metadata(
            dst_rank=self.dst_rank[rank],
            dst_offset=self.dst_offset[rank],
            seq_len=self.seq_len[rank],
            num_recv_tokens=self.num_recv_tokens[rank],
            normalized=self.normalized,
        )

    def normalize_dtype(self):
        return Metadata(
            dst_rank=self.dst_rank.to(torch.int32),
            dst_offset=self.dst_offset.to(torch.uint32),
            seq_len=self.seq_len.to(torch.uint32),
            num_recv_tokens=self.num_recv_tokens.to(torch.uint64),
            world_size=self.world_size,
            normalized=True,
        )

    def cuda(self):
        return Metadata(
            dst_rank=self.dst_rank.cuda().contiguous(),
            dst_offset=self.dst_offset.cuda().contiguous(),
            seq_len=self.seq_len.cuda().contiguous(),
            num_recv_tokens=self.num_recv_tokens.cuda().contiguous(),
            world_size=self.world_size,
            normalized=self.normalized,
        )


# NOTE(yonghao): this file will be later offloaded to a cuda kernel in a decentralized paradigm.
@torch.no_grad()
def compute_dst_offsets(
    glob_dst_id: torch.Tensor
) -> torch.Tensor:
    """
    Computes the destination offsets for a dispatch operation.

    This function calculates the correct offset for each token on its destination rank,
    ensuring that tokens are packed contiguously. It assumes tokens from lower-ranking
    source GPUs are placed before tokens from higher-ranking source GPUs.

    Args:
        glob_dst_id (torch.Tensor): A tensor of shape (world_size, num_tokens, ...)
                                    where each value is the destination rank for a token.
                                    Can be 2D for query or 3D for key_value.
                                    Value -1 indicates padding that should be ignored.

    Returns:
        torch.Tensor: A tensor of the same shape as glob_dst_id, containing the
                      calculated destination offset for each token.
    """

    original_shape = glob_dst_id.shape
    world_size = original_shape[0]

    # Flatten the tensor to 2D (world_size, total_tokens_per_rank) for generic processing
    glob_dst_id = glob_dst_id.reshape(world_size, -1)

    # Create mask for valid (non-padding) tokens
    valid_mask = glob_dst_id != -1

    # 1. Count how many tokens each rank sends to every other rank.
    # `counts[d, s]` will be the number of tokens sent from rank `s` to rank `d`.
    # Add world_size to glob_dst_id to handle -1, then subtract later
    shifted_dst_id = glob_dst_id + 1
    one_hot_dest = torch.nn.functional.one_hot(shifted_dst_id.long(), num_classes=world_size + 1).long()
    # Only keep the relevant classes (original 0 to world_size-1)
    one_hot_dest = one_hot_dest[:, :, 1:] * valid_mask.unsqueeze(-1)
    counts = torch.sum(one_hot_dest, dim=1).transpose(0, 1)

    # 2. Calculate the base offset for tokens from each source rank.
    # `base_offsets[d, s]` is the starting offset at destination `d` for all tokens from source `s`.
    # This is an exclusive cumsum along the source rank dimension.
    base_offsets = torch.cumsum(counts, dim=1) - counts

    # 3. Calculate the intra-rank offset for each token.
    # This is the offset within the group of tokens coming from the same source to the same destination.
    # It's a running count for each destination within each source rank.
    intra_rank_offsets = torch.cumsum(one_hot_dest, dim=1) - one_hot_dest
    intra_rank_offsets = torch.sum(intra_rank_offsets * one_hot_dest, dim=2) # Project back to get the final count

    # 4. Combine the base offset and intra-rank offset.
    # We use the destination IDs to gather the correct base offset for each token.
    # `base_offsets_gathered[s, t]` = base_offset at `dest_id[s,t]` for tokens from `s`
    src_rank_indices = torch.arange(world_size, device=glob_dst_id.device).unsqueeze(1)
    # For invalid tokens (padding), we'll gather from index 0 but mask it out later
    gather_ids = glob_dst_id * valid_mask
    base_offsets_gathered = base_offsets[gather_ids, src_rank_indices]

    glob_dst_offset_flat = base_offsets_gathered + intra_rank_offsets
    
    # Mask out offsets for padding tokens with -1
    glob_dst_offset_flat = glob_dst_offset_flat * valid_mask

    # Reshape back to the original input shape
    return glob_dst_offset_flat.reshape(original_shape).to(torch.uint32)


@torch.no_grad()
def compute_reverse_comm(
    fwd_dst_id: torch.Tensor,
    fwd_dst_offset: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the reverse communication pattern for the return trip of attention dispatching.

    Given the forward communication pattern (where tokens were sent), this function
    calculates the destination rank and offset for each result to be sent back to its
    original location.

    Args:
        fwd_dst_id (torch.Tensor): The forward destination rank for each token.
                                   Shape (world_size, num_tokens, ...).
                                   Value -1 indicates padding that should be ignored.
        fwd_dst_offset (torch.Tensor): The forward destination offset for each token.
                                       Shape (world_size, num_tokens, ...).

    Returns:
        A tuple containing:
        - rev_dst_id (torch.Tensor): The reverse destination (original source rank).
        - rev_dst_offset (torch.Tensor): The reverse offset (original token index).
    """

    original_shape = fwd_dst_id.shape
    world_size = original_shape[0]

    # Flatten to (world_size, total_tokens_per_rank)
    fwd_dst_id_flat = fwd_dst_id.reshape(world_size, -1)
    fwd_dst_offset_flat = fwd_dst_offset.reshape(world_size, -1)
    num_tokens_per_rank = fwd_dst_id_flat.shape[1]

    # Create mask for valid (non-padding) tokens
    valid_mask = fwd_dst_id_flat != -1

    # 1. Determine the number of tokens received by each rank to size the reverse tensors.
    one_hot_dest = torch.nn.functional.one_hot(torch.where(valid_mask, fwd_dst_id_flat, 0).long(), num_classes=world_size).long()
    one_hot_dest = one_hot_dest * valid_mask.unsqueeze(-1)  # Mask out padding tokens
    num_received = torch.sum(one_hot_dest, dim=(0, 1)) # Sum across both source and token dims
    max_received = torch.max(num_received)

    # 2. Create source rank and token indices that align with the flattened forward tensors.
    src_rank_indices = torch.arange(world_size, device=fwd_dst_id.device).unsqueeze(1).expand_as(fwd_dst_id_flat)
    src_token_indices = torch.arange(num_tokens_per_rank, device=fwd_dst_id.device).unsqueeze(0).expand_as(fwd_dst_id_flat)

    # 3. Create flat tensors for all communication info.
    fwd_d_flat = fwd_dst_id_flat.flatten().long()
    fwd_o_flat = fwd_dst_offset_flat.flatten().long()
    src_r_flat = src_rank_indices.flatten().long()
    src_t_flat = src_token_indices.flatten().long()
    valid_mask_flat = valid_mask.flatten()

    # 4. Perform the scatter operation to build the reverse map.
    # We use the forward destination (rank, offset) to compute a global destination index.
    # We then write the source (rank, token) into the reverse maps at that index.
    rev_dst_id = torch.zeros(world_size, max_received, dtype=torch.long, device=fwd_dst_id.device)
    rev_dst_offset = torch.zeros(world_size, max_received, dtype=torch.long, device=fwd_dst_id.device)

    # Only scatter valid tokens
    valid_indices = torch.where(valid_mask_flat)[0]
    valid_fwd_d = fwd_d_flat[valid_indices]
    valid_fwd_o = fwd_o_flat[valid_indices]
    valid_src_r = src_r_flat[valid_indices]
    valid_src_t = src_t_flat[valid_indices]

    # Scatter source ranks into the reverse destination id tensor
    rev_dst_id.view(-1).scatter_(0, valid_fwd_d * max_received + valid_fwd_o, valid_src_r)

    # Scatter source tokens into the reverse destination offset tensor
    rev_dst_offset.view(-1).scatter_(0, valid_fwd_d * max_received + valid_fwd_o, valid_src_t)

    # mask -1 based on num_received: those indices >= num_received should be masked
    rev_dst_id = rev_dst_id.masked_fill(num_received.unsqueeze(1) <= torch.arange(max_received, device=fwd_dst_id.device), -1).to(torch.int32)
    rev_dst_offset = rev_dst_offset.masked_fill(num_received.unsqueeze(1) <= torch.arange(max_received, device=fwd_dst_id.device), 0).to(torch.uint32)

    return rev_dst_id, rev_dst_offset


@torch.no_grad()
def compute_num_recv_tokens(
    dst_id: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        dst_id: tensor of shape (world_size, num_tokens, ...). The dst rank for each token. -1 is padding.
    Returns:
        num_recv_tokens: tensor of shape (world_size, world_size + 1). The number of tokens received by from rank.
        The last dimension is the total number of tokens received.
    """
    world_size = dst_id.shape[0]
    dst_id = dst_id.reshape(world_size, -1)
    one_hot_dst = torch.nn.functional.one_hot((dst_id + 1).long(), num_classes=world_size + 1).long()[:, :, 1:]
    num_recv_tokens = one_hot_dst.sum(dim=1).T
    num_recv_tokens = torch.cat([num_recv_tokens, num_recv_tokens.sum(dim=1, keepdim=True)], dim=1)
    return num_recv_tokens.long().to(torch.uint64)  # size_t


@torch.no_grad()
def compute_metadata(
    seq_len: torch.Tensor,  # shape of (world_size, max_num_local_seqs)
    global_dispatch: torch.Tensor,  # shape of (world_size, max_num_local_seqs, max_cp_degree)
):
    """
    Args:
        length (Tensor): shape (world_size, max_num_local_seqs). The length of each sequence.
        global_dispatch (Tensor): shape (world_size, max_num_local_seqs, max_cp_degree). value -1 is padding.
        The decision tensor. Recording the ranks that each sequence is dispatched to.
    Returns:
        dst_offset (Tensor): dst begin offset of each sequence (send side).
        rev_dst_rank (Tensor): dst ranks of each sequence (from recv side back to send side).
        rev_dst_offset (Tensor): dst offsets of each sequence (from recv side back to send side).
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

    fwd_metadata = Metadata(
        dst_rank=global_dispatch.reshape(dispatch_shape),
        dst_offset=seq_begin_offset.reshape(dispatch_shape),
        seq_len=seq_len,
        num_recv_tokens=num_recv_tokens,
        world_size=world_size,
    )
    rev_metadata = Metadata(
        dst_rank=rev_dst_rank,
        dst_offset=rev_dst_offset,
        seq_len=rev_seq_len,
        num_recv_tokens=rev_num_received_tokens,
        world_size=world_size,
    )
    return fwd_metadata, rev_metadata


######## Correctness testing tools ########
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


# Example Usage:
def test_v0():
    WORLD_SIZE = 4
    NUM_TOKENS = 8
    CP_DEGREE = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    def orchestrate(tensor: torch.Tensor, dst_id: torch.Tensor, dst_offset: torch.Tensor, dst_tensor: torch.Tensor | None = None):
        world_size = dst_id.shape[0]
        one_hot_dest = torch.nn.functional.one_hot(dst_id.reshape(world_size, -1) + 1, num_classes=world_size + 1).long()[:, :, 1:]
        num_received = torch.sum(one_hot_dest, dim=(0, 1)) # Sum across both source and token dims
        max_received = torch.max(num_received)
        if dst_tensor is None:
            dst_tensor = torch.zeros(world_size, max_received, dtype=tensor.dtype, device=tensor.device)
            orig_shape = dst_tensor.shape
        else:
            orig_shape = dst_tensor.shape
            dst_tensor = dst_tensor.reshape(world_size, -1)

        if dst_id.dim() == 3:
            sp_size = dst_id.shape[2]
        else:
            sp_size = None

        for i in range(world_size):
            if sp_size:
                for j in range(sp_size):
                    ids = dst_id[i, :, j]
                    offsets = dst_offset[i, :, j]
                    mask = ids != -1
                    ids = ids[mask]
                    offsets = offsets[mask]
                    dst_tensor[ids, offsets] = tensor[i][mask]
            else:
                ids = dst_id[i]
                offsets = dst_offset[i]
                # remove indices with value -1 (dummy value as masks)
                mask = ids != -1
                ids = ids[mask]
                offsets = offsets[mask]
                dst_tensor[ids, offsets] = tensor[i][mask]

        return dst_tensor.reshape(orig_shape)

    dst_id = torch.randint(-1, WORLD_SIZE, (WORLD_SIZE, NUM_TOKENS, CP_DEGREE), device=DEVICE)
    tensor = torch.randn(WORLD_SIZE, NUM_TOKENS, device=DEVICE)

    # forward communication
    dst_offset = compute_dst_offsets(dst_id)
    dst_tensor = orchestrate(tensor, dst_id, dst_offset)
    # reverse communication
    rev_dst_id, rev_dst_offset = compute_reverse_comm(dst_id, dst_offset)
    back_tensor = torch.zeros_like(dst_id, dtype=dst_tensor.dtype, device=DEVICE)
    back_tensor = orchestrate(dst_tensor, rev_dst_id, rev_dst_offset, dst_tensor=back_tensor)
    back_tensor_dedup = back_tensor.sum(dim=2) / (back_tensor != 0).sum(dim=2)
    assert torch.allclose(back_tensor_dedup.unsqueeze(2).repeat(1, 1, CP_DEGREE) * (back_tensor != 0), back_tensor)
    assert torch.allclose(tensor, back_tensor_dedup)


def gen_seq_lens(world_size: int, num_seqs: int, total_len: int) -> torch.Tensor:
    ratio = torch.rand((world_size, num_seqs)) + 0.25 / num_seqs   # Use a min value to guarantee that the sequence is not too short (0 after rounding)
    ratio = ratio / ratio.sum(dim=1, keepdim=True)
    seq_len = (ratio * total_len).round().int()
    seq_len_total = seq_len.sum(dim=1)
    seq_len_total_error = seq_len_total - total_len
    seq_len[:, -1] -= seq_len_total_error
    return seq_len


def test_v1():
    torch.manual_seed(0)
    WORLD_SIZE = 2
    NUM_SEQS = 3
    CP_DEGREE = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    HIDDEN_SIZE = 1
    TOTAL_SEQ_LEN = 1024
    seq_len = gen_seq_lens(WORLD_SIZE, NUM_SEQS, TOTAL_SEQ_LEN).long()
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
    print("test of v1 metadata passed")


if __name__ == '__main__':
    test_v1()