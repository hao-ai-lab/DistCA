import torch

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



# Example Usage:
if __name__ == '__main__':
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