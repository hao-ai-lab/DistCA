import torch

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

    Returns:
        torch.Tensor: A tensor of the same shape as glob_dst_id, containing the
                      calculated destination offset for each token.
    """
    if not glob_dst_id.is_cuda:
        raise TypeError("Input tensor must be a CUDA tensor.")

    original_shape = glob_dst_id.shape
    world_size = original_shape[0]
    
    # Flatten the tensor to 2D (world_size, total_tokens_per_rank) for generic processing
    if glob_dst_id.dim() > 2:
        glob_dst_id_flat = glob_dst_id.reshape(world_size, -1)
    else:
        glob_dst_id_flat = glob_dst_id

    num_tokens = glob_dst_id_flat.shape[1]

    # 1. Count how many tokens each rank sends to every other rank.
    # `counts[d, s]` will be the number of tokens sent from rank `s` to rank `d`.
    one_hot_dest = torch.nn.functional.one_hot(glob_dst_id_flat, num_classes=world_size).long()
    counts = torch.sum(one_hot_dest, dim=1).transpose(0, 1)

    # 2. Calculate the base offset for tokens from each source rank.
    # `base_offsets[d, s]` is the starting offset at destination `d` for all tokens from source `s`.
    # This is an exclusive cumsum along the source rank dimension.
    base_offsets = torch.cumsum(counts, dim=1) - counts

    # 3. Calculate the intra-rank offset for each token.
    # This is the offset within the group of tokens coming from the same source to the same destination.
    # It's a running count for each destination within each source rank.
    intra_rank_offsets = torch.cumsum(one_hot_dest, dim=1) - one_hot_dest
    intra_rank_offsets = torch.sum(intra_rank_offsets, dim=2) # Project back to get the final count

    # 4. Combine the base offset and intra-rank offset.
    # We use the destination IDs to gather the correct base offset for each token.
    # `base_offsets_gathered[s, t]` = base_offset at `dest_id[s,t]` for tokens from `s`
    src_rank_indices = torch.arange(world_size, device=glob_dst_id.device).unsqueeze(1)
    base_offsets_gathered = base_offsets[glob_dst_id_flat, src_rank_indices]
    
    glob_dst_offset_flat = base_offsets_gathered + intra_rank_offsets
    
    # Reshape back to the original input shape
    return glob_dst_offset_flat.reshape(original_shape)


def compute_reverse_comm(
    fwd_dst_id: torch.Tensor,
    fwd_dst_offset: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the reverse communication pattern for the return trip of MoE gating.

    Given the forward communication pattern (where tokens were sent), this function
    calculates the destination rank and offset for each result to be sent back to its
    original location.

    Args:
        fwd_dst_id (torch.Tensor): The forward destination rank for each token.
                                   Shape (world_size, num_tokens, ...).
        fwd_dst_offset (torch.Tensor): The forward destination offset for each token.
                                       Shape (world_size, num_tokens, ...).

    Returns:
        A tuple containing:
        - rev_dst_id (torch.Tensor): The reverse destination (original source rank).
        - rev_dst_offset (torch.Tensor): The reverse offset (original token index).
    """
    if not fwd_dst_id.is_cuda or not fwd_dst_offset.is_cuda:
        raise TypeError("Input tensors must be CUDA tensors.")
        
    original_shape = fwd_dst_id.shape
    world_size = original_shape[0]
    
    # Flatten to (world_size, total_tokens_per_rank)
    fwd_dst_id_flat = fwd_dst_id.reshape(world_size, -1)
    fwd_dst_offset_flat = fwd_dst_offset.reshape(world_size, -1)
    num_tokens_per_rank = fwd_dst_id_flat.shape[1]

    # 1. Determine the number of tokens received by each rank to size the reverse tensors.
    one_hot_dest = torch.nn.functional.one_hot(fwd_dst_id_flat, num_classes=world_size).long()
    num_received = torch.sum(one_hot_dest, dim=(0, 1)) # Sum across both source and token dims
    max_received = torch.max(num_received)

    # 2. Create source rank and token indices that align with the flattened forward tensors.
    src_rank_indices = torch.arange(world_size, device=fwd_dst_id.device).unsqueeze(1).expand_as(fwd_dst_id_flat)
    src_token_indices = torch.arange(num_tokens_per_rank, device=fwd_dst_id.device).unsqueeze(0).expand_as(fwd_dst_id_flat)

    # 3. Create flat tensors for all communication info.
    fwd_d_flat = fwd_dst_id_flat.flatten()
    fwd_o_flat = fwd_dst_offset_flat.flatten()
    src_r_flat = src_rank_indices.flatten()
    src_t_flat = src_token_indices.flatten()

    # 4. Perform the scatter operation to build the reverse map.
    # We use the forward destination (rank, offset) to compute a global destination index.
    # We then write the source (rank, token) into the reverse maps at that index.
    rev_dst_id = torch.zeros(world_size, max_received, dtype=torch.long, device=fwd_dst_id.device)
    rev_dst_offset = torch.zeros(world_size, max_received, dtype=torch.long, device=fwd_dst_id.device)
    
    # Scatter source ranks into the reverse destination id tensor
    rev_dst_id.view(-1).scatter_(0, fwd_d_flat * max_received + fwd_o_flat, src_r_flat)
    
    # Scatter source tokens into the reverse destination offset tensor
    rev_dst_offset.view(-1).scatter_(0, fwd_d_flat * max_received + fwd_o_flat, src_t_flat)

    return rev_dst_id, rev_dst_offset

# Example Usage:
if __name__ == '__main__':
    WORLD_SIZE = 4
    NUM_TOKENS = 8
    CP_DEGREE = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Test Offset Calculation ---
    # Mock global destination IDs for query (2D)
    glob_query_dst_id = torch.randint(0, WORLD_SIZE, (WORLD_SIZE, NUM_TOKENS), device=DEVICE)
    print("Global Query Destination IDs:\n", glob_query_dst_id)
    
    glob_query_dst_offset = compute_dst_offsets(glob_query_dst_id)
    print("\nGlobal Query Destination Offsets:\n", glob_query_dst_offset)

    # Mock global destination IDs for key_value (3D)
    glob_kv_dst_id = torch.randint(0, WORLD_SIZE, (WORLD_SIZE, NUM_TOKENS, CP_DEGREE), device=DEVICE)
    print("\nGlobal KV Destination IDs:\n", glob_kv_dst_id)
    
    glob_kv_dst_offset = compute_dst_offsets(glob_kv_dst_id)
    print("\nGlobal KV Destination Offsets:\n", glob_kv_dst_offset)
    
    # --- Test Reverse Communication Calculation ---
    print("\n--- Computing Reverse Communication ---")
    rev_id, rev_offset = compute_reverse_comm(glob_kv_dst_id, glob_kv_dst_offset)
    
    print("\nReverse Destination IDs (Original Source Ranks):\n", rev_id)
    print("\nReverse Destination Offsets (Original Token Indices):\n", rev_offset)