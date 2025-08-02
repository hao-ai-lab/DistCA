
"""
Assume we have 4 GPUs
- Rank 0: [1024] + [16 * K] # only the 16K needs to dispatch
- Rank 1: [1024]
- Rank 2: [1024]
- Rank 3: [1024]

Really, we only need to do CP on the 16K sequence. 
Assume we evenly split the 16K sequence into 8 shards to 4 GPUs.
"""
import torch

K = 1024
mlp_seq_len: torch.Tensor = torch.tensor([
    [1024, ] + [2 * K] * 8,
    [1024, ] + [0] * 8,
    [1024, ] + [0] * 8,
    [1024, ] + [0] * 8,
])

mlp_num_seqs: torch.Tensor = torch.tensor([
    [1 + 8],
    [1],
    [1],
    [1],
])

# How each sequence should dispatch 
mlp_q_dispatch = torch.tensor([
    [0,] + [0, 1, 2, 3, 3, 2, 1, 0],
    [1,] + [-1] * 8,
    [2,] + [-1] * 8,
    [3,] + [-1] * 8,
])

# (world_size, pad_len, max_cp_degree, 2)
world_size = 4
pad_len = 9
max_cp_degree = 8

# kv_to_q_mapping[src_rank, seq_id, cp_idx, 0] = q_rank (where the corresponding Q shard is)
# kv_to_q_mapping[src_rank, seq_id, cp_idx, 1] = q_seq_id (local sequence ID of the corresponding Q shard)
kv_to_q_mapping = torch.full((world_size, pad_len, max_cp_degree, 2), -1, dtype=torch.int64)

# For Rank 0: has context parallel sequences
# seq_id=0: [1024] -> no CP, maps to itself
kv_to_q_mapping[0, 0, 0, :] = torch.tensor([0, 0])

# seq_id=1-8: CP shards, each KV shard maps to all SUBSEQUENT Q shards
for seq_id in range(1, 9):  # seq_id 1-8 are the 8 CP shards
    kv_shard_idx = seq_id - 1  # 0-7 for the 8 shards
    # This KV shard corresponds to Q shards seq_id, seq_id+1, ..., 8
    for q_shard_offset in range(8 - kv_shard_idx):  # remaining Q shards
        if q_shard_offset < max_cp_degree:
            target_q_seq_id = seq_id + q_shard_offset
            if target_q_seq_id <= 8:
                kv_to_q_mapping[0, seq_id, q_shard_offset, :] = torch.tensor([0, target_q_seq_id])

# For other ranks: only have local sequences, no CP
for rank in range(1, 4):
    kv_to_q_mapping[rank, 0, 0, :] = torch.tensor([rank, 0])

# kv_to_q_rank[src_rank, seq_id, cp_idx] = position of this KV among all KVs mapping to the same Q
# Shape: (world_size, pad_len, max_cp_degree) - NOTE: 3D, not 4D!
kv_to_q_rank = torch.full((world_size, pad_len, max_cp_degree), -1, dtype=torch.int64)

# For Rank 0's CP sequences
for seq_id in range(1, 9):
    kv_shard_idx = seq_id - 1
    for q_shard_offset in range(8 - kv_shard_idx):  # remaining Q shards
        if q_shard_offset < max_cp_degree:
            target_q_seq_id = seq_id + q_shard_offset
            if target_q_seq_id <= 8:
                # This KV shard is at position kv_shard_idx among all KVs mapping to this Q
                kv_to_q_rank[0, seq_id, q_shard_offset] = kv_shard_idx

# For non-CP sequences
kv_to_q_rank[0, 0, 0] = 0  # Rank 0, seq_id=0
for rank in range(1, 4):
    kv_to_q_rank[rank, 0, 0] = 0


kv_context_size = torch.tensor([
    [0, 1024, 0, 2 * K, 4 * K, 6 * K, 8 * K, 10 * K, 12 * K, 14 * K],
    [0, 1024, 0, 0, 0, 0, 0, 0, 0],
    [0, 1024, 0, 0, 0, 0, 0, 0, 0],
    [0, 1024, 0, 0, 0, 0, 0, 0, 0],
])

q_to_num_kv_seq = num_kv_to_q = torch.tensor([
    [1] + [1, 2, 3, 4, 5, 6, 7, 8],
    [1] + [0] * 8,
    [1] + [0] * 8,
    [1] + [0] * 8,
])
q_to_num_kv_token = num_total_kv_to_q = torch.tensor([
    [1024] + [i * 2 * K for i in range(1, 9)],
    [1024] + [0] * 7,
    [1024] + [0] * 7,
    [1024] + [0] * 7,
])


compute_e2e_metadata(
    mlp_seq_len,  # shape of (world_size, max_num_local_seqs)
    mlp_num_seqs, # shape of (world_size,)
    mlp_q_dispatch,  # shape of (world_size, max_num_local_seqs)
    kv_to_q_mapping,
    kv_to_q_rank,
    kv_context_size,
    q_to_num_kv_seq, # or num_kv_to_q
    q_to_num_kv_token, # or num_total_kv_to_q
    return_intermediate=False,
)
