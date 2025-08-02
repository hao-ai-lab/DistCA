
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

kv_to_q_mapping: "torch.Tensor[world_size, pad_len, max_cp_degree, 2]" = torch.zeros(world_size, pad_len, max_cp_degree, 2)
for src_rank in range(world_size):
    for dst_rank in range(world_size):
        pass
        
# (world_size, pad_len, max_cp_degree)
kv_to_q_rank = "torch.Tensor[world_size, pad_len, max_cp_degree, 2]" = torch.zeros(world_size, pad_len, max_cp_degree, 2)

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
