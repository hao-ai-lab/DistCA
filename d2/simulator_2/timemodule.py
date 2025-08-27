from rich import print

K = 1024

base_seq_len = K * 64
attn_base_time = 12.5020
# linear_base_time = (13.5 + 8.5) # mlp + qkvo
# linear_base_time = (13.5 + 8.5)  # mlp + qkvo
mlp_base_time = 13.5  # assume expert parallel
qkvo_base_time = 8.5
linear_base_time = (mlp_base_time + qkvo_base_time)  # mlp + qkvo
# linear_base_time = 0
# linear_base_time = 0
total_ranks = 8


def get_attn_time(batch) -> float:
    total_time = 0
    for l in batch:
        ratio = l / base_seq_len
        total_time += attn_base_time * (ratio ** 2)
    return total_time


def get_mlp_time(batch) -> float:
    total_time = 0
    for l in batch:
        ratio = l / base_seq_len
        total_time += linear_base_time * (ratio)
    return total_time

def get_network_time(token_per_batch, cp_degree) -> float:
    base_token_per_batch = 512 * 1024
    if cp_degree == 1:
        return 0
    if cp_degree == 2:
        base_time = 8
    elif cp_degree == 4:
        base_time = 20
    elif cp_degree == 8:
        base_time = 46
    else:
        raise ValueError(f"Invalid cp_degree: {cp_degree}")

    total_time = base_time * (token_per_batch / base_token_per_batch)
    return total_time


def get_wlbllm_batch_time(batch: list[int], is_backward: bool = False, wlb_cp: int = 2, nlayers: int = 1) -> float:
    # TODO: Handle first and last layer latency
    token_per_batch = sum(batch)
    network_time = get_network_time(token_per_batch, wlb_cp)
    attn_time = get_attn_time(batch)
    mlp_time = get_mlp_time(batch)
    if is_backward:
        attn_time *= 2.5
        mlp_time *= 2
    compute_time = (attn_time + mlp_time) / wlb_cp
    total_time = compute_time + network_time
    total_time *= nlayers
    return total_time



def flatten(batch: list[list[int]]) -> list[int]:
    return [item for sublist in batch for item in sublist]


def window_slice(idx, size, max_idx):    
    lo = max(0, idx - size + 1)
    hi = min(max_idx, idx + 1)
    return slice(lo, hi)