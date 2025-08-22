# %%
import torch
import pandas as pd

def analyze_bandwidth(sz_bytes: torch.Tensor, time_ms: float = 27.0, bytes_per_GB: float = 1e9):
    """
    Args:
        sz_bytes: NxN tensor of sizes in BYTES.
        time_ms: elapsed time in milliseconds.
        bytes_per_GB: 1e9 for GB (decimal). Use 2**30 if you want GiB.
    Returns:
        dict with tensors:
          - sz_in_GB: NxN tensor
          - row_sum_offdiag_GB: N tensor
          - bw_in_GBps: N tensor (effective per-rank BW)
    """
    assert sz_bytes.dim() == 2 and sz_bytes.size(0) == sz_bytes.size(1), "sz must be square NxN"
    N = sz_bytes.size(0)
    # (1) bytes -> GB
    sz_in_GB = sz_bytes.to(torch.float64) / bytes_per_GB
    
    # (2) row-sum excluding diagonal
    row_sums = sz_in_GB.sum(dim=1) - sz_in_GB.diag()
    
    # (3) bandwidth in GB/s
    bw_in_GBps = row_sums / (time_ms / 1000.0)
    return {
        "sz_in_GB": sz_in_GB,
        "row_sum_offdiag_GB": row_sums,
        "bw_in_GBps": bw_in_GBps
    }

# --- Input tensors from your message ---
sender_transfer_sz = torch.tensor([
    [214452736,         0,  41617408,  41617408,         0,  40913408, 41614848,  41613312],
    [ 45773312, 219027968,  45770752,  45770752,         0,  45773312, 45770752,  45772800],
    [        0,         0, 155316736,         0,         0,         0, 97259008, 100665856],
    [        0,         0,  61671424, 196609024,         0,         0, 61673984,  61676544],
    [ 45728256,  45730816,  45730816,  45730816, 220687360,  45728256, 45730816,  45725696],
    [        0,         0,         0,         0,         0, 201326592,        0,         0],
    [        0,         0,         0,         0,         0,         0, 201326592,        0],
    [        0,         0,         0,         0,         0,         0,        0, 201326592]
])

recver_transfer_sz = torch.tensor([
    [214452736,  45773312,         0,         0,  45728256,         0,         0,         0],
    [        0, 219027968,         0,         0,  45730816,         0,         0,         0],
    [ 41617408,  45770752, 155316736,  61671424,  45730816,         0,         0,         0],
    [ 41617408,  45770752,         0, 196609024,  45730816,         0,         0,         0],
    [        0,         0,         0,         0, 220687360,         0,         0,         0],
    [ 40913408,  45773312,         0,         0,  45728256, 201326592,         0,         0],
    [ 41614848,  45770752,  97259008,  61673984,  45730816,         0, 201326592,         0],
    [ 41613312,  45772800, 100665856,  61676544,  45725696,         0,         0, 201326592]
])

# %%
# Analyze both matrices
sender_res = analyze_bandwidth(sender_transfer_sz, time_ms=27.0, bytes_per_GB=1e9)
recver_res = analyze_bandwidth(recver_transfer_sz, time_ms=27.0, bytes_per_GB=1e9)

# Build DataFrames for per-rank summaries
def to_rank_df(res_dict, title_prefix):
    N = res_dict["sz_in_GB"].size(0)
    df = pd.DataFrame({
        "rank": list(range(N)),
        "row_sum_offdiag_GB": res_dict["row_sum_offdiag_GB"].tolist(),
        "bw_in_GBps": res_dict["bw_in_GBps"].tolist()
    })
    return df

# %%
sender_df = to_rank_df(sender_res, "Sender")
recver_df = to_rank_df(recver_res, "Receiver")

# Also provide overall max off-diagonal element (GB) and total off-diagonal sum (GB)
def global_offdiag_stats(sz_bytes: torch.Tensor, bytes_per_GB: float = 1e9):
    N = sz_bytes.size(0)
    mask = ~torch.eye(N, dtype=torch.bool)
    off = sz_bytes[mask].to(torch.float64) / bytes_per_GB
    return off.max().item(), off.sum().item()

sender_max_GB, sender_sum_GB = global_offdiag_stats(sender_transfer_sz, 1e9)
recver_max_GB, recver_sum_GB = global_offdiag_stats(recver_transfer_sz, 1e9)

summary = pd.DataFrame({
    "matrix": ["Sender", "Receiver"],
    "max_offdiag_GB": [sender_max_GB, recver_max_GB],
    "sum_offdiag_GB": [sender_sum_GB, recver_sum_GB]
}).round(6)


# %%
sender_df

# %%
recver_df

# %%
summary

# %%