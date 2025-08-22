# %%
import torch
import pandas as pd

# %%
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


latency_ms_es = torch.tensor([
    25.9, # Rank 0
    5.94, # Rank 10
    13.9, # Rank 20
    16.4, # Rank 24
    6.57, # Rank 39
    29.6, # Rank 41
    29.4, # Rank 50
    29.3, # Rank 62
])
latency_s_es = latency_ms_es / 1000.0

# for i in range(sender_transfer_sz.shape[0]):
#     sender_transfer_sz[i, i] = 0
#     recver_transfer_sz[i, i] = 0

# %%
sender_row_sum = sender_transfer_sz.sum(dim=1)
sender_row_sum_in_GB = sender_row_sum / 1024**3
sender_row_bw_GBpS = sender_row_sum_in_GB / latency_s_es
sender_row_bw_GBpS
# %%
recver_row_sum = recver_transfer_sz.sum(dim=1)
recver_row_sum_in_GB = recver_row_sum / 1024**3
recver_row_bw_GBpS = recver_row_sum_in_GB / latency_s_es
recver_row_bw_GBpS
# %%
rich.print("sender_row_sum_in_GB", sender_row_sum_in_GB)
rich.print("recver_row_sum_in_GB", recver_row_sum_in_GB)
# %%
sender_row_sum_in_GB.sum(), recver_row_sum_in_GB.sum()
# %%
import rich
rich.print("sender_row_bw_GBpS", sender_row_bw_GBpS)
rich.print("recver_row_bw_GBpS", recver_row_bw_GBpS)
# %%