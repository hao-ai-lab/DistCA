# %%
import pandas as pd
import numpy as np
import json


# %%
with open("./data/fake.json", "r") as f:
    data = json.load(f)

# %%
data

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# %% 
def get_data(num_tokens_per_data, doc_dataset):
    data_budget = num_tokens_per_data
    doc_lens    = []

    for token_count in doc_dataset:
        # carve out as many full chunks as needed
        while token_count > data_budget:
            # consume whatever remains of this batch
            consumed = data_budget
            doc_lens.append(consumed)
            yield doc_lens

            # now subtract _that_ consumed amount, reset for the next batch
            token_count -= consumed
            data_budget = num_tokens_per_data
            doc_lens    = []

        # at this point token_count <= data_budget
        if token_count > 0:
            doc_lens.append(token_count)
            data_budget -= token_count

    # finally, if there are any leftover pieces, yield them too
    if doc_lens:
        yield doc_lens

# %%
K = 1024
batches = get_data(16 * K, data)
batches = list(batches)

# %%
batches

# %%
set(sum(i) for i in batches)

# %%
batch_size = 32
dp_degree = 2
tp_degree = 1
cp_degree = 1

dp_batches = [
    np.concatenate(batches[i:i + batch_size // dp_degree])
    for i in range(0, len(batches), batch_size // dp_degree)
]

# %%
dp_batches


# %%
import time_module.network as network
import time_module.compute as compute
import importlib
importlib.reload(compute)
importlib.reload(network)

# compute.attn_time(
#     gpu = "A100-SXM-80GB",
#     head_dim = 128,
#     nhead = 16,
#     tokens = tokens,
#     dtype = "half",
#     is_fwd=True
# )

# compute.gemm_time(
#     gpu = "A100-SXM-80GB",
#     m = 1024,
#     k = 1024,
#     n = 8192,
#     dtype = "half",
# )   

# %%
def get_batch_linear_time(
    batch, tp_degree,
    hqo, hkv, d, d1, d2 # original config
) -> float:
    """
    TotalLinearTime(T, hqo, hkv, d, d1, d2) 
        = M(T, d1, d2) + M(T, d2, d1) + M(T, d1, hqo * d) + 2 * M(T, hkv * d, d1)

    but need to modify from the original config such that:
        - d2 = d2 / tp_degree
        - hqo = hqo / tp_degree
        - hkv = hkv / tp_degree
    """
    d2 = d2 / tp_degree
    hqo = hqo / tp_degree
    hkv = hkv / tp_degree

    gpu = "A100-SXM-80GB"
    dtype = "half"
    T = sum(batch)

    def M(m, k, n):
        return compute.gemm_time(
            gpu = gpu,
            m = m, k = k, n = n, 
            dtype = dtype,
        )

    return M(T, d1, d2) + M(T, d2, d1) + M(T, d1, hqo * d) + 2 * M(T, hkv * d, d1)

# %%

# def get_dp_attn_balanced_time(dp_batches, tp_degree, cp_degree):
#     pass
hqo, hkv, d, d1, d2 = (32, 8, 128, 4096, int(4096 * 3.5))

mlp_time = [
    get_batch_linear_time(batch, tp_degree, hqo, hkv, d, d1, d2) 
    for batch in dp_batches
]
mlp_time

# %%

def get_batch_attn_time(
    batch, tp_degree, cp_degree,
    hqo, hkv, d,
) -> float:
    result = 0
    assert hqo % tp_degree == 0 and tp_degree <= hqo
    assert hkv % tp_degree == 0 and tp_degree <= hkv
    hqo = hqo // tp_degree
    hkv = hkv // tp_degree

    for idx, tokens in enumerate(batch):
        attn_time = compute.attn_time(
            gpu = "A100-SXM-80GB",
            cp = cp_degree,
            head_dim = d,
            nhead = hqo,
            tokens = tokens,
            dtype = "half",
            is_fwd = True,
        )
        result += attn_time
        print(f"attn_time[{idx}] = {attn_time}")
    return result

# %%
attn_time = [
    get_batch_attn_time(batch, tp_degree, cp_degree, hqo, hkv, d)
    for batch in dp_batches
]
attn_time
# %%
