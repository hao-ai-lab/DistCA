# %%
import argparse
from typing import List, Tuple

import numpy as np
import pulp

from datasets import load_dataset
from transformers import AutoTokenizer

# %%
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# %%
attn_time = np.load("data/attn_time.npy")
mlp_time = np.load("data/mlp_time.npy")
# %%
K = 1024
num_tokens_per_data = 16 * K
# %%
data_budget = num_tokens_per_data
# %%
import json
with open("./data/batches_16k.json", "r") as f:
    batches = json.load(f)
# %%
batches[0]
# %%
import numpy as np

dp_degree = 4
batch_size = 32
dp_batches = [
    np.concatenate(batches[i:i + batch_size // dp_degree])
    for i in range(0, batch_size, batch_size // dp_degree)
]

# %%
