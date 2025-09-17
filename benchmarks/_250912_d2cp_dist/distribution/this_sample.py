# %%
import d2.simulator.optimizers.samples
import importlib
importlib.reload(d2.simulator.optimizers.samples)
from d2.simulator.optimizers.samples import sample_prolong_docs, batch_documents
import json
from pathlib import Path
import numpy as np
import time
K = 1024
# %%# %%
filter_threshold = 64 * K
# max_ctx_length = 512 * K
max_ctx_length = 128 * K
docs = sample_prolong_docs(
    size=10000,
    seed=42,
    filter_threshold=filter_threshold,
    filter_ratio=0.90,
    upsample_long_factor=2,
    elongate_factor=8,
    change_long_doc_ratio=0.5,
)
batches = batch_documents(docs, max_ctx_length=max_ctx_length)
batches = list(batches)


for i in range(100):
    flops = sum(
        (l / (128*K)) ** 2 for l in batches[i]
    )
    print(
        flops, sorted(batches[i])[-5:]
    )
    time.sleep(0.5)
# %%
