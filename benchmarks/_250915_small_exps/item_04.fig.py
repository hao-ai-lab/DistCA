# %%
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

# %%
root = "/mnt/weka/home/hao.zhang/jd/d2/benchmarks/_250915_small_exps/logs.v1-item_04"
num_tokens = 16384
with open(os.path.join(root, f"num_tokens_{num_tokens}", "mem_snapshots", "memory_profile.rank0.pickle"), "rb") as f:
    data = pickle.load(f)

