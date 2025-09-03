# %%
import numpy as np
import json
import os
import pandas as pd
from IPython.display import display, Markdown
import matplotlib.pyplot as plt

# %%
! pwd

# %%
log_dir_base = f"/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250901_oom_inspection/logs/20250902155443_PST"
config = "mem.n4.n65536.b4.l4"
name = "d2_b1_1"
# %%
file_path = os.path.join(log_dir_base, config, name, "mem_snapshots", "memory_profile.pickle")
# %%
import pickle
with open(file_path, "rb") as f:
    data = pickle.load(f)
# %%
data.keys()

# %%

