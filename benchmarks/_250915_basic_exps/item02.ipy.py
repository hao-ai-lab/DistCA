# %%

import os
import json
import pandas as pd
import matplotlib.pyplot as plt

root_dir = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250915_basic_exps/item_02_wlb_memimbalance"
names = os.listdir(root_dir)

for name in names:
    mem_log_dir = os.path.join(root_dir, name, "mem-log")
    if not os.path.exists(mem_log_dir):
        continue
    mem_log_files = os.listdir(mem_log_dir)

    dfs = []
    max_mems = []
    plt.figure(figsize=(12, 6))
    for i, mem_log_file in enumerate(mem_log_files):
        mem_log_file_path = os.path.join(mem_log_dir, mem_log_file)
        # print(mem_log_file_path)
        with open(mem_log_file_path, "r") as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
        max_mem = df['allocated_cur'].max().item()
        max_mems.append(max_mem)
        plt.plot(df['allocated_cur'], label=f'R{i}')
    plt.legend(ncol=2, bbox_to_anchor=(1.05, 1), loc='best')
    plt.title(name)
    plt.xlabel('Step')
    plt.ylabel('Memory (GB)')
    plt.show()

    print("================================")
    print(name)
    print(f"{min(max_mems):.2f}", f"{max(max_mems):.2f}")


# %%



# %%
