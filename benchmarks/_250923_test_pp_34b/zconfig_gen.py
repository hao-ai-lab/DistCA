# %%

param_configs_cases = []
K = 1024
tp_size = 8
print("N bs mb   tok   M cp pp tp comment")
rows = []
# for seq_len in [128 * K, 256 * K, 384 * K, 512 * K]:
# for seq_len in [128 * K, ]:
for seq_len in [256 * K]:
# for seq_len in [384 * K, ]:
    # for nnodes in [64]:
    for nnodes in [32]:
        _ratios = {
            # 128 * K: [16],
            # 128 * K: [8],
            # 128 * K: [4],
            # 128 * K: [2],
            # 256 * K: [2],
            # 256 * K: [4],
            256 * K: [8],
            # 384 * K: [1],
            # 384 * K: [2],
            384 * K: [4],
            # 512 * K: [1, 2],
        }[seq_len]
        for ratio in _ratios:
            # if ratio * nnodes > 64, skip
            # if ratio * nnodes > 64:
            #     continue

            total_batch_size = ratio * nnodes // 8

            for batch_size in [0.5, 1, 2, 4, 8, 16, 32, 64]:
                for microbatch_size in [1, 2, 4, 8, 16, 32, 64]:

                    # if microbatch_size * batch_size > total_batch_size, skip
                    if microbatch_size * batch_size != total_batch_size:
                        continue

                    # for pp_size in [1, 2, 4, 8, 16, 32, 64]:
                    for pp_size in [1, 2, 4, 8]:

                        # if pp_size = 1, then microbatch_size must be 1
                        if pp_size == 1 and microbatch_size != 1:
                            continue
                                                        # microbatch_size >= pp_size * 2
                        if pp_size > 1 and microbatch_size < pp_size * 2:
                            continue


                        for mode in ["d2", "wlbllm"]:

                            # D2
                            if mode == "d2":
                                if batch_size == 0.5:
                                    continue
                                
                                cp_size = nnodes // pp_size

                                mode = "d2"

                                config = dict(
                                    ratio=ratio,
                                    seq_len=seq_len,
                                    nnodes=nnodes,
                                    batch_size=batch_size,
                                    microbatch_size=microbatch_size,
                                    mode=mode,
                                    cp_size=cp_size,
                                    pp_size=pp_size,
                                    tp_size=tp_size,
                                )
                                param_configs_cases.append(config)
                                continue

                            # WLBLLM
                            if mode == "wlbllm":
                                for cp_size in [1, 2, 4, 8, 16, 32, 64]:
                                    if pp_size * cp_size > nnodes:
                                        continue
                                    dp_size = nnodes // cp_size
                                    if dp_size > batch_size * 2:
                                        continue
                                    config = dict(
                                        ratio=ratio,
                                        seq_len=seq_len,
                                        nnodes=nnodes,
                                        batch_size=batch_size,
                                        microbatch_size=microbatch_size,
                                        mode=mode,
                                        cp_size=cp_size,
                                        pp_size=pp_size,
                                        tp_size=tp_size,
                                    )
                                    param_configs_cases.append(config)
                                continue

# Show how many configs survived
print(f"\nTotal valid configs: {len(param_configs_cases)}")
import pandas as pd
df = pd.DataFrame(param_configs_cases)

df = df.sort_values(by=[ "ratio", "seq_len", "nnodes", "batch_size", "microbatch_size", "cp_size", "pp_size"])
# df deduplicate
df = df.drop_duplicates().reset_index(drop=True)

# Set pandas options to display all rows
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Now display the DataFrame
df['tb'] = df['batch_size'] * df['microbatch_size']
# sort df by mode
df = df.sort_values(by=["mode", "pp_size"])
df
# %%
# Print in desired format
# # #   n   bs   mb     t       mode   cp   pp  tp     comment        env_var
print('    # n  bs  mb   t         mode   cp  pp tp    comment    env_var')
for _, row in df.iterrows():
    print(f"    {row['nnodes']:2d} {row['batch_size']:3g} {row['microbatch_size']:3d} {row['seq_len']:6d}    {row['mode']:<8} {row['cp_size']:2d} {row['pp_size']:2d} {row['tp_size']:2d} \'some_comment\'  \'\'")


# %%
