# %%
import d2.timemodule as tm

# %%

for x in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]:
    x = x * 1024
    mlp_time = tm.get_mlp_time(x, 8, 1)
    print(f"x: {x}, mlp_time: {mlp_time}")

# %%
