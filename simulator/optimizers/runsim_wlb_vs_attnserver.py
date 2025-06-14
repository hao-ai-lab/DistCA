# %%

import wlbllm as wlbllm
import attnserver as attnserver

from attnserver import AttnServerSolver
from wlbllm import WlbLlmSolver


# %%
K = 1024
# batch = [64, 32, 32, 32, 32, 10]
# batch = [64, 64, 32, 32, 1, 1, 2, 2, 3, 3,]
# batch = [64, 96, 1, 1, 2, 2, 3, 3,]
batch = [64, 32]
batch =  [i * K for i in batch]

num_workers = 4
num_total_devices = 32

print("Qwen3-235B")


# %%
import timemodule as tm

# %%
# tm.get_attn_time(64 * 1024, 8, 2)
for ctx_len in [1, 2, 4, 32, 64, 96]:
    # a = tm.get_attn_time(ctx_len * 1024, 8, 2)
    a = tm.get_attn_time(ctx_len * 1024, 8, 4)
    print(f"ctx_len: {ctx_len}, attn: {a}ms")

# %%

# %%
# %%
solver = WlbLlmSolver()
# solution = solver.solve(
#     batch, 
#     max_length=sum(batch), 
#     num_workers=num_workers, 
#     parallel_plan=(8, 2),
# )
solution = solver.solve(
    batch, 
    max_length=sum(batch), 
    # num_workers=num_workers, 
    num_workers=1, 
    parallel_plan=(8, 4),
)
solution.print_solution()

# %%
solver = AttnServerSolver()
solution = solver.solve(
    batch, 
    num_workers=num_total_devices, 
    num_total_devices=num_total_devices,
)
solution.print_solution()


# %%
for batch in distribution:
    pass