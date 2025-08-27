# %%
K = 1024

from d2.planner.equal_flops import (
    batch_to_items, 
    plan_relocation,
    item_to_intermediate_tensors,
    postprocess_items,
    calculate_flops_factor_in_each_gpu,
)

# %%
items_list =  [[10188, 19948, 100936], [131072], [131072], [10756, 20492, 14428, 2924, 16044, 1084,
10256, 8072, 920, 2268, 13740, 20848, 9240], [31304, 24316, 4140, 15204, 6480, 24552, 19208, 5868],
[12432, 21712, 24184, 19992, 4212, 14780, 19068, 14692], [1028, 51772, 12652, 23700, 24008, 17060,
852], [8852, 22444, 10972, 19900, 20736, 14284, 33884]]

for replan_iter in range(4):
    items = batch_to_items(items_list)
    for _ in range(replan_iter):
        items = plan_relocation(items, verbose=False, plot=False)
    if replan_iter > 0:
        items = postprocess_items(items)

    gpu_flops = calculate_flops_factor_in_each_gpu(items)
    print(f"replan_iter={replan_iter}")
    print(gpu_flops)
    print(max(gpu_flops) - min(gpu_flops))