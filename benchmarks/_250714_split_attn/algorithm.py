# %%
"""
Rebalance attention across batches.
-----

The purpose of the algorithm is to rebalacne the attention (flops) across the differnet bathces.

Given a particular batch, seuqence information and its GPU placement, 
```python
item = dict(q=int, kv=int, gpuid=int, seqid=int, dst_gpuid=int, flops=int, is_original=bool)
batch[0] = [all items where dst_gpuid == 0]
```
we want to rebalance the attentions across the batches. 

Specifically, given a particular batch information, we take the following steps:
1. Calculate the min/max across all batches.
2. Get the longest sequence which is the original sequence. 
3. Split the sequence into these batches such tht they get balanced as much as possible.
"""

# %%
from itertools import zip_longest
import rich
from copy import deepcopy

# %%
K = 1024

def get_flops(q=None, kv=None, **kwargs):
    assert q is not None and kv is not None, "q and kv must be provided"
    return sum(kv - i for i in range(q))

def get_number_of_work(unit_of_work: int, q=None, kv=None, **kwargs) -> int:
    flops = get_flops(q, kv)
    return flops // unit_of_work


def batch_to_items(batches):
    items = []
    for gpuid, batch in enumerate(batches):
        for seqid, seq_len in enumerate(batch):
            items.append(dict(
                q=seq_len, kv=seq_len, 
                gpuid=gpuid, seqid=seqid, src_gpuid=gpuid, 
                is_original=True
            ))
    return items


items = batch_to_items([
    [16 * K] * 4,
    # [64 * K], 
    [32 * K], 
    [8 * K] * 8,
    [4 * K] * 16,
])
rich.print(items)

# %%
def plot_batch_v1(items, title=None):
    ngpu = max(item["gpuid"] for item in items) + 1
    # plot a bar chart for each GPU - height as the sum of flops
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots()
    bottom = np.zeros(ngpu)

    x = range(ngpu)
    y = []
    for gpuid in range(ngpu):
        _y = []
        for item in items:
            if item["gpuid"] == gpuid:
                _y.append(get_flops(**item))
        y.append(_y)
    

    for idx, _ys in enumerate(zip_longest(*y, fillvalue=0)):
        ax.bar(x, _ys, bottom=bottom)
        bottom += _ys
    
    if title is not None:
        ax.set_title(title)
    
    plt.show()
    return

def plot_batch_v2(items, title=None):
    ngpu = max(item["gpuid"] for item in items) + 1
    # plot a bar chart for each GPU - height as the sum of flops
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots()
    bottom = np.zeros(ngpu)

    x = range(ngpu)
    y = []
    colors = []
    for gpuid in range(ngpu):
        _y = []
        _colors = []
        for item in items:
            if item["gpuid"] == gpuid:
                _y.append(get_flops(**item))
                _colors.append('blue' if item["is_original"] else 'orange')
        y.append(_y)
        colors.append(_colors)
    
    for idx, (_ys, _colors) in enumerate(zip(zip_longest(*y, fillvalue=0), zip_longest(*colors, fillvalue='blue'))):
        ax.bar(x, _ys, bottom=bottom, color=_colors)
        bottom += _ys
    
    if title is not None:
        ax.set_title(title)
    
    plt.show()
    return

plot_batch = plot_batch_v1

# %%
def get_oustanding_seq_v1(items):
    from collections import defaultdict

    ngpu = max(item["gpuid"] for item in items) + 1
    
    # Get each GPU's flops
    flops_per_gpu = [0] * ngpu
    for item in items:
        flops_per_gpu[item["gpuid"]] += get_flops(**item)
    
    # Get the GPU with the max flops
    max_flops_gpu_id = flops_per_gpu.index(max(flops_per_gpu))
    
    # Get the item with the max flops within that GPU
    max_flops_item = max([
        item for item in items if item["gpuid"] == max_flops_gpu_id
    ], key=lambda x: get_flops(**x))
    return max_flops_item

def get_oustanding_seq_v2(items):
    from collections import defaultdict

    # Get the item with the max flops within that GPU
    max_flops_item = max(items, key=lambda x: get_flops(**x))
    return max_flops_item

get_oustanding_seq = get_oustanding_seq_v1
get_oustanding_seq(items)


# %%
def plot_flops(items, plan_flops_per_gpu, title=None):
    fixed_flops_per_gpu = [0] * len(plan_flops_per_gpu)
    for item in items:
        fixed_flops_per_gpu[item["gpuid"]] += get_flops(**item)
    
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()
    x = range(len(plan_flops_per_gpu))
    ax.bar(x, fixed_flops_per_gpu, label="Fixed", color="orange")
    ax.bar(x, plan_flops_per_gpu, label="Plan", color="blue", bottom=fixed_flops_per_gpu)
    if title is not None:
        ax.set_title(title)
    ax.legend()
    plt.show()
        
    pass


# %%

# %%
def plan_relocation(items_, verbose=False, plot=True):
    items = deepcopy(items_)
    ngpu = max(item["gpuid"] for item in items) + 1

    def rlog(message):
        if verbose:
            rich.print(message)

    def rplot(data, title, plot_type='batch'):
        if plot:
            if plot_type == 'batch':
                plot_batch(data, title)
            elif plot_type == 'flops':
                plot_flops(data[0], data[1], title)

    rlog(items)
    rplot(items, f"Before removing the outstanding sequence", plot_type='batch')

    # Remove the outstanding sequence from the items
    outstanding_seq = get_oustanding_seq(items)
    items.remove(outstanding_seq)

    rlog(items)
    rplot(items, f"After removing the outstanding sequence", plot_type='batch')

    # ------------------------
    # Information Gathering
    # ------------------------
    # Get a bunch of informations:
    # - current flops for each GPU: `flops_per_gpu`
    # - max flops across all GPUs: `max_flops`
    # - deficit of each GPU compared to the max flops: `deficit = max_flops - flops`
    flops_per_gpu = [0] * ngpu
    for item in items:
        flops_per_gpu[item["gpuid"]] += get_flops(**item)
    max_flops = max(flops_per_gpu)
    deficit = [max_flops - flops for flops in flops_per_gpu]

    can_sequence_cover_deficit = get_flops(**outstanding_seq) > sum(deficit)
    rlog(f"outstanding_seq = {outstanding_seq}")
    rlog(f"get_flops(**outstanding_seq) = {get_flops(**outstanding_seq)}")
    rlog(f"flops_per_gpu = {flops_per_gpu}")
    rlog(f"max_flops = {max_flops}")
    rlog(f"deficit = {deficit}")
    rlog(f"sum(deficit) = {sum(deficit)}")
    rlog(f"can_sequence_cover_deficit = {can_sequence_cover_deficit}")
    # It should probably prefer to allocate the sequences to itself.

    # ------------------------
    # Flops Movement Planning
    # - Input:
    #   - `items`: the items we want to move the flops to the GPUs.
    #   - `outstanding_seq`: the sequence that we want to move the flops to the GPUs.
    # - Output:
    #   - `plan_to_move_flops`: the flops we want to move to each GPU.
    # ------------------------
    plan_to_move_flops = [0] * ngpu
    rplot((items, plan_to_move_flops), f"(Flops) Before planning the flops movement", plot_type='flops')
    remaining_flops = get_flops(**outstanding_seq)
    for k in range(2, ngpu + 1):
        # Find the topK largest deficit GPUs, 
        # and then try to fill them up with the remaining budgets.
        # The remaining budgets will try to equally allocate to these topK GPUs 
        # until the remaining budgets are all allocated.

        topK_deficit_gpus_ids = sorted(range(ngpu), key=lambda x: deficit[x], reverse=True)[:k]

        # Obtain the target GPU we want the deficit flops to go down to.
        water_level_gpu_id = topK_deficit_gpus_ids[-1]
        deficit_gpu_ids = topK_deficit_gpus_ids[:-1]

        # Calculate the water-level flops and the demand flops
        water_level_flops = deficit[water_level_gpu_id]
        deficit_flops = [deficit[gpu_id] for gpu_id in deficit_gpu_ids]
        deficit_diff_flops = [deficit[gpu_id] - water_level_flops for gpu_id in deficit_gpu_ids]
        total_deficit_flops = sum(deficit_diff_flops)

        if remaining_flops >= total_deficit_flops:
            for gpu_id, deficit_diff_flops in zip(deficit_gpu_ids, deficit_diff_flops):
                plan_to_move_flops[gpu_id] += deficit_diff_flops
                deficit[gpu_id] -= deficit_diff_flops
                remaining_flops -= deficit_diff_flops
        else:
            # Equally distribute the remaining flops to the deficit GPUs?
            equal_flops = remaining_flops / len(deficit_gpu_ids)
            for gpu_id in deficit_gpu_ids:
                plan_to_move_flops[gpu_id] += equal_flops
                deficit[gpu_id] -= equal_flops
                # remaining_flops -= equal_flops
            remaining_flops = 0

        rplot((items, plan_to_move_flops), f"(Flops) After planning the flops movement - Round {k}", plot_type='flops')
        if remaining_flops == 0:
            break

    if remaining_flops > 0:
        # Equally distribute the remaining flops to the GPUs.
        flops = remaining_flops / ngpu
        for gpu_id in range(ngpu):
            plan_to_move_flops[gpu_id] += flops
        remaining_flops = 0
        rplot((items, plan_to_move_flops), f"(Flops) After planning the flops movement - remainder", plot_type='flops')

    # Note: remaining_flops can be float.

    # ------------------------
    # Work Assignment
    # - Use `plan_to_move_flops` to assign the work to the GPUs.
    # - `outstanding_seq` is the sequence that we want to move the flops to the GPUs.
    # ------------------------
    def round_up(x): 
        if x > int(x): 
            x = x + 1
        return int(x)

    # Handle locality. First, we satisfy the need for the outstanding sequence's GPU.
    filling_gpu_ids = [outstanding_seq["gpuid"]] + [i for i in range(ngpu) if i != outstanding_seq["gpuid"]]
    rlog(f"filling_gpu_ids = {filling_gpu_ids}")

    # TODO: - Make this a generator -
    _current_work = 0
    _unit_of_work = outstanding_seq["kv"] + 1
    _max_work = outstanding_seq["kv"] // 2
    _remainder_work = outstanding_seq["kv"] % 2

    def get_next_work(flops: int, gpu_id: int):
        nonlocal _current_work
        works = flops // _unit_of_work
        if not works:
            return []

        # outstanding_seq = {
        #     'q': 32768, 'kv': 32768, 'gpuid': 1, 'seqid': 0, 
        #     'src_gpuid': 1, 'is_original': True
        # }

        if works + _current_work > _max_work:
            # assign everything to this GPU. 
            works = _max_work - _current_work
            head = outstanding_seq.copy()
            head.update(dict(
                q=works,
                kv=_current_work + works,
                gpuid=gpu_id,
                is_original=False
            ))
        else:
            head = outstanding_seq.copy()
            head.update(dict(
                q=works,
                kv=_current_work + works,
                gpuid=gpu_id,
                is_original=False
            ))
            tail = outstanding_seq.copy()
            tail.update(dict(
                q=works,
                kv=outstanding_seq["kv"] - _current_work,
                gpuid=gpu_id,
                is_original=False
            ))

        _current_work += works
        return [head, tail]

    rlog(f"outstanding_seq = {outstanding_seq}")
    rplot(items, f"Work assignment - Before", plot_type='batch')

    rlog(f"outstanding_seq = {outstanding_seq}")
    for idx, gpu_id in enumerate(filling_gpu_ids):
        rlog(f"[Round {idx}] gpu_id = {gpu_id}")
        flops_to_fill = plan_to_move_flops[gpu_id]
        flops_to_fill = round_up(flops_to_fill)

        rlog(f"[Round {idx}] flops_to_fill = {flops_to_fill}")

        if flops_to_fill <= 0:
            continue

        works = get_next_work(flops_to_fill, gpu_id)
        rlog(f"[Round {idx}] works = {works}")
        if works:
            items.extend(works)
        rplot(items, f"Work assignment - Round {idx}", plot_type='batch')

    # Handle remainder

    rplot(items, f"Result", plot_type='batch')
    rlog(items)
    return items



items = batch_to_items([
    # [16 * K] * 4,
    [64 * K], 
    [32 * K] * 2, 
    [8 * K] * 8,
    [4 * K] * 16,
])



# for k in range(3):
items = plan_relocation(
    items, 
    # verbose=True
)
# %%
items = plan_relocation(
    items, 
    # verbose=True
)