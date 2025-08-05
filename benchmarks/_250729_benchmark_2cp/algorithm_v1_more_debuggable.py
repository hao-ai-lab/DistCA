# %%
"""
A much more debugging friendly version of the rebalance algorithm -> CP Metadata.
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

from itertools import zip_longest
import rich
from copy import deepcopy

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



def test_create_batch_and_relocation():

    items = batch_to_items([
        [16 * K] * 4,
        # [64 * K], 
        [32 * K], 
        [8 * K] * 8,
        [4 * K] * 16,
    ])
    rich.print(items)
    
    items = plan_relocation(
        items, 
        verbose=True,
        plot=True,
    )
    return 

# %%
items = batch_to_items([
    [16 * K] * 4,
    # [64 * K], 
    [32 * K] * 2, 
    [8 * K] * 8,
    [4 * K] * 16,
])
rich.print(items)

# %%
items = plan_relocation(
    items, 
    # verbose=True,
    # plot=True,
    verbose=False,
    plot=False,
)
# %%
rich.print(items)

# %%
from collections import defaultdict
info_mapping = defaultdict(list)
for item in items:
    sid = (item["src_gpuid"], item["seqid"])
    info_mapping[sid].append(item)

info_list = list(info_mapping.items())
info_list.sort(
    key=lambda x: x[0]
)
for key, values in info_list:
    values.sort(key=lambda x: x["kv"])
    pass
rich.print(info_list)

# %%
world_size = max(item["src_gpuid"] for item in items) + 1
num_seqs = max(item["seqid"] for item in items) + 1
max_cp_degree = max(
    len(value)
    for value in info_mapping.values()
)
rich.print(dict(world_size=world_size, num_seqs=num_seqs, max_cp_degree=max_cp_degree))

# %%
import torch
seq_lens = torch.zeros((world_size, num_seqs), dtype=torch.int64)
for i in range(world_size):
    for j in range(num_seqs):
        seq_lens[i, j] = sum(item["q"] for item in info_mapping[(i, j)])
rich.print(seq_lens)
# %%
cp_num = torch.zeros((world_size, num_seqs), dtype=torch.int64)
for i in range(world_size):
    for j in range(num_seqs):
        cp_num[i, j] = len(info_mapping[(i, j)])
rich.print(cp_num)
# %%
cp_dst = torch.ones((world_size, num_seqs, max_cp_degree), dtype=torch.int64) * -1
for i in range(world_size):
    for j in range(num_seqs):
        cp_num_ = cp_num[i, j]
        for k in range(max_cp_degree):
            if k < cp_num_:
                cp_dst[i, j, k] = info_mapping[(i, j)][k]["gpuid"]
            else:
                cp_dst[i, j, k] = -1
rich.print(cp_dst)
# %%
seq_shard_lens = torch.zeros((world_size, num_seqs, max_cp_degree), dtype=torch.int64)
for i in range(world_size):
    for j in range(num_seqs):
        cp_num_ = cp_num[i, j]
        for k in range(max_cp_degree):
            if k < cp_num_:
                seq_shard_lens[i, j, k] = info_mapping[(i, j)][k]["q"]
            else:
                seq_shard_lens[i, j, k] = 0
rich.print(seq_shard_lens)



def item_to_intermediate_tensors(items, verbose=False):
    def print_if_verbose(message):
        if verbose:
            rich.print(message)

    from collections import defaultdict
    from copy import deepcopy
    
    # Prepare the info_mapping[(src_gpuid, seqid)] -> [items]
    items = deepcopy(items)
    info_mapping = defaultdict(list)
    for item in items:
        sid = (item["src_gpuid"], item["seqid"])
        info_mapping[sid].append(item)

    info_list = list(info_mapping.items())
    info_list.sort(
        key=lambda x: x[0]
    )
    for key, values in info_list:
        values.sort(key=lambda x: x["kv"])
        pass

    print_if_verbose(info_list)

    world_size = max(item["src_gpuid"] for item in items) + 1
    num_seqs = max(item["seqid"] for item in items) + 1
    max_cp_degree = max(
        len(value)
        for value in info_mapping.values()
    )
    world_info = dict(world_size=world_size, num_seqs=num_seqs, max_cp_degree=max_cp_degree)
    print_if_verbose(world_info)

    seq_lens = torch.zeros((world_size, num_seqs), dtype=torch.int64)
    for i in range(world_size):
        for j in range(num_seqs):
            seq_lens[i, j] = sum(item["q"] for item in info_mapping[(i, j)])
    print_if_verbose(seq_lens)

    cp_num = torch.zeros((world_size, num_seqs), dtype=torch.int64)
    for i in range(world_size):
        for j in range(num_seqs):
            cp_num[i, j] = len(info_mapping[(i, j)])
    print_if_verbose(cp_num)

    cp_dst = torch.ones((world_size, num_seqs, max_cp_degree), dtype=torch.int64) * -1
    for i in range(world_size):
        for j in range(num_seqs):
            cp_num_ = cp_num[i, j]
            for k in range(max_cp_degree):
                if k < cp_num_:
                    cp_dst[i, j, k] = info_mapping[(i, j)][k]["gpuid"]
                else:
                    cp_dst[i, j, k] = -1
    print_if_verbose(cp_dst)

    seq_shard_lens = torch.zeros((world_size, num_seqs, max_cp_degree), dtype=torch.int64)
    for i in range(world_size):
        for j in range(num_seqs):
            cp_num_ = cp_num[i, j]
            for k in range(max_cp_degree):
                if k < cp_num_:
                    seq_shard_lens[i, j, k] = info_mapping[(i, j)][k]["q"]
                else:
                    seq_shard_lens[i, j, k] = 0
    print_if_verbose(seq_shard_lens)

    return world_info, (items, info_mapping, info_list), (seq_lens, cp_num, cp_dst, seq_shard_lens)

# %%
items = [
    {'q': 16384, 'kv': 16384, 'gpuid': 0, 'seqid': 0, 'src_gpuid': 0, 'is_original': True},
    {'q': 16384, 'kv': 16384, 'gpuid': 0, 'seqid': 1, 'src_gpuid': 0, 'is_original': True},
    {'q': 16384, 'kv': 16384, 'gpuid': 0, 'seqid': 2, 'src_gpuid': 0, 'is_original': True},
    {'q': 16384, 'kv': 16384, 'gpuid': 0, 'seqid': 3, 'src_gpuid': 0, 'is_original': True},
    {'q': 32768, 'kv': 32768, 'gpuid': 1, 'seqid': 1, 'src_gpuid': 1, 'is_original': True},
    {'q': 8192, 'kv': 8192, 'gpuid': 2, 'seqid': 0, 'src_gpuid': 2, 'is_original': True},
    {'q': 8192, 'kv': 8192, 'gpuid': 2, 'seqid': 1, 'src_gpuid': 2, 'is_original': True},
    {'q': 8192, 'kv': 8192, 'gpuid': 2, 'seqid': 2, 'src_gpuid': 2, 'is_original': True},
    {'q': 8192, 'kv': 8192, 'gpuid': 2, 'seqid': 3, 'src_gpuid': 2, 'is_original': True},
    {'q': 8192, 'kv': 8192, 'gpuid': 2, 'seqid': 4, 'src_gpuid': 2, 'is_original': True},
    {'q': 8192, 'kv': 8192, 'gpuid': 2, 'seqid': 5, 'src_gpuid': 2, 'is_original': True},
    {'q': 8192, 'kv': 8192, 'gpuid': 2, 'seqid': 6, 'src_gpuid': 2, 'is_original': True},
    {'q': 8192, 'kv': 8192, 'gpuid': 2, 'seqid': 7, 'src_gpuid': 2, 'is_original': True},
    {'q': 4096, 'kv': 4096, 'gpuid': 3, 'seqid': 0, 'src_gpuid': 3, 'is_original': True},
    {'q': 4096, 'kv': 4096, 'gpuid': 3, 'seqid': 1, 'src_gpuid': 3, 'is_original': True},
    {'q': 4096, 'kv': 4096, 'gpuid': 3, 'seqid': 2, 'src_gpuid': 3, 'is_original': True},
    {'q': 4096, 'kv': 4096, 'gpuid': 3, 'seqid': 3, 'src_gpuid': 3, 'is_original': True},
    {'q': 4096, 'kv': 4096, 'gpuid': 3, 'seqid': 4, 'src_gpuid': 3, 'is_original': True},
    {'q': 4096, 'kv': 4096, 'gpuid': 3, 'seqid': 5, 'src_gpuid': 3, 'is_original': True},
    {'q': 4096, 'kv': 4096, 'gpuid': 3, 'seqid': 6, 'src_gpuid': 3, 'is_original': True},
    {'q': 4096, 'kv': 4096, 'gpuid': 3, 'seqid': 7, 'src_gpuid': 3, 'is_original': True},
    {'q': 4096, 'kv': 4096, 'gpuid': 3, 'seqid': 8, 'src_gpuid': 3, 'is_original': True},
    {'q': 4096, 'kv': 4096, 'gpuid': 3, 'seqid': 9, 'src_gpuid': 3, 'is_original': True},
    {'q': 4096, 'kv': 4096, 'gpuid': 3, 'seqid': 10, 'src_gpuid': 3, 'is_original': True},
    {'q': 4096, 'kv': 4096, 'gpuid': 3, 'seqid': 11, 'src_gpuid': 3, 'is_original': True},
    {'q': 4096, 'kv': 4096, 'gpuid': 3, 'seqid': 12, 'src_gpuid': 3, 'is_original': True},
    {'q': 4096, 'kv': 4096, 'gpuid': 3, 'seqid': 13, 'src_gpuid': 3, 'is_original': True},
    {'q': 4096, 'kv': 4096, 'gpuid': 3, 'seqid': 14, 'src_gpuid': 3, 'is_original': True},
    {'q': 4096, 'kv': 4096, 'gpuid': 3, 'seqid': 15, 'src_gpuid': 3, 'is_original': True},
    {'q': 6144, 'kv': 6144, 'gpuid': 2, 'seqid': 0, 'src_gpuid': 1, 'is_original': False},
    {'q': 6144, 'kv': 32768, 'gpuid': 2, 'seqid': 0, 'src_gpuid': 1, 'is_original': False},
    {'q': 10239, 'kv': 16383, 'gpuid': 3, 'seqid': 0, 'src_gpuid': 1, 'is_original': False},
    {'q': 10239, 'kv': 26624, 'gpuid': 3, 'seqid': 0, 'src_gpuid': 1, 'is_original': False}
]
# %%
# Postprocessing the planned item to add a "shard_id" field.
# The "shard_id" field is always 0 for the original sequence.
# For each non-original sequence, shard_id = how short the `kv` is among all the shards in the same sequence (ranking of `kv` sort ASC)
# - collect all the sequences that has the same `src_gpuid` and `seqid`
# - sort them by the `kv` to determine the shard id of that sequence.

for item in items:
    if item["is_original"]:
        item["shard_id"] = 0

# now handle the non-original sequences.
from collections import defaultdict
non_original_items = [item for item in items if not item["is_original"]]
src_gpuid_seqid_to_items = defaultdict(list)
for item in non_original_items:
    src_gpuid_seqid_to_items[(item["src_gpuid"], item["seqid"])].append(item)

for src_gpuid_seqid, items_ in src_gpuid_seqid_to_items.items():
    items_.sort(key=lambda x: x["kv"])
    for i, item in enumerate(items_):
        item["shard_id"] = i

rich.print(items)


# %%
def postprocess_items(items) -> list[dict]:
    """
    Postprocess the items to add a "shard_id" field.
    The "shard_id" field is always 0 for the original sequence.
    For each non-original sequence, shard_id = how short the `kv` is among all the shards in the same sequence (ranking of `kv` sort ASC)
    - collect all the sequences that has the same `src_gpuid` and `seqid`
    - sort them by the `kv` to determine the shard id of that sequence.
    """
    from copy import deepcopy
    items = deepcopy(items)

    for item in items:
        if item["is_original"]:
            item["shard_id"] = 0
    
    # now handle the non-original sequences.
    non_original_items = [item for item in items if not item["is_original"]]
    src_gpuid_seqid_to_items = defaultdict(list)
    for item in non_original_items:
        src_gpuid_seqid_to_items[(item["src_gpuid"], item["seqid"])].append(item)
    
    for src_gpuid_seqid, items_ in src_gpuid_seqid_to_items.items():
        items_.sort(key=lambda x: x["kv"])
        for i, item in enumerate(items_):
            item["shard_id"] = i
    return items

items = postprocess_items(items)
rich.print(items)

# %%
# Notes:
# 1. (src_gpuid, seqid) represents a sequence - this means the `seq_id` alone is not enough to represent a sequence's id.
# %%
world_size = len(set(item["gpuid"] for item in items))
max_cp_degree = world_size * 2
max_num_seqs = max(item["seqid"] for item in items) + 1

p = dict(world_size=world_size, max_cp_degree=max_cp_degree)
rich.print(p)


# %%
import torch
# %%
def exclusive_cumsum(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Cumsum but excluding itself."""
    cumsum = tensor.cumsum(dim=dim)
    zero = torch.zeros_like(tensor.select(dim, 0))
    return torch.cat([zero.unsqueeze(dim), cumsum.narrow(dim, 0, cumsum.size(dim) - 1)], dim=dim)

# %%
# item = {'q': 10239, 'kv': 26624, 'gpuid': 3, 'seqid': 0, 'src_gpuid': 1, 'is_original': False}
# %%
# seq_lens[src_gpuid, seqid] = sequence length of this sequence.
seq_lens = torch.zeros(world_size, max_num_seqs, dtype=torch.int64)
for item in items:
    seq_lens[item["src_gpuid"], item["seqid"]] += item["q"]

# %% 
shard_lens = torch.zeros(world_size, max_num_seqs, max_cp_degree, dtype=torch.int64)
for item in items:
    shard_lens[item["src_gpuid"], item["seqid"], item["shard_id"]] += item["q"]
rich.print("shard_lens = ", shard_lens)
# %%
rich.print("seq_lens =" , seq_lens)
# %%
# cp_num[src_gpuid, seqid] = number of destinations this sequence is dispatched to.
cp_num = torch.zeros(world_size, max_num_seqs, dtype=torch.int64)
for item in items:
    cp_num[item["src_gpuid"], item["seqid"]] += 1
rich.print("cp_num =", cp_num)

# %%
# cp_dst[src_gpuid, seqid, shard_id] = dst_rank that this shard should send to.
#    -1 means this shard is not dispatched.
cp_dst = torch.ones(world_size, max_num_seqs, max_cp_degree, dtype=torch.int64) * -1
for item in items:
    cp_dst[item["src_gpuid"], item["seqid"], item["shard_id"]] = item["gpuid"]

rich.print("cp_dst = ", cp_dst)
# %%
# num_cp_shards: for each source gpu, how many "total shards" are there to send to other GPUs
num_cp_shards = cp_num.sum(dim=1)
rich.print("num_cp_shards = ", num_cp_shards)
# %%
# take the max of num_cp_shards to get the proper dimension for the tensors below.
pad_len = torch.max(num_cp_shards)
rich.print("pad_len = ", pad_len)
# %%
# cp_seq_lens[src_gpuid, shard_id] = sequence length of this shard.
# Since we are summing a row, we cumulatively "move" the shard ids to the right place.
cp_seq_lens = torch.zeros(world_size, pad_len, dtype=torch.int64)
for i in range(world_size):
    start_shard_idx = 0
    for j in range(max_num_seqs):
        num_cp = int(cp_num[i, j].item())
        end_shard_idx = start_shard_idx + num_cp
        cp_seq_lens[i, start_shard_idx:end_shard_idx] = shard_lens[i, j, :num_cp]
        start_shard_idx = end_shard_idx
        pass
rich.print("cp_seq_lens = ", cp_seq_lens)
# %%
# cp_query_dst[src_gpuid, shard_id] = dst_rank that this shard should send to.
# init all to -1 as masking.
cp_query_dst = torch.ones(world_size, pad_len, dtype=torch.int64) * -1
for i in range(world_size):
    start_shard_idx = 0
    for j in range(max_num_seqs):
        num_cp = int(cp_num[i, j].item())
        end_shard_idx = start_shard_idx + num_cp
        cp_query_dst[i, start_shard_idx:end_shard_idx] = cp_dst[i, j, :num_cp]
        start_shard_idx = end_shard_idx
        pass
rich.print("cp_query_dst = ", cp_query_dst)
# %%
# kv_to_q_mapping[src_gpuid, shard_id, cp_id, 0] = kv shard id
# kv_to_q_mapping[src_gpuid, shard_id, cp_id, 1] = q shard id
# the k shard id that maps to the q shard id. this is essentially a mapping!
# init all to -1 as masking.

kv_to_q_mapping = torch.ones((world_size, pad_len, max_cp_degree, 2), dtype=torch.int64) * -1
kv_to_q_rank = torch.ones((world_size, pad_len, max_cp_degree), dtype=torch.int64) * -1
for i in range(world_size):
    start_shard_idx = 0
    for j in range(max_num_seqs):
        num_cp = int(cp_num[i, j].item())
        end_shard_idx = start_shard_idx + num_cp

        # Prepre the mask: a standard causal mask 
        row_indices = torch.arange(num_cp).view(-1, 1)
        col_indices = torch.arange(max_cp_degree).view(1, -1)
        mask = col_indices < (num_cp - row_indices)
        """
        ```python
        num_cp=4
        max_cp_degree=8
        row_indices = torch.arange(num_cp).view(-1, 1)
        col_indices = torch.arange(max_cp_degree).view(1, -1)
        mask = col_indices < (num_cp - row_indices)
        ```

        mask = [
            [ True,  True,  True,  True, False, False, False, False], # 0
            [ True,  True,  True, False, False, False, False, False], # 1
            [ True,  True, False, False, False, False, False, False], # 2 
            [ True, False, False, False, False, False, False, False], # 3
        ]
        """

        #### Compute kv_to_q_mapping.
        # - kv_to_q_mapping_seq[..., 0] = kv shard id
        kv_to_q_mapping_seq = torch.empty((num_cp, max_cp_degree, 2), dtype=torch.int64)
        kv_to_q_mapping_seq[..., 0] = torch.where(mask, i, -1)
        
        # - kv_to_q_mapping_seq[..., 1] = q shard id
        num_cp_shards_before = start_shard_idx
        vals_ch1 = row_indices + col_indices + num_cp_shards_before
        kv_to_q_mapping_seq[..., 1] = torch.where(mask, vals_ch1, -1)

        kv_to_q_mapping[i, start_shard_idx:end_shard_idx] = kv_to_q_mapping_seq

        start_shard_idx = end_shard_idx
        pass

rich.print("kv_to_q_mapping = ", kv_to_q_mapping)

# %%
# kv_to_q_rank[src_gpuid, shard_id, cp_id] = rank of the destination GPU that this shard should send to.
kv_to_q_rank = torch.ones((world_size, pad_len, max_cp_degree), dtype=torch.int64) * -1
for i in range(world_size):
    start_shard_idx = 0
    for j in range(max_num_seqs):
        num_cp = int(cp_num[i, j].item())
        end_shard_idx = start_shard_idx + num_cp

        # Prepre the mask: a standard causal mask 
        row_indices = torch.arange(num_cp).view(-1, 1)
        col_indices = torch.arange(num_cp).view(1, -1)
        mask = col_indices < (num_cp - row_indices)

        #### Compute kv_to_q_rank (Index of this KV to the query's dst).
        # kv_to_q_rank[src_rank, seq_id, shard_id] = dst_rank that this kv shard should send to.
        dst_rank = cp_dst[i, j, :num_cp].view(-1, 1). repeat(1, num_cp) * mask 
        dst_rank += mask.int() - 1
        kv_to_q_rank[i, start_shard_idx:end_shard_idx, :num_cp] = dst_rank

        start_shard_idx = end_shard_idx
        pass
rich.print("kv_to_q_rank = ", kv_to_q_rank)

# %%
# kv_context_size[src_gpuid, shard_id] = context size of this shard.
kv_context_size = torch.zeros((world_size, pad_len), dtype=torch.int64)
for i in range(world_size):
    start_shard_idx = 0
    for j in range(max_num_seqs):
        num_cp = int(cp_num[i, j].item())
        seq_len = seq_lens[i, j].item()
        if seq_len == 0:
            continue
        end_shard_idx = start_shard_idx + num_cp
        existing_ctx_lens = exclusive_cumsum(shard_lens[i, j, :num_cp], dim=0)
        # rich.print("existing_ctx_lens = ", existing_ctx_lens)
        kv_context_size[i, start_shard_idx:end_shard_idx] = existing_ctx_lens
        # rich.print("kv_context_size[i, start_shard_idx:end_shard_idx] = ", kv_context_size[i, start_shard_idx:end_shard_idx])
        start_shard_idx = end_shard_idx
        pass
rich.print("kv_context_size = ", kv_context_size)
# %%
# q_to_num_kv_seq[src_gpuid, shard_id] = number of kv sequences that this query should receive.
q_to_num_kv_seq = torch.zeros((world_size, pad_len), dtype=torch.int64)
for i in range(world_size):
    start_shard_idx = 0
    for j in range(max_num_seqs):
        num_cp = int(cp_num[i, j].item())
        end_shard_idx = start_shard_idx + num_cp
        q_to_num_kv_seq[i, start_shard_idx:end_shard_idx] = torch.arange(num_cp) + 1
        start_shard_idx = end_shard_idx
        pass
rich.print("q_to_num_kv_seq = ", q_to_num_kv_seq)

# %%
# q_to_num_kv_token[src_gpuid, shard_id] = number of kv tokens that this query should receive.
q_to_num_kv_token = torch.zeros((world_size, pad_len), dtype=torch.int64)
q_to_num_kv_tokens = kv_context_size + cp_seq_lens
rich.print("q_to_num_kv_tokens = ", q_to_num_kv_tokens)
# %%


# %% Converting the tensors into Metadata.
# Converting using the old method to Metadata.
from d2.runtime.inplace_metadata import compute_metadata, compute_metadata_kv, compute_attn_layout_seqlens
fwd_q_metadata, rev_q_metadata, q_intermediates = compute_metadata(
    cp_seq_lens, cp_query_dst, return_intermediate=True
)
_, q_seq_to_dst, _ = q_intermediates
fwd_k_metadata, rev_k_metadata, kv_intermediates = compute_metadata_kv(
    kv_to_q_mapping, kv_to_q_rank, kv_context_size, q_to_num_kv_seq,
    q_to_num_kv_tokens, cp_seq_lens, num_cp_shards, cp_query_dst,
    q_seq_to_dst.squeeze(2), pad_len,
    return_intermediate=True
)
attention_metadata = compute_attn_layout_seqlens(
    cp_seq_lens, q_to_num_kv_tokens, cp_query_dst, shard_to_tuple=True
)
ret = (
    fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata, attention_metadata
)

# %% # Converting using the new method to Metdata
from d2.runtime.inplace_metadata import compute_e2e_metadata
mlp_seq_len = seq_lens
mlp_num_seqs = cp_num
mlp_q_dispatch = cp_query_dst
mlp_kv_to_q_mapping = kv_to_q_mapping
mlp_kv_to_q_rank = kv_to_q_rank
mlp_kv_context_size = kv_context_size
mlp_q_to_num_kv_seq = q_to_num_kv_seq
mlp_q_to_num_kv_token = q_to_num_kv_tokens
ret = compute_e2e_metadata(
    mlp_seq_len, mlp_num_seqs, mlp_q_dispatch, mlp_kv_to_q_mapping,
    mlp_kv_to_q_rank, mlp_kv_context_size, mlp_q_to_num_kv_seq,
    mlp_q_to_num_kv_token, return_intermediate=True
)
pass
# %% # Converting using the fast all2all Metdata
from d2.runtime.fast_alltoall_metadata import compute_e2e_fa2a_metadata
mlp_seq_len = seq_lens
mlp_num_seqs = cp_num
mlp_q_dispatch = cp_query_dst
mlp_kv_to_q_mapping = kv_to_q_mapping
mlp_kv_to_q_rank = kv_to_q_rank
mlp_kv_context_size = kv_context_size
mlp_q_to_num_kv_seq = q_to_num_kv_seq
mlp_q_to_num_kv_token = q_to_num_kv_tokens
ret = compute_e2e_fa2a_metadata(
    mlp_seq_len=mlp_seq_len, 
    mlp_num_seqs=mlp_num_seqs, 
    mlp_q_dispatch=mlp_q_dispatch,
    mlp_kv_to_q_mapping=mlp_kv_to_q_mapping, 
    mlp_kv_to_q_rank=mlp_kv_to_q_rank,
    mlp_kv_context_size=mlp_kv_context_size, 
    mlp_q_to_num_kv_seq=mlp_q_to_num_kv_seq,
    mlp_q_to_num_kv_token=mlp_q_to_num_kv_token, 

    # This is just what we assumed...
    hidden_size_q=1024, 
    hidden_size_k=1024, 
    element_size=2 # float16
)
pass


# %% Compose the final function call to create the metadata out of the tensors.
def item_to_metadata(
    items,
    hidden_size_q=1024, 
    hidden_size_k=1024, 
    element_size=2 # float16
):
    """
    Convert the items from the replanning algorithm to the metadata objects.

    TODO: 
    1. Local vs remote sequence id arrangement. 
        Usually, we prefer to have the local sequence before the sequence that needed to be sent remotely. Reordering the sequence may be needed, but at which point do we reorder is a question. 
    2. 

    Usage:
    ```python
    K = 1024
    items = batch_to_items([
        [16 * K] * 4,
        [32 * K] * 2, 
        [8 * K] * 8,
        [4 * K] * 16,
    ])
    
    # Now you get the items to a list of dicts...
    items = [
        {'q': 16384, 'kv': 16384, 'gpuid': 0, 'seqid': 0, 'src_gpuid': 0, 'is_original': True},
        ...
        {'q': 4096, 'kv': 4096, 'gpuid': 3, 'seqid': 14, 'src_gpuid': 3, 'is_original': True},
        {'q': 4096, 'kv': 4096, 'gpuid': 3, 'seqid': 15, 'src_gpuid': 3, 'is_original': True},
        {'q': 6144, 'kv': 6144, 'gpuid': 2, 'seqid': 0, 'src_gpuid': 1, 'is_original': False},
        {'q': 6144, 'kv': 32768, 'gpuid': 2, 'seqid': 0, 'src_gpuid': 1, 'is_original': False},
        {'q': 10239, 'kv': 16383, 'gpuid': 3, 'seqid': 0, 'src_gpuid': 1, 'is_original': False},
        {'q': 10239, 'kv': 26624, 'gpuid': 3, 'seqid': 0, 'src_gpuid': 1, 'is_original': False}
    ]
    ```
    """
    import torch
    from collections import defaultdict
    from copy import deepcopy
    
    # Step 1: Postprocess items to add shard_id
    # - {'q': 4096, 'kv': 4096, 'gpuid': 3, 'seqid': 14, 'src_gpuid': 3, 'is_original': True}
    # + {'q': 4096, 'kv': 4096, 'gpuid': 3, 'seqid': 14, 'src_gpuid': 3, 'is_original': True, "shard_id": 0}
    items = postprocess_items(items)
    
    # Step 2: Calculate basic parameters
    world_size = len(set(item["gpuid"] for item in items))
    max_cp_degree = world_size * 2
    max_num_seqs = max(item["seqid"] for item in items) + 1
    
    # Step 3: Create tensors from items
    # seq_lens[src_gpuid, seqid] = sequence length of this sequence
    seq_lens = torch.zeros(world_size, max_num_seqs, dtype=torch.int64)
    for item in items:
        seq_lens[item["src_gpuid"], item["seqid"]] += item["q"]
    
    # shard_lens[src_gpuid, seqid, shard_id] = length of this shard
    shard_lens = torch.zeros(world_size, max_num_seqs, max_cp_degree, dtype=torch.int64)
    for item in items:
        shard_lens[item["src_gpuid"], item["seqid"], item["shard_id"]] += item["q"]
    
    # cp_num[src_gpuid, seqid] = number of destinations this sequence is dispatched to
    cp_num = torch.zeros(world_size, max_num_seqs, dtype=torch.int64)
    for item in items:
        cp_num[item["src_gpuid"], item["seqid"]] += 1
    
    # cp_dst[src_gpuid, seqid, shard_id] = dst_rank that this shard should send to
    cp_dst = torch.ones(world_size, max_num_seqs, max_cp_degree, dtype=torch.int64) * -1
    for item in items:
        cp_dst[item["src_gpuid"], item["seqid"], item["shard_id"]] = item["gpuid"]
    
    # num_cp_shards: for each source gpu, how many "total shards" are there to send to other GPUs
    num_cp_shards = cp_num.sum(dim=1)
    
    # take the max of num_cp_shards to get the proper dimension for the tensors below
    pad_len = torch.max(num_cp_shards)
    
    # cp_seq_lens[src_gpuid, shard_id] = sequence length of this shard
    cp_seq_lens = torch.zeros(world_size, pad_len, dtype=torch.int64)
    for i in range(world_size):
        start_shard_idx = 0
        for j in range(max_num_seqs):
            num_cp = int(cp_num[i, j].item())
            end_shard_idx = start_shard_idx + num_cp
            cp_seq_lens[i, start_shard_idx:end_shard_idx] = shard_lens[i, j, :num_cp]
            start_shard_idx = end_shard_idx
    
    # cp_query_dst[src_gpuid, shard_id] = dst_rank that this shard should send to
    cp_query_dst = torch.ones(world_size, pad_len, dtype=torch.int64) * -1
    for i in range(world_size):
        start_shard_idx = 0
        for j in range(max_num_seqs):
            num_cp = int(cp_num[i, j].item())
            end_shard_idx = start_shard_idx + num_cp
            cp_query_dst[i, start_shard_idx:end_shard_idx] = cp_dst[i, j, :num_cp]
            start_shard_idx = end_shard_idx
    
    # kv_to_q_mapping[src_gpuid, shard_id, cp_id, 0] = kv shard id
    # kv_to_q_mapping[src_gpuid, shard_id, cp_id, 1] = q shard id
    kv_to_q_mapping = torch.ones((world_size, pad_len, max_cp_degree, 2), dtype=torch.int64) * -1
    for i in range(world_size):
        start_shard_idx = 0
        for j in range(max_num_seqs):
            num_cp = int(cp_num[i, j].item())
            end_shard_idx = start_shard_idx + num_cp
            
            # Prepare the mask: a standard causal mask 
            row_indices = torch.arange(num_cp).view(-1, 1)
            col_indices = torch.arange(max_cp_degree).view(1, -1)
            mask = col_indices < (num_cp - row_indices)
            
            # Compute kv_to_q_mapping
            kv_to_q_mapping_seq = torch.empty((num_cp, max_cp_degree, 2), dtype=torch.int64)
            kv_to_q_mapping_seq[..., 0] = torch.where(mask, i, -1)
            
            num_cp_shards_before = start_shard_idx
            vals_ch1 = row_indices + col_indices + num_cp_shards_before
            kv_to_q_mapping_seq[..., 1] = torch.where(mask, vals_ch1, -1)
            
            kv_to_q_mapping[i, start_shard_idx:end_shard_idx] = kv_to_q_mapping_seq
            start_shard_idx = end_shard_idx
    
    # kv_to_q_rank[src_gpuid, shard_id, cp_id] = rank of the destination GPU
    kv_to_q_rank = torch.ones((world_size, pad_len, max_cp_degree), dtype=torch.int64) * -1
    for i in range(world_size):
        start_shard_idx = 0
        for j in range(max_num_seqs):
            num_cp = int(cp_num[i, j].item())
            end_shard_idx = start_shard_idx + num_cp
            
            # Prepare the mask: a standard causal mask 
            row_indices = torch.arange(num_cp).view(-1, 1)
            col_indices = torch.arange(num_cp).view(1, -1)
            mask = col_indices < (num_cp - row_indices)
            
            # Compute kv_to_q_rank
            dst_rank = cp_dst[i, j, :num_cp].view(-1, 1).repeat(1, num_cp) * mask 
            dst_rank += mask.int() - 1
            kv_to_q_rank[i, start_shard_idx:end_shard_idx, :num_cp] = dst_rank
            
            start_shard_idx = end_shard_idx
    
    # kv_context_size[src_gpuid, shard_id] = context size of this shard
    kv_context_size = torch.zeros((world_size, pad_len), dtype=torch.int64)
    for i in range(world_size):
        start_shard_idx = 0
        for j in range(max_num_seqs):
            num_cp = int(cp_num[i, j].item())
            seq_len = seq_lens[i, j].item()
            if seq_len == 0:
                continue
            end_shard_idx = start_shard_idx + num_cp
            existing_ctx_lens = exclusive_cumsum(shard_lens[i, j, :num_cp], dim=0)
            kv_context_size[i, start_shard_idx:end_shard_idx] = existing_ctx_lens
            start_shard_idx = end_shard_idx
    
    # q_to_num_kv_seq[src_gpuid, shard_id] = number of kv sequences that this query should receive
    q_to_num_kv_seq = torch.zeros((world_size, pad_len), dtype=torch.int64)
    for i in range(world_size):
        start_shard_idx = 0
        for j in range(max_num_seqs):
            num_cp = int(cp_num[i, j].item())
            end_shard_idx = start_shard_idx + num_cp
            q_to_num_kv_seq[i, start_shard_idx:end_shard_idx] = torch.arange(num_cp) + 1
            start_shard_idx = end_shard_idx
    
    # q_to_num_kv_token[src_gpuid, shard_id] = number of kv tokens that this query should receive
    q_to_num_kv_tokens = kv_context_size + cp_seq_lens
    
    # Step 4: Convert tensors to metadata using the fast alltoall method
    from d2.runtime.fast_alltoall_metadata import compute_e2e_fa2a_metadata
    
    ret = compute_e2e_fa2a_metadata(
        mlp_seq_len=seq_lens, 
        mlp_num_seqs=cp_num, 
        mlp_q_dispatch=cp_query_dst,
        mlp_kv_to_q_mapping=kv_to_q_mapping, 
        mlp_kv_to_q_rank=kv_to_q_rank,
        mlp_kv_context_size=kv_context_size, 
        mlp_q_to_num_kv_seq=q_to_num_kv_seq,
        mlp_q_to_num_kv_token=q_to_num_kv_tokens, 
        
        # These are assumptions for the metadata computation
        hidden_size_q=hidden_size_q, 
        hidden_size_k=hidden_size_k, 
        element_size=element_size  # float16
    )
    
    return ret


