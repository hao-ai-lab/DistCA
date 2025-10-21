from collections import defaultdict
from copy import deepcopy
import math
import random
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import time
import rich
import torch
from d2.runtime.compute_metadata import from_planner_output
from d2.runtime.shard_info import (ShardInfo, handle_planner_metadata,
                                   items_into_shardinfos)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

K = 1024

# This funciton is deprecated. As we use Item class not dict.
def batch_to_items(batches):
    items = []
    seqid = 0
    for gpuid, batch in enumerate(batches):
        for _, seq_len in enumerate(batch):
            items.append(dict(
                q=seq_len, kv=seq_len, 
                gpuid=gpuid, seqid=seqid, src_gpuid=gpuid, seq_len=seq_len,
                is_original=True
            ))
            seqid += 1
    return items

# Transfer batch to Item. Only for MLP-DP.
def batch_to_items_class(batches: list[list[int]], model_config=None):
    items = []
    seqid = 0
    for gpuid, batch in enumerate(batches):
        for seq_len in batch:
            item_dict = {'q': seq_len, 'kv': seq_len}
            item = Item(
                model_config,
                seq_len,
                seqid,
                gpuid,
                gpuid,
                item_dict,
                is_original=True
            )
            items.append(item)
            seqid += 1
    return items

# Can handle MLP DPCP and dummy doc.
def batch_to_items_with_dummy(batches: List[List[int]], num_tokens_per_rank: int, as_world_size: int, model_config: dict):

    items = []
    seqid = 0
    rank_budgets = [num_tokens_per_rank] * as_world_size
    current_rank_idx = 0

    for batch in batches:
        if len(batch) == 1 and batch[0] < num_tokens_per_rank:
            while current_rank_idx < as_world_size and rank_budgets[current_rank_idx] == 0:
                current_rank_idx += 1
            assert rank_budgets[current_rank_idx] == num_tokens_per_rank, "dummy doc should put on a empty rank"
            # dummy doc, this rank only put this dummy item.
            doc_len = batch[0]
            item_dict = {'q': doc_len, 'kv': doc_len}
            new_item = Item(model_config, doc_len, seqid, current_rank_idx, current_rank_idx,
                            item_dict, is_original=True)
            items.append(new_item)
            current_rank_idx += 1
            seqid += 1
        else:
            for doc_length in batch:
                doc_len = doc_length
                remaining_len = doc_len
                head_prefix_len = 0
                tail_suffix_start_pos = doc_len
                is_original_flag = True

                while remaining_len > 0:
                    while current_rank_idx < as_world_size and rank_budgets[current_rank_idx] == 0:
                        current_rank_idx += 1
                    if current_rank_idx >= as_world_size:
                        raise ValueError(f"Not enough space for seqid {seqid}, remaining_len {remaining_len}.")
                    
                    chunk_size = min(remaining_len, rank_budgets[current_rank_idx])
                    
                    if remaining_len == doc_len and chunk_size == doc_len:
                        item_dict = {'q': doc_len, 'kv': doc_len}
                        new_item = Item(model_config, doc_len, seqid, current_rank_idx, current_rank_idx,
                                        item_dict, is_original=True)
                        items.append(new_item)
                    else:
                        q_len_head = chunk_size // 2
                        q_len_tail = chunk_size - q_len_head
                        
                        new_head_kv = head_prefix_len + q_len_head
                        head_item = {'q': q_len_head, 'kv': new_head_kv}
                        tail_item = {'q': q_len_tail, 'kv': tail_suffix_start_pos}

                        new_item = Item(model_config, doc_len, seqid, current_rank_idx, current_rank_idx,
                                        head_item, tail_item, is_original=is_original_flag)
                        items.append(new_item)

                        head_prefix_len = new_head_kv
                        tail_suffix_start_pos -= q_len_tail
                        is_original_flag = False

                    rank_budgets[current_rank_idx] -= chunk_size
                    remaining_len -= chunk_size

                seqid += 1
        
    return items





"""
Partition and place a batch of variable-length documents onto GPU ranks under a
hybrid Data-Parallel (DP) and Context-Parallel (CP) policy, returning a list of
Item objects.(Can handle MLP DPCP. Can't handle dummy docs.)

Steps
1. Schedule each document to the next available rank in original order (DP).
2. If a document exceeds a rank's token budget, split it into head / tail
   chunks and keep placing until the entire document is covered (CP).

Args
  batches : List[List[int]]
      Outer list = per-sequence groups, inner list = document lengths in tokens.
  num_tokens_per_rank : int
      Maximum tokens that one rank can accept in this micro-batch.
  DP_degree : int
      Number of data-parallel ranks.

Returns
  List[Item]
      One Item per **placed chunk**.  
      • DP-fitting documents yield 1 Item.  
      • CP-split documents yield 2 Items (head + tail).

Examples
----------
Example 1 : documents fit into DP ranks
    batch           = [[16K], [8K, 8K], [4K]*4, [2K]*8]
    num_tokens_per_rank = 16K
    DP_degree         = 4

    Item0  q=16K kv=16K  gpu=0 seq=0 src=0 is_orig=True
    Item1  q= 8K kv= 8K  gpu=1 seq=1 src=1 is_orig=True
    ...
Example 2 : CP split required
    batch           = [[32K], [8K, 8K], [4K]*4]
    num_tokens_per_rank = 16K
    DP_degree         = 4

    # 32K doc split across 2 ranks (16K each)
    Item0  head: q=8K kv= 8K
           tail: q=8K kv=32K
           gpu=0 seq=0 src=0 is_orig=True

    Item1  head: q=8K kv=16K
           tail: q=8K kv=24K
           gpu=1 seq=0 src=1 is_orig=True

    # 8K docs fit directly
    Item2  q=8K kv=8K  gpu=2 seq=1 src=2 is_orig=True
    Item3  q=8K kv=8K  gpu=2 seq=2 src=2 is_orig=True
    ...
"""
def batch_to_items_general(batches: List[List[int]], num_tokens_per_rank: int, DP_degree: int, model_config: dict):
    """
    Put a batch of documents onto GPU ranks and return a list of Item objects.
    Args:
        batches: List[List[int]]
            Outer list = per-sequence groups, inner list = document lengths in tokens.
        num_tokens_per_rank: int
            Maximum tokens that one rank can accept in this micro-batch.
        DP_degree: int
            Number of data-parallel ranks.
        model_config: dict
            Model configuration.

    Returns:
        List[Item]
        One Item per **placed chunk**.  
        • DP-fitting documents yield 1 Item.  
        • CP-split documents yield 2 Items (head + tail).
    """
    items = []
    seqid = 0
    rank_budgets = [num_tokens_per_rank] * DP_degree
    current_rank_idx = 0

    # Flatten the batches into a list of dicts, each dict contains the length of the document.
    all_docs: list[dict] = []
    for _, batch in enumerate(batches):
        for doc_len in batch:
            all_docs.append({'len': doc_len})

    for doc_info in all_docs:
        doc_len = doc_info['len']
        remaining_len = doc_len
        head_prefix_len = 0
        tail_suffix_start_pos = doc_len
        is_original_flag = True

        while remaining_len > 0:
            while current_rank_idx < DP_degree and rank_budgets[current_rank_idx] == 0:
                current_rank_idx += 1
            if current_rank_idx >= DP_degree:
                raise ValueError(f"Not enough space for seqid {seqid}, remaining_len {remaining_len}.")
            
            chunk_size = min(remaining_len, rank_budgets[current_rank_idx])
            
            if remaining_len == doc_len and chunk_size == doc_len:
                item_dict = {'q': doc_len, 'kv': doc_len}
                new_item = Item(model_config, doc_len, seqid, current_rank_idx, current_rank_idx,
                                item_dict, is_original=True)
                items.append(new_item)
            else:
                q_len_head = chunk_size // 2
                q_len_tail = chunk_size - q_len_head
                
                new_head_kv = head_prefix_len + q_len_head
                head_item = {'q': q_len_head, 'kv': new_head_kv}
                tail_item = {'q': q_len_tail, 'kv': tail_suffix_start_pos}

                new_item = Item(model_config, doc_len, seqid, current_rank_idx, current_rank_idx,
                                head_item, tail_item, is_original=is_original_flag)
                items.append(new_item)

                head_prefix_len = new_head_kv
                tail_suffix_start_pos -= q_len_tail
                is_original_flag = False

            rank_budgets[current_rank_idx] -= chunk_size
            remaining_len -= chunk_size

        seqid += 1
        
    return items

from collections import deque


def cp_list_to_mlp_list(cp_rank_doc_lens: List[List[int]], as_world_size: int, num_token_per_rank: int) -> List[List[int]]:
    final_mlp_lists: List[List[int]] = []
    current_rank: List[int] = []

    for doc_list_per_rank in cp_rank_doc_lens:
        docs_queue = deque(doc_list_per_rank)

        while docs_queue:
            doc_len = docs_queue.popleft()
            is_split_doc = False

            while True:
                space_left = num_token_per_rank - sum(current_rank)

                if space_left == 0:
                    if current_rank:
                        final_mlp_lists.append(current_rank)
                    current_rank = []
                    space_left = num_token_per_rank

                if doc_len <= space_left:
                    if is_split_doc:
                        head = doc_len // 2
                        tail = doc_len - head
                        current_rank.extend([head, tail])
                    else:
                        current_rank.append(doc_len)
                    break

                if space_left > 0:
                    head1 = space_left // 2
                    tail1 = space_left - head1
                    current_rank.extend([head1, tail1])
                
                final_mlp_lists.append(current_rank)
                current_rank = []
                
                doc_len -= space_left
                is_split_doc = True

        if current_rank:
            final_mlp_lists.append(current_rank)
            current_rank = []

    while len(final_mlp_lists) < as_world_size:
        final_mlp_lists.append([])
    
    if len(final_mlp_lists) > as_world_size:
        final_mlp_lists = final_mlp_lists[:as_world_size]
    
    # for l in final_mlp_lists, sum(l) should be either num_token_per_rank or dummy(tp_size).
    assert len(final_mlp_lists) == as_world_size, f"final_mlp_lists should contain {as_world_size} number of List. But get {len(final_mlp_lists)}. final_mlp_list: {final_mlp_lists}"
    return final_mlp_lists



# Represent a part of sequence.
# This class can handle DP/CP MLP layout. 
# For DP layout, whole document will be put on one GPU.
# For CP layout, the document will be split into two chunks, head and tail on multiple GPUs.
class Item:
    def __init__(self, 
                model_config,
                seq_len, seqid, gpuid, src_gpuid,
                *item_dicts, 
                is_original = True):
        self.model_config = model_config
        assert len(item_dicts) == 1 or len(item_dicts) == 2, "item_dicts must contain exactly 1 or 2 items"
        for item in item_dicts:
            assert 'q' in item and 'kv' in item, "Each item must have 'q' and 'kv' keys"

        self.seq_len = seq_len
        self.seqid = seqid
        self.gpuid = gpuid
        self.src_gpuid = src_gpuid
        self.is_original = is_original
        self.shard_id = None

        self.items = [item for item in item_dicts]
        if len(self.items) == 1:
            self.complete = True
            self.complete_item = self.items[0]
        else:
            self.complete = False
            self.head = self.items[0]
            self.tail = self.items[1]
        self.total_flops = self.get_flops()

    def get_flops(self):
        flops = 0
        for item in self.items:
            q = int(item['q'])
            kv = int(item['kv'])
            # For each q token, it needs to attend to all previous kv tokens
            # So for q tokens from 0 to q-1, each token i attends to kv-i tokens
            # This gives us: sum(kv-i) for i in range(q)
            # Which is equivalent to: q*kv - sum(i) for i in range(q)
            # And sum(i) from 0 to q-1 is q*(q-1)/2
            # So final formula is: q*kv - q*(q-1)/2
            flops += q * kv - (q * (q - 1)) // 2
        return flops

    def get_q_size(self, length):
        hidden_size = self.model_config.hidden_size
        q_size = length * hidden_size
        return q_size

    def get_kv_size(self, length):
        hidden_size = self.model_config.hidden_size
        num_attention_heads = self.model_config.num_attention_heads
        num_key_value_heads = self.model_config.num_key_value_heads
        
        head_dim = hidden_size // num_attention_heads
        kv_size = 2 * length * num_key_value_heads * head_dim
        return kv_size

    def get_priority(self, local_surplus, remote_deficit):
        max_possible_flops = min(local_surplus, remote_deficit, self.total_flops)
        q_len = max_possible_flops / (self.seq_len + 1)
        kv_len = self.seq_len + 1       # this can be improve if use memcpy instead of transfering all kvs.
        communication_volume = self.get_q_size(q_len) + self.get_kv_size(kv_len)
        # latency = communication_volume / transfer_speed, if we know current rank_id, remote rank_id, network speed. We can compute communication latency.
        priority = communication_volume / max_possible_flops
        # maybe we can have a matrix in the planner to compute the latency of communication.
        return priority

    def split_item(self, flops, remote_gpuid, verbose = False):
        """
        Split the Item according to the specified amount of FLOPs.

        - If flops >= self.total_flops, move entire item to remote_gpuid.
        - If flops < self.total_flops, split the Item and return a new Item containing the head and tail chunks.
          The original Item (self) will be updated accordingly.

        Args:
            flops (float): The amount of computation (FLOPs) to move.
            remote_gpuid (int): The target GPU ID.

        Returns:
            Item or None: The newly created Item object; returns None if move entire item.
            actual_flops_moved (float): The actual amount of FLOPs moved.
        """
        # If the current Item is empty or has no FLOPs to move, splitting is not possible
        if not self.items or self.total_flops <= 0:
            return None

        def rlog(message):
            if verbose:
                rich.print(message)

        # If the requested flops is greater than or equal to the total FLOPs, move the whole Item
        if flops >= self.total_flops:
            # Update the gpuid and is_original status to reflect the move
            self.gpuid = remote_gpuid
            self.is_original = False
            # There is no new items. Return None.
            rlog(f"    - [bold]Move entire item[/bold]: Actual Moving ({self.total_flops:.2f} FLOPs) to satisfy need.")
            return None, self.total_flops

        else:
            flops_per_q = self.seq_len + 1
            q_to_move = int(flops / flops_per_q)

            if q_to_move <= 0:
                return None, 0
            moved_flops_actual = q_to_move * flops_per_q
            rlog(f"    - [bold]Splitting item[/bold]: Actual Moving q={q_to_move*2} ({moved_flops_actual:.2f} FLOPs) to satisfy need.")
            
            # TODO(Pb): Current flops_per_q only support even seq_len. Need to support odd seq_len in the future.
            if self.complete:
                # Split current Item
                head_dict = {}
                tail_dict = {}
                head_dict['q'] = (self.complete_item['q'] + 1) // 2  - q_to_move
                head_dict['kv'] = head_dict['q']

                tail_dict['q'] = self.complete_item['q'] // 2  - q_to_move
                tail_dict['kv'] = self.complete_item['kv']       # Unchanged

                self.head = head_dict
                self.tail = tail_dict
                self.complete = False
                self.complete_item = None
            else:
                # Split head
                self.head['q'] = self.head['q'] - q_to_move
                self.head['kv'] = self.head['kv'] - q_to_move

                # Split tail
                self.tail['q'] = self.tail['q'] - q_to_move
                self.tail['kv'] = self.tail['kv']   # Unchanged
            self.items = [self.head, self.tail]
            rlog(f"Origin flops {self.total_flops}, moved flops: {moved_flops_actual}, current flops: {self.get_flops()}")

            rlog(f"    - [debug] total_flops(before)={self.total_flops}, get_flops()={self.get_flops()}, moved_flops_actual={moved_flops_actual}, sum={self.get_flops() + moved_flops_actual}")
            assert self.total_flops == self.get_flops() + moved_flops_actual, f"Total flops should be equal. This error is mostly because of odd doc length. Currently, we only support even doc length. Please pad to even."
            self.total_flops = self.get_flops()
            rlog(f"    - [bold]Splitting item[/bold]: Actual Moving q={q_to_move} ({moved_flops_actual:.2f} FLOPs) to satisfy need.")

            new_head_dict = {}
            new_tail_dict = {}
            new_head_dict['q'] = q_to_move
            new_head_dict['kv'] = self.head['kv'] + new_head_dict['q']

            new_tail_dict['q'] = q_to_move
            new_tail_dict['kv'] = self.tail['kv'] - self.tail['q']

            rlog(f"    - Created head chunk: q={new_head_dict['q']}, kv={new_head_dict['kv']}, on GPU {remote_gpuid}")
            rlog(f"    - Created tail chunk: q={new_tail_dict['q']}, kv={new_tail_dict['kv']}, on GPU {remote_gpuid}")

            newly_split_item = Item(
                self.model_config,
                self.seq_len, self.seqid, remote_gpuid, self.src_gpuid, 
                new_head_dict, new_tail_dict,
                is_original=False
            )
            assert newly_split_item.total_flops == moved_flops_actual, "Total moved flops should be equal"
            return newly_split_item, moved_flops_actual

    # Transfer Item to List of dict(s).
    # Complete Item -> List[one dict]
    # Split Item -> List[two dicts]
    def to_dicts(self) -> List[Dict[str, Any]]:
        output_dicts = []
        for item_chunk in self.items:
            if item_chunk['q'] <= 0:
                continue
            item_dict = {
                'q': item_chunk['q'],
                'kv': item_chunk['kv'],
                'gpuid': self.gpuid,
                'seqid': self.seqid,
                'src_gpuid': self.src_gpuid,
                'is_original': self.is_original,
                'shard_id': self.shard_id
            }
            output_dicts.append(item_dict)
        return output_dicts

    def __repr__(self):
        if self.complete:
            return (f"<Item seqid={self.seqid} src_gpuid = {self.src_gpuid} gpuid={self.gpuid} complete="
                    f"q={self.complete_item['q']} kv={self.complete_item['kv']}>")
        else:
            return (f"<Item seqid={self.seqid} src_gpuid = {self.src_gpuid} gpuid={self.gpuid} split="
                    f"head(q={self.head['q']}, kv={self.head['kv']}) "
                    f"tail(q={self.tail['q']}, kv={self.tail['kv']})>")
    
    def __rich_console__(self, console: Console, options) -> rich.console.RenderResult:
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold magenta", justify="right")
        table.add_column()
        status = "[green]Original[/green]" if self.is_original else "[yellow]Replanned[/yellow]"
        table.add_row("Status:", status)
        if self.shard_id is not None:
            table.add_row("Shard ID:", str(self.shard_id))
        table.add_row("Seq Length:", str(self.seq_len))
        table.add_row("Total FLOPs:", f"{self.total_flops:,.2f}")
        table.add_row()
        if self.complete:
            item = self.complete_item
            table.add_row("[dim]Complete:", f"q={item['q']}, kv={item['kv']}")
        else:
            table.add_row("[dim]Head:", f"q={self.head['q']}, kv={self.head['kv']}")
            table.add_row("[dim]Tail:", f"q={self.tail['q']}, kv={self.tail['kv']}")
        yield Panel(
            table,
            title=f"[bold cyan]Item (seqid={self.seqid})[/bold cyan]",
            subtitle=f"GPU: {self.src_gpuid} -> {self.gpuid}",
            border_style="blue"
        )

def group_items_by_seqid(items: list[Item]) -> dict:
    doc_info = defaultdict(lambda: {'items': [], 'doc_len': -1, 'cp_group_index': -1})
    
    for item in items:
        if not doc_info[item.seqid]['items']:
            doc_info[item.seqid]['doc_len'] = item.seq_len
        
        doc_info[item.seqid]['items'].append(item)
        
    return doc_info


# {doc_id: {'items': [Item], 'doc_len':4096 , 'cp_group_index': 0}}
# cost = alpha * l(l-1)/2 / k + beta * (k-1)/k * l, l is doc_len, k is cp_degree.
def flex_sp(doc_info: dict, total_cp_degree: int, alpha: float = 1.0, beta: float = 1.0) -> tuple[list[list[int]], dict]:
    cp_groups = []
    import pulp
    group_sizes = [1]
    while group_sizes[-1] * 2 <= total_cp_degree:
        group_sizes.append(group_sizes[-1] * 2)
    max_group_sizes = {k: total_cp_degree // k for k in group_sizes}
    prob = pulp.LpProblem("DocAssignment", pulp.LpMinimize)
    C_max = pulp.LpVariable("C_max", lowBound=0, cat='Continuous')
    x = pulp.LpVariable.dicts(
        "AssignDoc",
        ((i, j, k) for i in doc_info for k in group_sizes for j in range(max_group_sizes[k])),
        cat='Binary',
    )
    y = pulp.LpVariable.dicts(
        "GroupUsed",
        ((j, k) for k in group_sizes for j in range(max_group_sizes[k])),
        cat='Binary',
    )
    prob += C_max
    prob += pulp.lpSum(k * y[j, k] for k in group_sizes for j in range(max_group_sizes[k])) == total_cp_degree
    for i in doc_info:
        prob += pulp.lpSum(x[i, j, k] for k in group_sizes for j in range(max_group_sizes[k])) == 1
    for k in group_sizes:
        for j in range(max_group_sizes[k]):
            prob += pulp.lpSum(alpha / 2 * (doc_info[i]['doc_len'] + 1) * doc_info[i]['doc_len'] / k * x[i, j, k] + beta * (k - 1) / k * doc_info[i]['doc_len'] for i in doc_info) <= C_max
            for i in doc_info:
                prob += x[i, j, k] <= y[j, k]
    prob.solve(solver=pulp.PULP_CBC_CMD(msg=False, timeLimit=60))

    cp_groups = []
    current = 0
    for k in reversed(group_sizes):
        for j in range(max_group_sizes[k]):
            if pulp.value(y[j, k]) > 0.5:
                cp_groups.append(list(range(current, current + k)))
                current += k
            for i in doc_info:
                if pulp.value(x[i, j, k]) > 0.5:
                    doc_info[i]['cp_group_index'] = len(cp_groups) - 1

    return cp_groups, doc_info


if __name__ == "__main__":
    docs = {}
    for i in range(36):
        docs[str(i)] = {'doc_len': int(math.floor(random.expovariate() % 10 / 10 * 65536)) + 1}
    cp_groups, doc_info = flex_sp(docs, 20)
    for i, group in enumerate(cp_groups):
        print(f"Group {i} (size {len(group)}): {[doc_info[d]['doc_len'] for d in doc_info if doc_info[d]['cp_group_index'] == i]}")


class Planner:
    def __init__(self,
                world_size: int,
                parallel_config,
                tolerance_factor: float = 0.1,
                model_config = None,
                dtype: torch.dtype = torch.bfloat16,
                planner_type: str = "greedy") -> None:
        self.model_config = model_config
        self.world_size = world_size
        self.parallel_config = parallel_config
        self.data_parallel = world_size // (parallel_config.pipeline_model_parallel_size * parallel_config.tensor_model_parallel_size)
        self.attention_server_world_size = self.data_parallel * parallel_config.pipeline_model_parallel_size
        self.dtype = dtype
        self.planner_type = planner_type
        self.tolerance_factor = tolerance_factor
        rich.print(f"[bold green] world_size: {self.world_size}, DP: {self.data_parallel}[/bold green], PP: {parallel_config.pipeline_model_parallel_size}, TP: {parallel_config.tensor_model_parallel_size}, attention_server_world_size: {self.attention_server_world_size}")
    # from item to metadata.
    def plan(self, items_: list[Item], verbose=False, plot=False, device="cuda", is_resend_qkv:bool=False):
        mlp_shard_len = self.items_to_mlp_doc_len(items_, device)
        planned_items: list[Item] = self.plan_items(items_, verbose, plot)
        # After ILP planner, source gpu id should equals to the doc's gpuid.
        if self.planner_type == "ilp":
            assert all(item.src_gpuid == item.gpuid for item in planned_items), "After ILP planner, source gpu id should equals to the doc's gpuid. But got {item.src_gpuid} != {item.gpuid} for item {item}."
            mlp_shard_len = self.items_to_mlp_doc_len(planned_items, device)
        planned_items: list[dict] = self.postprocess_items(planned_items) 
        planner_output: list[list[ShardInfo]] = self.items_into_shardinfos(planned_items)
        tp_size = self.parallel_config.tensor_model_parallel_size
        if self.parallel_config.pipeline_model_parallel_size == 1:
            hidden_size_q = self.model_config.hidden_size
            hidden_size_kv = hidden_size_q
            if hasattr(self.model_config, "num_key_value_heads"):
                hidden_size_kv = (hidden_size_kv * self.model_config.num_key_value_heads //
                                self.model_config.num_attention_heads)

            hidden_size_q_tp = hidden_size_q // tp_size
            hidden_size_k_tp = hidden_size_kv // tp_size

            lse_size = self.model_config.num_attention_heads // tp_size
            element_size = self.dtype.itemsize
            # TODO: We should get the transformer config
            if getattr(self.model_config, "attention_softmax_in_fp32", True): # if self.model_config.attention_softmax_in_fp32:
                # lse_size *= torch.float32.element_size // element_size
                lse_size *= torch.float32.itemsize // element_size

            (qkv_fwd_fa2a_metadata, qkv_rev_fa2a_metadata,
            attn_out_fwd_fa2a_metadata, attn_out_rev_fa2a_metadata,
            as_attn_metadata,
            ) = from_planner_output(
                self.attention_server_world_size, planner_output, hidden_size_q_tp, hidden_size_k_tp,
                lse_size, element_size, is_pipeline_tick=False, is_resend_qkv=is_resend_qkv,
            )
            fa2a_metadata = (
                qkv_fwd_fa2a_metadata, qkv_rev_fa2a_metadata,
                attn_out_fwd_fa2a_metadata, attn_out_rev_fa2a_metadata,
            )
            return fa2a_metadata, as_attn_metadata, mlp_shard_len
        else:
            # new metadata computation for pipeline parallel is in test_util. hard to import.
            # Now PP 3D parallel is directly support in: test_megatron_e2e_pipeline.py
            raise NotImplementedError("PP > 1 will be supported very soon.")
    
    # This function will be deprecated. As we don't need logical metadata anymore.
    def plan_to_raw_qkv_dispatch(self, items_: list[Item], verbose=False, plot=False, should_plan = True, return_items = False):
        # no plan for cp debug
        if should_plan == False:
            if verbose:
                rich.print("[bold yellow]Skip planning for CP debug.[/bold yellow]")
            items = deepcopy(items_)
        else:
            if verbose:
                rich.print("[bold green]Start planning.[/bold green]")
            items = self.plan_items(items_, verbose, plot)
        items = self.postprocess_items(items)
        shard_infos = self.items_into_shardinfos(items)
        
        ret = handle_planner_metadata(self.attention_server_world_size, shard_infos)
        if return_items:
            return ret, items
        return ret

    def plan_items(self, items_: list[Item], verbose=False, plot=False) -> list[Item]:
        if self.planner_type == "greedy":
            return self.plan_items_greedy(items_, verbose, plot)
        elif self.planner_type == "ilp":
            return self.plan_items_ilp(items_, verbose, plot)
        else:
            raise ValueError(f"Unknown planner_type: '{self.planner_type}'")

    # This is FlexSP ILP planner.
    # MLP layout is not changed. Only Attention in CP.
    # Documents' MLP GPU id is specified by Item's src_gpuid.
    # Documents' Attention GPU id is specified by Item's gpuid.
    def plan_items_ilp(self, items_: list[Item], verbose=False, plot=False) -> list[Item]:
        items = deepcopy(items_)
        doc_info = group_items_by_seqid(items)

        # For ILP planner, total_cp_degree can be treated as attention_server_world_size.
        # Aftr calling flex_sp functione
        # cp_groups: [[0, 1], [2,3,4,5], [6, 7, 8, 9], [10, 11]]
        # doc_info: {doc_id: {'items': [Item], 'doc_len':4096 , 'cp_group_index': cp_group_index_in_cp_groups}}
        cp_groups, doc_info = flex_sp(doc_info, self.attention_server_world_size)
        rich.print(f"🟡 cp_groups = {cp_groups}")

        # Split items according to flexsp plan.
        # TODO:Support MLP CP Split...
        final_items = []
        
        for doc_id, doc_info in doc_info.items():
            original_item = doc_info['items'][0]
            cp_index = doc_info['cp_group_index']
            cp_degree = len(cp_groups[cp_index])
            rich.print(f"🟡 item: {original_item}, cp_index: {cp_index}, cp_degree: {cp_degree}")
            total_flops = original_item.total_flops
            assert total_flops % cp_degree == 0, "Total flops should be divisible by cp_degree."
            flops_per_cp = total_flops / cp_degree

            split_items = doc_info['items']
            for _ in range(cp_degree-1):
                split_item, _ = original_item.split_item(flops_per_cp, -1, verbose=False)
                split_items.append(split_item)

            assert len(split_items) == cp_degree
            assert sum(item.total_flops for item in split_items) == total_flops
            assert all(item.total_flops == flops_per_cp for item in split_items), "Each split item should have the same total flops."

            cp_group = cp_groups[cp_index]
            for i, item in enumerate(split_items):
                item.gpuid = cp_group[i]
                item.src_gpuid = cp_group[i]    # in flex sp, mlp layout is similar to attention layout.
                final_items.append(item)
        rich.print(f"🟡 final_items = {final_items}")
        return final_items
    
    def plan_items_greedy(self, items_: list[Item], verbose=False, plot=False) -> list[Item]:
        items = deepcopy(items_)

        def rlog(message):
            if verbose:
                rich.print(message)
        
        def format_flops(flops):
            """Convert FLOPs to human-readable format (K, M, G, T, P)"""
            if flops == 0:
                return "0"
            
            abs_flops = abs(flops)
            if abs_flops >= 1e15:
                return f"{flops/1e15:.2f}P"
            elif abs_flops >= 1e12:
                return f"{flops/1e12:.2f}T"
            elif abs_flops >= 1e9:
                return f"{flops/1e9:.2f}G"
            elif abs_flops >= 1e6:
                return f"{flops/1e6:.2f}M"
            elif abs_flops >= 1e3:
                return f"{flops/1e3:.2f}K"
            else:
                return f"{flops:.2f}"

        flops_per_gpu = [0.0] * self.attention_server_world_size
        for item in items:
            flops_per_gpu[item.gpuid] += item.total_flops
        total_flops = sum(flops_per_gpu)
        
        avg_flops_per_gpu = total_flops / self.attention_server_world_size
        rlog(f"Total FLOPs: {format_flops(total_flops)}, Average FLOPs per GPU: {format_flops(avg_flops_per_gpu)}")
        
        surplus_deficit = [f - avg_flops_per_gpu for f in flops_per_gpu]
        threshold_flops = avg_flops_per_gpu * self.tolerance_factor

        recipients = sorted(
            [(i, deficit) for i, deficit in enumerate(surplus_deficit) if deficit < 0],
            key=lambda x: x[1]
        )
        rlog(f"Threshold FLOPs for moving: {format_flops(threshold_flops)}")
        for recipient_id, deficit in recipients:
            needed_flops = -deficit
            rlog(f"\n[bold yellow]Planning for GPU {recipient_id}[/bold yellow] (needs {format_flops(needed_flops)} FLOPs)")
            
            while needed_flops > threshold_flops:
                donor_gpus = {i for i, s in enumerate(surplus_deficit) if s > 0}
                if not donor_gpus:
                    rlog("[red]No more donor GPUs with surplus FLOPs. Stopping.[/red]")
                    break

                candidates = []
                for item in items:
                    if item.gpuid in donor_gpus:
                        donor_id = item.gpuid
                        if item.total_flops <= 0:
                            continue
                        priority = item.get_priority(surplus_deficit[donor_id], needed_flops)
                        max_flops_to_move = min(needed_flops, item.total_flops, surplus_deficit[donor_id])
                        
                        candidates.append({
                            'priority': priority,
                            'item': item,
                            'max_flops': max_flops_to_move
                        })
                
                if not candidates:
                    rlog("[yellow]No more candidate items to move. Stopping for this recipient.[/yellow]")
                    break
                
                candidates.sort(key=lambda x: x['priority'])
                best_candidate = candidates[0]
                item_to_move = best_candidate['item']
                donor_id = item_to_move.gpuid
                max_flops_to_move = best_candidate['max_flops']
                
                rlog(f"  - Candidate: {item_to_move} with priority {best_candidate['priority']:.4f}")
                rlog(f"    - Provides: {format_flops(item_to_move.total_flops)} FLOPs, Max possible to move: {format_flops(max_flops_to_move)}")

                newly_split_item, moved_flops_actual = item_to_move.split_item(max_flops_to_move, recipient_id, verbose=verbose)

                if moved_flops_actual > 0:
                    if newly_split_item:
                        items.append(newly_split_item)
                    
                    surplus_deficit[donor_id] -= moved_flops_actual
                    surplus_deficit[recipient_id] += moved_flops_actual
                    needed_flops -= moved_flops_actual
                else:
                    rlog(f"[yellow]Could not move any FLOPs from candidate {item_to_move}. Stopping for this recipient.[/yellow]")
                    break

        final_items = [item for item in items if item.total_flops > 0]

        if verbose:
            final_flops_per_gpu = [0.0] * self.attention_server_world_size
            for item in final_items:
                final_flops_per_gpu[item.gpuid] += item.get_flops()
            
            rlog("Final FLOPs distribution per GPU:")
            for i, f in enumerate(final_flops_per_gpu):
                rlog(f"  - GPU {i}: {format_flops(f)} FLOPs (Target: {format_flops(avg_flops_per_gpu)})")
        rlog("\n[bold green]Relocation planning finished.[/bold green]")
        return final_items
    

    def items_to_shardinfo(self, items_: list[Item], verbose=False) -> list[list[ShardInfo]]:
        planned_items = self.plan_items(items_, verbose)
        planned_items = self.postprocess_items(planned_items)
        shard_infos = self.items_into_shardinfos(planned_items)
        return shard_infos
    
    def postprocess_items(self, items: list[Item]) -> list[dict]:
        dict_items = []
        for item in items:
            dict_items.extend(item.to_dicts())
        
        src_gpuid_seqid_to_items = defaultdict(list)
        for item in dict_items:
            src_gpuid_seqid_to_items[item['seqid']].append(item)
        
        for _, items_in_seq in src_gpuid_seqid_to_items.items():
            items_in_seq.sort(key=lambda x: x['kv'])
            for i, item in enumerate(items_in_seq):
                item['shard_id'] = i
                
        return dict_items
    
    # Get mlp_shard_len from items. May need to change later. 
    # Currently, shards are put on MLP based on seq_id from small to big.
    def items_to_mlp_doc_len(self, items: list[Item], device: str = 'cuda') -> torch.Tensor:
        items_by_src_gpu = defaultdict(list)
        for item in items:
            items_by_src_gpu[item.src_gpuid].append(item)

        for src_gpuid in items_by_src_gpu:
            items_by_src_gpu[src_gpuid].sort(key=lambda x: x.seqid)

        final_shards_by_rank = [[] for _ in range(self.attention_server_world_size)]

        sorted_src_gpuids = sorted(items_by_src_gpu.keys())

        for src_gpuid in sorted_src_gpuids:
            sorted_items = items_by_src_gpu[src_gpuid]
            for item in sorted_items:
                if item.complete:
                    shard_len = item.complete_item['q'] # item.complete_item['q'] = item.seq_len
                    final_shards_by_rank[src_gpuid].append(shard_len)
                else:
                    head_shard_len = item.head['q']
                    final_shards_by_rank[src_gpuid].append(head_shard_len)

                    tail_shard_len = item.tail['q']
                    final_shards_by_rank[src_gpuid].append(tail_shard_len)

        doc_lens_per_rank = [
            torch.tensor(shard_len_list, dtype=torch.int32, device=device) for shard_len_list in final_shards_by_rank
            ]
        return doc_lens_per_rank



    def items_into_shardinfos(self, item_dicts):
        return items_into_shardinfos(item_dicts)
    
    @classmethod
    def from_individual_params(
        cls,
        tp_size: int,
        pp_size: int,
        dp_size: int,
        world_size: int,
        hidden_size_q: int,
        hidden_size_k: int,
    ):
        parallel_config = SimpleNamespace(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size
        )
        model_config = SimpleNamespace(
            hidden_size=hidden_size_q * tp_size,
            num_attention_heads = 1,
            num_key_value_heads = hidden_size_q // hidden_size_k,
            num_hidden_layers = 1
        )
        return cls(
            world_size=world_size,
            parallel_config=parallel_config,
            model_config=model_config)

    
def get_flops(q=None, kv=None, **kwargs):
    assert q is not None and kv is not None, "q and kv must be provided"
    return sum(kv - i for i in range(q))


def plot_flops(items, plan_flops_per_gpu, title=None):
    fixed_flops_per_gpu = [0] * len(plan_flops_per_gpu)
    for item in items:
        fixed_flops_per_gpu[item["gpuid"]] += get_flops(**item)
    
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    x = range(len(plan_flops_per_gpu))
    ax.bar(x, fixed_flops_per_gpu, label="Fixed", color="orange")
    ax.bar(x, plan_flops_per_gpu, label="Plan", color="blue", bottom=fixed_flops_per_gpu)
    if title is not None:
        ax.set_title(title)
    ax.legend()
    plt.show()
    return
