from collections import defaultdict
from copy import deepcopy
from typing import List

import rich
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from d2.runtime.shard_info import items_into_shardinfos, plan_to_metadata

K = 1024


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


def batch_to_items_class(batches, model_config=None):
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


"""
Partition and place a batch of variable-length documents onto GPU ranks under a
hybrid Data-Parallel (DP) and Context-Parallel (CP) policy, returning a list of
Item objects.

Steps
1. Schedule each document to the next available rank in original order (DP).
2. If a document exceeds a rank's token budget, split it into head / tail
   chunks and keep placing until the entire document is covered (CP).

Args
  batches : List[List[int]]
      Outer list = per-sequence groups, inner list = document lengths in tokens.
  num_batched_token : int
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
    num_batched_token = 16K
    DP_degree         = 4

    Item0  q=16K kv=16K  gpu=0 seq=0 src=0 is_orig=True
    Item1  q= 8K kv= 8K  gpu=1 seq=1 src=1 is_orig=True
    ...
Example 2 : CP split required
    batch           = [[32K], [8K, 8K], [4K]*4]
    num_batched_token = 16K
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
def batch_to_items_general(batches, num_batched_token, DP_degree, model_config):
    """
    Put a batch of documents onto GPU ranks and return a list of Item objects.
    """
    items = []
    seqid = 0
    rank_budgets = [num_batched_token] * DP_degree
    current_rank_idx = 0

    all_docs = []
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




# Represent a part of sequence.
# item_dicts: {'q': ... , 'kv': ..}
class Item:
    def __init__(self, 
                model_config,
                seq_len, seqid, gpuid, src_gpuid,
                *item_dicts, 
                is_original = True):
        self.model_config = model_config
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
            q = item['q']
            kv = item['kv']
            flops += sum(kv - i for i in range(q))
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
            assert self.total_flops == self.get_flops() + moved_flops_actual, f"Total flops should be equal"
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

    def to_dicts(self):
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

class Planner_DP:
    def __init__(self,
                world_size: int,
                parallel_config,
                tolerance_factor: float = 0.1,
                model_config = None) -> None:
        self.model_config = model_config
        self.world_size = world_size
        self.parallel_config = parallel_config
        self.data_parallel = world_size // (parallel_config.pipeline_model_parallel_size * parallel_config.tensor_model_parallel_size)
        self.num_dispatch_instances = self.data_parallel * parallel_config.pipeline_model_parallel_size

        self.tolerance_factor = tolerance_factor

    def plan(self, items_, verbose=False, plot=False):
        """
        Plan relocation of sequences across GPUs to balance FLOPs.
        """        
        items = self.plan_items(items_, verbose, plot)
        items = self.postprocess_items(items)
        metadata = self.item_to_metadata(items)
        return metadata

    def plan_items(self, items_, verbose=False, plot=False) -> list[dict]:
        """
        Plan relocation of sequences across GPUs to balance FLOPs.
        
        Args:
            items_: List of item dictionaries
            verbose: Whether to print verbose output
            
        Returns:
            List of item dictionaries after relocation planning
        """
        items = deepcopy(items_)

        def rlog(message):
            if verbose:
                rich.print(message)

        # Get total flops, and avg flops per GPU
        flops_per_gpu = [0.0] * self.num_dispatch_instances
        for item in items:
            flops_per_gpu[item['gpuid']] += get_flops(**item)
        total_flops = sum(flops_per_gpu)
        
        assert self.num_dispatch_instances > 0, "No worker to dispatch to."
        avg_flops_per_gpu = total_flops / self.num_dispatch_instances
        rlog(f"Total FLOPs: {total_flops:.2f}, Average FLOPs per GPU: {avg_flops_per_gpu:.2f}")
        
        surplus_deficit = [f - avg_flops_per_gpu for f in flops_per_gpu]

        recipients = sorted(
            [(i, deficit) for i, deficit in enumerate(surplus_deficit) if deficit < 0],
            key=lambda x: x[1]
        )
        rlog("\n[bold cyan]Balancing Plan[/bold cyan]")
        rlog(f"Average FLOPs Target: {avg_flops_per_gpu:.2f}")
        for gpu_id, deficit in recipients:
            rlog(f"  - GPU {gpu_id} needs {-deficit:.2f} FLOPs.")

        threshold_flops = avg_flops_per_gpu * self.tolerance_factor
        rlog(f"\n[bold cyan]Threshold for moving FLOPs: {threshold_flops:.2f}[/bold cyan]")
        
        for recipient_id, deficit in recipients:
            needed_flops = -deficit
            rlog(f"\n[bold yellow]Planning for GPU {recipient_id}[/bold yellow] (needs {needed_flops:.2f} FLOPs)")
            
            while abs(needed_flops) > threshold_flops:
                
                donor_gpus = {i for i, s in enumerate(surplus_deficit) if s > 0}
                if not donor_gpus:
                    rlog("[red]No more donor GPUs with surplus FLOPs. Stopping.[/red]")
                    break

                candidates = []
                for item in items:
                    if item['gpuid'] in donor_gpus:
                        seq_len = item['seq_len']
                        item_flops = get_flops(**item)

                        # Maximum amount of FLOPs could move for current item
                        max_flops_to_move = min(needed_flops, item_flops, surplus_deficit[item['gpuid']])

                        communication_cost = (max_flops_to_move / seq_len) + (2 * seq_len)
                        priority = communication_cost / max_flops_to_move

                        candidates.append({
                            'priority': priority,
                            'item': item,
                            'donor_id': item['gpuid'],
                            'max_flops': max_flops_to_move
                        })
                
                if not candidates:
                    rlog("[yellow]No more candidate items to move. Stopping for this recipient.[/yellow]")
                    break
                
                candidates.sort(key=lambda x: x['priority'])

                moved_something = False
                for best_candidate in candidates:
                    item_to_move = best_candidate['item']
                    donor_id = best_candidate['donor_id']
        
                    max_flops_to_move = best_candidate['max_flops']
                    item_total_flops = get_flops(**item_to_move)

                    rlog(f"  - Candidate: item (q={item_to_move['q']}, kv={item_to_move.get('kv')}, on_gpu={donor_id}) with priority {best_candidate['priority']:.4f}")
                    rlog(f"    - Provides: {item_total_flops:.2f} FLOPs, Max possible: {max_flops_to_move:.2f}, Recipient needs: {needed_flops:.2f} FLOPs, Difference: {max_flops_to_move - needed_flops:.2f} FLOPs")
                
                    # 3. If moving almost the entire item, just move it all
                    if item_total_flops <= max_flops_to_move:
                        rlog(f"    - [bold]Moving entire item[/bold] as its FLOPs ({max_flops_to_move:.2f}) are less than needed ({needed_flops:.2f}).")
                        
                        surplus_deficit[donor_id] -= max_flops_to_move
                        surplus_deficit[recipient_id] += max_flops_to_move
                        needed_flops -= max_flops_to_move
                        item_to_move['gpuid'] = recipient_id
                        item_to_move['is_original'] = False
                    else:
                        flops_per_q = item_to_move['seq_len'] + 1
                        q_to_move = int(max_flops_to_move / flops_per_q)

                        if q_to_move <= 0:
                            continue

                        moved_flops_actual = q_to_move * flops_per_q
                        original_q = item_to_move['q']
                        original_kv = item_to_move['kv']
                        rlog(f"    - [bold]Splitting item[/bold]: Actual Moving q={q_to_move*2} ({moved_flops_actual:.2f} FLOPs) to satisfy need.")

                        head_chunk = deepcopy(item_to_move)
                        head_chunk.update({'kv': original_kv - original_q + q_to_move, 'q': q_to_move, 'gpuid': recipient_id, 'is_original': False})
                        
                        tail_chunk = deepcopy(item_to_move)
                        tail_chunk.update({'kv': original_kv, 'q': q_to_move, 'gpuid': recipient_id, 'is_original': False})
                        rlog(f"    - Created head chunk: q={head_chunk['q']}, kv={head_chunk['kv']}, on GPU {recipient_id}")
                        rlog(f"    - Created tail chunk: q={tail_chunk['q']}, kv={tail_chunk['kv']}, on GPU {recipient_id}")
                        items.extend([head_chunk, tail_chunk])

                        item_to_move['q'] = original_q - (2 * q_to_move)
                        item_to_move['kv'] = original_kv - q_to_move
                        item_to_move['is_original'] = False

                        surplus_deficit[donor_id] -= moved_flops_actual
                        surplus_deficit[recipient_id] += moved_flops_actual
                        needed_flops -= moved_flops_actual
                        
                        rlog(f"    - [bold]Splitting item[/bold]: moved {moved_flops_actual:.2f} FLOPs (q={q_to_move*2}) to GPU {recipient_id}. Remaining q={item_to_move['q']} on GPU {donor_id}")
                                
                    moved_something = True
                    break

                if not moved_something:
                    rlog(f"[yellow]Could not find a suitable item to move for GPU {recipient_id}. Remaining need: {needed_flops:.2f}[/yellow]")
                    break
        
        final_items = [item for item in items if item['q'] > 0]
        post_processed_items = []
        for item in final_items:
            # Split dispatched sequences to two chunks.
            if item['is_original'] == False and item['gpuid'] == item['src_gpuid']:
                rlog(f"  - Found item to split on GPU {item['gpuid']}: q={item['q']}, kv={item['kv']}")
                
                half_q = item['q'] // 2
                
                head_chunk = deepcopy(item)
                head_chunk['q'] = half_q
                head_chunk['kv'] = item['kv'] - item['q'] + half_q
                
                if item['q'] % 2 != 0:
                    tail_chunk = deepcopy(item)
                    tail_chunk['q'] = half_q+1
                else:
                    tail_chunk = deepcopy(item)
                    tail_chunk['q'] = half_q
                    
                post_processed_items.extend([head_chunk, tail_chunk])
                rlog(f"    - [bold]Split into two chunks[/bold]:")
                rlog(f"      - Head: q={head_chunk['q']}, kv={head_chunk['kv']}")
                rlog(f"      - Tail: q={tail_chunk['q']}, kv={tail_chunk['kv']}")

            else:
                post_processed_items.append(item)
        final_items = post_processed_items
        rlog("\n[bold green]Relocation planning finished.[/bold green]")
        
        final_flops_per_gpu = [0.0] * self.num_dispatch_instances
        for item in final_items:
            final_flops_per_gpu[item['gpuid']] += get_flops(**item)
        
        rlog("Final FLOPs distribution per GPU:")
        for i, f in enumerate(final_flops_per_gpu):
            rlog(f"  - GPU {i}: {f:.2f} FLOPs (Target: {avg_flops_per_gpu:.2f})")

        return final_items    
    
    def postprocess_items(self, items) -> list[dict]:
        """
        Postprocess the items to add a "shard_id" field.
        The "shard_id" field is always 0 for the original sequence.
        For each non-original sequence, shard_id = how short the `kv` is among all the shards in the same sequence (ranking of `kv` sort ASC)
        - collect all the sequences that has the same `src_gpuid` and `seqid`
        - sort them by the `kv` to determine the shard id of that sequence.
        """
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
    
    def item_to_metadata(self, items):
        """
        Convert items to metadata objects.
        
        Args:
            items: List of item dictionaries
            hidden_size_q: Hidden size for query
            hidden_size_k: Hidden size for key/value
            element_size: Element size in bytes
            
        Returns:
            Metadata object for fast all-to-all communication
        """
        
        shard_infos = self.items_into_shardinfos(items)
        metadatas = plan_to_metadata(
        self.world_size, shard_infos, return_intermediate=True)
        return metadatas

    def items_into_shardinfos(self, items):
        """
        Convert the items to intermediate tensors for metadata generation.
        """
        
        return items_into_shardinfos(items)
    


class Planner:
    def __init__(self,
                world_size: int,
                parallel_config,
                tolerance_factor: float = 0.1,
                model_config = None) -> None:
        self.model_config = model_config
        self.world_size = world_size
        self.parallel_config = parallel_config
        self.data_parallel = world_size // (parallel_config.pipeline_model_parallel_size * parallel_config.tensor_model_parallel_size)
        self.num_dispatch_instances = self.data_parallel * parallel_config.pipeline_model_parallel_size
        self.tolerance_factor = tolerance_factor

    def plan(self, items_: list[Item], verbose=False, plot=False):
        items = self.plan_items(items_, verbose, plot)
        items = self.postprocess_items(items)

        metadata = self.item_to_metadata(items)
        return metadata
    
    def plan_to_raw_qkv_dispatch(self, items_: list[Item], verbose=False, plot=False):
        items = self.plan_items(items_, verbose, plot)
        items = self.postprocess_items(items)
        shard_infos = self.items_into_shardinfos(items)
        from d2.runtime.shard_info import handle_planner_metadata
        
        # (
        # mlp_num_seqs,
        # mlp_q_dispatch,
        # mlp_seq_lens,
        # kv_to_q_mapping,
        # kv_to_q_rank,
        # kv_context_size,
        # q_to_num_kv_seq,
        # q_to_num_kv_tokens,
        # ) 
        ret = handle_planner_metadata(self.num_dispatch_instances, shard_infos)
        return ret
    
    def plan_items(self, items_: list[Item], verbose=False, plot=False) -> list[Item]:
        items = deepcopy(items_)

        def rlog(message):
            if verbose:
                rich.print(message)

        flops_per_gpu = [0.0] * self.num_dispatch_instances
        for item in items:
            flops_per_gpu[item.gpuid] += item.total_flops
        total_flops = sum(flops_per_gpu)
        
        if self.num_dispatch_instances == 0:
            return []
        avg_flops_per_gpu = total_flops / self.num_dispatch_instances
        rlog(f"Total FLOPs: {total_flops:.2f}, Average FLOPs per GPU: {avg_flops_per_gpu:.2f}")
        
        surplus_deficit = [f - avg_flops_per_gpu for f in flops_per_gpu]
        threshold_flops = avg_flops_per_gpu * self.tolerance_factor

        recipients = sorted(
            [(i, deficit) for i, deficit in enumerate(surplus_deficit) if deficit < 0],
            key=lambda x: x[1]
        )
        rlog(f"Threshold FLOPs for moving: {threshold_flops:.2f}")
        for recipient_id, deficit in recipients:
            needed_flops = -deficit
            rlog(f"\n[bold yellow]Planning for GPU {recipient_id}[/bold yellow] (needs {needed_flops:.2f} FLOPs)")
            
            while needed_flops > threshold_flops:
                donor_gpus = {i for i, s in enumerate(surplus_deficit) if s > 0}
                if not donor_gpus:
                    rlog("[red]No more donor GPUs with surplus FLOPs. Stopping.[/red]")
                    break

                candidates = []
                for item in items:
                    if item.gpuid in donor_gpus:
                        donor_id = item.gpuid
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
                rlog(f"    - Provides: {item_to_move.total_flops:.2f} FLOPs, Max possible to move: {max_flops_to_move:.2f}")

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
            final_flops_per_gpu = [0.0] * self.num_dispatch_instances
            for item in final_items:
                final_flops_per_gpu[item.gpuid] += item.get_flops()
            
            rlog("Final FLOPs distribution per GPU:")
            for i, f in enumerate(final_flops_per_gpu):
                rlog(f"  - GPU {i}: {f:.2f} FLOPs (Target: {avg_flops_per_gpu:.2f})")
        rlog("\n[bold green]Relocation planning finished.[/bold green]")
        return final_items
    
    def postprocess_items(self, items: list[Item]) -> list[Item]:
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
    
    def item_to_metadata(self, items: list[dict]):
        shard_infos = self.items_into_shardinfos(items)
        metadatas = plan_to_metadata(
            self.num_dispatch_instances, shard_infos, return_intermediate=True
        )
        return metadatas

    def items_into_shardinfos(self, item_dicts):
        return items_into_shardinfos(item_dicts)

    
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
