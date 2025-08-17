from collections import defaultdict
from copy import deepcopy

import rich

from d2.runtime.shard_info import items_into_shardinfos, plan_to_metadata

K = 1024

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

    def plan(self, items_, verbose=False, plot=False):
        """
        Plan relocation of sequences across GPUs to balance FLOPs.
        """        
        items = self.plan_items(items_, verbose, plot)
        items = self.postprocess_items(items)
        return self.item_to_metadata(items)

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
        self.world_size, shard_infos, return_intermediate=True,
        return_mlp_no_shard_seq_lens=True)
        return metadatas

    def items_into_shardinfos(items, verbose=False):
        """
        Convert the items to intermediate tensors for metadata generation.
        """
        
        return items_into_shardinfos(items, verbose=verbose)
    
    
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
        
    pass
