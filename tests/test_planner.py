import collections
import random
import time

import rich
from rich.console import Console
from rich.table import Table
from test_util import ParallelConfig
from transformers import AutoConfig

from d2.planner.equal_flops import batch_to_items
from d2.planner.planner import Planner, get_flops

console = Console()


K = 1024

def verification_layout(originals, replans):
    """
    Verify whether replanned items meet expectations according to the specified algorithm.
    """
    console.print("[bold cyan]Running Replanning Verification Test...[/bold cyan]\n")

    # Build a mapping from (gpuid, seqid) to the original item for easy lookup
    original_map = {(item['gpuid'], item['seqid']): item for item in originals}

    # Group all replanned items by (src_gpuid, seqid)
    grouped_replans = collections.defaultdict(list)
    for item in replans:
        grouped_replans[(item['src_gpuid'], item['seqid'])].append(item)

    # Final test result flag
    overall_result = True

    # Iterate over each original sequence that appears in the replan results
    for (src_gpuid, seqid), items in grouped_replans.items():
        # According to the algorithm, filter out offloaded items (`is_original`: False and gpuid != src_gpuid)
        offloaded_items = [
            item for item in items if not item.get('is_original')
        ]
        
        # If a sequence has no offloaded parts, skip its verification
        if not offloaded_items:
            continue

        console.print(f"[bold]----- Verifying Sequence (src_gpuid={src_gpuid}, seqid={seqid}) -----[/bold]")
        
        # Find the corresponding original item
        original_item = original_map.get((src_gpuid, seqid))
        if not original_item:
            console.print(f"[bold red][FAIL][/bold red] Cannot find original item for (src_gpuid={src_gpuid}, seqid={seqid})")
            overall_result = False
            continue
        
        original_q = original_item['q']
        console.print(f"Original 'q': [bold yellow]{original_q}[/bold yellow]")
        
        # --- Check 1: Whether the sum of q equals the original q ---
        console.print("\n[bold]Check 1: Verifying 'q' sum conservation...[/bold]")
        replan_q_sum = sum(item['q'] for item in items)
        console.print(f"Sum of 'q' in all replanned parts for this sequence: [bold yellow]{replan_q_sum}[/bold yellow]")
        
        if replan_q_sum == original_q:
            console.print("[bold green][PASS][/bold green] 'q' sum is conserved.")
        else:
            console.print(f"[bold red][FAIL][/bold red] 'q' sum mismatch! Expected: {original_q}, Got: {replan_q_sum}")
            overall_result = False
            console.print("----- Verification Finished for this Sequence -----\n", style="bold")
            continue # If the q sum is incorrect, further checks are meaningless

        # --- Check 2: Whether the difference between adjacent kvs equals the q of the next item ---
        console.print("\n[bold]Check 2: Verifying 'kv' difference rule for offloaded items...[/bold]")
        
        # Sort the offloaded items by kv in ascending order
        sorted_offloaded = sorted(offloaded_items, key=lambda x: x['kv'])
        
        table = Table(title="Offloaded Items Sorted by 'kv'")
        table.add_column("Index", style="cyan")
        table.add_column("q", style="magenta")
        table.add_column("kv", style="green")
        table.add_column("gpuid", style="yellow")
        for i, item in enumerate(sorted_offloaded):
            table.add_row(str(i), str(item['q']), str(item['kv']), str(item['gpuid']))
        console.print(table)
        
        kv_check_passed = True
        if len(sorted_offloaded) > 1:
            for i in range(1, len(sorted_offloaded)):
                kv_prev = sorted_offloaded[i-1]['kv']
                kv_curr = sorted_offloaded[i]['kv']
                q_curr = sorted_offloaded[i]['q']
                kv_diff = kv_curr - kv_prev
                
                console.print(f"  - Checking item {i}: kv_diff ({kv_curr} - {kv_prev}) = [bold yellow]{kv_diff}[/bold yellow]. Comparing with current q: [bold yellow]{q_curr}[/bold yellow]")
                if kv_diff != q_curr:
                    console.print(f"    [bold red][FAIL][/bold red] Rule violated: kv_diff ({kv_diff}) != q ({q_curr})")
                    kv_check_passed = False
                    overall_result = False
                    #break # Once the rule is violated, stop checking this sequence
        
        if kv_check_passed:
             console.print("[bold green][PASS][/bold green] 'kv' difference rule holds for all adjacent pairs.")
        
        console.print("----- Verification Finished for this Sequence -----\n", style="bold")

    return overall_result


def run_flops_balance_test(originals, replans, tolerance):
    """
    Verify whether FLOPs are conserved and whether the load is balanced after replanning.
    """
    console = Console()
    console.print("[bold cyan]Running FLOPs Conservation and Load Balance Test...[/bold cyan]\n")

    # --- Check 1: The total FLOPs of original items and planned items are equal ---
    console.print("[bold]Check 1: Verifying Total FLOPs Conservation...[/bold]")
    total_original_flops = sum(get_flops(**item) for item in originals)
    total_replanned_flops = sum(get_flops(**item) for item in replans)

    console.print(f"Total Original FLOPs:  [yellow]{total_original_flops:,}[/yellow]")
    console.print(f"Total Replanned FLOPs: [yellow]{total_replanned_flops:,}[/yellow]")

    conservation_passed = (total_original_flops == total_replanned_flops)
    if conservation_passed:
        console.print("[bold green][PASS][/bold green] Total FLOPs are conserved.\n")
    else:
        console.print("[bold red][FAIL][/bold red] Total FLOPs do not match!\n")

    # --- Check 2: The load of planned items is balanced ---
    console.print(f"[bold]Check 2: Verifying Load Balancing (Tolerance = {tolerance * 100}%) ...[/bold]")

    # Group by gpuid and calculate FLOPs for each GPU
    gpu_flops = collections.defaultdict(int)
    gpu_ids = set()
    for item in replans:
        gpu_id = item['gpuid']
        gpu_flops[gpu_id] += get_flops(**item)
        gpu_ids.add(gpu_id)

    world_size = len(gpu_ids)
    if world_size == 0:
        console.print("[bold red]No GPUs found in replanned items. Cannot perform balance check.[/bold red]")
        return False

    avg_flops = total_replanned_flops / world_size
    lower_bound = avg_flops * (1 - tolerance)
    upper_bound = avg_flops * (1 + tolerance)

    console.print(f"Total GPUs (World Size): [bold blue]{world_size}[/bold blue]")
    console.print(f"Average FLOPs per GPU: [bold blue]{avg_flops:,.2f}[/bold blue]")
    console.print(f"Acceptable Range:      [bold blue][{lower_bound:,.2f}, {upper_bound:,.2f}][/bold blue]")

    # Create a table to display the results
    table = Table(title="GPU Load Balancing Analysis")
    table.add_column("GPU ID", style="cyan")
    table.add_column("Total FLOPs", style="magenta", justify="right")
    table.add_column("Deviation from Avg.", style="yellow", justify="right")
    table.add_column("Status", style="bold", justify="center")

    balancing_passed = True
    for gpu_id in sorted(gpu_flops.keys()):
        flops = gpu_flops[gpu_id]
        deviation = (flops - avg_flops) / avg_flops * 100
        if lower_bound <= flops <= upper_bound:
            status = "[green]PASS[/green]"
        else:
            status = "[red]FAIL[/red]"
            balancing_passed = False
        
        table.add_row(
            str(gpu_id),
            f"{flops:,}",
            f"{deviation:+.2f}%",
            status
        )
    
    console.print(table)
    if not balancing_passed:
         console.print("[bold red][FAIL][/bold red] At least one GPU is outside the tolerance range.\n")
    else:
        console.print("[bold green][PASS][/bold green] All GPUs are within the tolerance range.\n")


    return conservation_passed and balancing_passed




def generate_random_split(total_sum: int, num_sequences: int) -> list[int]:
    """
    Generate a list of random integers of specified length whose sum is total_sum.

    Args:
        total_sum (int): The total sum of all generated numbers.
        num_sequences (int): The number of random numbers to generate.

    Returns:
        list[int]: A list of `num_sequences` integers whose sum equals `total_sum`.
    """
    if num_sequences <= 0:
        return []
    if num_sequences == 1:
        return [total_sum]

    # 1. Generate a set of random floats as initial weights
    weights = [random.random() for _ in range(num_sequences)]
    total_weight = sum(weights)
    if total_weight == 0: # Handle rare case where all random numbers are 0
        proportions = [1/num_sequences] * num_sequences
    else:
        proportions = [w / total_weight for w in weights]
    float_values = [p * total_sum for p in proportions]
    int_values = [int(v) for v in float_values]
    remainder_to_distribute = total_sum - sum(int_values)
    remainders = [(i, v - int(v)) for i, v in enumerate(float_values)]
    remainders.sort(key=lambda x: x[1], reverse=True)
    for i in range(remainder_to_distribute):
        original_index = remainders[i][0]
        int_values[original_index] += 1
        
    return int_values

def generate_random_rank_batches(
    num_ranks: int, 
    total_tokens_per_rank: int, 
    max_sequences_per_rank: int
) -> list[list[int]]:
    """
    Generate a random batch for each rank.

    Args:
        num_ranks (int): Total number of ranks.
        total_tokens_per_rank (int): The total number of tokens each rank must have.
        max_sequences_per_rank (int): The maximum number of sequences allowed in each rank.

    Returns:
        list[list[int]]: A list containing num_ranks sublists, each representing a batch for a rank.
    """
    final_batches = []
    for _ in range(num_ranks):
        num_seq_for_this_rank = random.randint(1, max_sequences_per_rank)
        sequences_for_this_rank = generate_random_split(
            total_tokens_per_rank,
            num_seq_for_this_rank
        )
        final_batches.append(sequences_for_this_rank)
        
    return final_batches



def test_planner_equal_flops():
    rich.print("⚪ Testing planner equal flops...")
    num_seq = 4
    num_rank = 4
    batch = generate_random_rank_batches(num_rank, 32*K, num_seq)
    
    # items = batch_to_items([
    #     [16 * K] * 1,
    #     [8 * K] * 2,
    #     [4 * K] * 4,
    #     [2 * K] * 8, 
    # ])
    items = batch_to_items(batch)
    for item in items:
        item['seq_len'] = item['q']

    expected_items = [
        {'q': 16384, 'kv': 16384, 'gpuid': 0, 'seqid': 0, 'src_gpuid': 0, 'is_original': True},
        {'q': 8192, 'kv': 8192, 'gpuid': 1, 'seqid': 0, 'src_gpuid': 1, 'is_original': True},
        {'q': 8192, 'kv': 8192, 'gpuid': 1, 'seqid': 1, 'src_gpuid': 1, 'is_original': True},
        {'q': 4096, 'kv': 4096, 'gpuid': 2, 'seqid': 0, 'src_gpuid': 2, 'is_original': True},
        {'q': 4096, 'kv': 4096, 'gpuid': 2, 'seqid': 1, 'src_gpuid': 2, 'is_original': True},
        {'q': 4096, 'kv': 4096, 'gpuid': 2, 'seqid': 2, 'src_gpuid': 2, 'is_original': True},
        {'q': 4096, 'kv': 4096, 'gpuid': 2, 'seqid': 3, 'src_gpuid': 2, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 0, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 1, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 2, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 3, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 4, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 5, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 6, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 7, 'src_gpuid': 3, 'is_original': True}
    ]
    # for item in expected_items:
    #     assert item in items, f"item = {item} not in items: {expected_items = }\n{items = }"
    

    model_config = AutoConfig.from_pretrained("/mnt/moonfs/public-models-m2/meta-llama/Llama-3.1-8B/")
    parallel_config = ParallelConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    world_size = 6
    tolerance_factor = 0.01
    planner = Planner(world_size=world_size,
                      parallel_config=parallel_config,
                      model_config=model_config,
                      tolerance_factor=tolerance_factor)
    
    start_time = time.time()
    replanned_items = planner.plan_items(items, verbose=True, plot=True)
    end_time = time.time()
    rich.print(f"Time taken: {end_time - start_time} seconds")
    #rich.print(f"Replanned items: {replanned_items}")

    #replanned_items = plan_relocation(items, verbose=False, plot=False)
    expected_replanned_items = [
        {'q': 8192, 'kv': 8192, 'gpuid': 1, 'seqid': 0, 'src_gpuid': 1, 'is_original': True},
        {'q': 8192, 'kv': 8192, 'gpuid': 1, 'seqid': 1, 'src_gpuid': 1, 'is_original': True},
        {'q': 4096, 'kv': 4096, 'gpuid': 2, 'seqid': 0, 'src_gpuid': 2, 'is_original': True},
        {'q': 4096, 'kv': 4096, 'gpuid': 2, 'seqid': 1, 'src_gpuid': 2, 'is_original': True},
        {'q': 4096, 'kv': 4096, 'gpuid': 2, 'seqid': 2, 'src_gpuid': 2, 'is_original': True},
        {'q': 4096, 'kv': 4096, 'gpuid': 2, 'seqid': 3, 'src_gpuid': 2, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 0, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 1, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 2, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 3, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 4, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 5, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 6, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 7, 'src_gpuid': 3, 'is_original': True},
        {'q': 3754, 'kv': 3754, 'gpuid': 0, 'seqid': 0, 'src_gpuid': 0, 'is_original': False},
        {'q': 1706, 'kv': 5460, 'gpuid': 2, 'seqid': 0, 'src_gpuid': 0, 'is_original': False},
        {'q': 2730, 'kv': 8190, 'gpuid': 3, 'seqid': 0, 'src_gpuid': 0, 'is_original': False},
        {'q': 2734, 'kv': 10924, 'gpuid': 3, 'seqid': 0, 'src_gpuid': 0, 'is_original': False},
        {'q': 1706, 'kv': 12630, 'gpuid': 2, 'seqid': 0, 'src_gpuid': 0, 'is_original': False},
        {'q': 3754, 'kv': 16384, 'gpuid': 0, 'seqid': 0, 'src_gpuid': 0, 'is_original': False}
    ]
    verification_layout(items, replanned_items)
    run_flops_balance_test(items, replanned_items, tolerance=tolerance_factor)
    return


if __name__ == "__main__":
    iter = 1
    for _ in range(iter):
        test_planner_equal_flops()