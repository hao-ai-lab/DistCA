import collections
import random
import time
from typing import Any, Dict, List

import rich
import torch
from d2.planner.planner import (Planner, batch_to_items_general, get_flops)
from rich.console import Console
from rich.table import Table

from test_util import ParallelConfig

console = Console()

K = 1024

def verification_layout(
    originals: List[Dict[str, Any]], 
    replans: List[Dict[str, Any]], 
    verbose: bool = False
) -> bool:
    """
    Verify whether replanned items meet expectations according to the specified algorithm.
    Prints detailed steps only if verbose is True.
    """
    console = Console()
    
    def rlog(message):
        if verbose:
            console.print(message)

    console.print("[bold cyan]Running Replanning Verification Test...[/bold cyan]")

    original_q_totals = collections.defaultdict(int)
    for item in originals:
        original_q_totals[item['seqid']] += item['q']

    grouped_replans = collections.defaultdict(list)
    for item in replans:
        grouped_replans[item['seqid']].append(item)

    overall_result = True

    for seqid, items in grouped_replans.items():
        offloaded_items = [item for item in items if not item.get('is_original')]
        if not offloaded_items:
            continue

        rlog(f"\n[bold]----- Verifying Sequence (seqid={seqid}) -----[/bold]")
        
        original_q = original_q_totals.get(seqid)
        if original_q is None:
            console.print(f"[bold red][FAIL][/bold red] Cannot find original item(s) for (seqid={seqid})")
            overall_result = False
            continue
        
        rlog(f"Original total 'q' (summed across all ranks): [bold yellow]{original_q}[/bold yellow]")
        
        # Check 1: 'q' sum conservation
        rlog("\n[bold]Check 1: Verifying 'q' sum conservation...[/bold]")
        replan_q_sum = sum(item['q'] for item in items)
        rlog(f"Sum of 'q' in all replanned parts for this sequence: [bold yellow]{replan_q_sum}[/bold yellow]")
        
        if replan_q_sum == original_q:
            rlog("[bold green][PASS][/bold green] 'q' sum is conserved.")
        else:
            console.print(f"[bold red][FAIL][/bold red] 'q' sum mismatch for seqid={seqid}! Expected: {original_q}, Got: {replan_q_sum}")
            overall_result = False
            continue

        # Check 2: 'kv' difference rule for offloaded items
        rlog("\n[bold]Check 2: Verifying 'kv' difference rule for offloaded items...[/bold]")
        sorted_offloaded = sorted(offloaded_items, key=lambda x: x['kv'])
        
        kv_check_passed = True
        if len(sorted_offloaded) > 1:
            for i in range(1, len(sorted_offloaded)):
                kv_diff = sorted_offloaded[i]['kv'] - sorted_offloaded[i-1]['kv']
                q_curr = sorted_offloaded[i]['q']
                
                rlog(f"  - Checking item {i}: kv_diff ({kv_diff}) vs current q ({q_curr})")
                if kv_diff != q_curr:
                    console.print(f"    [bold red][FAIL][/bold red] Rule violated for seqid={seqid}: kv_diff ({kv_diff}) != q ({q_curr})")
                    kv_check_passed = False
                    overall_result = False

        if kv_check_passed:
            rlog("[bold green][PASS][/bold green] 'kv' difference rule holds for all adjacent pairs.")
        
        rlog("----- Verification Finished for this Sequence -----\n")

    if overall_result:
        console.print("[bold green]✅ Replanning Verification Test Passed[/bold green]\n")
    else:
        console.print("[bold red]❌ Replanning Verification Test Failed[/bold red]\n")

    return overall_result

def run_flops_balance_test(
    originals: List[Dict[str, Any]], 
    replans: List[Dict[str, Any]], 
    tolerance: float, 
    verbose: bool = False
) -> bool:
    """
    Verify FLOPs conservation and load balancing.
    Prints detailed steps only if verbose is True.
    """
    console = Console()

    def rlog(message):
        if verbose:
            console.print(message)

    console.print("[bold cyan]Running FLOPs Conservation and Load Balance Test...[/bold cyan]")

    # Check 1: FLOPs Conservation
    rlog("\n[bold]Check 1: Verifying Total FLOPs Conservation...[/bold]")
    total_original_flops = sum(get_flops(**item) for item in originals)
    total_replanned_flops = sum(get_flops(**item) for item in replans)

    rlog(f"Total Original FLOPs:  [yellow]{total_original_flops:,.2f}[/yellow]")
    rlog(f"Total Replanned FLOPs: [yellow]{total_replanned_flops:,.2f}[/yellow]")

    conservation_passed = (total_original_flops == total_replanned_flops)
    if conservation_passed:
        rlog("[bold green][PASS][/bold green] Total FLOPs are conserved.")
    else:
        console.print(f"[bold red][FAIL][/bold red] Total FLOPs do not match! Original: {total_original_flops:,.2f}, Replanned: {total_replanned_flops:,.2f}")

    # Check 2: Load Balancing
    rlog(f"\n[bold]Check 2: Verifying Load Balancing (Tolerance = {tolerance * 100}%) ...[/bold]")

    gpu_flops = collections.defaultdict(int)
    for item in replans:
        gpu_flops[item['gpuid']] += get_flops(**item)

    world_size = len(gpu_flops)
    balancing_passed = True
    if world_size > 0:
        avg_flops = total_replanned_flops / world_size
        lower_bound = avg_flops * (1 - tolerance)
        upper_bound = avg_flops * (1 + tolerance)

        rlog(f"Total GPUs (World Size): [bold blue]{world_size}[/bold blue]")
        rlog(f"Average FLOPs per GPU: [bold blue]{avg_flops:,.2f}[/bold blue]")
        rlog(f"Acceptable Range:      [bold blue][{lower_bound:,.2f}, {upper_bound:,.2f}][/bold blue]")

        table = Table(title="GPU Load Balancing Analysis")
        table.add_column("GPU ID", style="cyan")
        table.add_column("Total FLOPs", style="magenta", justify="right")
        table.add_column("Deviation from Avg.", style="yellow", justify="right")
        table.add_column("Status", style="bold", justify="center")

        for gpu_id in sorted(gpu_flops.keys()):
            flops = gpu_flops[gpu_id]
            deviation = (flops - avg_flops) / avg_flops * 100 if avg_flops > 0 else 0.0
            if not (lower_bound <= flops <= upper_bound):
                status = "[red]FAIL[/red]"
                balancing_passed = False
            else:
                status = "[green]PASS[/green]"
            table.add_row(str(gpu_id), f"{flops:,.2f}", f"{deviation:+.2f}%", status)
        
        if verbose or not balancing_passed:
            console.print(table)
    
    if balancing_passed:
        rlog("[bold green][PASS][/bold green] All GPUs are within the tolerance range.")
    else:
        console.print("[bold red][FAIL][/bold red] At least one GPU is outside the tolerance range.")

    overall_result = conservation_passed and balancing_passed
    
    if overall_result:
        console.print("[bold green]✅ FLOPs & Balance Test Passed[/bold green]\n")
    else:
        console.print("[bold red]❌ FLOPs & Balance Test Failed[/bold red]\n")

    return overall_result

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

    weights = [random.random() for _ in range(num_sequences)]
    total_weight = sum(weights)
    if total_weight == 0:
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

def generate_random_even_split(total_sum: int, num_sequences: int) -> list[int]:
    """
    Generate a list of random EVEN integers whose sum is total_sum.
    """
    if total_sum % 2 != 0:
        raise ValueError("total_sum must be an even number to be split into even sequences.")
        
    half_sum = total_sum // 2
    half_values = generate_random_split(half_sum, num_sequences)
    even_values = [v * 2 for v in half_values]
    return even_values

def generate_random_rank_even_batches(
    num_ranks: int, 
    total_tokens_per_rank: int, 
    max_sequences_per_rank: int
) -> list[list[int]]:
    """
    Generate a random batch for each rank, ensuring all sequence lengths are even.
    """
    final_batches = []
    for _ in range(num_ranks):
        num_seq_for_this_rank = random.randint(1, max_sequences_per_rank)
        sequences_for_this_rank = generate_random_even_split(
            total_tokens_per_rank,
            num_seq_for_this_rank
        )
        final_batches.append(sequences_for_this_rank)
        
    return final_batches

def _generate_random_split(total_sum: int, num_parts: int) -> list[int]:
    """
    Helper function: Randomly split an integer into a specified number of parts.
    """
    if num_parts <= 0:
        return []
    if num_parts == 1:
        return [total_sum]

    weights = [random.random() for _ in range(num_parts)]
    total_weight = sum(weights)
    if total_weight == 0:
        proportions = [1/num_parts] * num_parts
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

def generate_random_even_numbers_with_total_sum(
    num_ranks: int, 
    total_tokens_per_rank: int
) -> list[int]:
    """
    Generate a list of random even numbers whose sum equals num_ranks * total_tokens_per_rank.

    Args:
        num_ranks (int): DP degree.
        total_tokens_per_rank (int): Number of tokens per rank (num_batch_token).

    Returns:
        List[int]: A list of even numbers whose sum matches the requirement.
        
    Raises:
        ValueError: If num_ranks * total_tokens_per_rank is odd.
    """
    total_sum = num_ranks * total_tokens_per_rank

    if total_sum % 2 != 0:
        raise ValueError(
            f"Total sum ({total_sum}) is odd, cannot generate a list of all even numbers to satisfy the sum."
            " Please ensure the product of num_ranks and total_tokens_per_rank is even."
        )
        
    if total_sum == 0:
        return []

    half_sum = total_sum // 2
    num_even_numbers = random.randint(1, half_sum)
    half_values = _generate_random_split(half_sum, num_even_numbers)
    even_values = [val * 2 for val in half_values]
    return even_values

class MockConfig:
    def __init__(self):
        self.hidden_size = 4096
        self.num_attention_heads = 32
        self.num_key_value_heads = 8
        self.num_hidden_layers = 32

def test_cp_planner():
    rich.print("⚪ Testing planner CP/DP planner...")
    
    model_config = MockConfig()
    parallel_config = ParallelConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    world_size = 4
    tolerance_factor = 0.1
    planner = Planner(
        world_size=world_size,
        parallel_config=parallel_config,
        model_config=model_config,
        tolerance_factor=tolerance_factor
    )

    num_seq = 4
    dp_degree = 4
    num_batched_token = 16 * K
    # Random DP test.
    # batch = generate_random_rank_even_batches(dp_degree, num_batched_token, num_seq)

    # Random CP test.TODO(Pb):(Current CP layout only support even sequence_len.)
    # batch = [generate_random_even_numbers_with_total_sum(dp_degree, num_batched_token)]
    # rich.print(f"Number of Documents: {len(batch[0])}")
    # Classical test.
    batch = [
        [16 * K] * 1,
        [8 * K] * 2,
        [4 * K] * 4,
        [2 * K] * 8, 
    ]

    initial_items = batch_to_items_general(batch, num_batched_token, world_size, model_config)
    # Items => replanned_items
    start_time = time.time()
    replanned_items = planner.plan_items(initial_items, verbose=True, plot=True)
    end_time = time.time()
    rich.print(replanned_items)

    # Items => metadata
    start_time_plan = time.time()
    final_metadata = planner.plan(initial_items, verbose=False)
    end_time_plan = time.time()
    
    rich.print(f"\nPlanner execution time: {end_time - start_time:.4f} seconds")
    rich.print(f"Planner execution time: {end_time_plan - start_time_plan:.4f} seconds")

    initial_dict = []
    replan_dict = []
    for d in initial_items:
        initial_dict.extend(d.to_dicts())
    for d in replanned_items:
        replan_dict.extend(d.to_dicts())

    verification_layout(initial_dict, replan_dict)
    run_flops_balance_test(initial_dict, replan_dict, tolerance_factor)

def test_mlp_seq_len():
    model_config = MockConfig()
    parallel_config = ParallelConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    world_size = 4
    tolerance_factor = 0.1
    planner = Planner(
        world_size=world_size,
        parallel_config=parallel_config,
        model_config=model_config,
        tolerance_factor=tolerance_factor
    )
    # Test DP
    batches: List[List[int]] = [[256, 256],[128, 384],[512], [10, 502] ]
    num_batched_token = 512
    dp_degree = world_size // parallel_config.tensor_model_parallel_size // parallel_config.pipeline_model_parallel_size

    dp_cp_test_items = batch_to_items_general(batches, num_batched_token=num_batched_token, DP_degree=dp_degree, model_config = model_config)
    actual_output = planner.items_to_mlp_doc_len(dp_cp_test_items, device='cpu')

    expected_output = [
        torch.tensor([256, 256], dtype=torch.int32),
        torch.tensor([128, 384], dtype=torch.int32),
        torch.tensor([512], dtype=torch.int32),
        torch.tensor([10, 502], dtype=torch.int32),
    ]

    for i in range(world_size):
        actual_tensor = actual_output[i]
        expected_tensor = expected_output[i]
        assert torch.equal(actual_tensor, expected_tensor), \
            f"Rank {i} tensor is wrong\n Expected: {expected_tensor}\n actual: {actual_tensor}"
    rich.print(f"[bold green][PASS][/bold green] test_mlp_seq_len Passed MLP DP test")

    # Test CP
    batches: List[List[int]] = [[256, 1024],[256], [128, 384] ]
    num_batched_token = 512
    dp_degree = world_size // parallel_config.tensor_model_parallel_size // parallel_config.pipeline_model_parallel_size

    dp_cp_test_items = batch_to_items_general(batches, num_batched_token=num_batched_token, DP_degree=dp_degree, model_config = model_config)
    actual_output = planner.items_to_mlp_doc_len(dp_cp_test_items, device='cpu')

    expected_output = [
        torch.tensor([256, 128, 128], dtype=torch.int32),
        torch.tensor([256, 256], dtype=torch.int32),
        torch.tensor([128, 128, 256], dtype=torch.int32),
        torch.tensor([128, 384], dtype=torch.int32),
    ]

    for i in range(world_size):
        actual_tensor = actual_output[i]
        expected_tensor = expected_output[i]
        assert torch.equal(actual_tensor, expected_tensor), \
            f"Rank {i} tensor is wrong\n Expected: {expected_tensor}\n actual: {actual_tensor}"
    rich.print(f"[bold green][PASS][/bold green] test_mlp_seq_len Passed MLP CP test")
    return

if __name__ == "__main__":
    test_mlp_seq_len()
    iter = 1
    for _ in range(iter):
        test_dp_planner()
        test_cp_planner()