import collections
import random

from rich.console import Console
from rich.table import Table

from d2.planner.planner import Item

console = Console()

K = 1024

def verification_layout(originals, replans):
    """
    Verify whether replanned items meet expectations according to the specified algorithm.
    """
    console.print("[bold cyan]Running Replanning Verification Test...[/bold cyan]\n")

    original_map = {(item['gpuid'], item['seqid']): item for item in originals}

    grouped_replans = collections.defaultdict(list)
    for item in replans:
        grouped_replans[(item['src_gpuid'], item['seqid'])].append(item)

    overall_result = True

    for (src_gpuid, seqid), items in grouped_replans.items():
        offloaded_items = [
            item for item in items if not item.get('is_original')
        ]
        
        if not offloaded_items:
            continue

        console.print(f"[bold]----- Verifying Sequence (src_gpuid={src_gpuid}, seqid={seqid}) -----[/bold]")
        
        original_item = original_map.get((src_gpuid, seqid))
        if not original_item:
            console.print(f"[bold red][FAIL][/bold red] Cannot find original item for (src_gpuid={src_gpuid}, seqid={seqid})")
            overall_result = False
            continue
        
        original_q = original_item['q']
        console.print(f"Original 'q': [bold yellow]{original_q}[/bold yellow]")
        
        console.print("\n[bold]Check 1: Verifying 'q' sum conservation...[/bold]")

        replan_q_sum = sum(item['q'] for item in items)
        console.print(f"Sum of 'q' in all replanned parts for this sequence: [bold yellow]{replan_q_sum}[/bold yellow]")
        
        if replan_q_sum == original_q:
            console.print("[bold green][PASS][/bold green] 'q' sum is conserved.")
        else:
            console.print(f"[bold red][FAIL][/bold red] 'q' sum mismatch! Expected: {original_q}, Got: {replan_q_sum}")
            overall_result = False
            console.print("----- Verification Finished for this Sequence -----\n", style="bold")
            continue

        console.print("\n[bold]Check 2: Verifying 'kv' difference rule for offloaded items...[/bold]")
        
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
        
        if kv_check_passed:
             console.print("[bold green][PASS][/bold green] 'kv' difference rule holds for all adjacent pairs.")
        
        console.print("----- Verification Finished for this Sequence -----\n", style="bold")

    return overall_result

class MockConfig:
    def __init__(self):
        self.hidden_size = 4096
        self.num_attention_heads = 32
        self.num_key_value_heads = 8
        self.num_hidden_layers = 32


def test_general_split():
    model_config = MockConfig()

    seq_len = random.randint(20, 100) * K
    original_item_data = {'q': seq_len, 'kv': seq_len}

    item_to_split = Item(
        model_config,
        seq_len,
        1,
        0,
        0,
        original_item_data,
        is_original=True
    )
    
    original_total_flops = item_to_split.get_flops()
    
    console.print(f"[bold magenta] Starting General Split Test [/bold magenta]")
    console.print(f"[bold blue]Original Item on GPU {item_to_split.gpuid}:[/bold blue]")
    console.print(f"  q={item_to_split.complete_item['q']}, kv={item_to_split.complete_item['kv']}, flops={original_total_flops:.2f}\n")

    all_items = [item_to_split]
    
    num_splits = random.randint(2, 5)
    console.print(f"[bold yellow]Plan: Performing {num_splits} consecutive splits.[/bold yellow]\n")

    for i in range(num_splits):
        console.print(f"[bold]--------- Split Step {i + 1}/{num_splits} ---------[/bold]")
        
        flops_before_this_split = item_to_split.get_flops()

        if flops_before_this_split < 2:
            console.print(f"[yellow]Item has only {flops_before_this_split:.2f} FLOPs left. Stopping split process.[/yellow]")
            break

        flops_to_move = random.uniform(1, flops_before_this_split * 1.5)
        remote_gpuid = (item_to_split.gpuid + i + 1) % 4

        console.print(f"[bold blue]Action:[/bold blue] Splitting {flops_to_move:.2f} FLOPs from GPU {item_to_split.gpuid} to GPU {remote_gpuid}...")

        new_item, moved_flops = item_to_split.split_item(flops_to_move, remote_gpuid, verbose=True)

        if new_item:
            current_total_flops = item_to_split.get_flops() + new_item.get_flops()
            assert abs(current_total_flops - flops_before_this_split) < 1e-6, \
                f"FLOPs mismatch after split {i+1}! Before: {flops_before_this_split}, After Sum: {current_total_flops}"
            
            console.print(f"[green]✅ Step {i+1} Verification Passed:[/green] FLOPs are conserved for this split.")
            all_items.append(new_item)
        else:
            console.print(f"[yellow]Split {i+1} did not produce a new item. Probably requested FLOPs were too small/big.[/yellow]")
        
        if item_to_split.gpuid != item_to_split.src_gpuid:
            console.print(f"Entire item is moved")
            break

        console.print(f"[bold]------------------------------------[/bold]\n")

    final_total_flops = sum(item.get_flops() for item in all_items)
    
    console.print(f"[bold magenta]🏁 Final Verification 🏁[/bold magenta]")
    console.print(f"Original total FLOPs: {original_total_flops:.2f}")
    console.print(f"Sum of all final pieces' FLOPs: {final_total_flops:.2f}")

    assert abs(final_total_flops - original_total_flops) < 1e-6, \
        f"Total FLOPs mismatch! Original: {original_total_flops}, Final Sum: {final_total_flops}"

    console.print(f"[bold green]✅✅✅ All tests passed! Total FLOPs are conserved throughout the process.[/bold green]\n")
    
    originals_for_verification = [original_item_data]
    originals_for_verification[0].update({'gpuid': 0, 'seqid': 1})
    
    replans_for_verification = []
    for item in all_items:
        replans_for_verification.extend(item.to_dicts())
        
    verification_layout(originals_for_verification, replans_for_verification)

if __name__ == "__main__":
    iter = 150
    for _ in range(iter):
        test_general_split()
