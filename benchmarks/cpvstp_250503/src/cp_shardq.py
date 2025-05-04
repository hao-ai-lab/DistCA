import json
import torch
import flashinfer
import gc
import numpy as np
import multiprocessing as mp
from pathlib import Path
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console
from rich import print as rprint

K = 1024
M = K * K

def get_mask(qo_len, kv_len, rank, batch_size):
    # Dummy mask generation for illustration; replace with your actual logic.
    mask = torch.ones(batch_size * qo_len, kv_len, dtype=torch.bool)
    return mask

def visualize_mask_heatmap(mask):
    import matplotlib.pyplot as plt
    plt.imshow(mask.cpu().float(), cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title("Custom Attention Mask")
    plt.show()

def run_flash_attention(rank, batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim, device, queue):
    try:
        q = torch.randn(qo_len * batch_size, num_qo_heads, head_dim).half().to(device)
        k = torch.randn(kv_len, num_kv_heads, head_dim).half().to(device)
        v = torch.randn(kv_len, num_kv_heads, head_dim).half().to(device)
        mask = get_mask(qo_len, kv_len, rank, batch_size).to(device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        compute_times = []

        for _ in range(10):
            start_event.record()
            _ = flashinfer.single_prefill_with_kv_cache(q, k, v, custom_mask=mask)
            end_event.record()
            torch.cuda.synchronize()
            compute_times.append(start_event.elapsed_time(end_event))

        median_time = np.median(compute_times)
        result = dict(
            rank=rank,
            batch_size=batch_size,
            qo_len=qo_len,
            kv_len=kv_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            median_compute_time=median_time,
            all_compute_times=compute_times,
        )
        queue.put(result)
    except Exception as e:
        queue.put({"rank": rank, "error": str(e)})
    finally:
        gc.collect()
        torch.cuda.empty_cache()

def safe_run_flash_attention(rank, batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim=128, device="cuda"):
    queue = mp.Queue()
    p = mp.Process(
        target=run_flash_attention,
        args=(rank, batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim, device, queue)
    )
    p.start()
    p.join(timeout=60)

    if p.is_alive():
        p.terminate()
        return {"rank": rank, "error": "timeout"}

    return queue.get()

# Example batch run
if __name__ == "__main__":
    mp.set_start_method("spawn")  # Safe for CUDA
    cp_degrees = [1, 2, 4, 8]
    qo_lens = [8, 16, 32, 64, 128, 256]
    batch_sizes = [1, 2, 4, 8]
    result = []
    
    result_path = Path(__file__).parent.parent / 'results' / 'cp_shardq.jsonl'
    result_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results
    completed_runs = set()
    if result_path.exists():
        with open(result_path, "r") as f:
            for line in f:
                item = json.loads(line)
                key = (item['cp_degree'], item['rank'], item['batch_size'], item['qo_len'])
                completed_runs.add(key)

    total_tasks = len(batch_sizes) * len(qo_lens) * sum([
        cp_degree for cp_degree in cp_degrees
    ])
    console = Console()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running flash attention benchmarks...", total=total_tasks)

        for cp_degree in cp_degrees:
            for batch_size in batch_sizes:
                for qo_len in qo_lens:
                    for rank in range(cp_degree):
                        # Skip if already computed
                        if (cp_degree, rank, batch_size, qo_len) in completed_runs:
                            progress.console.print(f"[yellow]Skipping completed run: cp_degree={cp_degree}, rank={rank}, batch_size={batch_size}, qo_len={qo_len}[/yellow]")
                            progress.advance(task)
                            continue

                        adjusted_qo_len = K * (qo_len // 2 // cp_degree)
                        kv_len = K * qo_len
                        res = safe_run_flash_attention(rank, batch_size, adjusted_qo_len, kv_len, 32, 8)
                        progress.advance(task)
                        if "error" in res:
                            rprint(f"[red]Error in cp_degree={cp_degree}, rank={rank}, batch_size={batch_size}, qo_len={qo_len}: {res['error']}[/red]")
                            continue
                        
                        rprint(f"[green]cp_degree={cp_degree}, rank={rank}, batch_size={batch_size}, qo_len={qo_len}: {res['median_compute_time']:.2f} ms[/green]")
                    
                        # Add cp_degree to result
                        res['cp_degree'] = cp_degree
                        result.append(res)

                        with open(result_path, "a") as f:
                            json.dump(res, f)
                            f.write("\n")
                        