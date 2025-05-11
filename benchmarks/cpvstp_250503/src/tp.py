import json
from pathlib import Path
import torch
import flashinfer
import gc
from rich import print
import multiprocessing as mp
from collections import namedtuple
from time import sleep

# Set the start method to 'spawn' for CUDA compatibility

filename = 'tp'
result_path = Path(__file__).parent.parent / 'results' / f'{filename}.jsonl'
result_path.parent.mkdir(parents=True, exist_ok=True)

# Define Config namedtuple at module level
Config = namedtuple('Config', ['rank', 'batch_size', 'qo_len', 'kv_len', 'num_qo_heads', 'num_kv_heads', 'head_dim', 'tp_size'])

def get_mask(q_length, kv_length, rank, batch_size):
    a = torch.tril(torch.ones(q_length, kv_length, dtype=torch.bool))
    b = torch.cat([a] * batch_size, dim=0)
    return b

def run_flash_attention(rank=0, batch_size=1, qo_len=128, kv_len=4096, num_qo_heads=32, num_kv_heads=32, head_dim=128, tp_size=1,repeat=7, visualize_mask=False,device="cuda",return_tensors=False,verbose=False):
    def print_if_verbose(s):
        if verbose:
            print(s)
        return
    
    q = torch.randn(qo_len * batch_size, num_qo_heads // tp_size, head_dim, device=device, dtype=torch.float16)
    print(f"q.shape: {q.shape}. q size: {q.numel() * q.element_size() / 1024 ** 3} GB")
    k = torch.randn(kv_len, num_kv_heads // tp_size, head_dim, device=device, dtype=torch.float16)
    print(f"k.shape: {k.shape}. k size: {k.numel() * k.element_size() / 1024 ** 3} GB")
    v = torch.randn(kv_len, num_kv_heads // tp_size, head_dim, device=device, dtype=torch.float16)
    print(f"v.shape: {v.shape}. v size: {v.numel() * v.element_size() / 1024 ** 3} GB")
    if batch_size > 1:
        mask = get_mask(qo_len, kv_len, rank, batch_size)
        mask = mask.to(device)
        print(f"mask.shape: {mask.shape}. mask size: {mask.numel() * mask.element_size() / 1024 ** 3} GB")
    else:
        mask = None

    compute_times = []
    for _ in range(repeat):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        o_custom = flashinfer.single_prefill_with_kv_cache(q, k, v, custom_mask=mask)
        print(f"o_custom.shape: {o_custom.shape}. o_custom size: {o_custom.numel() * o_custom.element_size() / 1024 ** 3} GB")
        end_event.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        compute_times.append(elapsed_time_ms)
        print_if_verbose(f"Elapsed time: {elapsed_time_ms:.2f} ms")
        torch.cuda.empty_cache()
    
    median_compute_time = torch.tensor(compute_times).median()
    return_values = [None, compute_times, median_compute_time.item()]
    if return_tensors:
        return_values[0] = o_custom.cpu()
    return return_values

results = {}
tp_sizes = [4, 1, 2, 8]  # List of tp_sizes to try
tp_sizes = list(reversed(tp_sizes))

configs = dict(
    llama8b=dict(
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=128,
    ),
    llama70b=dict(
        num_qo_heads=64,
        num_kv_heads=8,
        head_dim=128,
    )
)

def run_benchmark(config, results_queue):
    try:
        if config['qo_len'] >= 2 ** 17 - 1:
            repeat = 3
        else:
            repeat = 7
        item = run_flash_attention(
            **config,
            repeat=repeat,
            return_tensors=False,
        )
        computed_time = item[-1]
        
        # Create Config instance with all required fields
        config_tuple = Config(
            rank=config['rank'],
            batch_size=config['batch_size'],
            qo_len=config['qo_len'],
            kv_len=config['kv_len'],
            num_qo_heads=config['num_qo_heads'],
            num_kv_heads=config['num_kv_heads'],
            head_dim=config['head_dim'],
            tp_size=config['tp_size'],
        )
        results_queue.put((config_tuple, computed_time))
    except Exception as e:
        print(f"Error: {e}")
        print(f"Config: {config}")
        import traceback
        error_message = f"{str(e)}\n{traceback.format_exc()}"
        results_queue.put((None, error_message))
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    completed_runs = set()
    if result_path.exists():
        with open(result_path, "r") as f:
            for line in f:
                item = json.loads(line)
                config_tuple = Config(
                    rank=item['rank'],
                    batch_size=item['batch_size'],
                    qo_len=item['qo_len'],
                    kv_len=item['kv_len'],
                    num_qo_heads=item['num_qo_heads'],
                    num_kv_heads=item['num_kv_heads'],
                    head_dim=item['head_dim'],
                    tp_size=item['tp_size']
                )
                completed_runs.add(config_tuple)
    
    for batch_size in [1, 2, 4, 8]:    
        for name, model_config in configs.items():
            for tp_size in tp_sizes:
                for k in range(10, 20 + 1):
                    qo_len = kv_len = 2 ** k
                    config = dict(
                        rank=0,
                        batch_size=batch_size,
                        qo_len=qo_len,
                        kv_len=kv_len,
                        num_qo_heads=model_config['num_qo_heads'],
                        num_kv_heads=model_config['num_kv_heads'],
                        head_dim=model_config['head_dim'],
                        tp_size=tp_size,
                    )
                    config_tuple = Config(
                        rank=config['rank'],
                        batch_size=config['batch_size'],
                        qo_len=config['qo_len'],
                        kv_len=config['kv_len'],
                        num_qo_heads=config['num_qo_heads'],
                        num_kv_heads=config['num_kv_heads'],
                        head_dim=config['head_dim'],
                        tp_size=tp_size
                    )
                    if config_tuple in completed_runs:
                        print(f"Skipping already computed config: {config_tuple}")
                        continue

                    print(f"Running {name} with tp_size={tp_size}, k={k}, qo_len {qo_len}, kv_len {kv_len}, num_qo_heads {model_config['num_qo_heads'] // tp_size}, num_kv_heads {model_config['num_kv_heads'] // tp_size}, head_dim {model_config['head_dim']}")
                    
                    results_queue = mp.Queue()
                    p = mp.Process(target=run_benchmark, args=(config, results_queue))
                    p.start()
                    p.join()

                    if p.is_alive():
                        p.terminate()
                        continue

                    config_result, result = results_queue.get()
                    if config_result is None:
                        print(f"Failed {config} run with error: {result}")
                        sleep(3)
                        continue
                    results[config_result] = result
                    print(f"Finished {config_result} with result: {result}")

                    result_dict = dict(
                        **config, 
                        computed_time=result,
                    )
                    with open(result_path, 'a') as f:
                        f.write(json.dumps(result_dict) + '\n')