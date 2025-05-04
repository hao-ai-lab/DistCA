import torch
import flashinfer
import gc
from rich import print
import multiprocessing as mp
from collections import namedtuple

# Set the start method to 'spawn' for CUDA compatibility
mp.set_start_method('spawn', force=True)

# Define Config namedtuple at module level
Config = namedtuple('Config', ['rank', 'batch_size', 'qo_len', 'kv_len', 'num_qo_heads', 'num_kv_heads', 'head_dim', 'tp_size'])

def get_mask(q_length, kv_length, rank, batch_size):
    a = torch.tril(torch.ones(q_length, kv_length))
    b = torch.cat([a] * batch_size, dim=0)
    return b
    

def run_flash_attention(rank=0, batch_size=1, qo_len=128, kv_len=4096, num_qo_heads=32, num_kv_heads=32, head_dim=128, repeat=7, visualize_mask=False,device="cuda",return_tensors=False,verbose=False):
    def print_if_verbose(s):
        if verbose:
            print(s)
        return
    
    q = torch.randn(qo_len * batch_size, num_qo_heads, head_dim).half().to(device)
    k = torch.randn(kv_len, num_kv_heads, head_dim).half().to(device)
    v = torch.randn(kv_len, num_kv_heads, head_dim).half().to(device)
    mask = get_mask(qo_len, kv_len, rank, batch_size)
    mask = mask.to(device)

    compute_times = []
    for _ in range(repeat):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        o_custom = flashinfer.single_prefill_with_kv_cache(q, k, v, custom_mask=mask)
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
tp_size = 4

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
        item = run_flash_attention(
            **config,
            repeat=7,
            return_tensors=False,
        )
        config['tp_size'] = tp_size
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
            tp_size=tp_size
        )
        results_queue.put((config_tuple, computed_time))
    except Exception as e:
        print(f"Error: {e}")
        print(f"Config: {config}")
        results_queue.put((None, str(e)))
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    results_queue = mp.Queue()
    
    for name, model_config in configs.items():
        for k in range(10, 20 + 1):
            qo_len = kv_len = 2 ** k
            config = dict(
                rank=0,
                batch_size=1,
                qo_len=qo_len,
                kv_len=kv_len,
                num_qo_heads=model_config['num_qo_heads'] // tp_size,
                num_kv_heads=model_config['num_kv_heads'] // tp_size,
                head_dim=model_config['head_dim'],
            )
            print(f"Running {name} with qo_len {qo_len}, kv_len {kv_len}, num_qo_heads {model_config['num_qo_heads'] // tp_size}, num_kv_heads {model_config['num_kv_heads'] // tp_size}, head_dim {model_config['head_dim']}")
            
            p = mp.Process(target=run_benchmark, args=(config, results_queue))
            p.start()
            p.join()

            config_result, result = results_queue.get()
            if config_result is None:
                print(f"Failed {config} run with error: {result}")
                continue
            results[config_result] = result
            print(f"Finished {config_result} with result: {result}")