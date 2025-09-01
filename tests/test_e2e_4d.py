"""
Benchmark 4D Parallel
- MegatronWorker - with pipeline paralleism.
- Entrypoint:
    - Specify the dp,cp,pp,tp size
    - initialize the paralleism degree


Usage
```bash
NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 \
torchrun --nnodes=1 --nproc_per_node=8 test_e2e_4d.py \
    --tp-size=8 --num-tokens 1024 
```
"""
import os
import rich
import torch
import argparse
import contextlib
from typing import Iterable, List, Optional

K = 1024

try:
    import wlbllm
    import wlbllm.utils
    import wlbllm.registry
except ImportError:
    print("""⚠️ WLBLLM is not installed. This only affects if you're testing WLBLLM tests. To install:

    cd d2/baseline/wlbllm_original
    pip install -e .
    """)
    pass


# ---------------------
# Debug Utilities
# ---------------------

def debug_print(*args, **kwargs):
    if os.getenv("D2_DEBUG_PRINT", "0") == "1":
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            rich.print(f"[Rank {rank}]", *args, **kwargs)
    return


# ---------------------
# EnvVar Registry
# ---------------------


# ---------------------
# Dataset Utilities
# ---------------------
from d2.simulator.optimizers.samples import (
    sample_wlbllm_docs_upsample, 
    batch_documents,
)
ITERATION_ID = 0
GLOBAL_BATCH: Optional[Iterable[List[int]]] = None
iterated_samples = []
modified_batches = []
fa2a_metadata_list = []

# TODO(GindaChen)(Refactor): Move it to a particualr dataset provider file (we already have it).
# TODO(GindaChen)(Refactor): Ensure the GLOBAL_BATCH and ITERATION_ID global varaibles have protected accesses.
def setup_global_batch(
    tokens_per_batch, 
    up_sample_factor=2,
    elongate_factor=1,
    filter_threshold=64 * 1024,
    filter_ratio=0.90,
    should_add_debug_cases=False,
):
    global GLOBAL_BATCH
    if GLOBAL_BATCH is not None:
        return

    GLOBAL_BATCH = batch_documents(
        sample_wlbllm_docs_upsample(
            size=10000,
            filter_threshold=filter_threshold,
            filter_ratio=filter_ratio,
            upsample_long_factor=up_sample_factor,
            elongate_factor=elongate_factor,
        ), max_ctx_length=tokens_per_batch
    )
    if should_add_debug_cases:
        GLOBAL_BATCH = list(GLOBAL_BATCH)
        manual_case = [
            [tokens_per_batch // 4 * 3 - 512, 512, tokens_per_batch // 4],
            [tokens_per_batch // 4 * 3 - 512, 512, tokens_per_batch // 4],
        ]
        GLOBAL_BATCH = manual_case + GLOBAL_BATCH
        GLOBAL_BATCH = iter(GLOBAL_BATCH)
    return

def get_next_batch(num_batches: int) -> Iterable[List[List[int]]]:
    global GLOBAL_BATCH
    global ITERATION_ID
    global iterated_samples
    # get num_batches number of batches 
    batches = []
    for _ in range(num_batches):    
        batches.append(next(GLOBAL_BATCH))
    ITERATION_ID += 1
    iterated_samples.append(batches)
    return batches



# ---------------------
# Main Functions
# ---------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["baseline", "d2", "wlbllm"], default="baseline", 
                        help="Test mode: 'baseline' for simple batch generation, 'd2' for balanced flops planning, 'wlbllm' for wlbllm")
    
    # Model Args
    parser.add_argument("--model-path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--num-layers", type=str, default=None, help="Number of layers. Default: None (read from hf model config). Override by environment variable `NUM_LAYERS`.")

    # Dataset Args
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--max-sample-id", type=int, default=10)
    parser.add_argument("--up-sample-factor", type=int, default=2)
    parser.add_argument("--elongate-factor", type=int, default=1)
    parser.add_argument("--filter-threshold", type=int, default=64 * 1024)
    parser.add_argument("--filter-ratio", type=float, default=0.90)
    parser.add_argument("--should-add-debug-cases", action="store_true")
    
    # Setup Args
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--should-profile-memory", type=str, default=None)
    
    # Parallelism Args
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)

    # Distributed Args
    parser.add_argument("--num-nodes", type=int, default=None, help="Number of nodes. Default: calculated from WORLD_SIZE and LOCAL_WORLD_SIZE")
    parser.add_argument("--num-gpus-per-node", type=int, default=None, help="Number of GPUs per node. Default: calculated from WORLD_SIZE and LOCAL_WORLD_SIZE")


    args = parser.parse_args()

    # ---------------------
    # Modify the arguments 
    # ---------------------

    def check_and_modify_model_args(args):
        if args.num_layers is None:
            args.num_layers = os.environ.get("NUM_LAYERS", None)
        return args

    # - Distributed Args
    def check_and_modify_dist_args(args):
        # assert either num_nodes or num_gpus_per_node is both None or both having values
        if args.num_nodes is None and args.num_gpus_per_node is None:
            pass
        elif args.num_nodes is not None and args.num_gpus_per_node is not None:
            pass
        else:
            raise ValueError("Either num_nodes or num_gpus_per_node must be specified")

        # torchrun env variables
        # calculate the values from WORLD_SIZE and LOCAL_WORLD_SIZE
        world_size = int(os.environ.get("WORLD_SIZE", None))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", None))
        num_nodes = world_size // local_world_size
        num_gpus_per_node = local_world_size

        args.num_nodes = num_nodes
        args.num_gpus_per_node = num_gpus_per_node
        return args
    

    # - Setup Output Dir
    def check_and_modify_output_dir(args):
        if args.output_dir is None:
            args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
            pass
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        return args


    args = check_and_modify_model_args(args)
    args = check_and_modify_dist_args(args)
    args = check_and_modify_output_dir(args)

    return args

def main():
    def check_if_torch_distributed_env_variables_are_set():
        if not os.environ.get("WORLD_SIZE", None) or not os.environ.get("LOCAL_WORLD_SIZE", None):
            raise ValueError("WORLD_SIZE and LOCAL_WORLD_SIZE must be set")
    
    check_if_torch_distributed_env_variables_are_set()

    args = parse_args()
    print(args)
    pass

if __name__ == "__main__":
    main()