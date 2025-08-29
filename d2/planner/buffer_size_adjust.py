from dataclasses import dataclass
import rich
import torch
from typing import Iterable, List, Optional
from dataclasses import dataclass
from transformers import AutoConfig
from d2.simulator.optimizers.samples import (
    batch_documents, 
    sample_wlbllm_docs_upsample
)
from d2.planner.planner import Planner, batch_to_items_general

K = 1024


def get_next_batch(global_batch, dp_size) -> Iterable[List[List[int]]]:
    batches = []
    for _ in range(dp_size):    
        batches.append(next(global_batch))
    return batches

def chunk(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

def overflow_info(fa2a_metadata, as_rank):
    send_sz = [torch.sum(m.fa2a_metadata[1][as_rank]).item() for m in fa2a_metadata]
    # send_sz + sender_recv_offset = sender_recv_last_token
    send_last_offset = [(m.fa2a_metadata[1] + m.fa2a_metadata[2])[as_rank] for m in fa2a_metadata]
    recv_sz = [torch.sum(m.fa2a_metadata[3][as_rank]).item() for m in fa2a_metadata]
    max_send_sz_mb = max(send_sz) // 1024**2
    max_recv_sz_mb = max(recv_sz) // 1024**2

    max_send_last_offset = max(torch.max(o).item() for o in send_last_offset) // 1024**2
    return max_send_sz_mb, max_recv_sz_mb, max_send_last_offset

def calculate_recommended_buffer_size(
    global_batch,
    per_batch_seq_len,
    parallel_config,
    model_config,
    batch_size=4,
    as_world_size=8,
    world_size=8,
    num_iterations=100,
    verbose=False,
):
    """
    Calculate the recommended buffer size based on the global batch.
    
    Args:
        global_batch: Output from setup_global_batch
        batch_size: Batch size for processing
        world_size: Number of parallel workers
        num_iterations: Number of iterations to run
        model_path: Path to the model
        verbose: Whether to print detailed information
        
    Returns:
        int: Recommended buffer size in GB
    """
    # Load model configuration    
    
    # Setup parameters
    num_batched_token_per_as_rank = per_batch_seq_len * batch_size // as_world_size
    
    max_send_sz_mb_overall = 0
    max_recv_sz_mb_overall = 0
    max_send_last_offset_overall = 0
    
    for i in range(num_iterations):
        if verbose:
            rich.print(f"---------------\nIteration {i}\n---------------")
        try:
            _seq_lens = get_next_batch(global_batch, 2 * batch_size)
        except StopIteration:
            break
            
        seq_lens_0 = _seq_lens[:batch_size]
        seq_lens_1 = _seq_lens[batch_size:]

        _items_0 = batch_to_items_general(seq_lens_0, num_batched_token_per_as_rank, as_world_size, model_config)
        _items_1 = batch_to_items_general(seq_lens_1, num_batched_token_per_as_rank, as_world_size, model_config)

        planner = Planner(world_size, parallel_config, model_config=hf_config)

        fa2a_metadata_0, _, _ = planner.plan(_items_0)
        fa2a_metadata_1, _, _ = planner.plan(_items_1)

        max_send_sz_mb = 0
        max_recv_sz_mb = 0
        max_send_last_offset = 0
        
        for rank in range(world_size):
            infos_0 = overflow_info(fa2a_metadata_0, rank)
            infos_1 = overflow_info(fa2a_metadata_1, rank)
            if verbose:
                rich.print(f"rank {rank}: {infos_0}, {infos_1}")
            max_send_sz_mb = max(max_send_sz_mb, infos_0[0], infos_1[0])
            max_recv_sz_mb = max(max_recv_sz_mb, infos_0[1], infos_1[1])
            max_send_last_offset = max(max_send_last_offset, infos_0[2], infos_1[2])
            
        if verbose:
            rich.print(f"Max size: ({max_send_sz_mb}, {max_recv_sz_mb}, {max_send_last_offset})")
            
        max_send_sz_mb_overall = max(max_send_sz_mb_overall, max_send_sz_mb)
        max_recv_sz_mb_overall = max(max_recv_sz_mb_overall, max_recv_sz_mb)
        max_send_last_offset_overall = max(max_send_last_offset_overall, max_send_last_offset)

    if verbose:
        rich.print(f"Max size overall: ({max_send_sz_mb_overall}, {max_recv_sz_mb_overall}, {max_send_last_offset_overall})")
        
    recommended_buffer_size = max(max_send_sz_mb_overall, max_recv_sz_mb_overall, max_send_last_offset_overall)
    recommended_buffer_size = (recommended_buffer_size + 1024) // 1024  # MB -> GB
    
    if verbose:
        rich.print(f"buffer_size = {recommended_buffer_size} * GB")
        
    return recommended_buffer_size

# Example usage
if __name__ == "__main__":
    # Setup parameters
    per_batch_seq_len = 256 * K
    batch_size = 4
    world_size = 8
    
    # Create global batch
    
    global_batch = batch_documents(
        sample_wlbllm_docs_upsample(
            size=10000,
            filter_threshold=64 * 1024,
            filter_ratio=0.90,
            upsample_long_factor=2,
            elongate_factor=4,
        ), max_ctx_length=per_batch_seq_len
    )
    
    # Calculate recommended buffer size
    model_config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    
    @dataclass
    class ParallelConfig:
        tensor_model_parallel_size: int = 1
        pipeline_model_parallel_size: int = 1
        virtual_pipeline_model_parallel_size: Optional[int] = None
        context_parallel_size: int = 1
        expert_model_parallel_size: int = 1
        expert_tensor_parallel_size: Optional[int] = None

    parallel_config = ParallelConfig(
        pipeline_model_parallel_size=1,
        tensor_model_parallel_size=1,
    )
    buffer_size = calculate_recommended_buffer_size(
        global_batch,
        per_batch_seq_len,
        parallel_config,
        model_config,
        batch_size=batch_size,
        as_world_size=world_size,
        world_size=world_size,
        num_iterations=100,
        verbose=True
    )
    
    rich.print(f"Recommended buffer size: {buffer_size} GB")
