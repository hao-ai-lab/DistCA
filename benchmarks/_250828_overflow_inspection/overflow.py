# %%

import rich
from typing import Iterable, List
from d2.simulator.optimizers.samples import (
    batch_documents, 
    sample_wlbllm_docs_upsample
)


def setup_global_batch(
    per_batch_seq_len, 
    up_sample_factor=2,
    elongate_factor=1,
    filter_threshold=64 * 1024,
    filter_ratio=0.90,
    should_add_debug_cases=False,
):
    GLOBAL_BATCH = batch_documents(
        sample_wlbllm_docs_upsample(
            size=10000,
            filter_threshold=filter_threshold,
            filter_ratio=filter_ratio,
            upsample_long_factor=up_sample_factor,
            elongate_factor=elongate_factor,
        ), max_ctx_length=per_batch_seq_len
    )

    if should_add_debug_cases:
        GLOBAL_BATCH = list(GLOBAL_BATCH)
        manual_case = [
            [per_batch_seq_len // 4 * 3 - 512, 512, per_batch_seq_len // 4],
            [per_batch_seq_len // 4 * 3 - 512, 512, per_batch_seq_len // 4],
        ]
        GLOBAL_BATCH = manual_case + GLOBAL_BATCH
        GLOBAL_BATCH = iter(GLOBAL_BATCH)
    return GLOBAL_BATCH

def get_next_batch(dp_size) -> Iterable[List[List[int]]]:
    global GLOBAL_BATCH
    batches = []
    for _ in range(dp_size):    
        batches.append(next(GLOBAL_BATCH))
    return batches

def chunk(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

# %%
from transformers import AutoConfig
from dataclasses import dataclass
from typing import Optional

@dataclass
class ParallelConfig:
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: Optional[int] = None
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: Optional[int] = None




model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
hf_config = AutoConfig.from_pretrained(model_path)

model_config = hf_config
parallel_config = ParallelConfig(
    pipeline_model_parallel_size=1,
    tensor_model_parallel_size=1,
)


# %%
from d2.planner.planner import Planner

# %%
K = 1024


import torch
import rich

def overflow_info(fa2a_metadata, as_rank):
    send_sz = [torch.sum(m.fa2a_metadata[1][as_rank]).item() for m in fa2a_metadata]
    # send_sz + sender_recv_offset = sender_recv_last_token
    send_last_offset = [(m.fa2a_metadata[1] + m.fa2a_metadata[2])[as_rank] for m in fa2a_metadata]
    recv_sz = [torch.sum(m.fa2a_metadata[3][as_rank]).item() for m in fa2a_metadata]
    max_send_sz_mb = max(send_sz) // 1024**2
    max_recv_sz_mb = max(recv_sz) // 1024**2

    max_send_last_offset = max(torch.max(o).item() for o in send_last_offset) // 1024**2
    return max_send_sz_mb, max_recv_sz_mb, max_send_last_offset

# %%
batch_size = 4
world_size = 8
per_batch_seq_len = 256 * K

as_world_size = world_size 
num_batched_token_per_as_rank = per_batch_seq_len * batch_size // as_world_size

GLOBAL_BATCH = setup_global_batch(
    per_batch_seq_len=per_batch_seq_len,
    should_add_debug_cases=True,
    elongate_factor=4,
)


max_send_sz_mb_overall: int = 0
max_recv_sz_mb_overall: int = 0
max_send_last_offset_overall: int = 0
for i in range(100):
    rich.print(f"---------------\nIteration {i}\n---------------")
    try:
        _seq_lens: list[list[int]] = get_next_batch(2 * batch_size)
    except StopIteration:
        break
    seq_lens_0: list[list[int]] = _seq_lens[:batch_size]
    seq_lens_1: list[list[int]] = _seq_lens[batch_size:]

    from d2.planner.planner import batch_to_items_general

    _items_0: list['Item'] = batch_to_items_general(seq_lens_0, num_batched_token_per_as_rank, as_world_size, model_config)
    _items_1: list['Item'] = batch_to_items_general(seq_lens_1, num_batched_token_per_as_rank, as_world_size, model_config)

    planner = Planner(world_size, parallel_config, model_config=hf_config)

    fa2a_metadata_0, as_attn_metadata_0, mlp_shard_len_0 = planner.plan(_items_0)
    fa2a_metadata_1, as_attn_metadata_1, mlp_shard_len_1 = planner.plan(_items_1)

    max_send_sz_mb = 0
    max_recv_sz_mb = 0
    max_send_last_offset = 0
    for rank in range(world_size):
        infos_0 = overflow_info(fa2a_metadata_0, rank)
        infos_1 = overflow_info(fa2a_metadata_1, rank)
        rich.print(f"rank {rank}: {infos_0}, {infos_1}")
        max_send_sz_mb = max(max_send_sz_mb, infos_0[0])
        max_recv_sz_mb = max(max_recv_sz_mb, infos_0[1])
        max_send_last_offset = max(max_send_last_offset, infos_0[2])
    rich.print(f"Max size: ({max_send_sz_mb}, {max_recv_sz_mb}, {max_send_last_offset})")
    max_send_sz_mb_overall = max(max_send_sz_mb_overall, max_send_sz_mb)
    max_recv_sz_mb_overall = max(max_recv_sz_mb_overall, max_recv_sz_mb)
    max_send_last_offset_overall = max(max_send_last_offset_overall, max_send_last_offset)

rich.print(f"Max size overall: ({max_send_sz_mb_overall}, {max_recv_sz_mb_overall}, {max_send_last_offset_overall})")




