from typing import Iterable, List, Optional
from d2.simulator.optimizers.samples import (
    sample_wlbllm_docs_upsample,
    sample_prolong_docs, 
    batch_documents,
)
from collections import deque


def balance_ping_pong_with_mb(
    seq_lens: list[list[int]],
    mb: int, batch_size: int,
) -> list[list[int]]:
    """
    Co-design with the `create_pp_microbatches` and `create_qkv_dispatch_pipeline_tick`.
    
    In flie `test_megatron_e2e_pipeline_with_cp.py`, the outer loop logic is roughly:
    
    ```python
    for sample_idx in range(max_sample_id):
        microbatches_0, tick_per_rank_doc_lens_0 = create_pp_microbatches(mb, batch_size)
        microbatches_1, tick_per_rank_doc_lens_1 = create_pp_microbatches(mb, batch_size)
        pass
    ```

    We first take the ping batch (of mb * batch_size number of batches), then the pong batch.
    Sometimes, ping and pong are not balanced.
    This function aims to balance the two batches in terms of attention flops. 
    
    The procedure of this function looks like this:
    1. Take the `mb * batch_size * 2` number of batches.
    2. Reorganize is such that we have
    -------------------------------------------------------------------------------------------
      ping_b[0] | ping_b[1] | ... | ping_b[bs-1] | pong_b[0] | pong_b[1] | ... | pong_b[bs-1]
    -------------------------------------------------------------------------------------------


    """
    
    def batch_flops(batch):
        return sum(y ** 2 // 2 for y in batch)

    def zigzag(batches: list[list[int]]) -> list[list[int]]:
        """
        Convert each inner list into a zigzag pattern.
        Example (single inner list shown):
        8 7 6 5 4 3 2 1 -> 8 6 4 2 1 3 5 7
        Rule: take elements at even indices in order, then odd indices in reverse.
        """
        out: list[list[int]] = []
        for batch in batches:
            left = batch[0::2]            # even indices
            right = batch[1::2][::-1]     # odd indices, reversed
            out.append(left + right)
        return out

    assert len(seq_lens) % 2 == 0, f"ping pong should have even number of batches, but got {len(seq_lens)} batches, seq_lens={seq_lens}"
    assert len(seq_lens) == (mb * batch_size * 2), f"seq_lens should be divisible by {mb * batch_size * 2}, but got {len(seq_lens)}"
    sorted_batches = sorted(seq_lens, key=batch_flops, reverse=True)
    sorted_batches_deque = deque(sorted_batches)
    
    # Now sorted_batches_deque is sorted by flops >
    microbatch_results: 'list[list[int]]' = []

    for _ in range(mb):
        batches = []
        for _ in range(batch_size * 2):
            batches.append(sorted_batches_deque.popleft())

        # Now do ping/pong balance with the batches
        ping, pong = [], []
        ping_flops, pong_flops = 0, 0
        avg_num_batches = len(batches) // 2

        for batch in batches:
            if (ping_flops <= pong_flops and len(ping) < avg_num_batches) or len(pong) >= avg_num_batches:
                ping.append(batch)
                ping_flops += batch_flops(batch)
            else:
                pong.append(batch)
                pong_flops += batch_flops(batch)
        
        # Feed the ping and pong back to the results
        ping = zigzag(ping)
        pong = zigzag(pong)
        microbatch_results.extend(ping + pong)

    return microbatch_results


ITERATION_ID = 0
iterated_samples = []
GLOBAL_BATCH: Optional[Iterable[List[int]]] = None

def setup_global_batch(
    total_seq_len, 
    up_sample_factor=2,
    elongate_factor=1,
    filter_threshold=64 * 1024,
    filter_ratio=0.90,
    # should_add_debug_cases=True,
    should_add_debug_cases=False,
    change_long_doc_ratio=0.0,
    sample_name='wlbllm',
    balance_ping_pong_batch_size: None | dict[str, int] =None,
):
    global GLOBAL_BATCH
    if GLOBAL_BATCH is not None:
        return

    assert elongate_factor > 0, f"elongate_factor: {elongate_factor} must be greater than 0"

    if sample_name == 'wlbllm':
        sample_func = sample_wlbllm_docs_upsample
    elif sample_name == 'prolong':
        sample_func = sample_prolong_docs
    else:
        raise ValueError(f"Invalid sample_name: {sample_name}")

    GLOBAL_BATCH = batch_documents(
        sample_func(
            size=10000,
            filter_threshold=filter_threshold,
            filter_ratio=filter_ratio,
            upsample_long_factor=up_sample_factor,
            elongate_factor=elongate_factor,
            change_long_doc_ratio=change_long_doc_ratio,
        ), max_ctx_length=total_seq_len
    )
    GLOBAL_BATCH = list(GLOBAL_BATCH)
    
    if should_add_debug_cases:
        GLOBAL_BATCH = list(GLOBAL_BATCH)
        manual_case = [
            [total_seq_len], 
            [total_seq_len], 
            [total_seq_len], 
            [total_seq_len], 
            [total_seq_len], 
            [total_seq_len], 
            [total_seq_len], 
            [total_seq_len], 
            [total_seq_len], 
            [total_seq_len], 
            [total_seq_len], 
            [total_seq_len], 
            # [total_seq_len // 8] * 8,
            # [total_seq_len], [total_seq_len // 8] * 8,
            # [total_seq_len], [total_seq_len // 8] * 8,
            # [total_seq_len], [total_seq_len // 8] * 8,
        ]
        GLOBAL_BATCH = manual_case * 100 + GLOBAL_BATCH
    
    if balance_ping_pong_batch_size:
        # Construct a new global batch array such that
        # we take `balance_ping_pong_batch_size x 2` every time, 
        # and then balance the ping/pong.
        new_global_batch = []
        mb = balance_ping_pong_batch_size['mb']
        batch_size = balance_ping_pong_batch_size['batch_size']
        for star_idx in range(0, len(GLOBAL_BATCH), mb * batch_size * 2):
            end_idx = star_idx + mb * batch_size * 2
            if end_idx > len(GLOBAL_BATCH):
                new_global_batch.extend(GLOBAL_BATCH[star_idx:])
                continue
            else:
                new_data_chunk = balance_ping_pong_with_mb(
                    GLOBAL_BATCH[star_idx:end_idx],
                    mb=mb,
                    batch_size=batch_size,
                )
                new_global_batch.extend(new_data_chunk)
        GLOBAL_BATCH = new_global_batch
    
    GLOBAL_BATCH = iter(GLOBAL_BATCH)
    return


def get_next_batch(dp_size):
    global GLOBAL_BATCH
    global ITERATION_ID
    global iterated_samples
    
    batches = []
    if GLOBAL_BATCH is None:
        raise RuntimeError("GLOBAL_BATCH has not been initialized. Call setup_global_batch() first.")
        
    for _ in range(dp_size):    
        batches.append(next(GLOBAL_BATCH))
    ITERATION_ID += 1
    iterated_samples.append(batches)
    return batches