from typing import Iterable, List, Optional
from d2.simulator.optimizers.samples import sample_wlbllm_docs_upsample, batch_documents

ITERATION_ID = 0
iterated_samples = []
GLOBAL_BATCH: Optional[Iterable[List[int]]] = None

def setup_global_batch(
    total_seq_len, 
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
        ), max_ctx_length=total_seq_len
    )
    if should_add_debug_cases:
        GLOBAL_BATCH = list(GLOBAL_BATCH)
        manual_case = [
            [total_seq_len], [total_seq_len // 8] * 8,
            [total_seq_len], [total_seq_len // 8] * 8,
        ]
        GLOBAL_BATCH = manual_case * 4 + GLOBAL_BATCH
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