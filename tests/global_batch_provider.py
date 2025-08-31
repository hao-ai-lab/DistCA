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


_dp_rank = 0
_dp_size = 1
def set_dp_size_and_rank(rank: int, size: int):
    global _dp_size
    global _dp_rank
    _dp_size = size
    _dp_rank = rank
    print(f"Set dp size and rank: {_dp_size = }, {_dp_rank = }")
    return


def get_next_batch(num_batches: int) -> List[int]:
    global GLOBAL_BATCH
    global ITERATION_ID
    global iterated_samples
    global _dp_rank
    global _dp_size

    if GLOBAL_BATCH is None:
        raise RuntimeError("GLOBAL_BATCH has not been initialized. Call setup_global_batch() first.")

    all_batches: List[List[int]] = []
    for _ in range(_dp_size):
        batches: List[int] = []
        for _ in range(num_batches):
            batches.append(next(GLOBAL_BATCH))
        all_batches.append(batches)
    print(f"{all_batches = }")
    print(f"{len(all_batches) = }")

    batches = all_batches[_dp_rank]

    ITERATION_ID += 1
    iterated_samples.append(batches)
    return batches


def get_iterated_samples():
    global iterated_samples
    return iterated_samples