# TODO: Write unit test for the WLBLLM Planner (no defer)


import rich
K = 1024

def get_length(micro_batch: list[int]) -> int:
    return sum(micro_batch)

def get_workload(micro_batch: list[int]) -> int:
    # TODO: Fix this get_workload function to calculate the `breakpoint` of a model.
    a = [ i / (64 * K) for i in micro_batch]
    return sum(i ** 2 + i for i in a)

def flatten(seq_lens: list[list[int]]) -> list[int]:
    return [item for sublist in seq_lens for item in sublist]

def balance_data_for_wlbllm(
    dp_size, dp_rank, total_seq_len, batch_size, rank, 
    _seq_lens, 
    ENABLE_BALANCED_FLOS_NO_DEFER = True,
):
    """Balance data across DP ranks for WLBLLM planner.
    
    Args:
        dp_size: Number of data parallel ranks
        dp_rank: Current data parallel rank
        total_seq_len: Maximum sequence length
        batch_size: Global batch size
        rank: Global rank
        _seq_lens: Original sequence lengths
        K: Model constant for workload calculation
        
    Returns:
        seq_lens: Balanced sequence lengths for current rank
    """
    # Balance the data here for WLBLLM.
    # TODO: This only works for DP+CP.
    # ENABLE_BALANCED_FLOS_NO_DEFER = False
    
    if ENABLE_BALANCED_FLOS_NO_DEFER and dp_size > 1:
        # how many tokens per dp replicate (with cp) can hold?
        # max_seq_len_without_cp * cp_size * 2 (ping pong)
        Lmax = total_seq_len * 2 * batch_size // dp_size
        rich.print(f"🟡 Lmax = total_seq_len({total_seq_len}) * 2 * batch_size({batch_size}) // dp_size({dp_size}) = Lmax({Lmax})")

        all_docs = flatten(_seq_lens)
        all_docs.sort(reverse=True)
        new_batch = []
        for r in range(dp_size):
            new_batch.append([])
        
        # Step 1: Pack the docs into the new batch.
        remained_docs = []
        for doc in all_docs:
            workloads = [get_workload(batch) for batch in new_batch]
            lengths = [get_length(batch) for batch in new_batch]
            min_workload_idx = workloads.index(min(workloads))
            min_length_idx = lengths.index(min(lengths))
            
            if lengths[min_workload_idx] + doc <= Lmax:
                new_batch[min_workload_idx].append(doc)
            else:
                if lengths[min_length_idx] + doc <= Lmax:
                    new_batch[min_length_idx].append(doc)
                else:
                    remained_docs.append(doc)
            pass

        # Step 2: Pack the remained docs, by workload.
        for doc in remained_docs:
            workloads = [get_workload(batch) for batch in new_batch]
            lengths = [get_length(batch) for batch in new_batch]
            min_workload_idx = workloads.index(min(workloads))
            new_batch[min_workload_idx].append(doc)


        seq_lens = [new_batch[dp_rank]]

    else:
        seq_lens = _seq_lens
        new_batch = seq_lens
        
    return seq_lens, new_batch


