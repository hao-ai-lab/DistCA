# TODO: Write unit test for the WLBLLM Planner (no defer)


import rich
import os
K = 1024

def get_length(micro_batch: list[int]) -> int:
    return sum(micro_batch)
# Store warning state at module level
_warning_shown = False

def get_workload(micro_batch: list[int], model_config: dict | None = None) -> int:
    # TODO: Fix this get_workload function to calculate the `breakpoint` of a model.
    global _warning_shown
    attn_linear_breakpoint = 128 * K
    try:
        attn_linear_breakpoint = int(os.environ["ATTN_LINEAR_BREAKPOINT"])
    except Exception:
        pass
    if model_config is not None and not _warning_shown:
        print(f"⚠️ In `get_workload`, model_config is not used right now (becuase we have not implemented model config -> attn linear breakpoint yet). Constant is set to {attn_linear_breakpoint} for now.")
        _warning_shown = True
    a = [ i / (attn_linear_breakpoint) for i in micro_batch]
    return sum(i ** 2 + i for i in a)

def flatten(seq_lens: list) -> list:
    flattened = []
    for item in seq_lens:
        if isinstance(item, list):
            flattened.extend(flatten(item))
        else:
            flattened.append(item)
    return flattened

def balance_data_for_wlbllm(
    dp_size, dp_rank: int | list[int], total_seq_len, batch_size, 
    _seq_lens, 
    ENABLE_BALANCED_FLOS_NO_DEFER = True,
    model_config: dict | None = None,
):
    """Balance data across DP ranks for WLBLLM planner.
    
    Args:
        dp_size: Number of data parallel ranks
        dp_rank: Current data parallel rank(s)
        total_seq_len: Maximum sequence length
        batch_size: Global batch size
        rank: Global rank
        _seq_lens: Original sequence lengths
        
    Returns:
        seq_lens: Balanced sequence lengths for current rank
    """
    # Balance the data here for WLBLLM.
    # TODO: This only works for DP+CP.
    # ENABLE_BALANCED_FLOS_NO_DEFER = False
    if not ENABLE_BALANCED_FLOS_NO_DEFER:
        raise NotImplementedError(
            "Balance data for WLBLLM with defer is not implemented yet. "
            "The defer implementation requries a state to keep track of a window of requests."
        )
    
    if ENABLE_BALANCED_FLOS_NO_DEFER and dp_size > 1:
        # how many tokens per dp replicate (with cp) can hold?
        # max_seq_len_without_cp * cp_size * 2 (ping pong)
        # TODO: (Refactor) Be careful when refactoring this function! This function will cause performance regression!
        # The function introduces `batchsize` and `dpsize`, just to balance the tokens across differnet DP 
        #   (or DP*PP, depending on the outer function calls - if you don't understand, see WLBLLM paper 
        #   and understand how it works by taking DP * PP batches each time). 
        # TODO: Sorting / swapping batches may have perforamnce regression. Be careful of what you wish for.
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
            workloads = [get_workload(batch, model_config) for batch in new_batch]
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

        # rich.print(f"🟡 wlbllm replanning: new_batch = {new_batch}, new_workloads = {workloads}, new_lengths = {lengths}")

        # Check each batch and ensure it is not empty.
        for i, batch in enumerate(new_batch):
            if len(batch) == 0:
                pad = 128
                batch.append(pad) # Just append a padded doc.
                print(f"⚠️ wlbllm replanning: new_batch[{i}] is empty, padded with {pad}. This means some DP ranks are empty.")


        if isinstance(dp_rank, int):
            seq_lens = [new_batch[dp_rank]]
        else:
            seq_lens = [new_batch[rank] for rank in dp_rank]

    else:
        # Ensure this is just "one bucket" out.
        seq_lens = [flatten(_seq_lens)]
        new_batch = seq_lens
        
    return seq_lens, new_batch


