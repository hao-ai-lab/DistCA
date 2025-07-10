import wandb
import os
import json
import numpy as np
from typing import Deque, Iterable, List, Sequence
from collections import deque
from pathlib import Path
import matplotlib.pyplot as plt

data_folder = Path("data")
data_folder.absolute()


from samples import (
    batch_documents, 
    sample_wlbllm_docs_upsample,
)


def inspect_docdist(
    *,
    batch_size: int = 64 * 1024,
    upsample_long_factor: int = 2,
    sample_size: int = 2000,
    filter_threshold: int = 10000,
    filter_ratio: float = 1.0,
):
    docs = sample_wlbllm_docs_upsample(
        size=sample_size,
        upsample_long_factor=upsample_long_factor,
        filter_threshold=filter_threshold,
        filter_ratio=filter_ratio,
    )
    batches = batch_documents(docs, max_ctx_length=batch_size)
    batches = list(batches)

    run = wandb.init(
        project="doc-wlbllm-upsample",
        name=f"wlb-f{upsample_long_factor}-bs{batch_size}-ss{sample_size}",
    )

    run.config.update({
        "upsample_long_factor": upsample_long_factor,
        "batch_size": batch_size,
        "sample_size": sample_size,
        "upsample_long_factor": upsample_long_factor,
    })

    table = wandb.Table(columns=[
        "long_doc_count", "short_doc_count", "long_doc_tokens", "short_doc_tokens", 
        "long_doc_count_over_all_ratio", "long_doc_tokens_over_all_ratio", 
        "batch_size", "total_tokens", "batch"
    ])

    for batch in batches:
        print(batch)
        long_doc_count = sum(1 for doc in batch if doc >= 10000)
        short_doc_count = sum(1 for doc in batch if doc < 10000)
        long_doc_tokens = sum(doc for doc in batch if doc >= 10000)
        short_doc_tokens = sum(doc for doc in batch if doc < 10000)
        long_doc_count_over_all_ratio = (long_doc_count) / (long_doc_count + short_doc_count)
        long_doc_tokens_over_all_ratio = (long_doc_tokens) / (long_doc_tokens + short_doc_tokens)
        total_tokens = sum(batch)
        run.log({
            "long_doc_count": long_doc_count,
            "short_doc_count": short_doc_count,
            "long_doc_tokens": long_doc_tokens,
            "short_doc_tokens": short_doc_tokens,
            "long_doc_count_over_all_ratio": long_doc_count_over_all_ratio,
            "long_doc_tokens_over_all_ratio": long_doc_tokens_over_all_ratio,
            "batch_size": len(batch),
            "total_tokens": total_tokens,
        })
        table.add_data(long_doc_count, short_doc_count, long_doc_tokens, short_doc_tokens, 
                    long_doc_count_over_all_ratio, long_doc_tokens_over_all_ratio, 
                    len(batch), total_tokens, batch)


    run.log({"table": table})

    short_doc_count = len([doc for doc in docs if doc < filter_threshold])
    long_doc_count = len([doc for doc in docs if doc >= filter_threshold])
    short_doc_tokens = sum([doc for doc in docs if doc < filter_threshold])
    long_doc_tokens = sum([doc for doc in docs if doc >= filter_threshold])

    doc_ratio = 100 * long_doc_count / (short_doc_count + long_doc_count)
    token_ratio = 100 * long_doc_tokens / (short_doc_tokens + long_doc_tokens)
    run.summary.update({
        "long_doc_count": long_doc_count,
        "short_doc_count": short_doc_count,
        "long_doc_tokens": long_doc_tokens,
        "short_doc_tokens": short_doc_tokens,
        "long_doc_count_over_all_ratio": doc_ratio,
        "long_doc_tokens_over_all_ratio": token_ratio,
    })
    run.finish()
    return


def main():
    for batch_size in [64 * 1024, 128 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024]:
        for upsample_long_factor in [1, 2, 4, 8, 16, 32, 64]:
            print(f"Starting WLB upsampling inspection: batch_size: {batch_size}, upsample_long_factor: {upsample_long_factor}")
            inspect_docdist(
                batch_size=batch_size,
                upsample_long_factor=upsample_long_factor,
            )
    pass

if __name__ == "__main__":
    main()