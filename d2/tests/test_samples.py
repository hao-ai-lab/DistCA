# import matplotlib.pyplot as plt
from d2.simulator.optimizers.samples import (
    sample_wlbllm_docs_altered,
    sample_multimodal_gaussian,
    sample_random_docs,
    sample_wlbllm_docs,
    batch_documents,
)
# for i in [64,128,256,512,1024,2048]:
#     docs = sample_wlbllm_docs_altered(size=i)
#     plt.hist(docs, bins=100, edgecolor='black')
#     plt.show()

import os
from pathlib import Path
import matplotlib.pyplot as plt
from d2.simulator.optimizers.samples import (
    sample_wlbllm_docs_altered,
    sample_multimodal_gaussian,
    sample_random_docs,
    sample_wlbllm_docs,
    batch_documents,
)
K = 1024


this_dir = Path(__file__).parent

def make_flat_batches(batches):
    return [x for xs in batches for x in xs]


def plot_distributions(docs, max_seq_length:int = 64 * K):
    batches = list(batch_documents(docs, max_ctx_length=max_seq_length))
    flatted_batches = make_flat_batches(batches)

    fig = plt.figure(figsize=(10, 8))

    # Upper subplot for docs
    plt.subplot(2, 1, 1)
    plt.hist(docs, bins=100, edgecolor='black')
    plt.title('Distribution of Docs')
    plt.xlabel('Document Length')
    plt.ylabel('Frequency')

    # Lower subplot for flatted_batches
    plt.subplot(2, 1, 2)
    plt.hist(flatted_batches, bins=100, edgecolor='black')
    plt.title('Distribution of Flatted Batches')
    plt.xlabel('Batch Length')
    plt.ylabel('Frequency')

    plt.tight_layout()
    return fig
