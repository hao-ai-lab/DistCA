# %%

from d2.simulator.optimizers.samples import (
    sample_wlbllm_docs_altered,
    sample_multimodal_gaussian,
    sample_random_docs,
    sample_wlbllm_docs,
    batch_documents,
)

# %%

docs = sample_wlbllm_docs_altered(
    size = 10000,
    filter_threshold=10000,
    filter_ratio=0.09,
)
import matplotlib.pyplot as plt
plt.hist(docs, bins=100, edgecolor='black')
plt.show()

# %%
K = 1024
batches = list(batch_documents(
    docs, 
    max_ctx_length=64 * K * 4,
))

# %%
i = 5
len(batches[i]), batches[i]


# %%



# %%
