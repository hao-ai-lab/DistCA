# WLBLLM Optimal Baseline

This is a low-fidelity (more optimized but not accurate implementation) reproduction of the WLBLLM Optimal baseline.

# Quick Start


```bash
# You may want to have the wlbllm_optimal package available in the shared file system.
pip install -e .
bash run.sh
```

# Implementation

## `pretrain_gpt.py` 

The main entry point. There includes a few hacks:
- `get_batch()`: hack to intercept the data loader and get a batch to forward. This function will get the appropriate (tp, cp)-tuned batch for the worker.
- `prepare_packed_seq_params()`: Prepare the metadata for the sequence parameter object.

## `context_parallel.py`

The package to handle the context parallel strategy (and kv) for the context parallel for WLBLLM.