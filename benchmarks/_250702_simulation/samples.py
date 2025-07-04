import os
import json
import numpy as np
from typing import Deque, Iterable, List, Sequence
from collections import deque
from pathlib import Path


def sample_random_docs(
    *,
    max_ctx_length: int,
    size: int,
    seed: int = 42
) -> list[int]:
    """
    Create a deque of `size` random document lengths in (1, max_ctx_length].

    Parameters
    ----------
    max_ctx_length : int
        Upper bound for any single-document length.
    size : int
        Number of documents to generate.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    deque[int]
        A deque containing the generated document lengths.
    """
    rng = np.random.default_rng(seed)
    docs = rng.integers(1, max_ctx_length + 1, size=size)
    docs = np.abs(docs).astype(int)
    return docs.tolist()


def sample_multimodal_gaussian(
    means:   Sequence[float],
    sigmas:  Sequence[float],
    weights: Sequence[float],
    size:    int,
    seed:    int = 42
) -> list[int]:
    """
    Draw `size` samples from a mixture of k Gaussians.

    Parameters
    ----------
    means : sequence of float
        The mean (μ) of each component.
    sigmas : sequence of float
        The standard deviation (σ) of each component.
    weights : sequence of float
        The (non-negative) weight of each component. Need not sum to 1.
    size : int
        Number of samples to draw.
    seed : int, optional
        RNG seed.

    Returns
    -------
    np.ndarray
        Samples of shape (size,), clipped to ≥0.
    """
    assert len(means) == len(sigmas) == len(weights), "lengths must match"
    w = np.array(weights, dtype=float)
    p = w / w.sum()

    rng = np.random.default_rng(seed)
    # 1) Choose a component index for each sample
    components = rng.choice(len(means), size=size, p=p)

    # 2) For each component, draw as many normals as were assigned to it
    samples = np.empty(size, dtype=float)
    for k, (μ, q) in enumerate(zip(means, sigmas)):
        idx = components == k
        if np.any(idx):
            samples[idx] = rng.normal(loc=μ, scale=q, size=idx.sum())

    # 3) If negative values don't make sense, clip at 0
    docs = samples.clip(min=0).astype(int)
    docs = np.abs(docs).astype(int)
    return docs.tolist()


def sample_wlbllm_docs(
    *,
    size: int,
    seed: int = 42
) -> list[int]:
    """
    Sample `size` documents from the WLB-LLM distribution.

    Parameters
    ----------
    size : int
        Number of documents to sample.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    list[int]
        A list of document lengths.
    """
    thisdir = Path(os.path.dirname(__file__))
    docpath = thisdir.parent / "data" / "dist_wlbllm.json"
    with open(docpath, "r") as f:
        docs = json.load(f)
    rng = np.random.default_rng(seed)
    docs = rng.choice(docs, size=size, replace=False)
    return docs.tolist()


def sample_wlbllm_docs_altered(
    *,
    size: int,
    seed: int = 42,
    filter_threshold: int = 10000,
    filter_ratio: float = 0.1,
) -> list[int]:
    """
    Sample `size` documents from the WLB-LLM distribution.

    Parameters
    ----------
    size : int
        Number of documents to sample.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    list[int]
        A list of document lengths.
    """
    thisdir = Path(os.path.dirname(__file__))
    docpath = thisdir.parent / "data" / "dist_wlbllm.json"
    with open(docpath, "r") as f:
        docs = json.load(f)
    shorter_docs = [doc for doc in docs if doc < filter_threshold]
    longer_docs = [doc for doc in docs if doc >= filter_threshold]
    
    docs = shorter_docs[:int(filter_ratio * len(shorter_docs))] + longer_docs

    rng = np.random.default_rng(seed)
    docs = rng.choice(docs, size=size, replace=False)
    return docs.tolist()


def batch_documents(
    docs: Sequence[int],
    *,
    max_ctx_length: int
) -> Iterable[List[int]]:
    """
    Yield batches whose token-sums never exceed `max_ctx_length`.
    If a document would overflow the current batch, split it
    (carry the remainder forward).

    Parameters
    ----------
    docs : deque[int]
        Document lengths (will be consumed in-place).
    max_ctx_length : int
        Capacity of each batch.

    Yields
    ------
    List[int]
        A batch (list of token counts) whose sum ≤ `max_ctx_length`.
        All batches except possibly the last have sum == `max_ctx_length`.
    """
    batch: List[int] = []

    if not isinstance(docs, deque):
        docs = deque(docs)

    while docs:
        doc = docs.popleft()

        while doc:                                  # may need multiple splits
            space_left = max_ctx_length - sum(batch)

            # If it fits, drop it in and move on
            if doc <= space_left:
                batch.append(doc)
                doc = 0
            else:
                # Fill the batch, yield it, and keep the remainder
                if space_left:                      # avoid 0-length chunks
                    batch.append(space_left)
                    doc -= space_left
                yield batch
                batch = []

    if batch:                                       # flush tail
        yield batch

