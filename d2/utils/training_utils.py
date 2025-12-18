"""
Training utilities for data loading and batch management.

This module contains all data loading, tokenization, and batch preparation logic
for the D2 training pipeline.
"""

from typing import Iterable, Iterator, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import torch
from d2.simulator.optimizers.samples import sample_wlbllm_docs_upsample, batch_documents, sample_prolong_docs
from d2.planner.planner import Item


@dataclass
class BatchEntry:
    """Represents a single batch entry with sequence lengths and optional token chunks."""
    lengths: List[int]
    token_chunks: Optional[List[torch.Tensor]] = None


@dataclass
class SequenceRecord:
    """Records tokens and positions for a sequence, with a cursor for incremental reading."""
    tokens: torch.Tensor
    positions: torch.Tensor
    cursor: int = 0

    def take(self, length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Take the next `length` tokens and positions from this sequence.
        
        Parameters
        ----------
        length : int
            Number of tokens to take
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Token slice and position slice
        """
        end = self.cursor + length
        if end > self.tokens.size(0):
            raise RuntimeError(
                f"Requested {length} tokens but only {self.tokens.size(0) - self.cursor} remain."
            )
        token_slice = self.tokens[self.cursor:end]
        position_slice = self.positions[self.cursor:end]
        self.cursor = end
        return token_slice, position_slice


# Global state for batch generation and tokenization
ITERATION_ID = 0
GLOBAL_BATCH: Optional[Iterable[BatchEntry]] = None
GLOBAL_TOKENIZED_DATA: Optional[List[torch.Tensor]] = None  # Store tokenized sequences
TOKENIZER = None  # Store tokenizer
TOKEN_PAD_ID = 0
GLOBAL_DOC_INDEX = 0
GLOBAL_DOC_OFFSET = 0


def _set_tokenizer_reference(tokenizer):
    """Set the global tokenizer reference and determine the pad token ID."""
    global TOKENIZER, TOKEN_PAD_ID
    TOKENIZER = tokenizer
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None or pad_id < 0:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None or pad_id < 0:
        pad_id = 0
    TOKEN_PAD_ID = int(pad_id)


def _reset_token_cursor():
    """Reset the global document cursor to the beginning."""
    global GLOBAL_DOC_INDEX, GLOBAL_DOC_OFFSET
    GLOBAL_DOC_INDEX = 0
    GLOBAL_DOC_OFFSET = 0


def lazy_token_iterator_from_texts(
    texts: Iterable[str],
    tokenizer,
    max_samples: int = 10000,
) -> Iterator[torch.Tensor]:
    """
    Lazily tokenize an iterable of raw text strings.

    This is a dependency-light, streaming-friendly iterator:
      - Does NOT store all tokenized sequences in memory.
      - Tokenizes on-the-fly as you iterate.
      - Skips empty/whitespace-only strings.

    Parameters
    ----------
    texts : Iterable[str]
        Iterable of raw text strings.
    tokenizer :
        Any object with an `encode(text, add_special_tokens=True, truncation=False)` method
        (e.g., a HuggingFace tokenizer or a simple dummy tokenizer in tests).
    max_samples : int
        Maximum number of non-empty samples to yield.

    Yields
    ------
    torch.Tensor
        1D tensor of token IDs for each non-empty input text.
    """
    count = 0
    for text in texts:
        if count >= max_samples:
            break
        if not isinstance(text, str):
            continue
        stripped = text.strip()
        if not stripped:
            continue
        tokens = tokenizer.encode(stripped, add_special_tokens=True, truncation=False)
        if not tokens:
            continue
        yield torch.tensor(tokens, dtype=torch.long)
        count += 1


def _advance_doc_cursor():
    """Move to the next document in the tokenized dataset."""
    global GLOBAL_DOC_INDEX, GLOBAL_DOC_OFFSET
    if not GLOBAL_TOKENIZED_DATA:
        raise RuntimeError("Tokenized dataset is empty.")
    GLOBAL_DOC_INDEX = (GLOBAL_DOC_INDEX + 1) % len(GLOBAL_TOKENIZED_DATA)
    GLOBAL_DOC_OFFSET = 0


def _next_token_chunk(target_len: int) -> torch.Tensor:
    """
    Extract the next chunk of tokens from the global tokenized dataset.
    
    This function handles reading tokens across document boundaries and
    maintains the global cursor position.
    
    Parameters
    ----------
    target_len : int
        Number of tokens to extract
        
    Returns
    -------
    torch.Tensor
        Tensor of token IDs with shape (target_len,)
    """
    if GLOBAL_TOKENIZED_DATA is None:
        raise RuntimeError("Real dataset tokens are not initialized.")
    if target_len <= 0:
        return torch.empty(0, dtype=torch.long)
    global GLOBAL_DOC_INDEX, GLOBAL_DOC_OFFSET
    chunks: List[torch.Tensor] = []
    remaining = target_len
    while remaining > 0:
        if not GLOBAL_TOKENIZED_DATA:
            raise RuntimeError("Tokenized dataset is empty.")
        doc = GLOBAL_TOKENIZED_DATA[GLOBAL_DOC_INDEX]
        if doc.numel() == 0:
            _advance_doc_cursor()
            continue
        available = doc.numel() - GLOBAL_DOC_OFFSET
        if available <= 0:
            _advance_doc_cursor()
            continue
        take = min(available, remaining)
        chunks.append(doc[GLOBAL_DOC_OFFSET:GLOBAL_DOC_OFFSET + take])
        GLOBAL_DOC_OFFSET += take
        remaining -= take
        if GLOBAL_DOC_OFFSET == doc.numel():
            _advance_doc_cursor()
    if len(chunks) == 1:
        return chunks[0].clone()
    return torch.cat(chunks, dim=0)


def _build_real_batch_generator(doc_iter: Iterable[List[int]]):
    """
    Build a batch generator that extracts real tokens from the dataset.
    
    Parameters
    ----------
    doc_iter : Iterable[List[int]]
        Iterator over batches of document lengths
        
    Returns
    -------
    Generator
        Generator yielding BatchEntry objects with real tokens
    """
    def generator():
        for batch in doc_iter:
            lengths = [int(val) for val in batch]
            token_chunks = [_next_token_chunk(length) for length in lengths]
            yield BatchEntry(lengths=lengths, token_chunks=token_chunks)
    return generator()


def _build_synthetic_batch_generator(doc_iter: Iterable[List[int]]):
    """
    Build a batch generator for synthetic data (no real tokens).
    
    Parameters
    ----------
    doc_iter : Iterable[List[int]]
        Iterator over batches of document lengths
        
    Returns
    -------
    Generator
        Generator yielding BatchEntry objects without tokens
    """
    def generator():
        for batch in doc_iter:
            lengths = [int(val) for val in batch]
            yield BatchEntry(lengths=lengths, token_chunks=None)
    return generator()


def load_and_tokenize_dataset(
    dataset_name: str,
    tokenizer,
    seed: int = 42,
    max_total_tokens: Optional[int] = None,
) -> List[torch.Tensor]:
    """
    Load a dataset and tokenize it.

    Currently supports:
      - 'wikitext'      → `wikitext`, config 'wikitext-103-v1', split 'train'
      - 'allenai/c4' or 'c4' → `allenai/c4`, config 'en', split 'train'

    The interface is kept generic, but we only route these two names.
    """
    from datasets import load_dataset
    import numpy as np

    print(f"Loading dataset: {dataset_name}...")

    # Route dataset_name to specific HuggingFace datasets
    if dataset_name == "wikitext":
        hf_name = "wikitext"
        hf_config = "wikitext-103-v1"
        hf_split = "train"
        is_streaming = False  # small enough to load normally
        text_column = "text"
    elif dataset_name in ("allenai/c4", "c4"):
        hf_name = "allenai/c4"
        hf_config = "en"
        hf_split = "train"
        is_streaming = True   # very large; use streaming
        text_column = "text"
    else:
        raise ValueError(
            f"Unsupported dataset_name '{dataset_name}'. "
            f"Only 'wikitext' and 'allenai/c4' (or 'c4') are supported here."
        )

    # Load dataset
    print(f"Loading dataset '{hf_name}' with config '{hf_config}' and split '{hf_split}'...")
    if is_streaming:
        dataset = load_dataset(hf_name, hf_config, split=hf_split, streaming=True)
    else:
        dataset = load_dataset(hf_name, hf_config, split=hf_split)

    tokenized_sequences: List[torch.Tensor] = []
    rng = np.random.default_rng(seed)
    total_tokens = 0  # track total tokens yielded for optional capping
    
    print("Starting loading dataset...")
    if is_streaming:
        # Streaming: process examples sequentially, respecting both max_total_tokens
        print(f"Tokenizing streaming dataset '{hf_name}'...")
        for i, example in enumerate(dataset):
            if max_total_tokens is not None and total_tokens >= max_total_tokens:
                print(f"Reached max_total_tokens={max_total_tokens} at sample_id={i}, total_tokens={total_tokens}")
                break
            text = example.get(text_column)
            if not (isinstance(text, str) and text.strip()):
                continue
            tokens = tokenizer.encode(text, add_special_tokens=True, truncation=False)
            if not tokens:
                continue
            # Enforce global token budget if set
            if max_total_tokens is not None:
                remaining = max_total_tokens - total_tokens
                if remaining <= 0:
                    print(f"Reached max_total_tokens={max_total_tokens} at sample_id={i}, total_tokens={total_tokens}")
                    break
                if len(tokens) > remaining:
                    tokens = tokens[:remaining]
            token_tensor = torch.tensor(tokens, dtype=torch.long)
            tokenized_sequences.append(token_tensor)
            total_tokens += token_tensor.numel()
            if (i + 1) % 1000 == 0:
                print(f"  Tokenized {i + 1} samples, {len(tokenized_sequences)} sequences, total_tokens={total_tokens}")
    else:
        # Non-streaming: sample randomly, but still respect max_total_tokens if provided
        dataset_size = len(dataset)
        indices = rng.choice(dataset_size, size=dataset_size, replace=False)
        print(f"Tokenizing {dataset_size} randomly sampled documents from {dataset_size} total in '{hf_name}'...")
        for idx, i in enumerate(indices):
            if max_total_tokens is not None and total_tokens >= max_total_tokens:
                print(f"Reached max_total_tokens={max_total_tokens} at sample_id={idx}, total_tokens={total_tokens}")
                break
            example = dataset[int(i)]
            text = example.get(text_column)
            if not (isinstance(text, str) and text.strip()):
                continue
            tokens = tokenizer.encode(text, add_special_tokens=True, truncation=False)
            if not tokens:
                continue
            if max_total_tokens is not None:
                remaining = max_total_tokens - total_tokens
                if remaining <= 0:
                    break
                if len(tokens) > remaining:
                    tokens = tokens[:remaining]
            token_tensor = torch.tensor(tokens, dtype=torch.long)
            tokenized_sequences.append(token_tensor)
            total_tokens += token_tensor.numel()
            if (idx + 1) % 1000 == 0:
                print(f"  Tokenized {idx + 1}/{dataset_size} samples, total_tokens={total_tokens}")
    
    print(
        f"🟢 Loaded and tokenized {len(tokenized_sequences)} sequences "
        f"from '{hf_name}' (total_tokens={total_tokens})"
    )
    return tokenized_sequences


def setup_global_batch(
    total_seq_len, 
    up_sample_factor=2,
    elongate_factor=1,
    filter_threshold=64 * 1024,
    filter_ratio=0.90,
    should_add_debug_cases=False,
    change_long_doc_ratio=0.0,
    sample_name='wlbllm',
    tokenizer=None,
    max_total_tokens: Optional[int] = None,
):
    """
    Setup global batch data for training.
    
    This function can either:
    1. Load synthetic sequence length distributions ('wlbllm', 'prolong')
    2. Load real datasets with actual tokens ('bookcorpus', 'wikitext', 'openwebtext', 'c4')
    
    Parameters
    ----------
    total_seq_len : int
        Maximum sequence length per batch
    up_sample_factor : int
        Up-sampling factor for long sequences
    elongate_factor : int
        Factor to elongate sequences
    filter_threshold : int
        Threshold for filtering long sequences
    filter_ratio : float
        Ratio for filtering sequences
    should_add_debug_cases : bool
        Whether to add debug cases (not supported)
    change_long_doc_ratio : float
        Ratio for changing long documents
    sample_name : str
        Dataset name. Options:
        - 'wlbllm', 'prolong': Synthetic sequence length distributions (no real tokens)
        - 'bookcorpus', 'wikitext', 'openwebtext', 'c4': Real datasets with actual tokens
    tokenizer : AutoTokenizer, optional
        Required when using real datasets to tokenize the text
        
    Usage Example
    -------------
    # To use with a real dataset like bookcorpus:
    # python training_3d.py --sample-name bookcorpus ...
    
    # To use with synthetic data (default):
    # python training_3d.py --sample-name wlbllm ...
    """
    global GLOBAL_BATCH
    global GLOBAL_TOKENIZED_DATA
    global TOKENIZER
    
    if GLOBAL_BATCH is not None:
        return

    assert elongate_factor > 0, f"elongate_factor: {elongate_factor} must be greater than 0"

    # Check if we're loading a real dataset
    real_datasets = ['bookcorpus', 'wikitext', 'openwebtext', 'c4']
    is_real_dataset = sample_name not in ['wlbllm', 'prolong']
    
    if is_real_dataset:
        # Load and tokenize real dataset
        if tokenizer is None:
            raise ValueError(f"Tokenizer must be provided when using dataset: {sample_name}")
        
        _set_tokenizer_reference(tokenizer)
        _reset_token_cursor()
        GLOBAL_TOKENIZED_DATA = load_and_tokenize_dataset(
            dataset_name=sample_name,
            tokenizer=tokenizer,
            seed=42,
            max_total_tokens=max_total_tokens,
        )
        
        # Get document lengths from tokenized data
        doc_lengths = [len(seq) for seq in GLOBAL_TOKENIZED_DATA]
        
        # Create batches based on actual token lengths
        doc_iter = batch_documents(doc_lengths, max_ctx_length=total_seq_len)
        GLOBAL_BATCH = _build_real_batch_generator(doc_iter)
        
    elif sample_name == 'wlbllm':
        sample_func = sample_wlbllm_docs_upsample
        doc_iter = batch_documents(
            sample_func(
                size=10000,
                filter_threshold=filter_threshold,
                filter_ratio=filter_ratio,
                upsample_long_factor=up_sample_factor,
                elongate_factor=elongate_factor,
                change_long_doc_ratio=change_long_doc_ratio,
            ), max_ctx_length=total_seq_len
        )
        GLOBAL_BATCH = _build_synthetic_batch_generator(doc_iter)
    elif sample_name == 'prolong':
        sample_func = sample_prolong_docs
        doc_iter = batch_documents(
            sample_func(
                size=10000,
                filter_threshold=filter_threshold,
                filter_ratio=filter_ratio,
                upsample_long_factor=up_sample_factor,
                elongate_factor=elongate_factor,
                change_long_doc_ratio=change_long_doc_ratio,
            ), max_ctx_length=total_seq_len
        )
        GLOBAL_BATCH = _build_synthetic_batch_generator(doc_iter)
    else:
        raise ValueError(f"Invalid sample_name: {sample_name}")
    
    if should_add_debug_cases:
        raise NotImplementedError("Debug cases are not supported with the new batch generator.")
    return


def get_next_batch(dp_size, iterated_samples) -> Tuple[List[List[int]], List[Optional[List[torch.Tensor]]]]:
    """
    Get next batch of sequence lengths and corresponding tokens.
    
    Parameters
    ----------
    dp_size : int
        Data parallel size (number of batches to get)
    iterated_samples : list
        List to append the batches to (for tracking)
    
    Returns
    -------
    Tuple[List[List[int]], List[Optional[List[torch.Tensor]]]]
        - List of batch structures (sequence lengths)
        - List of token tensors for each batch (or None for synthetic data)
    """
    global GLOBAL_BATCH
    global ITERATION_ID
    
    # get dp_size number of batches 
    batches = []
    batch_tokens = []
    
    for _ in range(dp_size):    
        batch_entry = next(GLOBAL_BATCH)
        batches.append(batch_entry.lengths)
        batch_tokens.append(batch_entry.token_chunks)
    
    ITERATION_ID += 1
    iterated_samples.append(batches)
    return batches, batch_tokens


def build_sequence_records(
    seq_lens_half: List[List[int]],
    batch_tokens_half: List[Optional[List[torch.Tensor]]],
) -> List[SequenceRecord]:
    """
    Build sequence records from sequence lengths and token chunks.
    
    Parameters
    ----------
    seq_lens_half : List[List[int]]
        List of batches, each containing sequence lengths
    batch_tokens_half : List[Optional[List[torch.Tensor]]]
        List of token chunks for each batch
        
    Returns
    -------
    List[SequenceRecord]
        List of sequence records with tokens and positions
    """
    if any(tokens is None for tokens in batch_tokens_half):
        raise RuntimeError("Real dataset tokens are required for all sequences.")
    sequence_records: List[SequenceRecord] = []
    for lengths, tokens in zip(seq_lens_half, batch_tokens_half):
        assert tokens is not None
        if len(lengths) != len(tokens):
            raise RuntimeError("Mismatch between lengths and token chunks.")
        for length, chunk in zip(lengths, tokens):
            chunk = chunk.to(dtype=torch.long).clone()
            if chunk.numel() < length:
                pad = torch.full(
                    (length - chunk.numel(),),
                    TOKEN_PAD_ID,
                    dtype=torch.long,
                )
                chunk = torch.cat([chunk, pad], dim=0)
            elif chunk.numel() > length:
                chunk = chunk[:length]
            positions = torch.arange(chunk.size(0), dtype=torch.long)
            sequence_records.append(SequenceRecord(tokens=chunk, positions=positions))
    return sequence_records


def build_rank_shards(
    items: List[Item],
    sequence_records: List[SequenceRecord],
    num_ranks: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Build token and position shards for each rank based on planner items.
    
    This function distributes tokens and positions to ranks according to the
    planner's scheduling decisions (stored in Item objects).
    
    Parameters
    ----------
    items : List[Item]
        List of planned items from the planner
    sequence_records : List[SequenceRecord]
        List of sequence records containing tokens and positions
    num_ranks : int
        Number of ranks (attention server world size)
        
    Returns
    -------
    Tuple[List[torch.Tensor], List[torch.Tensor]]
        - List of token tensors, one per rank
        - List of position tensors, one per rank
    """
    items_by_src = defaultdict(list)
    for item in items:
        items_by_src[item.src_gpuid].append(item)

    tokens_per_rank: List[torch.Tensor] = []
    positions_per_rank: List[torch.Tensor] = []

    for rank in range(num_ranks):
        rank_items = items_by_src.get(rank, [])
        rank_items.sort(key=lambda x: x.seqid)
        token_chunks: List[torch.Tensor] = []
        position_chunks: List[torch.Tensor] = []
        for item in rank_items:
            if item.complete:
                chunk_lengths = [int(item.complete_item["q"])]
            else:
                chunk_lengths = [int(item.head["q"]), int(item.tail["q"])]
            for chunk_len in chunk_lengths:
                tokens_slice, pos_slice = sequence_records[item.seqid].take(chunk_len)
                token_chunks.append(tokens_slice)
                position_chunks.append(pos_slice)
        if token_chunks:
            tokens_per_rank.append(torch.cat(token_chunks, dim=0))
            positions_per_rank.append(torch.cat(position_chunks, dim=0))
        else:
            tokens_per_rank.append(torch.empty(0, dtype=torch.long))
            positions_per_rank.append(torch.empty(0, dtype=torch.long))
    return tokens_per_rank, positions_per_rank

