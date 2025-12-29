"""Mock dataset utilities for DistCA GPT training.

This module provides:
1. DistCAMockGPTLowLevelDataset - A mock low-level dataset that generates tokenized data
2. DistCAMockGPTDataset - A mock GPT dataset wrapper
3. train_valid_test_datasets_provider - Dataset provider for train/valid/test splits
4. GlobalBatchPeeker - A utility to inspect global batch layout from rank 0
"""

import numpy
import logging
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

from megatron.training.tokenizer.tokenizer import MegatronTokenizer
from megatron.core.datasets.utils import Split
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.training.global_vars import get_args
from megatron.training import get_tokenizer
from megatron.training.utils import get_blend_and_blend_per_split
from megatron.core import parallel_state as mpu
from megatron.core.num_microbatches_calculator import get_num_microbatches

logger = logging.getLogger(__name__)


class DistCAMockGPTLowLevelDataset:
    """The mock GPT low level dataset

    This class is meant to generate tokenized data in the classic "Megatron-LM" GPT style. Notably,
    we add the end of document token to each element indexed in __getitem__

    Args:
        tokenizer (MegatronTokenizer): The tokenizer the special token information of which we use
            to augment the mock data.
    """

    seed: int = 0
    """The hard-coded random seed to use to set the NumPy RNG"""

    size: int = 100000
    """The hard-coded number of samples to generate"""

    max_sequence_length: int = 512
    """The hard-coded max sequence length to generate"""

    def __init__(self, tokenizer: MegatronTokenizer) -> None:
        self.tokenizer = tokenizer
        rng = numpy.random.default_rng(seed=self.seed)
        self.sequence_lengths = rng.integers(
            low=1, high=self.max_sequence_length, 
            size=self.size, dtype=numpy.int32
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> numpy.number:
        length = self.sequence_lengths[idx]
        sample = numpy.int64(
            numpy.concatenate([numpy.arange(length - 1) + 1, [self.tokenizer.eod]])
        )
        return sample

    def get(self, idx: int, offset: int = 0, length: Optional[int] = None) -> numpy.ndarray:
        """This function is an abstraction over __getitem__ with support for slicing

        Args:
            idx (int): The index into the dataset

            offset (int): The integer token offset in the sequence

            length (Optional[int]): The number of tokens to grab from the sequence

        Returns:
            numpy.ndarray: The sequence tokens at the index
        """
        if length is None:
            length = self.sequence_lengths[idx] - offset
        return self[idx][offset : offset + length]


class DistCAMockGPTDataset(GPTDataset):
    """The mock GPT dataset

    Args:
        indexed_dataset (MockGPTLowLevelDataset): The MockGPTLowLevelDataset around which to build
            the MockGPTDataset

        dataset_path (Optional[str]): This argument is of no consequence for the MockGPTDataset

        indices (numpy.ndarray): The set of the dataset indices to expose

        num_samples (int): The number of samples to draw from the dataset

        index_split (Split): The indices Split

        config (GPTDatasetConfig): The config
    """

    def __init__(
        self,
        dataset: DistCAMockGPTLowLevelDataset,
        dataset_path: Optional[str],
        indices: numpy.ndarray,
        num_samples: int,
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        assert config.mock

        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: DistCAMockGPTLowLevelDataset) -> int:
        """Abstract method implementation

        Args:
            low_level_dataset (DistCAMockGPTLowLevelDataset): The underlying DistCAMockGPTLowLevelDataset

        Returns:
            int: The number of unique elements in the underlying DistCAMockGPTLowLevelDataset
        """
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(
        dataset_path: Optional[str], config: GPTDatasetConfig
    ) -> DistCAMockGPTLowLevelDataset:
        """Abstract method implementation

        Args:
            dataset_path (Optional[str]): This argument is of no consequence for the
                DistCAMockGPTLowLevelDataset

            config (GPTDatasetConfig): The config

        Returns:
            DistCAMockGPTLowLevelDataset: The underlying DistCAMockGPTLowLevelDataset
        """
        return DistCAMockGPTLowLevelDataset(config.tokenizer)


def core_gpt_dataset_config_from_args(args):
    """Adapt from pretrain_gpt.py"""
    tokenizer = get_tokenizer()

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        s3_cache_path=args.s3_cache_path,
    )


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)
    logger.info(f"> GPTDatasetConfig: {config}")

    if args.mock_data:
        dataset_type = DistCAMockGPTDataset
    else:
        dataset_type = GPTDataset

    logger.info("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    logger.info("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


# Mark as distributed for Megatron
train_valid_test_datasets_provider.is_distributed = True


@dataclass
class DocumentInfo:
    """Information about a document within a sample."""
    doc_id: int
    start_offset: int  # Token offset within the sample
    length: int  # Number of tokens from this document


@dataclass  
class SampleInfo:
    """Information about a single sample in the dataset."""
    sample_idx: int
    total_tokens: int
    documents: List[DocumentInfo]
    
    def __repr__(self):
        doc_lens = [d.length for d in self.documents]
        return f"Sample(idx={self.sample_idx}, tokens={self.total_tokens}, docs={len(self.documents)}, doc_lens={doc_lens})"


@dataclass
class MicrobatchInfo:
    """Information about a single microbatch."""
    microbatch_idx: int
    dp_rank: int
    samples: List[SampleInfo]
    
    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.samples)
    
    @property
    def num_documents(self) -> int:
        return sum(len(s.documents) for s in self.samples)
    
    def __repr__(self):
        sample_lens = [s.total_tokens for s in self.samples]
        return f"Microbatch(idx={self.microbatch_idx}, dp_rank={self.dp_rank}, samples={len(self.samples)}, sample_lens={sample_lens})"


@dataclass
class GlobalBatchLayout:
    """Layout of a global batch across all DP ranks and microbatches."""
    iteration: int
    num_microbatches: int
    micro_batch_size: int
    dp_size: int
    seq_length: int
    microbatches: List[MicrobatchInfo]  # Indexed as [dp_rank * num_microbatches + microbatch_idx]
    
    def get_microbatch(self, dp_rank: int, microbatch_idx: int) -> MicrobatchInfo:
        """Get microbatch info for a specific DP rank and microbatch index."""
        return self.microbatches[dp_rank * self.num_microbatches + microbatch_idx]
    
    def summary(self) -> str:
        """Return a summary string of the global batch layout."""
        lines = [
            f"=== Global Batch Layout (Iteration {self.iteration}) ===",
            f"  num_microbatches: {self.num_microbatches}",
            f"  micro_batch_size: {self.micro_batch_size}",
            f"  dp_size: {self.dp_size}",
            f"  seq_length: {self.seq_length}",
            f"  total_samples: {self.num_microbatches * self.micro_batch_size * self.dp_size}",
            "",
        ]
        
        for dp_rank in range(self.dp_size):
            lines.append(f"  DP Rank {dp_rank}:")
            for mb_idx in range(self.num_microbatches):
                mb = self.get_microbatch(dp_rank, mb_idx)
                doc_lens_per_sample = []
                for s in mb.samples:
                    doc_lens_per_sample.append([d.length for d in s.documents])
                lines.append(f"    Microbatch {mb_idx}: {len(mb.samples)} samples, docs_per_sample={doc_lens_per_sample}")
        
        return "\n".join(lines)
    
    def document_length_histogram(self) -> Dict[str, Any]:
        """Get histogram of document lengths across the global batch."""
        all_doc_lengths = []
        for mb in self.microbatches:
            for sample in mb.samples:
                for doc in sample.documents:
                    all_doc_lengths.append(doc.length)
        
        if not all_doc_lengths:
            return {"min": 0, "max": 0, "mean": 0, "count": 0}
        
        return {
            "min": min(all_doc_lengths),
            "max": max(all_doc_lengths),
            "mean": sum(all_doc_lengths) / len(all_doc_lengths),
            "count": len(all_doc_lengths),
            "lengths": all_doc_lengths,
        }


class GlobalBatchPeeker:
    """A utility to inspect global batch layout from rank 0.
    
    This class allows you to "peek" at what the global batch looks like
    before or during training, showing:
    - Document lengths in each microbatch
    - Sample distribution across DP ranks
    - Document packing/padding patterns
    
    Example usage:
        ```python
        peeker = GlobalBatchPeeker(train_dataset)
        
        # Peek at a specific iteration
        layout = peeker.peek_iteration(iteration=0)
        print(layout.summary())
        
        # Peek at next N global batches
        for layout in peeker.peek_n_iterations(n=5):
            print(layout.summary())
        ```
    
    Args:
        dataset: The training dataset (GPTDataset or DistCAMockGPTDataset)
        micro_batch_size: Micro batch size (default: from args)
        seq_length: Sequence length (default: from args)
    """
    
    def __init__(
        self, 
        dataset,
        micro_batch_size: Optional[int] = None,
        seq_length: Optional[int] = None,
        num_microbatches: Optional[int] = None,
        dp_size: Optional[int] = None,
    ):
        self.dataset = dataset
        args = get_args()
        
        self.micro_batch_size = micro_batch_size or args.micro_batch_size
        self.seq_length = seq_length or args.seq_length
        self.num_microbatches = num_microbatches or get_num_microbatches()
        self.dp_size = dp_size or mpu.get_data_parallel_world_size()
        
        # Samples per global batch
        self.samples_per_global_batch = (
            self.micro_batch_size * self.num_microbatches * self.dp_size
        )
        
        logger.info(f"GlobalBatchPeeker initialized:")
        logger.info(f"  micro_batch_size: {self.micro_batch_size}")
        logger.info(f"  num_microbatches: {self.num_microbatches}")
        logger.info(f"  dp_size: {self.dp_size}")
        logger.info(f"  samples_per_global_batch: {self.samples_per_global_batch}")
    
    def _get_sample_info(self, sample_idx: int) -> SampleInfo:
        """Extract document information from a dataset sample."""
        # Use internal method to get document structure
        if hasattr(self.dataset, '_query_document_sample_shuffle_indices'):
            text, document_ids = self.dataset._query_document_sample_shuffle_indices(sample_idx)
            
            # Parse document boundaries from the sample
            # document_ids contains the document IDs that contributed to this sample
            documents = []
            if len(document_ids) == 0:
                # Single document spans entire sample
                documents.append(DocumentInfo(
                    doc_id=-1,
                    start_offset=0,
                    length=len(text)
                ))
            else:
                # Multiple documents - need to find boundaries
                # The document_ids array tells us which documents contributed
                # We need to look at EOD tokens to find boundaries
                tokenizer = get_tokenizer()
                eod_token = tokenizer.eod
                
                # Find EOD positions in text
                eod_positions = numpy.where(text == eod_token)[0].tolist()
                
                start = 0
                for i, doc_id in enumerate(document_ids):
                    if i < len(eod_positions):
                        end = eod_positions[i] + 1  # Include EOD token
                    else:
                        end = len(text)
                    
                    documents.append(DocumentInfo(
                        doc_id=doc_id,
                        start_offset=start,
                        length=end - start
                    ))
                    start = end
            
            return SampleInfo(
                sample_idx=sample_idx,
                total_tokens=len(text),
                documents=documents
            )
        else:
            # Fallback for datasets without _query_document_sample_shuffle_indices
            sample = self.dataset[sample_idx]
            if isinstance(sample, dict) and 'tokens' in sample:
                tokens = sample['tokens']
                total_tokens = len(tokens) if hasattr(tokens, '__len__') else tokens.numel()
            else:
                total_tokens = self.seq_length
            
            return SampleInfo(
                sample_idx=sample_idx,
                total_tokens=total_tokens,
                documents=[DocumentInfo(doc_id=-1, start_offset=0, length=total_tokens)]
            )
    
    def peek_iteration(self, iteration: int = 0) -> GlobalBatchLayout:
        """Peek at the global batch layout for a specific iteration.
        
        Args:
            iteration: The training iteration to peek at (0-indexed)
            
        Returns:
            GlobalBatchLayout with details about the batch structure
        """
        start_sample = iteration * self.samples_per_global_batch
        
        microbatches = []
        
        # Simulate how samples are distributed across DP ranks and microbatches
        # The ordering is: for each microbatch, samples go to DP ranks in order
        sample_offset = start_sample
        
        for dp_rank in range(self.dp_size):
            for mb_idx in range(self.num_microbatches):
                samples = []
                for _ in range(self.micro_batch_size):
                    if sample_offset < len(self.dataset):
                        sample_info = self._get_sample_info(sample_offset)
                        samples.append(sample_info)
                    sample_offset += 1
                
                microbatches.append(MicrobatchInfo(
                    microbatch_idx=mb_idx,
                    dp_rank=dp_rank,
                    samples=samples
                ))
        
        return GlobalBatchLayout(
            iteration=iteration,
            num_microbatches=self.num_microbatches,
            micro_batch_size=self.micro_batch_size,
            dp_size=self.dp_size,
            seq_length=self.seq_length,
            microbatches=microbatches
        )
    
    def peek_n_iterations(self, n: int = 1, start_iteration: int = 0):
        """Generator that yields GlobalBatchLayout for n consecutive iterations.
        
        Args:
            n: Number of iterations to peek
            start_iteration: Starting iteration index
            
        Yields:
            GlobalBatchLayout for each iteration
        """
        for i in range(n):
            yield self.peek_iteration(start_iteration + i)
    
    def print_layout(self, iteration: int = 0, detailed: bool = False):
        """Print the global batch layout for an iteration.
        
        Args:
            iteration: The training iteration to inspect
            detailed: If True, print detailed document info
        """
        layout = self.peek_iteration(iteration)
        print(layout.summary())
        
        if detailed:
            print("\n--- Detailed Document Info ---")
            for mb in layout.microbatches:
                print(f"\nDP Rank {mb.dp_rank}, Microbatch {mb.microbatch_idx}:")
                for sample in mb.samples:
                    print(f"  {sample}")
                    for doc in sample.documents:
                        print(f"    Doc {doc.doc_id}: offset={doc.start_offset}, len={doc.length}")





