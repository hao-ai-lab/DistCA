"""
Token and document monitoring utilities for distributed training.

This module provides tools to inspect what tokens and documents are being
processed during training iterations.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional

import torch

from .logging import get_logger

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TokenMonitorConfig:
    """Configuration for token monitoring."""
    enabled: bool = True
    max_tokens_to_decode: int = 200  # Max tokens to decode per sample for text preview
    max_samples_to_log: int = 2  # Max samples to log detailed info per batch
    max_docs_to_preview: int = 10  # Max documents to show previews for
    doc_preview_chars: int = 50  # Characters to show per document preview (first N + last N)
    preview_edge_chars: int = 20  # Characters to show from start and end of document text


# Global config instance
_config = TokenMonitorConfig()

# Mutable counter for tracking iterations
_iteration_counter = [0]


def get_token_monitor_config() -> TokenMonitorConfig:
    """Get the current token monitor configuration."""
    return _config


def set_token_monitor_config(
    enabled: Optional[bool] = None,
    max_tokens_to_decode: Optional[int] = None,
    max_samples_to_log: Optional[int] = None,
    max_docs_to_preview: Optional[int] = None,
    doc_preview_chars: Optional[int] = None,
    preview_edge_chars: Optional[int] = None,
) -> None:
    """Update token monitor configuration."""
    global _config
    if enabled is not None:
        _config.enabled = enabled
    if max_tokens_to_decode is not None:
        _config.max_tokens_to_decode = max_tokens_to_decode
    if max_samples_to_log is not None:
        _config.max_samples_to_log = max_samples_to_log
    if max_docs_to_preview is not None:
        _config.max_docs_to_preview = max_docs_to_preview
    if doc_preview_chars is not None:
        _config.doc_preview_chars = doc_preview_chars
    if preview_edge_chars is not None:
        _config.preview_edge_chars = preview_edge_chars


def reset_iteration_counter() -> None:
    """Reset the iteration counter to 0."""
    _iteration_counter[0] = 0


def get_iteration_count() -> int:
    """Get the current iteration count."""
    return _iteration_counter[0]


def increment_iteration_counter() -> int:
    """Increment and return the new iteration count."""
    _iteration_counter[0] += 1
    return _iteration_counter[0]


# =============================================================================
# Document Analysis
# =============================================================================

@dataclass
class DocumentInfo:
    """Information about documents within a token sequence."""
    num_documents: int
    tokens_per_doc: List[int]
    eod_positions: List[int]


def analyze_documents_in_sequence(token_ids: list, eod_token_id: int) -> DocumentInfo:
    """Analyze document structure within a token sequence.
    
    Documents are separated by EOD (end-of-document) tokens. This function
    finds document boundaries and calculates per-document token counts.
    
    Args:
        token_ids: List of token IDs
        eod_token_id: The end-of-document token ID
        
    Returns:
        DocumentInfo with num_documents, tokens_per_doc, and eod_positions
    """
    eod_positions = [i for i, tok in enumerate(token_ids) if tok == eod_token_id]
    
    if not eod_positions:
        # No EOD found - treat entire sequence as one document
        return DocumentInfo(
            num_documents=1,
            tokens_per_doc=[len(token_ids)],
            eod_positions=[],
        )
    
    # Calculate tokens per document
    # Documents are: [0:eod_0+1], [eod_0+1:eod_1+1], ...
    tokens_per_doc = []
    prev_end = 0
    for eod_pos in eod_positions:
        doc_length = eod_pos - prev_end + 1  # Include the EOD token
        tokens_per_doc.append(doc_length)
        prev_end = eod_pos + 1
    
    # If there are tokens after the last EOD, count them as a partial document
    if prev_end < len(token_ids):
        tokens_per_doc.append(len(token_ids) - prev_end)
    
    num_docs = len(eod_positions) + (1 if prev_end < len(token_ids) else 0)
    
    return DocumentInfo(
        num_documents=num_docs,
        tokens_per_doc=tokens_per_doc,
        eod_positions=eod_positions,
    )


# =============================================================================
# Mask and Position Analysis
# =============================================================================

def analyze_loss_mask_for_doc(
    loss_mask: torch.Tensor,
    start_pos: int,
    end_pos: int,
) -> dict:
    """Analyze loss mask for a document range.
    
    Args:
        loss_mask: Loss mask tensor of shape [seq_length] (1D, for one sample)
        start_pos: Start position of the document
        end_pos: End position of the document (exclusive)
        
    Returns:
        Dict with 'zeros', 'ones', 'total', 'loss_ratio'
    """
    doc_mask = loss_mask[start_pos:end_pos]
    ones = int((doc_mask == 1).sum().item())
    zeros = int((doc_mask == 0).sum().item())
    total = len(doc_mask)
    loss_ratio = ones / total if total > 0 else 0.0
    return {
        'zeros': zeros,
        'ones': ones,
        'total': total,
        'loss_ratio': loss_ratio,
    }


def analyze_position_ids_for_doc(
    position_ids: torch.Tensor,
    start_pos: int,
    end_pos: int,
) -> dict:
    """Analyze position IDs for a document range.
    
    Args:
        position_ids: Position IDs tensor of shape [seq_length] (1D, for one sample)
        start_pos: Start position of the document
        end_pos: End position of the document (exclusive)
        
    Returns:
        Dict with 'min', 'max', 'is_contiguous', 'resets_at_doc_start'
    """
    doc_pos = position_ids[start_pos:end_pos]
    if len(doc_pos) == 0:
        return {'min': None, 'max': None, 'is_contiguous': True, 'resets_at_doc_start': False}
    
    min_pos = int(doc_pos.min().item())
    max_pos = int(doc_pos.max().item())
    
    # Check if positions are contiguous (0, 1, 2, ... or start, start+1, ...)
    expected = torch.arange(min_pos, min_pos + len(doc_pos), device=doc_pos.device, dtype=doc_pos.dtype)
    is_contiguous = torch.equal(doc_pos, expected)
    
    # Check if position resets to 0 at document start
    resets_at_doc_start = (min_pos == 0)
    
    return {
        'min': min_pos,
        'max': max_pos,
        'is_contiguous': is_contiguous,
        'resets_at_doc_start': resets_at_doc_start,
    }


def describe_attention_mask(
    attention_mask: Optional[torch.Tensor],
    seq_length: int,
    log: logging.Logger,
) -> str:
    """Describe the attention mask pattern.
    
    Args:
        attention_mask: Attention mask tensor. Can be:
            - None (no mask / full attention)
            - 2D: [seq_len, seq_len] per sample
            - 3D: [batch, seq_len, seq_len]
            - 4D: [batch, num_heads, seq_len, seq_len]
        seq_length: Expected sequence length
        log: Logger for output
        
    Returns:
        String description of the mask pattern
    """
    if attention_mask is None:
        return "None (full causal attention handled by backend)"
    
    shape = tuple(attention_mask.shape)
    dtype = attention_mask.dtype
    
    # Get a 2D slice for analysis
    if len(shape) == 2:
        mask_2d = attention_mask
    elif len(shape) == 3:
        mask_2d = attention_mask[0]  # First batch
    elif len(shape) == 4:
        mask_2d = attention_mask[0, 0]  # First batch, first head
    else:
        return f"Unknown shape {shape}"
    
    # Analyze the mask pattern
    seq_len = mask_2d.shape[0]
    
    # Check if it's a causal (lower triangular) mask
    # In causal mask: mask[i, j] = 1 if j <= i, else 0 (or True/False)
    is_bool = dtype == torch.bool
    
    # Create expected causal mask (always as float for comparison, then convert)
    causal_float = torch.tril(torch.ones(seq_len, seq_len, device=mask_2d.device, dtype=torch.float32))
    
    # Convert mask to float for comparison
    mask_float = mask_2d.float()
    
    if torch.equal(mask_float, causal_float):
        pattern = "causal (lower triangular)"
    elif torch.equal(mask_float, 1.0 - causal_float):
        pattern = "causal (upper triangular - inverted)"
    elif torch.all(mask_float == 1.0):
        pattern = "full attention (all ones/True)"
    elif torch.all(mask_float == 0.0):
        pattern = "no attention (all zeros/False)"
    else:
        # Check if it's block-diagonal or has other structure
        nonzero_ratio = mask_float.mean().item()
        # Check if lower triangular part is all ones (causal with possible padding)
        lower_tri = torch.tril(mask_float)
        lower_ratio = lower_tri.sum().item() / (seq_len * (seq_len + 1) / 2)
        
        if lower_ratio > 0.99:
            pattern = f"mostly causal (lower_tri_ratio={lower_ratio:.3f})"
        else:
            pattern = f"custom pattern (nonzero_ratio={nonzero_ratio:.3f})"
    
    return f"shape={shape}, dtype={dtype}, pattern={pattern}"


# =============================================================================
# Token Logging
# =============================================================================

def log_tokens_text(
    tokens: torch.Tensor, 
    tokenizer, 
    iteration: Optional[int] = None,
    loss_mask: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    logger: Optional[logging.Logger] = None,
    config: Optional[TokenMonitorConfig] = None,
) -> None:
    """Log decoded text from token IDs for monitoring.
    
    Logs document structure, token previews, text, and mask/position info for each sample.
    
    Args:
        tokens: Token tensor of shape [batch_size, seq_length]
        tokenizer: The tokenizer to decode tokens (must have .eod and .detokenize())
        iteration: Current training iteration. If None, uses internal counter.
        loss_mask: Optional loss mask tensor of shape [batch_size, seq_length]
        attention_mask: Optional attention mask tensor
        position_ids: Optional position IDs tensor of shape [batch_size, seq_length]
        logger: Logger to use. If None, uses the module logger.
        config: TokenMonitorConfig. If None, uses global config.
    """
    if tokens is None:
        return
    
    cfg = config or _config
    if not cfg.enabled:
        return
    
    log = logger or get_logger()
    iter_num = iteration if iteration is not None else _iteration_counter[0]
    
    batch_size = tokens.shape[0]
    seq_length = tokens.shape[1] if len(tokens.shape) > 1 else len(tokens)
    eod_token_id = tokenizer.eod
    
    log.info(f"=== Iteration {iter_num} Token Monitor (batch_size={batch_size}, seq_len={seq_length}, eod_id={eod_token_id}) ===")
    
    # Describe attention mask once (usually same for all samples)
    attn_desc = describe_attention_mask(attention_mask, seq_length, log)
    log.info(f"  Attention mask: {attn_desc}")
    
    # Aggregate document stats across all samples in batch
    total_docs = 0
    all_doc_lengths = []
    
    for sample_idx in range(batch_size):
        sample_tokens = tokens[sample_idx] if len(tokens.shape) > 1 else tokens
        full_sample_tokens = sample_tokens.tolist()
        
        # Get per-sample loss_mask and position_ids if available
        sample_loss_mask = None
        if loss_mask is not None:
            if len(loss_mask.shape) == 1:
                sample_loss_mask = loss_mask
            else:
                sample_loss_mask = loss_mask[sample_idx]
        
        sample_position_ids = None
        if position_ids is not None:
            if len(position_ids.shape) == 1:
                sample_position_ids = position_ids
            else:
                sample_position_ids = position_ids[sample_idx]
        
        # Analyze document structure for the full sequence
        doc_info = analyze_documents_in_sequence(full_sample_tokens, eod_token_id)
        total_docs += doc_info.num_documents
        all_doc_lengths.extend(doc_info.tokens_per_doc)
        
        # Only log detailed text for first few samples
        if sample_idx < cfg.max_samples_to_log:
            truncated_tokens = full_sample_tokens[:cfg.max_tokens_to_decode]
            
            try:
                # Decode tokens to text
                decoded_text = tokenizer.detokenize(truncated_tokens)
                # Truncate if too long and add ellipsis
                if len(decoded_text) > 500:
                    decoded_text = decoded_text[:500] + "..."
                
                log.info(f"  Sample {sample_idx}:")
                log.info(f"    Documents in sample: {doc_info.num_documents}")
                log.info(f"    Tokens per doc: {doc_info.tokens_per_doc[:10]}{'...' if len(doc_info.tokens_per_doc) > 10 else ''}")
                log.info(f"    EOD positions: {doc_info.eod_positions[:10]}{'...' if len(doc_info.eod_positions) > 10 else ''}")
                log.info(f"    Token IDs (first 20): {truncated_tokens[:20]}{'...' if len(truncated_tokens) > 20 else ''}")
                log.info(f"    Text (first {len(truncated_tokens)} tokens): {repr(decoded_text)}")
                
                # Calculate document start/end positions
                doc_starts = [0] + [pos + 1 for pos in doc_info.eod_positions if pos + 1 < len(full_sample_tokens)]
                doc_ends = [pos + 1 for pos in doc_info.eod_positions] + ([len(full_sample_tokens)] if doc_info.eod_positions and doc_info.eod_positions[-1] + 1 < len(full_sample_tokens) else [])
                if not doc_info.eod_positions:
                    doc_ends = [len(full_sample_tokens)]
                
                # Print document details with mask and position info
                log.info(f"    --- Document Details ---")
                for doc_idx, start_pos in enumerate(doc_starts[:cfg.max_docs_to_preview]):
                    end_pos = doc_ends[doc_idx] if doc_idx < len(doc_ends) else len(full_sample_tokens)
                    doc_len = end_pos - start_pos
                    
                    # Get document text preview (first N and last N chars)
                    # Decode enough tokens to get the full document (or a reasonable amount)
                    doc_tokens = full_sample_tokens[start_pos:end_pos]
                    try:
                        doc_text = tokenizer.detokenize(doc_tokens)
                        doc_text_clean = doc_text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                        
                        edge = cfg.preview_edge_chars
                        if len(doc_text_clean) <= edge * 2 + 5:
                            # Short enough to show full text
                            doc_preview = doc_text_clean
                        else:
                            # Show first N ... last N
                            doc_preview = f"{doc_text_clean[:edge]} ... {doc_text_clean[-edge:]}"
                    except Exception:
                        doc_preview = "[decode failed]"
                    
                    # Build info string
                    info_parts = [f"Doc {doc_idx} (@pos {start_pos}-{end_pos}, len={doc_len})"]
                    
                    # Loss mask info
                    if sample_loss_mask is not None:
                        loss_info = analyze_loss_mask_for_doc(sample_loss_mask, start_pos, end_pos)
                        info_parts.append(f"loss_mask: {loss_info['ones']}/{loss_info['total']} active ({loss_info['loss_ratio']:.1%})")
                    
                    # Position IDs info
                    if sample_position_ids is not None:
                        pos_info = analyze_position_ids_for_doc(sample_position_ids, start_pos, end_pos)
                        pos_str = f"pos_ids: [{pos_info['min']}-{pos_info['max']}]"
                        if pos_info['resets_at_doc_start']:
                            pos_str += " (resets)"
                        if not pos_info['is_contiguous']:
                            pos_str += " (non-contiguous!)"
                        info_parts.append(pos_str)
                    
                    log.info(f"      {' | '.join(info_parts)}")
                    log.info(f"        Text: {repr(doc_preview)}")
                
                if len(doc_starts) > cfg.max_docs_to_preview:
                    log.info(f"      ... and {len(doc_starts) - cfg.max_docs_to_preview} more documents")
                    
            except Exception as e:
                log.warning(f"  Sample {sample_idx}: Failed to decode tokens: {e}")
    
    # Summary stats for entire microbatch
    if all_doc_lengths:
        avg_doc_len = sum(all_doc_lengths) / len(all_doc_lengths)
        min_doc_len = min(all_doc_lengths)
        max_doc_len = max(all_doc_lengths)
        log.info(f"  --- Microbatch Summary ---")
        log.info(f"    Total documents in microbatch: {total_docs}")
        log.info(f"    Avg tokens/doc: {avg_doc_len:.1f}, Min: {min_doc_len}, Max: {max_doc_len}")
        
        # Overall loss mask summary
        if loss_mask is not None:
            total_ones = int((loss_mask == 1).sum().item())
            total_elements = loss_mask.numel()
            log.info(f"    Loss mask: {total_ones}/{total_elements} active ({total_ones/total_elements:.1%})")
    
    log.info("=" * 60)


def monitor_batch_tokens(
    tokens: torch.Tensor,
    tokenizer,
    loss_mask: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Convenience function to monitor tokens in a batch.
    
    Automatically increments the iteration counter and logs token info.
    
    Args:
        tokens: Token tensor of shape [batch_size, seq_length]
        tokenizer: The tokenizer to decode tokens
        loss_mask: Optional loss mask tensor of shape [batch_size, seq_length]
        attention_mask: Optional attention mask tensor
        position_ids: Optional position IDs tensor of shape [batch_size, seq_length]
        logger: Logger to use. If None, uses the module logger.
    """
    if not _config.enabled:
        return
    
    iteration = increment_iteration_counter()
    log_tokens_text(
        tokens, 
        tokenizer, 
        iteration=iteration,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        position_ids=position_ids,
        logger=logger,
    )

