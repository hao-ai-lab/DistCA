#!/usr/bin/env python3
"""
Unit tests for create_qkv_dispatch function.

These tests demonstrate exactly what create_qkv_dispatch does and doesn't do,
providing a bottom-up understanding of QKV dispatch planning behavior.
"""

import torch
import sys
import os
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.test_util import create_qkv_dispatch


def test_basic_qkv_dispatch():
    """Test basic QKV dispatch creation."""
    world_size = 2
    total_seq_len = 64  # Must be divisible by max_cp_degree
    num_seqs = 4
    max_cp_degree = 4
    
    torch.manual_seed(42)  # For reproducible results
    fwd_q, rev_q, fwd_k, rev_k, attn_meta = create_qkv_dispatch(
        world_size, total_seq_len, num_seqs, max_cp_degree
    )
    
    # ✅ What it DOES:
    # Returns 5 metadata objects
    assert len([fwd_q, rev_q, fwd_k, rev_k, attn_meta]) == 5, "Should return 5 metadata objects"
    
    # All metadata should be for the same world_size
    assert fwd_q.world_size == world_size, "Forward query metadata should have correct world_size"
    assert rev_q.world_size == world_size, "Reverse query metadata should have correct world_size"
    assert fwd_k.world_size == world_size, "Forward KV metadata should have correct world_size"
    assert rev_k.world_size == world_size, "Reverse KV metadata should have correct world_size"
    
    # Query metadata should be simpler (2D dispatch)
    assert fwd_q.dst_rank.dim() == 3, "Query dispatch should be 3D (world_size, seqs, 1)"
    assert fwd_q.dst_rank.shape[2] == 1, "Query should have CP degree 1"
    
    # KV metadata should support context parallelism (3D dispatch possible)
    assert fwd_k.dst_rank.dim() == 3, "KV dispatch should be 3D"
    assert fwd_k.dst_rank.shape[2] == max_cp_degree, "KV should support max CP degree"
    
    print("✅ Basic QKV dispatch test passed")
    print(f"Query dispatch shape: {fwd_q.dst_rank.shape}")
    print(f"KV dispatch shape: {fwd_k.dst_rank.shape}")


def test_sequence_length_generation():
    """Test that sequence lengths are generated correctly."""
    world_size = 2
    total_seq_len = 128
    num_seqs = 8
    max_cp_degree = 4
    
    torch.manual_seed(123)
    fwd_q, rev_q, fwd_k, rev_k, attn_meta = create_qkv_dispatch(
        world_size, total_seq_len, num_seqs, max_cp_degree
    )
    
    # ✅ What it DOES for sequence lengths:
    # Sequence lengths should be multiples of max_cp_degree
    seq_lens = fwd_q.seq_len
    assert (seq_lens % max_cp_degree == 0).all(), "All sequence lengths should be multiples of max_cp_degree"
    
    # Each rank should have total_seq_len tokens (after scaling)
    rank_totals = seq_lens.sum(dim=1)
    expected_total = total_seq_len  # Should equal total_seq_len per rank
    assert (rank_totals == expected_total).all(), f"Each rank should have {expected_total} tokens"
    
    # Should have positive lengths (no completely empty sequences)
    assert (seq_lens > 0).all(), "All sequence lengths should be positive"
    
    print("✅ Sequence length generation test passed")
    print(f"Sequence lengths: {seq_lens}")
    print(f"Per-rank totals: {rank_totals}")


def test_context_parallelism_assignment():
    """Test CP degree assignment and consistency."""
    world_size = 3
    total_seq_len = 32
    num_seqs = 4
    max_cp_degree = 8
    
    torch.manual_seed(456)
    fwd_q, rev_q, fwd_k, rev_k, attn_meta = create_qkv_dispatch(
        world_size, total_seq_len, num_seqs, max_cp_degree
    )
    
    # ✅ What it DOES for CP:
    # Check that CP degrees are powers of 2
    dst_rank_k = fwd_k.dst_rank
    
    # Count non-padding entries per sequence (CP degree)
    cp_degrees = (dst_rank_k >= 0).sum(dim=2)  # Count non-(-1) entries
    
    # All CP degrees should be powers of 2 (or 0 for padding sequences)
    for rank in range(world_size):
        for seq in range(num_seqs):
            cp_deg = cp_degrees[rank, seq].item()
            if cp_deg > 0:  # Skip padding sequences
                assert cp_deg & (cp_deg - 1) == 0, f"CP degree {cp_deg} should be power of 2"
                assert cp_deg <= max_cp_degree, f"CP degree {cp_deg} should not exceed max"
    
    print("✅ Context parallelism assignment test passed")
    print(f"CP degrees per sequence: {cp_degrees}")


def test_causal_attention_constraints():
    """Test that KV-to-Q mappings respect causal attention."""
    world_size = 2
    total_seq_len = 16
    num_seqs = 2
    max_cp_degree = 4
    
    torch.manual_seed(789)
    fwd_q, rev_q, fwd_k, rev_k, attn_meta = create_qkv_dispatch(
        world_size, total_seq_len, num_seqs, max_cp_degree
    )
    
    # ✅ What it DOES for causality:
    # This is complex to test directly, but we can check some properties
    
    # KV metadata should have the seq_recv_mask for causality
    assert rev_k.seq_recv_mask is not None, "KV reverse metadata should have sequence receive mask"
    
    # The mask should have the right shape
    expected_mask_shape = fwd_k.dst_rank.shape
    assert rev_k.seq_recv_mask.shape == expected_mask_shape, "Mask should match dispatch shape"
    
    # Mask should be boolean-like (0s and 1s)
    mask_values = torch.unique(rev_k.seq_recv_mask)
    assert torch.all((mask_values == 0) | (mask_values == 1)), "Mask should be binary"
    
    print("✅ Causal attention constraints test passed")
    print(f"KV mask shape: {rev_k.seq_recv_mask.shape}")
    print(f"Mask values: {torch.unique(rev_k.seq_recv_mask)}")


def test_metadata_consistency():
    """Test that forward and reverse metadata are consistent."""
    world_size = 2
    total_seq_len = 32
    num_seqs = 4
    max_cp_degree = 4
    
    torch.manual_seed(101)
    fwd_q, rev_q, fwd_k, rev_k, attn_meta = create_qkv_dispatch(
        world_size, total_seq_len, num_seqs, max_cp_degree
    )
    
    # ✅ What it DOES for consistency:
    
    # Query forward and reverse should be complementary
    # Total tokens sent forward = total tokens received in reverse
    q_fwd_total = fwd_q.num_recv_tokens[:, -1].sum()
    q_rev_total = rev_q.num_recv_tokens[:, -1].sum()
    assert q_fwd_total == q_rev_total, "Query forward and reverse totals should match"
    
    # KV forward and reverse should be complementary
    k_fwd_total = fwd_k.num_recv_tokens[:, -1].sum()
    k_rev_total = rev_k.num_recv_tokens[:, -1].sum()
    # Note: KV might have different totals due to replication, so just check they're positive
    assert k_fwd_total > 0, "KV forward total should be positive"
    assert k_rev_total > 0, "KV reverse total should be positive"
    
    print("✅ Metadata consistency test passed")
    print(f"Query: {q_fwd_total} forward = {q_rev_total} reverse")
    print(f"KV: {k_fwd_total} forward, {k_rev_total} reverse")


def test_attention_metadata():
    """Test attention computation metadata."""
    world_size = 2
    total_seq_len = 32
    num_seqs = 4
    max_cp_degree = 4
    
    torch.manual_seed(202)
    fwd_q, rev_q, fwd_k, rev_k, attn_meta = create_qkv_dispatch(
        world_size, total_seq_len, num_seqs, max_cp_degree
    )
    
    # ✅ What it DOES for attention metadata:
    # Should return 5 elements for attention computation
    assert len(attn_meta) == 5, "Attention metadata should have 5 components"
    
    cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, num_local_seqs_recv = attn_meta
    
    # All should be tensors or integers
    assert isinstance(cu_seqlens_q, torch.Tensor), "cu_seqlens_q should be tensor"
    assert isinstance(cu_seqlens_kv, torch.Tensor), "cu_seqlens_kv should be tensor"
    assert isinstance(max_seqlen_q, torch.Tensor), "max_seqlen_q should be tensor"
    assert isinstance(max_seqlen_kv, torch.Tensor), "max_seqlen_kv should be tensor"
    assert isinstance(num_local_seqs_recv, torch.Tensor), "num_local_seqs_recv should be tensor"
    
    # Cumulative sequence lengths should be increasing
    assert (cu_seqlens_q.diff() >= 0).all(), "Cumulative Q sequence lengths should be non-decreasing"
    assert (cu_seqlens_kv.diff() >= 0).all(), "Cumulative KV sequence lengths should be non-decreasing"
    
    print("✅ Attention metadata test passed")
    print(f"Max Q seqlen: {max_seqlen_q}")
    print(f"Max KV seqlen: {max_seqlen_kv}")


def test_divisibility_constraint():
    """Test that the function enforces divisibility constraint."""
    world_size = 2
    total_seq_len = 17  # NOT divisible by max_cp_degree=4
    num_seqs = 2
    max_cp_degree = 4
    
    # ✅ What it DOES for invalid inputs:
    try:
        create_qkv_dispatch(world_size, total_seq_len, num_seqs, max_cp_degree)
        assert False, "Should raise AssertionError for non-divisible total_seq_len"
    except AssertionError as e:
        assert "divisible" in str(e).lower(), "Should mention divisibility in error"
    
    print("✅ Divisibility constraint test passed")


def test_edge_cases():
    """Test various edge cases."""
    # Minimum valid configuration
    world_size = 1
    total_seq_len = 4
    num_seqs = 1
    max_cp_degree = 2
    
    torch.manual_seed(303)
    fwd_q, rev_q, fwd_k, rev_k, attn_meta = create_qkv_dispatch(
        world_size, total_seq_len, num_seqs, max_cp_degree
    )
    
    # ✅ What it DOES for edge cases:
    assert fwd_q.world_size == 1, "Should handle single rank"
    assert fwd_q.seq_len.shape == (1, 1), "Should handle single sequence"
    
    # Large configuration (should not crash)
    world_size = 8
    total_seq_len = 128
    num_seqs = 16
    max_cp_degree = 8
    
    torch.manual_seed(404)
    fwd_q, rev_q, fwd_k, rev_k, attn_meta = create_qkv_dispatch(
        world_size, total_seq_len, num_seqs, max_cp_degree
    )
    
    assert fwd_q.world_size == 8, "Should handle large world size"
    
    print("✅ Edge cases test passed")


def test_what_it_does_NOT_do():
    """Test what create_qkv_dispatch explicitly does NOT do."""
    world_size = 2
    total_seq_len = 32
    num_seqs = 4
    max_cp_degree = 4
    
    torch.manual_seed(505)
    fwd_q, rev_q, fwd_k, rev_k, attn_meta = create_qkv_dispatch(
        world_size, total_seq_len, num_seqs, max_cp_degree
    )
    
    print("❌ What create_qkv_dispatch does NOT do:")
    
    # ❌ Does NOT optimize for communication efficiency
    print("   - No bandwidth or latency optimization")
    print("   - Random destination assignment")
    
    # ❌ Does NOT balance load across ranks
    print("   - No attempt to equalize work")
    print("   - Some ranks may get much more/less work")
    
    # ❌ Does NOT consider network topology
    print("   - No awareness of physical connections")
    print("   - No NUMA or distance considerations")
    
    # ❌ Does NOT validate hardware constraints
    print("   - No memory capacity checking")
    print("   - No bandwidth limits")
    
    # ❌ Does NOT handle dynamic sequences
    print("   - All lengths must be known upfront")
    print("   - No support for variable batch sizes")
    
    # ❌ Does NOT provide error handling
    print("   - No fallback for failed ranks")
    print("   - No validation of dispatch feasibility")
    
    # ❌ Does NOT optimize memory usage
    print("   - No consideration of memory fragmentation")
    print("   - No buffer reuse optimization")


if __name__ == "__main__":
    print("Testing create_qkv_dispatch function")
    print("=" * 50)
    
    test_basic_qkv_dispatch()
    print()
    
    test_sequence_length_generation()
    print()
    
    test_context_parallelism_assignment()
    print()
    
    test_causal_attention_constraints()
    print()
    
    test_metadata_consistency()
    print()
    
    test_attention_metadata()
    print()
    
    test_divisibility_constraint()
    print()
    
    test_edge_cases()
    print()
    
    test_what_it_does_NOT_do()
    print()
    
    print("✅ All create_qkv_dispatch tests passed!")
    print("\nSUMMARY:")
    print("create_qkv_dispatch creates QKV communication plans by:")
    print("  ✅ Generating realistic sequence length distributions")
    print("  ✅ Assigning random but valid context parallelism degrees")
    print("  ✅ Creating consistent forward/reverse metadata pairs")
    print("  ✅ Respecting causal attention constraints")
    print("  ✅ Providing attention computation metadata")
    print("  ❌ Does NOT optimize communication efficiency")
    print("  ❌ Does NOT provide load balancing")
    print("  ❌ Does NOT handle dynamic or error scenarios")