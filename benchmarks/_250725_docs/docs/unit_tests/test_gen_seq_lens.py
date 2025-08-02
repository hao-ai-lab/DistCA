#!/usr/bin/env python3
"""
Unit tests for gen_seq_lens function.

These tests demonstrate exactly what gen_seq_lens does and doesn't do,
providing a bottom-up understanding of its behavior.
"""

import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.test_util import gen_seq_lens


def test_basic_functionality():
    """Test that gen_seq_lens produces correct basic output."""
    world_size = 2
    num_seqs = 3
    total_len = 120
    
    seq_lens = gen_seq_lens(world_size, num_seqs, total_len)
    
    # ✅ What it DOES:
    assert seq_lens.shape == (world_size, num_seqs), f"Shape should be ({world_size}, {num_seqs})"
    assert seq_lens.dtype == torch.int32, "Should return integer lengths"
    assert (seq_lens.sum(dim=1) == total_len).all(), "Each rank should sum to total_len"
    assert (seq_lens > 0).all(), "All sequences should have positive length"
    
    print("✅ Basic functionality test passed")
    print(f"Generated lengths: {seq_lens}")
    print(f"Row sums: {seq_lens.sum(dim=1)}")


def test_single_sequence():
    """Test edge case with only one sequence per rank."""
    world_size = 3
    num_seqs = 1
    total_len = 100
    
    seq_lens = gen_seq_lens(world_size, num_seqs, total_len)
    
    # ✅ What it DOES:
    assert seq_lens.shape == (3, 1)
    assert torch.equal(seq_lens, torch.tensor([[100], [100], [100]])), "Single sequence should get all tokens"
    
    print("✅ Single sequence test passed")
    print(f"Single sequence result: {seq_lens}")


def test_many_sequences():
    """Test behavior with many small sequences."""
    world_size = 1
    num_seqs = 10
    total_len = 50
    
    seq_lens = gen_seq_lens(world_size, num_seqs, total_len)
    
    # ✅ What it DOES:
    assert seq_lens.shape == (1, 10)
    assert seq_lens.sum() == 50, "Should sum to total_len"
    assert (seq_lens > 0).all(), "All sequences should be positive"
    # Minimum length should be at least 1 due to the 0.25/num_seqs offset
    min_expected = max(1, int(0.25 * total_len / num_seqs))
    assert seq_lens.min() >= min_expected, f"Minimum length should be at least {min_expected}"
    
    print("✅ Many sequences test passed")
    print(f"Lengths: {seq_lens}")
    print(f"Min length: {seq_lens.min()}, Max length: {seq_lens.max()}")


def test_zero_total_length():
    """Test edge case with zero total length."""
    world_size = 2
    num_seqs = 3
    total_len = 0
    
    seq_lens = gen_seq_lens(world_size, num_seqs, total_len)
    
    # ✅ What it DOES:
    assert seq_lens.shape == (2, 3)
    assert torch.equal(seq_lens, torch.zeros(2, 3, dtype=torch.int32)), "Zero total should give zero lengths"
    
    print("✅ Zero total length test passed")


def test_reproducibility():
    """Test that results are reproducible with same seed."""
    world_size = 2
    num_seqs = 4
    total_len = 200
    
    # ✅ What it DOES:
    torch.manual_seed(42)
    seq_lens1 = gen_seq_lens(world_size, num_seqs, total_len)
    
    torch.manual_seed(42)
    seq_lens2 = gen_seq_lens(world_size, num_seqs, total_len)
    
    assert torch.equal(seq_lens1, seq_lens2), "Same seed should produce identical results"
    
    # Different seeds should (usually) produce different results
    torch.manual_seed(43)
    seq_lens3 = gen_seq_lens(world_size, num_seqs, total_len)
    assert not torch.equal(seq_lens1, seq_lens3), "Different seeds should produce different results"
    
    print("✅ Reproducibility test passed")
    print(f"Seed 42 result: {seq_lens1}")
    print(f"Seed 43 result: {seq_lens3}")


def test_rounding_correction():
    """Test that rounding errors are properly corrected."""
    world_size = 1
    num_seqs = 3
    total_len = 10  # Forces rounding issues
    
    seq_lens = gen_seq_lens(world_size, num_seqs, total_len)
    
    # ✅ What it DOES:
    assert seq_lens.sum() == 10, "Must sum exactly to total_len despite rounding"
    assert (seq_lens > 0).all(), "All sequences must be positive"
    
    # The last sequence absorbs rounding errors, so it might be different
    # from what pure ratio would suggest
    print("✅ Rounding correction test passed")
    print(f"Lengths with rounding correction: {seq_lens}")
    
    # Show what happens without correction (simulation)
    torch.manual_seed(42)
    ratio = torch.rand((1, 3)) + 0.25 / 3
    ratio = ratio / ratio.sum(dim=1, keepdim=True)
    uncorrected = (ratio * 10).round().int()
    print(f"Without correction would sum to: {uncorrected.sum()}")
    print(f"Actual corrected sum: {seq_lens.sum()}")


def test_distribution_properties():
    """Test statistical properties of the distribution."""
    world_size = 1
    num_seqs = 100
    total_len = 10000
    
    torch.manual_seed(123)
    seq_lens = gen_seq_lens(world_size, num_seqs, total_len)
    
    # ✅ What it DOES:
    mean_length = seq_lens.float().mean()
    expected_mean = total_len / num_seqs
    assert abs(mean_length - expected_mean) < 5, f"Mean should be close to {expected_mean}"
    
    # Should have reasonable variance (not all the same)
    std_length = seq_lens.float().std()
    assert std_length > 1, "Should have some variance in lengths"
    
    print("✅ Distribution properties test passed")
    print(f"Mean length: {mean_length:.2f} (expected: {expected_mean})")
    print(f"Std deviation: {std_length:.2f}")
    print(f"Min: {seq_lens.min()}, Max: {seq_lens.max()}")


def test_what_it_does_NOT_do():
    """Explicitly test limitations and what the function does NOT guarantee."""
    world_size = 2
    num_seqs = 5
    total_len = 1000
    
    seq_lens = gen_seq_lens(world_size, num_seqs, total_len)
    
    print("❌ What gen_seq_lens does NOT do:")
    
    # ❌ Does NOT guarantee uniform distribution
    print(f"   - Lengths are NOT uniform: {seq_lens[0]}")
    uniform_length = total_len // num_seqs
    assert not torch.equal(seq_lens[0], torch.tensor([uniform_length] * num_seqs)), \
        "Should NOT produce uniform lengths"
    
    # ❌ Does NOT guarantee specific minimum/maximum bounds
    print(f"   - No control over min/max bounds beyond positivity")
    print(f"   - Min: {seq_lens.min()}, Max: {seq_lens.max()}")
    
    # ❌ Does NOT guarantee load balancing across ranks
    print(f"   - Different ranks can have very different distributions:")
    print(f"   - Rank 0: {seq_lens[0]}")
    print(f"   - Rank 1: {seq_lens[1]}")
    
    # ❌ Does NOT follow any specific statistical distribution
    print(f"   - Not Gaussian, not exponential, just ratio-based random")
    
    # ❌ Does NOT optimize for any computational property
    print(f"   - No consideration of memory alignment, cache efficiency, etc.")


if __name__ == "__main__":
    print("Testing gen_seq_lens function")
    print("=" * 50)
    
    test_basic_functionality()
    print()
    
    test_single_sequence()
    print()
    
    test_many_sequences()
    print()
    
    test_zero_total_length()
    print()
    
    test_reproducibility()
    print()
    
    test_rounding_correction()
    print()
    
    test_distribution_properties()
    print()
    
    test_what_it_does_NOT_do()
    print()
    
    print("✅ All gen_seq_lens tests passed!")
    print("\nSUMMARY:")
    print("gen_seq_lens creates random sequence length distributions that:")
    print("  ✅ Sum exactly to the target total per rank")
    print("  ✅ Are all positive integers")
    print("  ✅ Are reproducible with seeds")
    print("  ✅ Handle edge cases gracefully")
    print("  ❌ Do NOT guarantee uniform distributions")
    print("  ❌ Do NOT provide load balancing")
    print("  ❌ Do NOT follow specific statistical models")