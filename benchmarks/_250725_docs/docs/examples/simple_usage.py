#!/usr/bin/env python3
"""
Simple usage examples for the communication metadata system.

These examples show how to use the key functions in practical scenarios,
building understanding from simple to complex use cases.
"""

import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.test_util import gen_seq_lens, create_qkv_dispatch
from tests.test_comm_metadata import orchestrate_simulate
from d2.runtime.inplace_metadata import compute_metadata, Metadata


def example_1_sequence_generation():
    """Example 1: Generate realistic sequence lengths for testing."""
    print("=" * 60)
    print("EXAMPLE 1: Generating Realistic Sequence Lengths")
    print("=" * 60)
    
    # Scenario: 4 GPU ranks, each with up to 6 sequences, 2048 tokens per rank
    world_size = 4
    num_seqs = 6
    tokens_per_rank = 2048
    
    print(f"Setup: {world_size} ranks, {num_seqs} sequences each, {tokens_per_rank} tokens per rank")
    
    # Generate sequences with reproducible random seed
    torch.manual_seed(42)
    seq_lens = gen_seq_lens(world_size, num_seqs, tokens_per_rank)
    
    print(f"\nGenerated sequence lengths:")
    for rank in range(world_size):
        total = seq_lens[rank].sum().item()
        print(f"Rank {rank}: {seq_lens[rank].tolist()} (total: {total})")
    
    # Show statistics
    all_lens = seq_lens.flatten()
    print(f"\nStatistics:")
    print(f"  Min length: {all_lens.min()}")
    print(f"  Max length: {all_lens.max()}")
    print(f"  Mean length: {all_lens.float().mean():.1f}")
    print(f"  Std deviation: {all_lens.float().std():.1f}")
    
    print("\n💡 Key insight: Lengths are random but realistic, always sum correctly")


def example_2_simple_communication():
    """Example 2: Simple rank-to-rank communication."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Simple Rank-to-Rank Communication")
    print("=" * 60)
    
    # Scenario: 2 ranks swap their data
    world_size = 2
    
    # Create simple dispatch: each rank sends to the other
    seq_len = torch.tensor([[128, 64], [96, 112]])  # Different lengths per rank
    dispatch = torch.tensor([[1, 1], [0, 0]])       # R0→R1, R1→R0
    
    print(f"Setup: Rank 0 has sequences of length {seq_len[0].tolist()}")
    print(f"       Rank 1 has sequences of length {seq_len[1].tolist()}")
    print(f"       Dispatch plan: {dispatch.tolist()}")
    
    # Compute communication metadata
    fwd_meta, rev_meta = compute_metadata(seq_len, dispatch)
    
    print(f"\nForward metadata:")
    print(f"  Destinations: {fwd_meta.dst_rank.squeeze().tolist()}")
    print(f"  Offsets: {fwd_meta.dst_offset.squeeze().tolist()}")
    print(f"  Receive counts: {fwd_meta.num_recv_tokens.tolist()}")
    
    # Create test data
    hidden_dim = 32
    tensor = torch.zeros(world_size, 256, hidden_dim)
    
    # Fill with distinctive patterns
    tensor[0, :128] = 1.0   # Rank 0, seq 0
    tensor[0, 128:192] = 2.0  # Rank 0, seq 1
    tensor[1, :96] = 10.0    # Rank 1, seq 0
    tensor[1, 96:208] = 20.0  # Rank 1, seq 1
    
    # Simulate forward communication
    max_recv = fwd_meta.num_recv_tokens.max()
    output = torch.zeros(world_size, max_recv, hidden_dim)
    output = orchestrate_simulate(tensor, output, fwd_meta)
    
    print(f"\nAfter forward communication:")
    print(f"  Rank 0 received data with values: {torch.unique(output[0][output[0].sum(dim=1) != 0])}")
    print(f"  Rank 1 received data with values: {torch.unique(output[1][output[1].sum(dim=1) != 0])}")
    
    # Simulate reverse communication  
    rev_output = torch.zeros(world_size, 256, hidden_dim)
    rev_output = orchestrate_simulate(output, rev_output, rev_meta)
    
    # Verify perfect reconstruction
    error = (tensor - rev_output).abs().max()
    print(f"\nAfter reverse communication:")
    print(f"  Max reconstruction error: {error} (should be 0)")
    
    print("\n💡 Key insight: Forward + reverse = perfect reconstruction")


def example_3_context_parallelism():
    """Example 3: Context parallelism with broadcast."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Context Parallelism (Broadcast)")
    print("=" * 60)
    
    # Scenario: Sequence shards broadcasted to multiple ranks
    world_size = 3
    
    # Setup where sequences are sent to multiple destinations
    seq_len = torch.tensor([[64, 32], [48, 80], [96, 16]])
    
    # 3D dispatch: each sequence shard can go to multiple ranks
    dispatch = torch.tensor([
        [[1, 2], [0, -1]],    # R0: seq0→[R1,R2], seq1→[R0]
        [[0, -1], [2, 1]],    # R1: seq0→[R0], seq1→[R2,R1]  
        [[1, 0], [0, 1]]      # R2: seq0→[R1,R0], seq1→[R0,R1]
    ])
    
    print(f"Setup: 3D dispatch for context parallelism")
    print(f"  Rank 0 broadcasts seq0 to ranks [1,2], sends seq1 to rank 0")
    print(f"  Rank 1 sends seq0 to rank 0, broadcasts seq1 to ranks [2,1]")
    print(f"  Rank 2 broadcasts seq0 to ranks [1,0], broadcasts seq1 to ranks [0,1]")
    
    # Compute metadata
    fwd_meta, rev_meta = compute_metadata(seq_len, dispatch)
    
    print(f"\nReceive token counts (including broadcasts):")
    for rank in range(world_size):
        print(f"  Rank {rank}: {fwd_meta.num_recv_tokens[rank].tolist()}")
    
    # Create test data with distinctive patterns
    hidden_dim = 16
    max_tokens = seq_len.sum(dim=1).max()
    tensor = torch.zeros(world_size, max_tokens, hidden_dim)
    
    # Fill each rank's sequences with unique values
    offset = 0
    for rank in range(world_size):
        for seq in range(2):
            length = seq_len[rank, seq].item()
            if length > 0:
                tensor[rank, offset:offset+length] = rank * 10 + seq + 1
                offset += length
        offset = 0  # Reset for next rank
    
    # Simulate communication
    max_recv = fwd_meta.num_recv_tokens.max()
    output = torch.zeros(world_size, max_recv, hidden_dim)
    output = orchestrate_simulate(tensor, output, fwd_meta)
    
    print(f"\nAfter forward communication (with broadcasts):")
    for rank in range(world_size):
        received_values = torch.unique(output[rank][output[rank].sum(dim=1) != 0])
        print(f"  Rank {rank} received data from sequences: {received_values.tolist()}")
    
    print("\n💡 Key insight: Context parallelism broadcasts data to multiple ranks")


def example_4_full_qkv_pipeline():
    """Example 4: Complete QKV dispatch pipeline."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Complete QKV Dispatch Pipeline")
    print("=" * 60)
    
    # Realistic scenario: 4 ranks, moderate-sized sequences
    world_size = 4
    total_seq_len = 256  # Must be divisible by max_cp_degree
    num_seqs = 8
    max_cp_degree = 4
    hidden_dim = 64
    
    print(f"Setup: {world_size} ranks, {num_seqs} sequences, {total_seq_len} tokens each")
    print(f"       Max context parallelism degree: {max_cp_degree}")
    print(f"       Hidden dimension: {hidden_dim}")
    
    # Create complete QKV dispatch plan
    torch.manual_seed(123)
    fwd_q_meta, rev_q_meta, fwd_k_meta, rev_k_meta, attn_meta = create_qkv_dispatch(
        world_size, total_seq_len, num_seqs, max_cp_degree
    )
    
    print(f"\nGenerated metadata:")
    print(f"  Query forward shape: {fwd_q_meta.dst_rank.shape}")
    print(f"  KV forward shape: {fwd_k_meta.dst_rank.shape}")
    print(f"  Query max receive tokens: {fwd_q_meta.num_recv_tokens[:, -1].tolist()}")
    print(f"  KV max receive tokens: {fwd_k_meta.num_recv_tokens[:, -1].tolist()}")
    
    # Create test tensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    query_tensor = torch.randn(world_size, total_seq_len, hidden_dim, device=device)
    kv_tensor = torch.randn(world_size, total_seq_len, hidden_dim, device=device)
    
    print(f"Created tensors on device: {device}")
    
    # Test Query path
    print(f"\nTesting Query path:")
    
    # Forward query
    q_max_recv = fwd_q_meta.num_recv_tokens.max()
    q_output = torch.zeros(world_size, q_max_recv, hidden_dim, device=device)
    q_output = orchestrate_simulate(query_tensor, q_output, fwd_q_meta)
    print(f"  Query forward: {query_tensor.shape} → {q_output.shape}")
    
    # Reverse query
    q_reconstructed = torch.zeros(world_size, total_seq_len, hidden_dim, device=device)
    q_reconstructed = orchestrate_simulate(q_output, q_reconstructed, rev_q_meta)
    q_error = (query_tensor - q_reconstructed).abs().max()
    print(f"  Query reverse: {q_output.shape} → {q_reconstructed.shape}")
    print(f"  Query reconstruction error: {q_error}")
    
    # Test KV path  
    print(f"\nTesting KV path:")
    
    # Forward KV
    kv_max_recv = fwd_k_meta.num_recv_tokens.max()
    kv_output = torch.zeros(world_size, kv_max_recv, hidden_dim, device=device)
    kv_output = orchestrate_simulate(kv_tensor, kv_output, fwd_k_meta)
    print(f"  KV forward: {kv_tensor.shape} → {kv_output.shape}")
    
    # Reverse KV (with deduplication)
    kv_reconstructed = torch.zeros(world_size, total_seq_len * max_cp_degree, hidden_dim, device=device)
    kv_reconstructed = orchestrate_simulate(kv_output, kv_reconstructed, rev_k_meta)
    kv_reconstructed = kv_reconstructed.reshape(world_size, max_cp_degree, total_seq_len, hidden_dim)
    print(f"  KV reverse: {kv_output.shape} → {kv_reconstructed.shape}")
    
    # Deduplicate KV
    kv_mask = kv_reconstructed != 0
    kv_dedup = kv_reconstructed.sum(dim=1) / kv_mask.int().sum(dim=1).clamp(min=1)
    kv_error = (kv_tensor - kv_dedup).abs().max()
    print(f"  KV after deduplication: {kv_dedup.shape}")
    print(f"  KV reconstruction error: {kv_error}")
    
    # Show attention metadata
    cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, num_local_seqs = attn_meta
    print(f"\nAttention metadata:")
    print(f"  Max Q sequence length: {max_seqlen_q}")
    print(f"  Max KV sequence length: {max_seqlen_kv}")
    print(f"  Number of local sequences per rank: {num_local_seqs}")
    
    print("\n💡 Key insight: Complete pipeline handles both Q and KV with different patterns")


def example_5_debugging_metadata():
    """Example 5: How to debug metadata issues."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Debugging Metadata Issues")
    print("=" * 60)
    
    # Create a scenario that might have issues
    world_size = 2
    seq_len = torch.tensor([[100, 50], [75, 25]])
    dispatch = torch.tensor([[1, 0], [0, -1]])  # Note: second seq of rank 1 is padding
    
    print("Debugging scenario with padding:")
    print(f"  Sequence lengths: {seq_len.tolist()}")
    print(f"  Dispatch plan: {dispatch.tolist()}")
    
    fwd_meta, rev_meta = compute_metadata(seq_len, dispatch)
    
    # Debug checklist
    print(f"\n🔍 Debug Checklist:")
    
    # 1. Check shapes
    print(f"1. Shape consistency:")
    print(f"   Forward dst_rank: {fwd_meta.dst_rank.shape}")
    print(f"   Forward dst_offset: {fwd_meta.dst_offset.shape}")
    print(f"   Forward seq_len: {fwd_meta.seq_len.shape}")
    
    # 2. Check padding handling
    print(f"2. Padding handling:")
    padding_mask = dispatch == -1
    print(f"   Padding positions: {padding_mask.tolist()}")
    print(f"   Corresponding seq_lens: {seq_len[padding_mask].tolist()}")
    
    # 3. Check token conservation
    print(f"3. Token conservation:")
    total_sent = seq_len.sum()
    total_recv = fwd_meta.num_recv_tokens[:, -1].sum()
    print(f"   Total sent: {total_sent}")
    print(f"   Total received: {total_recv}")
    print(f"   Conservation: {'✅' if total_sent == total_recv else '❌'}")
    
    # 4. Check offset validity
    print(f"4. Offset validity:")
    max_offsets = fwd_meta.dst_offset.max(dim=1)[0]
    max_recv_per_rank = fwd_meta.num_recv_tokens[:, -1]
    print(f"   Max offsets: {max_offsets.tolist()}")
    print(f"   Max receives: {max_recv_per_rank.tolist()}")
    
    # 5. Check for overwrites
    print(f"5. Overwrite detection:")
    for rank in range(world_size):
        rank_dispatch = fwd_meta.dst_rank.squeeze()
        rank_offsets = fwd_meta.dst_offset.squeeze()
        rank_lens = fwd_meta.seq_len
        
        # Find sequences going to this rank
        to_rank = (rank_dispatch == rank)
        if to_rank.any():
            offsets = rank_offsets[to_rank]
            lengths = rank_lens[to_rank]
            end_positions = offsets + lengths
            
            # Check for overlaps
            for i in range(len(offsets)):
                for j in range(i+1, len(offsets)):
                    overlap = (offsets[i] < offsets[j] + lengths[j]) and (offsets[j] < offsets[i] + lengths[i])
                    if overlap:
                        print(f"   ❌ Overlap detected at rank {rank}: seq {i} and {j}")
                    
            print(f"   Rank {rank}: sequences at {offsets.tolist()} with lengths {lengths.tolist()}")
    
    print("\n💡 Key insight: Systematic debugging reveals metadata structure and potential issues")


if __name__ == "__main__":
    print("Communication Metadata System - Practical Examples")
    print("🚀 Building understanding from simple to complex scenarios")
    
    example_1_sequence_generation()
    example_2_simple_communication()
    example_3_context_parallelism()
    example_4_full_qkv_pipeline()
    example_5_debugging_metadata()
    
    print("\n" + "=" * 60)
    print("🎉 EXAMPLES COMPLETE")
    print("=" * 60)
    print("You now have practical knowledge of:")
    print("  ✅ Generating realistic sequence distributions")
    print("  ✅ Setting up simple rank-to-rank communication")  
    print("  ✅ Using context parallelism for broadcasting")
    print("  ✅ Running complete QKV dispatch pipelines")
    print("  ✅ Debugging metadata issues systematically")
    print("\nNext steps: Try modifying these examples with your own parameters!")