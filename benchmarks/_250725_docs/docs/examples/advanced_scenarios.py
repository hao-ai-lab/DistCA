#!/usr/bin/env python3
"""
Advanced usage scenarios for the communication metadata system.

These examples show complex real-world usage patterns and edge cases
that demonstrate the full capability of the system.
"""

import torch
import math
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.test_util import gen_seq_lens, create_qkv_dispatch
from tests.test_comm_metadata import orchestrate_simulate  
from d2.runtime.inplace_metadata import compute_metadata, Metadata


def scenario_1_large_scale_deployment():
    """Scenario 1: Large-scale deployment with many ranks."""
    print("=" * 70)
    print("ADVANCED SCENARIO 1: Large-Scale Deployment")
    print("=" * 70)
    
    # Simulate a large deployment: 64 GPUs, long sequences
    world_size = 64
    total_seq_len = 4096  # 4K tokens per rank
    num_seqs = 32         # 32 sequences per rank
    max_cp_degree = 16    # High parallelism
    hidden_dim = 128
    
    print(f"🚀 Large-scale setup:")
    print(f"   World size: {world_size} GPUs")
    print(f"   Sequence length: {total_seq_len} tokens per GPU")
    print(f"   Sequences per GPU: {num_seqs}")
    print(f"   Max CP degree: {max_cp_degree}")
    print(f"   Total system tokens: {world_size * total_seq_len:,}")
    
    # This would be expensive to actually run, so we'll just show the setup
    print(f"\n📊 Memory estimates:")
    
    # Estimate metadata memory usage
    max_seqs_padded = num_seqs  # Assuming no padding needed
    
    # Query metadata: (world_size, max_seqs, 1) for dst_rank, dst_offset, etc.
    query_meta_size = world_size * max_seqs_padded * 3 * 4  # 3 tensors, 4 bytes each
    
    # KV metadata: (world_size, max_seqs, max_cp_degree) 
    kv_meta_size = world_size * max_seqs_padded * max_cp_degree * 3 * 4
    
    # Receive token matrices: (world_size, world_size+1)
    recv_tokens_size = world_size * (world_size + 1) * 8  # 8 bytes for int64
    
    total_meta_mb = (query_meta_size + kv_meta_size + recv_tokens_size * 4) / (1024 * 1024)
    
    print(f"   Metadata memory: ~{total_meta_mb:.1f} MB")
    
    # Estimate communication volumes
    avg_tokens_per_seq = total_seq_len / num_seqs
    total_query_tokens = world_size * total_seq_len
    
    # KV might be replicated due to CP
    est_kv_replication = max_cp_degree / 4  # Rough estimate
    total_kv_tokens = total_query_tokens * est_kv_replication
    
    tensor_memory_gb = (total_query_tokens + total_kv_tokens) * hidden_dim * 4 / (1024**3)
    print(f"   Peak tensor memory: ~{tensor_memory_gb:.1f} GB")
    
    print(f"\n💡 Insight: Large-scale deployments need careful memory planning")
    print(f"   - Metadata scales as O(world_size²)")
    print(f"   - Tensor memory scales as O(world_size × sequence_length)")
    print(f"   - Context parallelism adds replication overhead")


def scenario_2_heterogeneous_sequences():
    """Scenario 2: Highly heterogeneous sequence lengths."""
    print("\n" + "=" * 70)
    print("ADVANCED SCENARIO 2: Heterogeneous Sequence Lengths")
    print("=" * 70)
    
    # Simulate a workload with very different sequence lengths
    world_size = 4
    
    # Manually create heterogeneous sequences (not using gen_seq_lens)
    seq_len = torch.tensor([
        [2048, 512, 128, 32, 16, 8],      # Rank 0: Very mixed lengths
        [1024, 1024, 1024, 256, 0, 0],   # Rank 1: More uniform, with padding
        [4096, 0, 0, 0, 0, 0],           # Rank 2: One very long sequence
        [64, 64, 64, 64, 64, 64]         # Rank 3: Many small sequences
    ])
    
    print(f"🔄 Heterogeneous sequence distribution:")
    for rank in range(world_size):
        active_seqs = (seq_len[rank] > 0).sum().item()
        total_tokens = seq_len[rank].sum().item()
        min_len = seq_len[rank][seq_len[rank] > 0].min().item() if active_seqs > 0 else 0
        max_len = seq_len[rank].max().item()
        print(f"   Rank {rank}: {active_seqs} sequences, {total_tokens:4d} tokens, range [{min_len:4d}-{max_len:4d}]")
    
    # Create a dispatch plan that tries to balance load
    # Send long sequences to ranks with short sequences
    dispatch = torch.tensor([
        [3, 2, 1, 0, -1, -1],  # Rank 0: send long seqs to others
        [3, 2, 0, 1, -1, -1],  # Rank 1: similar strategy
        [1, -1, -1, -1, -1, -1],  # Rank 2: send big seq to rank 1
        [0, 1, 2, 0, 1, 2]     # Rank 3: distribute evenly
    ])
    
    print(f"\n📊 Load balancing dispatch:")
    print(f"   Trying to send long sequences to ranks with shorter sequences")
    
    # Compute metadata
    fwd_meta, rev_meta = compute_metadata(seq_len, dispatch)
    
    # Analyze load distribution after dispatch
    print(f"\n📈 After dispatch load distribution:")
    for rank in range(world_size):
        received_tokens = fwd_meta.num_recv_tokens[rank, -1].item()  # Total received
        print(f"   Rank {rank} will receive: {received_tokens:4d} tokens")
    
    # Calculate load balance metrics
    recv_tokens = fwd_meta.num_recv_tokens[:, -1]
    load_mean = recv_tokens.float().mean()
    load_std = recv_tokens.float().std()
    load_imbalance = load_std / load_mean
    
    print(f"\n⚖️  Load balance metrics:")
    print(f"   Mean load: {load_mean:.1f} tokens")
    print(f"   Std deviation: {load_std:.1f} tokens")
    print(f"   Imbalance ratio: {load_imbalance:.3f} (lower is better)")
    
    print(f"\n💡 Insight: Heterogeneous workloads need careful dispatch planning")
    print(f"   - Simple random dispatch can create severe imbalance")
    print(f"   - Load-aware dispatch helps but adds complexity")


def scenario_3_memory_constrained_environment():
    """Scenario 3: Memory-constrained environment optimization."""
    print("\n" + "=" * 70)
    print("ADVANCED SCENARIO 3: Memory-Constrained Environment")
    print("=" * 70)
    
    # Simulate running on smaller GPUs with memory constraints
    world_size = 8
    base_seq_len = 1024
    num_seqs = 16
    max_cp_degree = 4
    hidden_dim = 256
    
    # Available memory per GPU (simulated)
    gpu_memory_gb = 8  # 8GB GPU
    available_memory_bytes = gpu_memory_gb * (1024**3) * 0.7  # 70% available
    
    print(f"💾 Memory-constrained setup:")
    print(f"   GPU memory: {gpu_memory_gb} GB per GPU")
    print(f"   Available for tensors: {available_memory_bytes / (1024**3):.1f} GB per GPU")
    
    # Calculate memory requirements for different CP degrees
    print(f"\n📊 Memory usage analysis for different CP degrees:")
    
    for cp_degree in [1, 2, 4, 8]:
        if cp_degree > max_cp_degree:
            continue
            
        # Estimate memory usage
        base_tokens = base_seq_len
        
        # Query tensors: original + attention layout
        query_memory = base_tokens * hidden_dim * 4 * 2  # FP32, 2 buffers
        
        # KV tensors: original + attention layout + replication
        kv_replication_factor = (cp_degree + 1) / 2  # Rough estimate
        kv_memory = base_tokens * hidden_dim * 4 * 2 * kv_replication_factor
        
        # Metadata memory
        metadata_memory = num_seqs * cp_degree * 12  # Rough estimate
        
        total_memory = query_memory + kv_memory + metadata_memory
        memory_gb = total_memory / (1024**3)
        
        fits = total_memory < available_memory_bytes
        status = "✅" if fits else "❌"
        
        print(f"   CP degree {cp_degree}: {memory_gb:.2f} GB {status}")
    
    # Show memory optimization strategies
    print(f"\n⚡ Memory optimization strategies:")
    print(f"   1. Reduce CP degree when memory is tight")
    print(f"   2. Use gradient checkpointing")
    print(f"   3. Sequence-level batching instead of token-level")
    print(f"   4. Mixed precision (FP16/BF16)")
    print(f"   5. Offload inactive tensors to CPU")
    
    # Demonstrate adaptive CP degree selection
    optimal_cp = 2  # From analysis above
    
    print(f"\n🎯 Adaptive configuration:")
    print(f"   Selected CP degree: {optimal_cp} (fits in memory)")
    
    # Create configuration with reduced CP degree
    torch.manual_seed(42)
    fwd_q, rev_q, fwd_k, rev_k, attn_meta = create_qkv_dispatch(
        world_size, base_seq_len, num_seqs, optimal_cp
    )
    
    print(f"   Generated metadata shapes:")
    print(f"     Query: {fwd_q.dst_rank.shape}")
    print(f"     KV: {fwd_k.dst_rank.shape}")
    
    print(f"\n💡 Insight: Memory constraints require adaptive configuration")
    

def scenario_4_fault_tolerance_simulation():
    """Scenario 4: Simulating fault tolerance scenarios."""
    print("\n" + "=" * 70)
    print("ADVANCED SCENARIO 4: Fault Tolerance Simulation")
    print("=" * 70)
    
    # Setup: Simulate what happens when ranks fail
    world_size = 6
    total_seq_len = 128
    num_seqs = 8
    max_cp_degree = 4
    
    print(f"🔧 Fault tolerance scenario:")
    print(f"   Original configuration: {world_size} ranks")
    
    # Create original dispatch
    torch.manual_seed(123)
    original_fwd_q, _, original_fwd_k, _, _ = create_qkv_dispatch(
        world_size, total_seq_len, num_seqs, max_cp_degree
    )
    
    # Simulate rank 2 and rank 4 failing
    failed_ranks = [2, 4]
    surviving_ranks = [0, 1, 3, 5]
    
    print(f"   Failed ranks: {failed_ranks}")
    print(f"   Surviving ranks: {surviving_ranks}")
    
    # Analyze impact on communication
    print(f"\n📊 Impact analysis:")
    
    # Check what data was supposed to go to failed ranks
    q_to_failed = 0
    k_to_failed = 0
    
    for failed_rank in failed_ranks:
        # Count sequences destined for failed rank
        q_mask = (original_fwd_q.dst_rank.squeeze() == failed_rank)
        q_tokens = original_fwd_q.seq_len[q_mask].sum().item()
        q_to_failed += q_tokens
        
        k_mask = (original_fwd_k.dst_rank == failed_rank)
        k_tokens = original_fwd_k.seq_len.unsqueeze(-1).expand_as(original_fwd_k.dst_rank)[k_mask].sum().item()
        k_to_failed += k_tokens
    
    total_q_tokens = original_fwd_q.seq_len.sum().item()
    total_k_tokens = original_fwd_k.seq_len.sum().item() * max_cp_degree  # Rough estimate
    
    q_loss_percent = (q_to_failed / total_q_tokens) * 100
    k_loss_percent = (k_to_failed / total_k_tokens) * 100
    
    print(f"   Query tokens lost: {q_to_failed}/{total_q_tokens} ({q_loss_percent:.1f}%)")
    print(f"   KV tokens affected: {k_to_failed} ({k_loss_percent:.1f}%)")
    
    # Simulate recovery strategies
    print(f"\n🔄 Recovery strategies:")
    
    print(f"   1. Redistribute to surviving ranks:")
    # Simple redistribution: map failed ranks to surviving ones
    rank_mapping = {}
    for i, failed_rank in enumerate(failed_ranks):
        new_rank = surviving_ranks[i % len(surviving_ranks)]
        rank_mapping[failed_rank] = new_rank
        print(f"      Rank {failed_rank} → Rank {new_rank}")
    
    print(f"   2. Reconfigure with smaller world size:")
    new_world_size = len(surviving_ranks)
    print(f"      New world size: {new_world_size}")
    
    try:
        # Create new configuration
        torch.manual_seed(123)  # Same seed for comparison
        new_fwd_q, _, new_fwd_k, _, _ = create_qkv_dispatch(
            new_world_size, total_seq_len, num_seqs, max_cp_degree
        )
        
        print(f"      New configuration created successfully")
        
        # Compare load distribution
        old_max_recv = original_fwd_q.num_recv_tokens[:, -1].max().item()
        new_max_recv = new_fwd_q.num_recv_tokens[:, -1].max().item()
        
        print(f"      Load increase: {new_max_recv}/{old_max_recv} = {new_max_recv/old_max_recv:.2f}x")
        
    except Exception as e:
        print(f"      Reconfiguration failed: {e}")
    
    print(f"\n💡 Insight: Fault tolerance requires dynamic reconfiguration")
    print(f"   - Failed ranks cause data loss and load redistribution")
    print(f"   - Recovery strategies have different trade-offs")
    print(f"   - Dynamic world size changes require system support")


def scenario_5_performance_optimization():
    """Scenario 5: Performance optimization analysis."""
    print("\n" + "=" * 70)
    print("ADVANCED SCENARIO 5: Performance Optimization")
    print("=" * 70)
    
    # Analyze performance characteristics of different configurations
    world_size = 8
    base_seq_len = 512
    num_seqs = 16
    hidden_dim = 128
    
    print(f"⚡ Performance optimization analysis:")
    print(f"   Base configuration: {world_size} ranks, {base_seq_len} tokens/rank")
    
    configurations = [
        {"cp_degree": 1, "name": "No CP"},
        {"cp_degree": 2, "name": "CP-2"},
        {"cp_degree": 4, "name": "CP-4"},
        {"cp_degree": 8, "name": "CP-8"},
    ]
    
    print(f"\n📊 Comparing different CP configurations:")
    print(f"{'Config':<8} {'Comm Vol':<10} {'Max Load':<10} {'Imbalance':<12} {'Memory':<10}")
    print(f"-" * 60)
    
    for config in configurations:
        cp_degree = config["cp_degree"]
        name = config["name"]
        
        try:
            torch.manual_seed(42)  # Consistent comparison
            fwd_q, _, fwd_k, _, _ = create_qkv_dispatch(
                world_size, base_seq_len, num_seqs, cp_degree
            )
            
            # Communication volume (total tokens exchanged)
            comm_vol = fwd_q.num_recv_tokens[:, -1].sum().item()
            comm_vol_k = fwd_k.num_recv_tokens[:, -1].sum().item()
            total_comm = comm_vol + comm_vol_k
            
            # Load balance
            q_loads = fwd_q.num_recv_tokens[:, -1]
            max_load = q_loads.max().item()
            load_imbalance = q_loads.float().std() / q_loads.float().mean()
            
            # Memory estimate (rough)
            memory_factor = 1 + (cp_degree - 1) * 0.5  # KV replication
            
            print(f"{name:<8} {total_comm:<10d} {max_load:<10d} {load_imbalance:<12.3f} {memory_factor:<10.1f}x")
            
        except Exception as e:
            print(f"{name:<8} {'Error':<10} {'Error':<10} {'Error':<12} {'Error':<10}")
    
    # Analyze communication patterns
    print(f"\n🔀 Communication pattern analysis:")
    
    torch.manual_seed(42)
    fwd_q, _, fwd_k, _, _ = create_qkv_dispatch(world_size, base_seq_len, num_seqs, 4)
    
    # Count all-to-all vs point-to-point
    q_dispatch = fwd_q.dst_rank.squeeze()
    unique_dests_per_rank = []
    
    for rank in range(world_size):
        dests = q_dispatch[rank][q_dispatch[rank] >= 0]  # Remove padding
        unique_dests = len(torch.unique(dests))
        unique_dests_per_rank.append(unique_dests)
    
    avg_fanout = sum(unique_dests_per_rank) / len(unique_dests_per_rank)
    
    print(f"   Average communication fanout: {avg_fanout:.1f} ranks")
    print(f"   Pattern: {'All-to-all like' if avg_fanout > world_size/2 else 'Point-to-point like'}")
    
    # Bandwidth utilization estimate
    total_data_mb = (fwd_q.num_recv_tokens[:, -1].sum() + fwd_k.num_recv_tokens[:, -1].sum()) * hidden_dim * 4 / (1024**2)
    theoretical_bandwidth_gbps = 100  # Assume 100 Gbps network
    transfer_time_ms = (total_data_mb * 8) / (theoretical_bandwidth_gbps * 1000) * 1000
    
    print(f"   Total data transfer: {total_data_mb:.1f} MB")
    print(f"   Estimated transfer time: {transfer_time_ms:.2f} ms @ 100 Gbps")
    
    print(f"\n💡 Insight: Performance optimization requires multi-dimensional analysis")
    print(f"   - Higher CP degree increases communication volume")
    print(f"   - Load imbalance affects overall throughput")  
    print(f"   - Memory usage grows with replication")
    print(f"   - Network topology affects actual performance")


if __name__ == "__main__":
    print("Communication Metadata System - Advanced Scenarios")
    print("🚀 Exploring complex real-world usage patterns")
    
    scenario_1_large_scale_deployment()
    scenario_2_heterogeneous_sequences()
    scenario_3_memory_constrained_environment()
    scenario_4_fault_tolerance_simulation()
    scenario_5_performance_optimization()
    
    print("\n" + "=" * 70)
    print("🏆 ADVANCED SCENARIOS COMPLETE")
    print("=" * 70)
    print("You now understand advanced concepts:")
    print("  ✅ Large-scale deployment considerations")
    print("  ✅ Handling heterogeneous workloads")
    print("  ✅ Memory-constrained optimization")
    print("  ✅ Fault tolerance strategies")
    print("  ✅ Performance optimization trade-offs")
    print("\nThese scenarios demonstrate the complexity of real-world distributed systems!")