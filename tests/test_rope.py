import torch
import rich
from unittest.mock import patch
from dataclasses import dataclass
from typing import List, Optional

from test_util import ParallelConfig
from d2.planner.planner import batch_to_items_general, Planner
from d2.runtime.megatron.d2_rope import apply_rotary_pos_emb_d2


@dataclass
class MockTransformerConfig:
    rotary_interleaved: bool = False
    multi_latent_attention: bool = False


def mock_apply_rotary_pos_emb_bshd(t, freqs, rotary_interleaved, multi_latent_attention, mscale=1.0):
    return t + freqs


def test_rope_with_logical_range():
    rich.print("[bold yellow]🟡 Starting RoPE Logical Range Unit Test...[/bold yellow]")
    
    config = MockTransformerConfig()
    device = "cpu"
    hidden_dim = 4
    num_heads = 1
    
    max_seq_len = 2000
    freqs = torch.arange(max_seq_len, dtype=torch.float32).view(max_seq_len, 1, 1, 1).expand(max_seq_len, 1, 1, hidden_dim)

    rich.print("  Checking Discontinuous Logical Ranges...")
    
    t = torch.zeros(5, num_heads, hidden_dim)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
    
    shard_logical_range = torch.tensor([
        [100, 102],
        [0, 3]
    ], dtype=torch.long)
    
    patch_path = 'd2.runtime.megatron.d2_rope._apply_rotary_pos_emb_bshd'
    
    with patch(patch_path, side_effect=mock_apply_rotary_pos_emb_bshd):
        output = apply_rotary_pos_emb_d2(
            t=t,
            freqs=freqs,
            config=config,
            cu_seqlens=cu_seqlens,
            shard_logical_range=shard_logical_range
        )
        
        assert output.shape == t.shape, f"Shape mismatch. Got {output.shape}"
        
        val_0 = output[0][0,0].item()
        assert torch.allclose(output[0], torch.full_like(output[0], 100.0)), \
            f"Token 0 error. Expected 100.0, got {val_0}"
        
        assert torch.allclose(output[1], torch.full_like(output[1], 101.0)), "Token 1 error"
        
        val_2 = output[2][0,0].item()
        assert torch.allclose(output[2], torch.full_like(output[2], 0.0)), \
            f"Token 2 error (Jump check). Expected 0.0, got {val_2}"
        
        assert torch.allclose(output[3], torch.full_like(output[3], 1.0)), "Token 3 error"
        
        assert torch.allclose(output[4], torch.full_like(output[4], 2.0)), "Token 4 error"

    rich.print(f"[bold green][PASS][/bold green] Discontinuous Ranges check passed.")

    rich.print("  Checking Length Mismatch Error validation...")
    
    t_mismatch = torch.zeros(5, 1, 4)
    cu_seqlens_mismatch = torch.tensor([0, 2, 5], dtype=torch.int32)
    shard_logical_range_mismatch = torch.tensor([[100, 105], [0, 3]], dtype=torch.long)
    
    error_raised = False
    try:
        with patch(patch_path, side_effect=mock_apply_rotary_pos_emb_bshd):
            apply_rotary_pos_emb_d2(
                t=t_mismatch, freqs=freqs, config=config,
                cu_seqlens=cu_seqlens_mismatch, shard_logical_range=shard_logical_range_mismatch
            )
    except ValueError as e:
        error_raised = True
    except Exception as e:
        rich.print(f"[bold red]FAILED[/bold red] Caught unexpected exception type: {type(e)}")
        raise e

    assert error_raised, "Function failed to raise ValueError for length mismatch!"
    
    rich.print(f"[bold green][PASS][/bold green] Mismatch Error validation passed.")
    rich.print(f"[bold green][PASS][/bold green] test_rope_with_logical_range Completed Successfully")


class MockConfig:
    def __init__(self):
        self.hidden_size = 4096
        self.num_attention_heads = 32
        self.num_key_value_heads = 8
        self.num_hidden_layers = 32


def test_mlp_physical_length_and_logical_range():
    model_config = MockConfig()
    parallel_config = ParallelConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    world_size = 4
    tolerance_factor = 0.1
    planner = Planner(
        world_size=world_size,
        parallel_config=parallel_config,
        model_config=model_config,
        tolerance_factor=tolerance_factor
    )
    rich.print("  Checking DP Scenario...")
    batches: List[List[int]] = [[256, 256],[128, 384],[512], [10, 502]]
    num_batched_token = 512
    dp_degree = world_size // parallel_config.tensor_model_parallel_size // parallel_config.pipeline_model_parallel_size

    dp_cp_test_items = batch_to_items_general(batches, num_tokens_per_rank=num_batched_token, DP_degree=dp_degree, model_config = model_config) 
    fa2a_metadata_0, as_attn_metadata_0, mlp_shard_len_0, shard_logical_range_0 = planner.plan(dp_cp_test_items, device='cpu')
    
    expected_output_len = [
        torch.tensor([256, 256], dtype=torch.int32),
        torch.tensor([128, 384], dtype=torch.int32),
        torch.tensor([512], dtype=torch.int32),
        torch.tensor([10, 502], dtype=torch.int32),
    ]

    for i in range(world_size):
        assert torch.equal(mlp_shard_len_0[i], expected_output_len[i]), \
            f"[DP] Rank {i} length tensor is wrong\n Expected: {expected_output_len[i]}\n Actual: {mlp_shard_len_0[i]}"
    
    expected_output_range = [
        torch.tensor([[0, 256], [0, 256]], dtype=torch.long),
        torch.tensor([[0, 128], [0, 384]], dtype=torch.long),
        torch.tensor([[0, 512]], dtype=torch.long),
        torch.tensor([[0, 10], [0, 502]], dtype=torch.long),
    ]

    for i in range(world_size):
        actual = shard_logical_range_0[i].to(torch.long)
        expected = expected_output_range[i].to(torch.long)
        assert torch.equal(actual, expected), \
            f"[DP] Rank {i} logical range is wrong\n Expected: {expected}\n Actual: {actual}"

    rich.print(f"[bold green][PASS][/bold green] test_mlp_seq_len Passed MLP DP test")

    rich.print("  Checking DPCP Scenario...")
    batches: List[List[int]] = [[256, 1024],[256], [128, 384] ]
    num_batched_token = 512
    dp_degree = world_size // parallel_config.tensor_model_parallel_size // parallel_config.pipeline_model_parallel_size

    dp_cp_test_items = batch_to_items_general(batches, num_tokens_per_rank=num_batched_token, DP_degree=dp_degree, model_config = model_config)
    fa2a_metadata_0, as_attn_metadata_0, mlp_shard_len_0, shard_logical_range_0 = planner.plan(dp_cp_test_items, device='cpu')

    expected_output_len = [
        torch.tensor([256, 128, 128], dtype=torch.int32), 
        torch.tensor([256, 256], dtype=torch.int32),
        torch.tensor([128, 128, 256], dtype=torch.int32),
        torch.tensor([128, 384], dtype=torch.int32),
    ]

    for i in range(world_size):
        assert torch.equal(mlp_shard_len_0[i], expected_output_len[i]), \
            f"[CP] Rank {i} length tensor is wrong\n Expected: {expected_output_len[i]}\n Actual: {mlp_shard_len_0[i]}"

    expected_output_range = [
        # Rank 0:
        # 1. Doc 0: [0, 256)
        # 2. Doc 1 Head 1: [0, 128)
        # 3. Doc 1 Tail 1: [896, 1024)
        torch.tensor([[0, 256], [0, 128], [896, 1024]], dtype=torch.long),
        # Rank 1:
        # 1. Doc 1 Head 2: [128, 384)
        # 2. Doc 1 Tail 2: [640, 896)
        torch.tensor([[128, 384], [640, 896]], dtype=torch.long),
        # Rank 2:
        # 1. Doc 1 Head 3: [384, 512)
        # 2. Doc 1 Tail 3: [512, 640)
        # 3. Doc 2: [0, 256)
        torch.tensor([[384, 512], [512, 640], [0, 256]], dtype=torch.long),
        # Rank 3:
        # 1. Doc 3: [0, 128)
        # 2. Doc 4: [0, 384)
        torch.tensor([[0, 128], [0, 384]], dtype=torch.long),
    ]
    for i in range(world_size):
        actual = shard_logical_range_0[i].to(torch.long)
        expected = expected_output_range[i].to(torch.long)
        assert torch.equal(actual, expected), \
            f"[CP] Rank {i} logical range is wrong\n Expected: {expected}\n Actual: {actual}"

    rich.print(f"[bold green][PASS][/bold green] test_mlp_seq_len Passed MLP CP test (Head/Tail Policy)")
    return


def test_end_to_end_rope_verification():
    rich.print("[bold yellow]🟡 Starting End-to-End RoPE Verification (Planner -> RoPE)...[/bold yellow]")
    
    model_config = MockConfig()
    transformer_config = MockTransformerConfig()
    parallel_config = ParallelConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    world_size = 4
    tolerance_factor = 0.1
    planner = Planner(
        world_size=world_size,
        parallel_config=parallel_config,
        model_config=model_config,
        tolerance_factor=tolerance_factor
    )
    
    device = "cpu"
    hidden_dim = 4
    num_heads = 1
    
    batches: List[List[int]] = [[256, 1024], [256], [128, 384]]
    num_batched_token = 512
    dp_degree = world_size

    dp_cp_items = batch_to_items_general(batches, num_tokens_per_rank=num_batched_token, DP_degree=dp_degree, model_config=model_config)
    
    _, _, mlp_shard_lens, shard_logical_ranges = planner.plan(dp_cp_items, device=device)
    
    max_seq_len = 2048
    freqs = torch.arange(max_seq_len, dtype=torch.float32).view(max_seq_len, 1, 1, 1).expand(max_seq_len, 1, 1, hidden_dim)

    patch_path = 'd2.runtime.megatron.d2_rope._apply_rotary_pos_emb_bshd'
    with patch(patch_path, side_effect=mock_apply_rotary_pos_emb_bshd):
        for rank_id in range(world_size):
            rich.print(f"    Verifying Rank {rank_id}...")
            
            phys_lens = mlp_shard_lens[rank_id]
            cu_seqlens = torch.zeros(len(phys_lens) + 1, dtype=torch.int32)
            torch.cumsum(phys_lens, dim=0, out=cu_seqlens[1:])
            
            total_len = cu_seqlens[-1].item()
            t_in = torch.zeros(total_len, num_heads, hidden_dim)
            
            logical_range = shard_logical_ranges[rank_id]
            
            output = apply_rotary_pos_emb_d2(
                t=t_in,
                freqs=freqs,
                config=transformer_config,
                cu_seqlens=cu_seqlens,
                shard_logical_range=logical_range
            )
            
            expected_pos_ids = []
            
            if rank_id == 0:
                expected_pos_ids.append(torch.arange(0, 256))
                expected_pos_ids.append(torch.arange(0, 128))
                expected_pos_ids.append(torch.arange(896, 1024))
                
            elif rank_id == 1:
                expected_pos_ids.append(torch.arange(128, 384))
                expected_pos_ids.append(torch.arange(640, 896))
                
            elif rank_id == 2:
                expected_pos_ids.append(torch.arange(384, 512))
                expected_pos_ids.append(torch.arange(512, 640))
                expected_pos_ids.append(torch.arange(0, 256))
                
            elif rank_id == 3:
                expected_pos_ids.append(torch.arange(0, 128))
                expected_pos_ids.append(torch.arange(0, 384))

            expected_tensor = torch.cat(expected_pos_ids).float()
            
            assert output.shape[0] == expected_tensor.shape[0], f"Rank {rank_id} shape mismatch"
            
            actual_vals = output[:, 0, 0]
            
            if not torch.allclose(actual_vals, expected_tensor):
                diff_indices = torch.nonzero(actual_vals != expected_tensor).flatten()
                first_diff = diff_indices[0].item()
                rich.print(f"[bold red]FAIL[/bold red] Rank {rank_id} Value Mismatch at index {first_diff}")
                rich.print(f"  Expected: {expected_tensor[first_diff]}")
                rich.print(f"  Actual:   {actual_vals[first_diff]}")
                shard_idx = 0
                for i, l in enumerate(phys_lens):
                    if first_diff < l:
                        shard_idx = i
                        break
                    first_diff -= l
                rich.print(f"  Error occurred in Shard {shard_idx} (PhysLen: {phys_lens[shard_idx]})")
                raise AssertionError(f"Rank {rank_id} RoPE verification failed")

            rich.print(f"    [green]OK[/green] Rank {rank_id} passed.")

    rich.print(f"[bold green][PASS][/bold green] End-to-End RoPE Verification Completed Successfully.")


if __name__ == '__main__':
    test_mlp_physical_length_and_logical_range()
    test_rope_with_logical_range()
    test_end_to_end_rope_verification()
