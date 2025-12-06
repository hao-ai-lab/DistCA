
import torch
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import Optional
from d2.runtime.megatron.d2_rope import apply_rotary_pos_emb_d2

@dataclass
class MockTransformerConfig:
    rotary_interleaved: bool = False
    multi_latent_attention: bool = False

def mock_apply_rotary_pos_emb_bshd(t, freqs, rotary_interleaved, multi_latent_attention, mscale=1.0):
    return t + freqs


class TestRoPEWithLogicalRange(unittest.TestCase):
    
    def setUp(self):
        self.config = MockTransformerConfig()
        self.device = "cpu"
        self.hidden_dim = 4
        self.num_heads = 1
        
        max_seq_len = 2000
        self.freqs = torch.arange(max_seq_len, dtype=torch.float32).view(max_seq_len, 1, 1, 1).expand(max_seq_len, 1, 1, self.hidden_dim)

    # Patch the mglm _apply_rotary_pos_emb_bshd RoPE function to check the correctness of the logical position.
    @patch('d2.runtime.megatron.d2_rope._apply_rotary_pos_emb_bshd', mock_apply_rotary_pos_emb_bshd)
    def test_discontinuous_logical_ranges(self):
        
        t = torch.zeros(5, self.num_heads, self.hidden_dim)
        
        cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
        shard_logical_range = torch.tensor([
            [100, 102],
            [0, 3]
        ], dtype=torch.long)
        
        output = apply_rotary_pos_emb_d2(
            t=t,
            freqs=self.freqs,
            config=self.config,
            cu_seqlens=cu_seqlens,
            shard_logical_range=shard_logical_range
        )
        

        self.assertEqual(output.shape, t.shape)
        
        print(f"Checking Token 0 (Expect Logical Pos 100): {output[0][0,0].item()}")
        self.assertTrue(torch.allclose(output[0], torch.full_like(output[0], 100.0)))
        
        self.assertTrue(torch.allclose(output[1], torch.full_like(output[1], 101.0)))
        
        print(f"Checking Token 2 (Expect Logical Pos 0): {output[2][0,0].item()}")
        self.assertTrue(torch.allclose(output[2], torch.full_like(output[2], 0.0)))
        
        self.assertTrue(torch.allclose(output[3], torch.full_like(output[3], 1.0)))
        
        self.assertTrue(torch.allclose(output[4], torch.full_like(output[4], 2.0)))

    @patch('d2.runtime.megatron.d2_rope._apply_rotary_pos_emb_bshd', mock_apply_rotary_pos_emb_bshd)
    def test_mismatch_error(self):
        t = torch.zeros(5, 1, 4)
        cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32) # Shard 0 len=2
        
        shard_logical_range = torch.tensor([[100, 105], [0, 3]], dtype=torch.long)
        
        with self.assertRaises(ValueError):
            apply_rotary_pos_emb_d2(
                t=t, freqs=self.freqs, config=self.config,
                cu_seqlens=cu_seqlens, shard_logical_range=shard_logical_range
            )

if __name__ == '__main__':
    unittest.main()