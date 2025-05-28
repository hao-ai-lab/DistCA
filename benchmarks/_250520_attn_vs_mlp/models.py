
import time_module
from time_module import compute, network
from dataclasses import dataclass

@dataclass
class Llama8B:
    tp: int = 1
    cp: int = 1
    head_dim: int = 128
    nhqo: int = 32
    nhkv: int = 8
    dtype = "half"

    gpu: str = 'A100-SXM-80GB'

    hidden_size = 4096
    intermediate_size = int(4096 * 3.5)

    def _mlp(self, T: int):
        """Get MLP time for 1 sequence (with T tokens)"""
        def M(m, k, n):
            return compute.gemm_time(
                gpu=self.gpu,
                m=m, k=k, n=n,
                dtype=self.dtype,
            )
        
        d1 = self.hidden_size
        d2 = self.intermediate_size // self.tp
        hdim = self.head_dim
    
        # up-project
        mlp_up = M(T, d1, d2)
    
        # down-project
        mlp_down = M(T, d2, d1)

        # attention qkv
        q_proj = M(T // self.cp, d1, hdim * self.nhqo // self.tp)
        kv_proj = M(T // self.cp, d1, hdim * self.nhkv // self.tp)

        return mlp_up + mlp_down + q_proj + 2 * kv_proj

    def _attn(self, T: int):
        """Get Attention time for 1 sequence (with T tokens)"""
        hqo = self.nhqo // self.tp
        hkv = self.nhkv // self.tp
        return compute.attn_time(
            gpu=self.gpu, 
            cp=self.cp,
            head_dim=self.head_dim,
            nhead=hqo,
            tokens=T,
            dtype=self.dtype,
            is_fwd=True,
        )
    
    pass
