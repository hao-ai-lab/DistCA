from dataclasses import dataclass
import os
import socket
from typing import Optional
import math

from megatron.core import parallel_state as mpu
import ray
import torch

import d2.runtime.inplace_metadata_cp
from d2.runtime.inplace_metadata_cp import create_qkv_dispatch_2cp_with_explicit_sharding

import rich


def print_results(results):
    rich.print("fwd_q_metadata: \n", results[0], "\n")
    rich.print("rev_q_metadata: \n", results[1], "\n")
    rich.print("fwd_k_metadata: \n", results[2], "\n")
    rich.print("rev_k_metadata: \n", results[3], "\n")
    rich.print("attention_metadata: \n", results[4], "\n")

def test_case_1():
    rich.print("Test case 1: equal sharding, with only rank = 0 have sequence to dispatch and all others are 0.")

    results = create_qkv_dispatch_2cp_with_explicit_sharding(
        world_size=4,
        
        # # seq_lens[rank, seq_id] = length of this sequence
        # assert that after the first 0, all the rest are 0.
        seq_lens=torch.tensor([
            [1024, 2048], 
            [0, 0],
            [0, 0],
            [0, 0],
        ]), 
        
        # # cp_dst[rank, seq_id, shard_id] = dst_rank
        cp_dst=torch.tensor([
            [[0, 1, 2, 3, 3, 2, 1, 0],         [0, 1, 2, 3, 3, 2, 1, 0],       ],
            [[-1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1]],
            [[-1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1]],
            [[-1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1]],
        ]),
        
        # # cp_shards[rank, seq_id] = length of this shard to that rank
        cp_shards=torch.tensor([
            # unequal sharding, sum to 1024             # unequal sharding, sum to 2048
            [[64, 128, 192, 128, 64, 128, 192, 128],    [128, 256, 384, 256, 128, 256, 384, 256], ],
            # [[128, 128, 128, 128, 128, 128, 128, 128],  [256, 256, 256, 256, 256, 256, 256, 256],],
            [[-1, -1, -1, -1, -1, -1, -1, -1],          [-1, -1, -1, -1, -1, -1, -1, -1]],
            [[-1, -1, -1, -1, -1, -1, -1, -1],          [-1, -1, -1, -1, -1, -1, -1, -1]],
            [[-1, -1, -1, -1, -1, -1, -1, -1],          [-1, -1, -1, -1, -1, -1, -1, -1]],
        ]),
    )
    print_results(results)
    rich.print("Test case 1: done")

def test_case_2():
    rich.print("Test case 2: unequal sharding, with only rank = 0 have sequence to dispatch and all others are 0.")
    results = create_qkv_dispatch_2cp_with_explicit_sharding(
        world_size=4,
        # Define how many sequences are we going to dispatch
        seq_lens=torch.tensor([
            [1024, 2048], 
            [128, 0],
            [128, 0],
            [128, 0],
        ]), 
        cp_dst=torch.tensor([
            [[0, 1, 2, 3, 3, 2, 1, 0],         [0, 1, 2, 3, 3, 2, 1, 0],       ],
            [[1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1]],
            [[2, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1]],
            [[3, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1]],
        ]),
        # # cp_shards[rank, seq_id] = length of this shard to that rank
        # cp_shards=torch.tensor([
        #     [128, 128, 128, 128, 128, 128, 128, 128], # equal sharding
        #     [256, 256, 256, 256, 256, 256, 256, 256], # equal sharding
        # ]),
        cp_shards=torch.tensor([
            # unequal sharding, sum to 1024            # unequal sharding, sum to 2048
            [[64, 128, 192, 128, 64, 128, 192, 128],   [128, 256, 384, 256, 128, 256, 384, 256], ],
            [[128, -1, -1, -1, -1, -1, -1, -1],         [-1, -1, -1, -1, -1, -1, -1, -1]],
            [[128, -1, -1, -1, -1, -1, -1, -1],         [-1, -1, -1, -1, -1, -1, -1, -1]],
            [[128, -1, -1, -1, -1, -1, -1, -1],         [-1, -1, -1, -1, -1, -1, -1, -1]],
        ]),
    )
    print_results(results)  
    rich.print("Test case 2: done")

if __name__ == "__main__":
    # d2.runtime.inplace_metadata_cp.VERBOSE = True
    # d2.runtime.inplace_metadata_cp.DEBUG = True

    test_case_1()
    test_case_2()