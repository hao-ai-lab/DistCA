import os
import sys
import random
from itertools import accumulate
import argparse
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rich import print

def run(rank: int, world_size: int):
    print(f"Running on rank {rank} of {world_size}")
    pass

if __name__ == "__main__":
    world_size = 4
    mp.spawn(
        run,
        nprocs=world_size,
        args=(world_size,),
        join=True,
    )