"""
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
torchrun --nnodes 1 --nproc_per_node 4 test_raw_alltoall.py \
    --world-size 4
"""

import torch
from d2.runtime.attn_kernels.ops import (
    nvshmem_init
)
from d2.runtime.attn_kernels.dispatch import (
    fast_a2a, FastDispatcherWrapper ,
)
from d2.runtime.attn_kernels.dispatch import fast_a2a

from test_util import BaseWorker

import os
from torch import tensor


def test_raw_alltoall():
    instance_id = 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    buffer_size = 109051904

    assert world_size == 4, f"This test is only designed for world size 4"

    worker = BaseWorker(rank, world_size)
    worker.init_comm(buffer_size=buffer_size)

    fa2a_metadata = (
        tensor([
            [       0,  6116352,  6116352, 11184128],
            [       0,        0,  6291456,  6291456],
            [       0,        0,        0,  6291456],
            [       0,        0,        0,        0]]),
        tensor([
            [6116352,       0, 5067776, 5592576],
            [      0, 6291456,       0,       0],
            [      0,       0, 6291456,       0],
            [      0,       0,       0, 6291456]]),
        tensor([
            [       0,        0,        0,        0],
            [ 6116352,        0,  5067776,  5592576],
            [ 6116352,  6291456,  5067776,  5592576],
            [ 6116352,  6291456, 11359232,  5592576]]),
        tensor([
            [6116352,       0,       0,       0],
            [      0, 6291456,       0,       0],
            [5067776,       0, 6291456,       0],
            [5592576,       0,       0, 6291456]])
    )

    sender_send_disp, sender_transfer_sz, sender_recv_disp, recver_transfer_sz = [
        t[rank].cuda().to(torch.uint64) 
        for t in fa2a_metadata
    ]

    my_rank_send_offset = sender_send_disp[rank].item()
    my_rank_recv_offset = sender_recv_disp[rank].item()
    my_rank_send_sz = sender_transfer_sz[rank].item()


    ret = fast_a2a(
        sender_send_disp, sender_transfer_sz,
        sender_recv_disp, recver_transfer_sz,
        my_rank_send_offset, my_rank_recv_offset, my_rank_send_sz,
    )

    torch.cuda.synchronize()
    torch.distributed.barrier()

    import rich
    if rank == 0:
        rich.print(ret)
        rich.print(f"🟢 Test raw alltoall passed (ignore all warnings on destructor - it's a known issue)")


if __name__ == "__main__":
    test_raw_alltoall()