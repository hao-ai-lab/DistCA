from typing import List

import ray
import torch

from attn_kernels.ops import dispatch, nvshmem_finalize, nvshmem_init, nvshmem_my_pe, nvshmem_barrier_all, nvshmem_get_unique_id
from inplace_metadata import compute_dst_offsets, compute_reverse_comm, compute_num_recv_tokens

@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.nvshmem_initialized = False
        self.nvshmem_pe = None

    def init_comm(self, uid: torch.Tensor):
        nvshmem_init(uid, self.rank, self.world_size)
        self.nvshmem_initialized = True
        self.nvshmem_pe = nvshmem_my_pe()

    def test(self, seed: int, tensor,
             dst_id, dst_offset, num_received_per_rank,
             rev_dst_id, rev_dst_offset, rev_num_received_per_rank):
        tensor = tensor.cuda()
        dst_id = dst_id.cuda()
        dst_offset = dst_offset.cuda().to(torch.uint32)
        num_received_per_rank = num_received_per_rank.cuda().to(torch.uint64)
        rev_dst_id = rev_dst_id.cuda()
        rev_dst_offset = rev_dst_offset.cuda().to(torch.uint32)
        rev_num_received_per_rank = rev_num_received_per_rank.cuda().to(torch.uint64)
        torch.manual_seed(seed)

        cp_degree = dst_id.shape[2] if dst_id.dim() == 3 else 1

        # reverse communication
        back_tensor = torch.zeros_like(
            dst_id, dtype=tensor.dtype, device=tensor.device
        )
        num_received = num_received_per_rank[self.world_size]
        # forward communication
        dst_tensor = torch.zeros(
            (num_received,), dtype=tensor.dtype, device=tensor.device
        )
        dst_tensor = self.orchestrate(
            tensor, dst_id, dst_offset,
            dst_tensor=dst_tensor,
            num_recv_tokens=num_received_per_rank
        )
        back_tensor = self.orchestrate(
            dst_tensor, rev_dst_id, rev_dst_offset,
            dst_tensor=back_tensor,
            num_recv_tokens=rev_num_received_per_rank
        )
        back_tensor_dedup = back_tensor.sum(dim=2) / (back_tensor != 0).sum(dim=2)
        assert torch.allclose(
            back_tensor_dedup.unsqueeze(2).repeat(1, 1, cp_degree) * (back_tensor != 0), back_tensor
        )
        # assert torch.allclose(tensor, back_tensor_dedup)
        return dst_tensor, back_tensor_dedup

    @torch.no_grad()
    def orchestrate(self,
                    tensor: torch.Tensor,
                    dst_id: torch.Tensor,
                    dst_offset: torch.Tensor,
                    dst_tensor: torch.Tensor,
                    num_recv_tokens: torch.Tensor):
        print(f"rank {self.rank} orchestrate tensor {tensor[:, :2]=}, {dst_id=}, {dst_offset=}, {dst_tensor=}, {num_recv_tokens=}")
        dispatch(dst_tensor, None, tensor, None, dst_id, dst_offset, None, None, num_recv_tokens, None)
        nvshmem_barrier_all()

    def __del__(self):
        if getattr(self, "nvshmem_initialized", False):
            self.shutdown()

    def shutdown(self):
        if self.nvshmem_initialized:
            nvshmem_finalize()
            self.nvshmem_initialized = False


def orchestrate(tensor: torch.Tensor, dst_id: torch.Tensor, dst_offset: torch.Tensor, dst_tensor: torch.Tensor | None = None):
    world_size = dst_id.shape[0]
    dst_id = dst_id.long()
    dst_offset = dst_offset.long()
    one_hot_dest = torch.nn.functional.one_hot(dst_id.reshape(world_size, -1) + 1, num_classes=world_size + 1).long()[:, :, 1:]
    num_received = torch.sum(one_hot_dest, dim=(0, 1)) # Sum across both source and token dims
    max_received = torch.max(num_received)
    if dst_tensor is None:
        dst_tensor = torch.zeros(world_size, max_received, tensor.shape[2], dtype=tensor.dtype, device=tensor.device)
        orig_shape = dst_tensor.shape
    else:
        orig_shape = dst_tensor.shape
        dst_tensor = dst_tensor.reshape(world_size, -1, tensor.shape[2])

    if dst_id.dim() == 3:
        sp_size = dst_id.shape[2]
    else:
        sp_size = None

    for i in range(world_size):
        if sp_size:
            for j in range(sp_size):
                ids = dst_id[i, :, j]
                offsets = dst_offset[i, :, j]
                mask = ids != -1
                ids = ids[mask]
                offsets = offsets[mask]
                dst_tensor[ids, offsets] = tensor[i][mask]
        else:
            ids = dst_id[i]
            offsets = dst_offset[i]
            # remove indices with value -1 (dummy value as masks)
            mask = ids != -1
            ids = ids[mask]
            offsets = offsets[mask]
            dst_tensor[ids, offsets] = tensor[i][mask]

    return dst_tensor.reshape(orig_shape)


def create_testcase(seed: int, world_size: int, num_tokens: int, cp_degree: int, hidden_size):
    torch.manual_seed(seed)
    dst_id = torch.randint(
        -1, world_size, (world_size, num_tokens, cp_degree), dtype=torch.int32
    )
    if cp_degree == 1:
        dst_id = dst_id.reshape(world_size, num_tokens)

    dst_offset = compute_dst_offsets(dst_id)
    rev_dst_id, rev_dst_offset = compute_reverse_comm(dst_id, dst_offset)
    num_recv_tokens = compute_num_recv_tokens(dst_id)
    rev_num_recv_tokens = compute_num_recv_tokens(rev_dst_id)
    print(f"{dst_id=},\n{dst_offset=},\n{rev_dst_id=},\n{rev_dst_offset=},\n{num_recv_tokens=},\n{rev_num_recv_tokens=}\n" + "=" * 20)

    tensor = torch.randn(world_size, num_tokens, hidden_size)
    dst_tensor = orchestrate(tensor, dst_id, dst_offset)
    back_tensor = torch.zeros_like(dst_id, dtype=dst_tensor.dtype, device=tensor.device)
    back_tensor = back_tensor.repeat(1, 1, hidden_size).reshape(back_tensor.shape + (hidden_size,))
    back_tensor = orchestrate(dst_tensor, rev_dst_id, rev_dst_offset, dst_tensor=back_tensor)
    back_tensor_dedup = back_tensor.sum(dim=2) / (back_tensor != 0).sum(dim=2)
    return dst_id, dst_offset, rev_dst_id, rev_dst_offset, dst_tensor, back_tensor_dedup, num_recv_tokens, rev_num_recv_tokens

def test(seed, world_size, num_tokens, cp_degree, workers: List[ray.ObjectRef], hidden_size: int=128):
    (dst_id, dst_offset, rev_dst_id, rev_dst_offset,
     dst_tensor, back_tensor_dedup, num_recv_tokens, rev_num_recv_tokens) = create_testcase(seed, world_size, num_tokens, cp_degree, hidden_size)
    assert len(workers) == world_size
    refs = []

    for rank, worker in enumerate(workers):
        tensor_slice = dst_tensor[rank]
        assert tensor_slice.shape == (num_tokens, hidden_size)
        dst_id_slice = dst_id[rank]
        dst_offset_slice = dst_offset[rank].long()
        rev_dst_id_slice = rev_dst_id[rank]
        rev_dst_offset_slice = rev_dst_offset[rank].long()
        num_received_slice = num_recv_tokens[rank].long()
        rev_num_received_slice = rev_num_recv_tokens[rank].long()
        ref = worker.test.remote(
            seed, tensor_slice,
            dst_id_slice, dst_offset_slice, num_received_slice,
            rev_dst_id_slice, rev_dst_offset_slice, rev_num_received_slice
        )
        refs.append(ref)

    results = ray.get(refs)
    dst_tensor_out = torch.cat([r[0] for r in results], dim=0)
    back_tensor_out = torch.cat([r[1] for r in results], dim=0)
    assert torch.allclose(dst_tensor_out, dst_tensor)
    assert torch.allclose(back_tensor_out, back_tensor_dedup)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--cp-degree", type=int, default=1)
    args = parser.parse_args()
    ray.init()
    workers = [
        Worker.remote(i, args.world_size)
        for i in range(args.world_size)
    ]

    uid = nvshmem_get_unique_id()
    ray.get([
        worker.init_comm.remote(uid)
        for worker in workers
    ])

    print("init done")
    test(0, args.world_size, args.num_tokens, args.cp_degree, workers)
    print("test done")
    ray.get([
        worker.shutdown.remote()
        for worker in workers
    ])

