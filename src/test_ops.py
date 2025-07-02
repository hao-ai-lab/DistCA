from typing import List

import ray
import torch

from attn_kernels.ops import dispatch, nvshmem_finalize, nvshmem_init, nvshmem_my_pe, nvshmem_barrier_all, nvshmem_get_unique_id, DispatcherWrapper
from inplace_metadata import compute_dst_offsets, compute_reverse_comm, compute_num_recv_tokens

@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.nvshmem_initialized = False
        self.nvshmem_pe = None
        self.dispatcher = None

    def init_comm(self, uid: torch.Tensor, stride: int, max_tokens_query: int, max_tokens_key_value: int):
        nvshmem_init(uid, self.rank, self.world_size)
        self.nvshmem_initialized = True
        self.nvshmem_pe = nvshmem_my_pe()
        self.dispatcher = DispatcherWrapper(
            q_stride=stride,
            kv_stride=stride,
            max_tokens_query=max_tokens_query,
            max_tokens_key_value=max_tokens_key_value,
            rank=self.rank,
            world_size=self.world_size,
        )

    def test(self, seed: int, tensor: torch.Tensor,
             dst_id: torch.Tensor, dst_offset: torch.Tensor, num_received_per_rank: torch.Tensor,
             rev_dst_id: torch.Tensor, rev_dst_offset: torch.Tensor, rev_num_received_per_rank: torch.Tensor):
        tensor = tensor.cuda().contiguous()
        dst_id = dst_id.cuda().contiguous()
        dst_offset = dst_offset.cuda().to(torch.uint32).contiguous()
        num_received_per_rank = num_received_per_rank.cuda().to(torch.uint64).contiguous()
        rev_dst_id = rev_dst_id.cuda().contiguous()
        rev_dst_offset = rev_dst_offset.cuda().to(torch.uint32).contiguous()
        rev_num_received_per_rank = rev_num_received_per_rank.cuda().to(torch.uint64).contiguous()
        hidden_size = tensor.shape[-1]
        torch.manual_seed(seed)

        cp_degree = dst_id.shape[2] if dst_id.dim() == 3 else 1

        # reverse communication
        back_tensor = torch.zeros_like(
            dst_id, dtype=tensor.dtype, device=tensor.device
        ).unsqueeze(-1).repeat(*((1,)* dst_id.ndim), hidden_size)
        num_received = num_received_per_rank[self.world_size]
        # forward communication
        dst_tensor = torch.zeros(
            (num_received, hidden_size), dtype=tensor.dtype, device=tensor.device
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
        if cp_degree > 1:
            back_tensor_dedup = back_tensor.sum(dim=2) / (back_tensor != 0).sum(dim=2)
            assert torch.allclose(
                back_tensor_dedup.unsqueeze(2).repeat(1, 1, cp_degree) * (back_tensor != 0), back_tensor
            )
        else:
            back_tensor_dedup = back_tensor
        assert torch.allclose(tensor, back_tensor_dedup)
        print(f"rank {self.rank} test done")
        return dst_tensor, back_tensor_dedup

    @torch.no_grad()
    def orchestrate(self,
                    tensor: torch.Tensor,
                    dst_id: torch.Tensor,
                    dst_offset: torch.Tensor,
                    dst_tensor: torch.Tensor,
                    num_recv_tokens: torch.Tensor):
        # print(f"rank {self.rank} orchestrate tensor {tensor[:, :2]=}, {dst_id=}, {dst_offset=}, {dst_tensor.shape=}, {num_recv_tokens=}")
        dispatch(dst_tensor, None, tensor, None, dst_id, dst_offset, None, None, num_recv_tokens, None, self.dispatcher)
        nvshmem_barrier_all()
        return dst_tensor

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
        assert dst_tensor.dtype == tensor.dtype and dst_tensor.device == tensor.device

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
        0, world_size, (world_size, num_tokens, cp_degree), dtype=torch.int32
    )
    if cp_degree == 1:
        dst_id = dst_id.reshape(world_size, num_tokens)

    dst_offset = compute_dst_offsets(dst_id)
    rev_dst_id, rev_dst_offset = compute_reverse_comm(dst_id, dst_offset)
    num_recv_tokens = compute_num_recv_tokens(dst_id)
    rev_num_recv_tokens = compute_num_recv_tokens(rev_dst_id)
    # print(f"{dst_id=},\n{dst_offset=},\n{rev_dst_id=},\n{rev_dst_offset=},\n{num_recv_tokens=},\n{rev_num_recv_tokens=}\n" + "=" * 20)

    tensor = torch.randn(world_size, num_tokens, hidden_size, dtype=torch.float16)
    dst_tensor = orchestrate(tensor, dst_id, dst_offset)
    back_tensor = torch.zeros_like(dst_id, dtype=dst_tensor.dtype, device=tensor.device)
    back_tensor = back_tensor.unsqueeze(-1).repeat(*((1,) * dst_id.ndim), hidden_size)
    back_tensor = orchestrate(dst_tensor, rev_dst_id, rev_dst_offset, dst_tensor=back_tensor)
    if cp_degree > 1:
        back_tensor_dedup = back_tensor.sum(dim=2) / (back_tensor != 0).sum(dim=2)
    else:
        back_tensor_dedup = back_tensor
    return dst_id, dst_offset, rev_dst_id, rev_dst_offset, tensor, dst_tensor, back_tensor_dedup, num_recv_tokens, rev_num_recv_tokens

@torch.no_grad()
def test(seed, world_size, num_tokens, cp_degree, workers: List[ray.ObjectRef], hidden_size: int=128):
    (dst_id, dst_offset, rev_dst_id, rev_dst_offset,
     tensor, dst_tensor, back_tensor_dedup, num_recv_tokens, rev_num_recv_tokens) = create_testcase(seed, world_size, num_tokens, cp_degree, hidden_size)
    assert len(workers) == world_size
    refs = []

    for rank, worker in enumerate(workers):
        tensor_slice = tensor[rank]
        assert tensor_slice.shape == (num_tokens, hidden_size), tensor_slice.shape
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
    dst_tensor_outs = [r[0] for r in results]
    # The first dimension of each value in the list should be padded to the same as dst_tensor's dim 1
    max_dim1 = dst_tensor.shape[1]
    dst_tensor_outs = [
        torch.cat([r, torch.zeros(max_dim1 - r.shape[0], *r.shape[1:], dtype=r.dtype, device=r.device)], dim=0).unsqueeze(0)
        for r in dst_tensor_outs
    ]
    dst_tensor_out = torch.cat(dst_tensor_outs, dim=0)
    assert dst_tensor_out.shape == dst_tensor.shape
    assert torch.allclose(dst_tensor_out, dst_tensor.to(dst_tensor_out.device))
    back_tensor_out = torch.cat([r[1].unsqueeze(0) for r in results], dim=0)
    # assert torch.allclose(dst_tensor_out, dst_tensor)
    assert torch.allclose(back_tensor_out, back_tensor_dedup.to(back_tensor_out.device))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--cp-degree", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=128)
    args = parser.parse_args()
    ray.init()
    workers = [
        Worker.remote(i, args.world_size)
        for i in range(args.world_size)
    ]

    uid = nvshmem_get_unique_id()
    stride = args.hidden_size * torch.float16.itemsize
    max_tokens_query = args.num_tokens * args.world_size
    max_tokens_key_value = args.num_tokens * args.world_size
    ray.get([
        worker.init_comm.remote(uid, stride, max_tokens_query, max_tokens_key_value)
        for worker in workers
    ])

    print("init done")
    test(0, args.world_size, args.num_tokens, args.cp_degree, workers, args.hidden_size)
    print("test done")
    ray.get([
        worker.shutdown.remote()
        for worker in workers
    ])

