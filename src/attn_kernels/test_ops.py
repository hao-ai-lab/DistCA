from ops import dispatch, nvshmem_finalize, nvshmem_init, nvshmem_my_pe, nvshmem_barrier_all

import ray
import torch

from ..inplace_metadata import compute_dst_offsets, compute_reverse_comm, compute_num_recv_tokens

class Worker:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.nvshmem_initialized = False
        self.nvshmem_pe = None

    def init_comm(self):
        nvshmem_init(self.rank, self.world_size)
        self.nvshmem_initialized = True
        self.nvshmem_pe = nvshmem_my_pe()

    def test(self, seed: int, tensor,
             dst_id, dst_offset, num_received_per_rank,
             rev_dst_id, rev_dst_offset, rev_num_received_per_rank):
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

    def orchestrate(self,
                    tensor: torch.Tensor,
                    dst_id: torch.Tensor,
                    dst_offset: torch.Tensor,
                    dst_tensor: torch.Tensor,
                    num_recv_tokens: torch.Tensor):
        dispatch(dst_tensor, None, tensor, None, dst_id, dst_offset, None, None, num_recv_tokens, None)
        nvshmem_barrier_all()

    def __del__(self):
        if self.nvshmem_initialized:
            nvshmem_finalize()

def orchestrate(tensor: torch.Tensor, dst_id: torch.Tensor, dst_offset: torch.Tensor, dst_tensor: torch.Tensor | None = None):
        world_size = dst_id.shape[0]
        one_hot_dest = torch.nn.functional.one_hot(dst_id.reshape(world_size, -1) + 1, num_classes=world_size + 1).long()[:, :, 1:]
        num_received = torch.sum(one_hot_dest, dim=(0, 1)) # Sum across both source and token dims
        max_received = torch.max(num_received)
        if dst_tensor is None:
            dst_tensor = torch.zeros(world_size, max_received, dtype=tensor.dtype, device=tensor.device)
            orig_shape = dst_tensor.shape
        else:
            orig_shape = dst_tensor.shape
            dst_tensor = dst_tensor.reshape(world_size, -1)

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

def create_testcase(seed: int, world_size: int, num_tokens: int, cp_degree: int):
    torch.manual_seed(seed)
    dst_id = torch.randint(
        -1, world_size, (world_size, num_tokens, cp_degree), dtype=torch.uint32
    )
    dst_offset = compute_dst_offsets(dst_id)
    rev_dst_id, rev_dst_offset = compute_reverse_comm(dst_id, dst_offset)
    num_recv_tokens = compute_num_recv_tokens(dst_id)
    rev_num_recv_tokens = compute_num_recv_tokens(rev_dst_id)

    tensor = torch.randn(world_size, num_tokens, device="cuda")
    dst_tensor = orchestrate(tensor, dst_id, dst_offset)
    back_tensor = torch.zeros_like(dst_id, dtype=dst_tensor.dtype, device=tensor.device)
    back_tensor = orchestrate(dst_tensor, rev_dst_id, rev_dst_offset, dst_tensor=back_tensor)
    back_tensor_dedup = back_tensor.sum(dim=2) / (back_tensor != 0).sum(dim=2)
    return dst_id, dst_offset, rev_dst_id, rev_dst_offset, dst_tensor, back_tensor_dedup, num_recv_tokens, rev_num_recv_tokens

def test(seed, world_size, num_tokens, cp_degree, workers):
    dst_id, dst_offset, rev_dst_id, rev_dst_offset, dst_tensor, back_tensor_dedup, num_recv_tokens, rev_num_recv_tokens = create_testcase(seed, world_size, num_tokens, cp_degree)
    assert len(workers) == world_size
    refs = []

    for rank, worker in enumerate(workers):
        tensor_slice = dst_tensor[rank]
        dst_id_slice = dst_id[rank]
        dst_offset_slice = dst_offset[rank]
        rev_dst_id_slice = rev_dst_id[rank]
        rev_dst_offset_slice = rev_dst_offset[rank]
        num_received_slice = num_recv_tokens[rank]
        rev_num_received_slice = rev_num_recv_tokens[rank]
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
