#include <cuda.h>
#include <cuda_runtime.h>

#include "core/cuda_utils.h"

namespace attn{
namespace {
template <bool TO_NVSHMEM>
__global__ void memcpy_nvshmem_non_cp(
  uint8_t *tensor,
  uint8_t *nvshmem_buffer,
  const int64_t *seq_nvshmem_offset,
  const int64_t *seq_tokens,
  const int64_t token_bytes,
  const int32_t total_num_tokens
) {

  size_t cur_seq = 0;
  size_t cur_seq_end = __ldg(&seq_tokens[0]);
  size_t cur_seq_start_token_id = 0;
  size_t cur_seq_start_offset = __ldg(&seq_nvshmem_offset[0]);

  for (size_t token_idx = blockIdx.x; token_idx < total_num_tokens;
       token_idx += gridDim.x) {
    while (token_idx > cur_seq_end) {
      // Move to the next sequence
      cur_seq += 1;
      cur_seq_start_offset = __ldg(&seq_nvshmem_offset[cur_seq]);

      cur_seq_start_token_id = cur_seq_end;
      cur_seq_end += __ldg(&seq_tokens[cur_seq]);
    }

    const uint8_t *src_ = tensor + token_idx * token_bytes;
    uint8_t *nvshmem_ = (
      // sequence start (bytes)
      nvshmem_buffer + cur_seq_start_offset +
      // token offset (bytes)
      (token_idx - cur_seq_start_token_id) * token_bytes
    );

    for (size_t i = threadIdx.x; i * sizeof(int4) < token_bytes;
         i += blockDim.x) {
      // i-th byte's offset in src
      if constexpr (TO_NVSHMEM) {
        ((int4*)nvshmem_)[i] = ((int4*)src_)[i];
      } else {
        // Copying from nvshmem to src
        ((int4*)src_)[i] = ((int4*)nvshmem_)[i];
      }
    }
  }
}


__global__ void memcpy_to_nvshmem_cp(
  uint8_t *tensor,
  uint8_t *nvshmem_buffer,
  const int8_t *do_replica,
  const int64_t *seq_nvshmem_offset,
  const int64_t *seq_tokens,
  const int64_t token_bytes,
  const int64_t total_num_tokens,
  const int64_t max_num_cp,
  const int64_t num_seq
) {
  const int64_t all_replica_num_tokens = total_num_tokens * max_num_cp;

  size_t cur_cp = 0;
  size_t cur_seq = 0; // NOTE: physical sequence id
  // logical id. token_i - cur_seq_start_token_id == token_id_in_seq.
  size_t cur_seq_start_token_id = 0;
  // logical id.
  size_t cur_seq_end = __ldg(&seq_tokens[0]);
  // physical offset of nvshmem buffer.
  size_t cur_seq_start_offset = __ldg(&seq_nvshmem_offset[0]);
  size_t cur_shard = 0; // flatten (cp_id, seq_id) index.
  int8_t cur_do_replica = __ldg(&do_replica[0]);

  for (size_t token_idx = blockIdx.x; token_idx < all_replica_num_tokens;
       token_idx += gridDim.x) {
    // NOTE: token_idx is a logical index on the (cp_id, seq_id, token_id) order.
    while (token_idx > cur_seq_end) {
      // Move to the next sequence
      cur_seq += 1;
      cur_shard += 1;
      if (cur_seq == num_seq) {
        cur_seq = 0;
        cur_cp += 1;
      }
      const size_t seq_len = __ldg(&seq_tokens[cur_seq]);
      // update the logical start id
      cur_seq_start_token_id = cur_seq_end;
      cur_seq_end += seq_len;

      if (token_idx < cur_seq_end) {
        cur_seq_start_offset = __ldg(&seq_nvshmem_offset[cur_shard]);
        cur_do_replica = __ldg(&do_replica[cur_seq * max_num_cp + cur_cp]);
      }
    }

    if (cur_do_replica == 0) {
      continue;
    }

    // token_idx is the logical token of multiple cp replicas.
    // its % total_num_tokens is the token id of this cp relica.
    const int4 *send = (int4 *)(tensor + (
      token_idx % total_num_tokens
    ) * token_bytes);
    uint8_t *recv =  (
      // sequence start (bytes)
      nvshmem_buffer + cur_seq_start_offset +
      // token offset (bytes)
      (token_idx - cur_seq_start_token_id) * token_bytes
    );

    for (size_t i = threadIdx.x; i * sizeof(int4) < token_bytes;
         i += blockDim.x) {
      // i-th byte's offset in src
      ((int4*)recv)[i] = send[i];
    }
  }
}
};  // namespace

void launch_memcpy_non_cp(
  uint8_t *tensor,
  uint8_t *nvshmem_buffer,
  const int64_t *seq_nvshmem_offset,
  const int64_t *seq_tokens,
  const int64_t token_bytes,
  const int32_t total_num_tokens,
  const bool to_nvshmem,
  cudaStream_t stream
) {
  const int num_blocks = std::max(total_num_tokens / 128, 128);
  constexpr unsigned NUM_WARPS = 10;
  dim3 dimGrid(num_blocks, 1, 1);
  dim3 dimBlock(NUM_WARPS * 32, 1, 1);
  const size_t sharedMemory = 0;
  void *args[] = {
    &tensor,
    &nvshmem_buffer,
    const_cast<int64_t **>(&seq_nvshmem_offset),
    const_cast<int64_t **>(&seq_tokens),
    const_cast<int64_t *>(&token_bytes),
    const_cast<int32_t *>(&total_num_tokens)
  };
  if (to_nvshmem) {
    CUDACHECK(cudaLaunchKernel(
      (void *)memcpy_nvshmem_non_cp<true>,
      dimGrid,
      dimBlock,
      args,
      sharedMemory,
      stream
    ));
  } else {
    CUDACHECK(cudaLaunchKernel(
      (void *)memcpy_nvshmem_non_cp<false>,
      dimGrid,
      dimBlock,
      args,
      sharedMemory,
      stream
    ));
  }
}

};  // namespace attn
