#include <cuda.h>
#include <cuda_runtime.h>

#include "core/cuda_utils.h"

// TODO: support memcpy of strided tensor.
// This is because on the sender side, the model uses qkv_proj
// and split output to q,k,v.
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
    while (token_idx >= cur_seq_end) {
      // Move to the next sequence
      cur_seq += 1;
      cur_seq_start_token_id = cur_seq_end;
      cur_seq_end += __ldg(&seq_tokens[cur_seq]);
      if (token_idx < cur_seq_end) {
        cur_seq_start_offset = __ldg(&seq_nvshmem_offset[cur_seq]);
      }
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


template <bool TO_NVSHMEM>
__global__ void memcpy_nvshmem_cp(
  uint8_t *tensor,
  uint8_t *nvshmem_buffer,
  const int8_t *do_shard,
  const int64_t *seq_nvshmem_offset,
  const int64_t *seq_tokens,
  const int64_t token_bytes,
  const int32_t total_num_tokens,
  const int32_t max_num_cp,
  const int64_t num_seq
) {
  // each kv replica is a logical sequence.
  // logical token layout: (cp_degree, num_token)
  const int64_t logical_num_tokens = total_num_tokens * max_num_cp;

  size_t cur_cp = 0;
  size_t cur_seq = 0; // NOTE: physical sequence id
  // logical id. token_i - cur_seq_start_token_id == token_id_in_seq.
  size_t cur_seq_start_token_id = 0;
  // logical id.
  size_t cur_seq_end = __ldg(&seq_tokens[0]);
  // physical offset of nvshmem buffer.
  size_t cur_seq_start_offset = __ldg(&seq_nvshmem_offset[0]);
  size_t cur_shard = 0; // flatten (cp_id, seq_id) index.
  int8_t cur_do_shard = __ldg(&do_shard[0]);

  for (size_t token_idx = blockIdx.x; token_idx < logical_num_tokens;
       token_idx += gridDim.x) {
    // NOTE: token_idx is a logical index on the (cp_id, seq_id, token_id) order.
    while (token_idx >= cur_seq_end) {
      // Move to the next sequence
      cur_seq += 1;
      cur_shard += 1;
      if (cur_seq == num_seq) {
        cur_seq = 0;
        cur_cp += 1;
      }
      // update the logical start id
      cur_seq_start_token_id = cur_seq_end;
      cur_seq_end += __ldg(&seq_tokens[cur_seq]);

      if (token_idx < cur_seq_end) {
        cur_seq_start_offset = __ldg(&seq_nvshmem_offset[cur_shard]);
        cur_do_shard = __ldg(&do_shard[cur_seq * max_num_cp + cur_cp]);
      }
    }

    if (cur_do_shard == 0) {
      continue;
    }

    // token_idx is the logical token of multiple cp replicas.
    // For sending to nvshmem, it does not have the CP dim,
    // its % total_num_tokens is the token id of this cp relica.
    // For the reverse side, it has the CP dim, so logical index
    // is the physical index
    const size_t token_physical_offset_on_tensor = (
      TO_NVSHMEM ? (token_idx % total_num_tokens) : token_idx
    ) * token_bytes;
    uint8_t *tensor_token = tensor + token_physical_offset_on_tensor;
    uint8_t *buffer_token =  (
      // sequence start (bytes)
      nvshmem_buffer + cur_seq_start_offset +
      // token offset (bytes)
      (token_idx - cur_seq_start_token_id) * token_bytes
    );

    for (size_t i = threadIdx.x; i * sizeof(int4) < token_bytes;
         i += blockDim.x) {
      // i-th byte's offset in src
      if constexpr (TO_NVSHMEM) {
        ((int4*)buffer_token)[i] = ((int4*)tensor_token)[i];
      } else {
        ((int4*)tensor_token)[i] = ((int4*)buffer_token)[i];
      }
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
  const int num_blocks = std::max(total_num_tokens / 32, 128);
  constexpr int WARP_SIZE = 32;
  int NUM_WARPS = std::min(
    10,  // at most 10 warps
    std::max( // at least one warp. at most each thread only transfer one int4
      1, (int)(token_bytes / sizeof(int4) / WARP_SIZE)
    )
  );

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


void launch_memcpy_cp(
  uint8_t *tensor,
  uint8_t *nvshmem_buffer,
  const int8_t *do_shard,
  const int64_t *seq_nvshmem_offset,
  const int64_t *seq_tokens,
  const int64_t token_bytes,
  const int32_t total_num_tokens,
  const int32_t max_num_cp,
  const int64_t num_seq,
  const bool to_nvshmem,
  cudaStream_t stream
) {
  const int num_blocks = std::max(total_num_tokens * max_num_cp / 32, 128);
  // As a kv token's hidden can be as small as 128 elements (256 Bytes)
  // we do not need that much of threads.
  // 256 / 16(int4 size) / 32 = 1
  constexpr int WARP_SIZE = 32;
  int NUM_WARPS = std::min(
    10,  // at most 10 warps
    std::max( // at least one warp. at most each thread only transfer one int4
      1, (int)(token_bytes / sizeof(int4) / WARP_SIZE)
    )
  );
  dim3 dimGrid(num_blocks, 1, 1);
  dim3 dimBlock(NUM_WARPS * WARP_SIZE, 1, 1);
  const size_t sharedMemory = 0;

  void *args[] = {
    &tensor,
    &nvshmem_buffer,
    const_cast<int8_t **>(&do_shard),
    const_cast<int64_t **>(&seq_nvshmem_offset),
    const_cast<int64_t **>(&seq_tokens),
    const_cast<int64_t *>(&token_bytes),
    const_cast<int32_t *>(&total_num_tokens),
    const_cast<int32_t *>(&max_num_cp),
    const_cast<int64_t *>(&num_seq)
  };
  if (to_nvshmem) {
    CUDACHECK(cudaLaunchKernel(
      (void *)memcpy_nvshmem_cp<true>,
      dimGrid,
      dimBlock,
      args,
      sharedMemory,
      stream
    ));
  } else {
    CUDACHECK(cudaLaunchKernel(
      (void *)memcpy_nvshmem_cp<false>,
      dimGrid,
      dimBlock,
      args,
      sharedMemory,
      stream
    ));
  }
}

};  // namespace attn
