#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>

#include "wlb_memcpy.h"

#define CUDACHECK(cmd)                                                                             \
  do {                                                                                             \
    cudaError_t e = cmd;                                                                           \
    if (e != cudaSuccess) {                                                                        \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));        \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)

namespace wlb {

namespace {
template <bool IS_FORWARD>
__global__ void wlb_shuffle_memcpy_kernel(
  uint8_t *gathered,
  uint8_t *shuffled,
  const uint64_t *shard_lens,
  const uint64_t *shard_gathered_offsets,
  const size_t num_docs,
  const size_t num_total_tokens,
  const size_t hidden_bytes
) {
  /* gathered layout: (num_shards, num_tokens_per_shard, hidden_bytes)
      for each shard, display doc_0_this_shard, doc_1_this_shard, ...
     shuffled layout: (num_total_shuffled_tokens, hidden_bytes)
      num_total_tokens roughly equals num_shards * num_tokens_per_shard
      but may have some remainders.
      display in doc_0, doc_1, ...
     shard_len layout: (num_docs, num_shards)
   */

  // The global shard id w.r.t. shard_len.flatten()'s layout (i.e. (num_docs, num_shards))
  size_t cur_shard_glob_id = 0;
  // The global start and end token id of the current shard
  size_t cur_shard_start_token_id = 0;
  size_t cur_shard_end_token_id = __ldg(shard_lens + cur_shard_glob_id);

  size_t cur_shard_start_gathered_id = 0;

  // iterating in the shuffled layout because the gathered layout may have padding.
  for (size_t token_id = blockIdx.x; token_id < num_total_tokens;
       token_id += gridDim.x) {

    // if run out of the current shard, goes to a new one
    while (token_id > cur_shard_end_token_id) {
      // update the shard start and end token id
      cur_shard_glob_id++;
      size_t shard_len = __ldg(shard_lens + cur_shard_glob_id);
      cur_shard_start_token_id = cur_shard_end_token_id;
      cur_shard_end_token_id += shard_len;

      // update the shuffled layout id
      if (token_id <= cur_shard_end_token_id) {
        cur_shard_start_gathered_id = __ldg(shard_gathered_offsets + cur_shard_glob_id);
        break;
      }
    }

    uint8_t *gathered_token = gathered + (
      cur_shard_start_gathered_id + (token_id - cur_shard_start_token_id)
    ) * hidden_bytes;
    uint8_t *shuffled_token = shuffled + token_id * hidden_bytes;
    for (size_t i = threadIdx.x; i * sizeof(int4) < hidden_bytes;
         i += blockDim.x) {
      if constexpr (IS_FORWARD) {
        ((int4 *)shuffled_token)[i] = ((int4 *)gathered_token)[i];
      } else {
        ((int4 *)gathered_token)[i] = ((int4 *)shuffled_token)[i];
      }
    }
  }
}
}; // namespace


void launch_wlb_shuffle_memcpy(
  uint8_t *gathered,
  uint8_t *shuffled,
  const uint64_t *shard_lens,
  const uint64_t *shard_gathered_offsets,
  const size_t num_docs,
  const size_t num_total_tokens,
  const size_t hidden_bytes,
  const bool is_forward,
  const cudaStream_t stream
) {
  const int num_blocks = std::max(
    (int)num_total_tokens / 32, 128
  );
  // As a kv token's hidden can be as small as 128 elements (256 Bytes)
  // we do not need that much of threads.
  // 256 / 16(int4 size) / 32 = 1
  constexpr int WARP_SIZE = 32;
  int NUM_WARPS = std::min(
    10,  // at most 10 warps
    std::max( // at least one warp. at most each thread only transfer one int4
      1, (int)(hidden_bytes / sizeof(int4) / WARP_SIZE)
    )
  );
  dim3 dimGrid(num_blocks, 1, 1);
  dim3 dimBlock(NUM_WARPS * WARP_SIZE, 1, 1);
  const size_t sharedMemory = 0;

  void *args[] = {
    &gathered,
    &shuffled,
    const_cast<uint64_t **>(&shard_lens),
    const_cast<uint64_t **>(&shard_gathered_offsets),
    const_cast<size_t *>(&num_docs),
    const_cast<size_t *>(&num_total_tokens),
    const_cast<size_t *>(&hidden_bytes)
  };
  if (is_forward) {
    CUDACHECK(cudaLaunchKernel(
      (void *)wlb_shuffle_memcpy_kernel<true>,
      dimGrid,
      dimBlock,
      args,
      sharedMemory,
      stream
    ));
  } else {
    CUDACHECK(cudaLaunchKernel(
      (void *)wlb_shuffle_memcpy_kernel<false>,
      dimGrid,
      dimBlock,
      args,
      sharedMemory,
      stream
    ));
  }
}
};  // namespace wlb