#include <cuda.h>
#include <cuda_runtime.h>

namespace attn {
void launch_memcpy_non_cp(
  uint8_t *src_buffer,
  uint8_t *nvshmem_buffer,
  const int64_t *seq_nvshmem_offset,
  const int64_t *seq_tokens,
  const int8_t *copy_seq_mask,
  const int64_t token_bytes,
  const int32_t total_num_tokens,
  const bool to_nvshmem,
  cudaStream_t stream
);

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
);

template <typename T>
void launch_inplace_grad_sum(
  T* data,
  const int32_t* num_copies,
  const int64_t* copy_start_id,
  const int64_t* seq_lens,
  const int64_t num_tokens,
  // NOTE: unlike other kernels, hidden is not in terms of bytes,
  // but instead in terms of elements
  const int64_t hidden,
  const int64_t num_seq,
  const int64_t num_max_copies,
  cudaStream_t stream
);
};  // namespace attn