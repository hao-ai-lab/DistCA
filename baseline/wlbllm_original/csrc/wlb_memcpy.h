#include <cuda.h>
#include <cuda_runtime.h>

namespace wlb {

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
);
};  // namespace wlb
