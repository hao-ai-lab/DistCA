#include <cooperative_groups.h>
#include <cuda.h>
#include <nvshmem.h>

namespace attn {

// A helper class to store nvshmem buffers and reuse it for all dispatch calls.
class DispatchHelper {
public:
  DispatchHelper(
    size_t q_stride,
    size_t kv_stride,
    size_t max_tokens_query,
    size_t max_tokens_key_value,
    unsigned rank,
    unsigned world_size
  );

  ~DispatchHelper();

  void set_num_sms(const size_t num_sms);

  void dispatch(
    // Input and output tensors
    const std::byte *send_tensor,
    std::byte *recv_tensor,
    const std::byte *kv_send_tensor,
    std::byte *kv_recv_tensor,
    // Metadata tensors
    const int32_t *dst_ranks,
    const uint32_t *dst_offsets,
    const uint64_t *num_recv_tokens,
    const uint32_t *seq_lens,
    //
    const int32_t *kv_dst_ranks,
    const uint32_t *kv_dst_offsets,
    const uint64_t *kv_num_recv_tokens,
    // Metadata
    const size_t num_tokens,
    const size_t num_sequence,
    const size_t max_cp_degree,
    const size_t stride,
    const size_t kv_stride,
    cudaStream_t stream,
    // recv kv backward special metadata
    const uint32_t *seq_recv_mask,
    const uint32_t *recv_seq_lens,
    const size_t kv_backward_num_tokens
  );

  unsigned rank() const {
    return _rank;
  }

  unsigned world_size() const {
    return _world_size;
  }

private:
  const unsigned _rank;
  const unsigned _world_size;

  std::byte *q_send_buffer;
  std::byte *q_recv_buffer;
  std::byte *kv_send_buffer;
  std::byte *kv_recv_buffer;
  uint64_t *q_signal_buffer;
  uint64_t *kv_signal_buffer;
  int _numSMs;
};
};  // namespace attn
