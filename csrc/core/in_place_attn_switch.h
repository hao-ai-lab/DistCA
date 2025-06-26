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

    void dispatch(
        std::byte *query_out,
        std::byte *key_value_out,
        const std::byte *query_in,
        const std::byte *key_value_in,
        const int32_t *query_dst_id,
        const uint32_t *query_dst_offset,
        const int32_t *key_value_dst_id,
        const uint32_t *key_value_dst_offset,
        const size_t num_send_tokens,
        const uint64_t *num_recv_tokens_query,
        const uint64_t *num_recv_tokens_key_value,
        const size_t max_cp_degree,
        const size_t q_stride,
        const size_t kv_stride,
        cudaStream_t stream
    );

private:
    const unsigned rank;
    const unsigned world_size;

    std::byte *q_send_buffer;
    std::byte *q_recv_buffer;
    std::byte *kv_send_buffer;
    std::byte *kv_recv_buffer;
    uint64_t *q_signal_buffer;
    uint64_t *kv_signal_buffer;
};
};  // namespace attn
