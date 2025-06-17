#include <cooperative_groups.h>
#include <cuda.h>
#include <nvshmem.h>

#include "core/common_utils.h"
#include "core/cuda_utils.h"
#include "core/in_place_attn_switch.h"

namespace {
template <bool KEY_VALUE>
__global__ void qkv_dispatch_kernel(
    std::byte* query_out,
    std::byte* key_value_out,
    const std::byte* query_in,
    const std::byte* key_value_in,
    const int32_t* query_dst_id,
    const int32_t* query_dst_offset,
    const int32_t* key_value_dst_id,
    const int32_t* key_value_dst_offset,
    size_t num_tokens,
    size_t q_stride,
    size_t kv_stride,
    uint32_t max_cp_degree,
    // nvshmem buffers
    std::byte* q_send_buffer,
    std::byte* q_recv_buffer,
    std::byte* kv_send_buffer,
    std::byte* kv_recv_buffer,
    // receive info
    int num_q_to_recv,
    int num_kv_to_recv
) {
    // --- Calculate thread/warp IDs based on the new launch grid ---
    const unsigned WARP_SIZE = 32;
    const unsigned NUM_WARPS = blockDim.x / WARP_SIZE;

    // NOTE(yonghao): a warp is the minimum unit of token-level communication.
    const unsigned lane_id = threadIdx.x % WARP_SIZE;
    const unsigned warp_id = threadIdx.x / WARP_SIZE;
    // NOTE(yonghao): a warp group is responsible for one token. (potentially multiple destinations)
    const unsigned warp_group_id = blockIdx.x;
    // NOTE(yonghao): We may later use a warp for metadata, and then this is different from blockIdx.x
    const unsigned warp_group_size = NUM_WARPS * WARP_SIZE;
    const unsigned num_warp_groups = gridDim.x;
    // NOTE(yonghao): we may put some metadata for each token's send buffer.
    const unsigned Q_BUFFER_STRIDE = q_stride;
    const unsigned KV_BUFFER_STRIDE = kv_stride;

    // --- SENDER-SIDE LOGIC with Warp-Level Grid-Stride Loop ---
    
    // Each warp group processes one token at a time and strides through the entire sequence.
    // This allows a grid of any size to process all tokens.
    for (int token_idx = warp_group_id; token_idx < num_tokens; token_idx += num_warp_groups) {

        const int4* query_token = (int4*)(query_in + token_idx * q_stride);
        std::byte* q_send_buffer_token = q_send_buffer + token_idx * Q_BUFFER_STRIDE;

        const int4* key_value_token = KEY_VALUE ? (int4*)(key_value_in + token_idx * kv_stride) : nullptr;
        std::byte* kv_send_buffer_token = nullptr;
        if constexpr (KEY_VALUE) {
            kv_send_buffer_token = kv_send_buffer + token_idx * KV_BUFFER_STRIDE;
        }

        // Perform warp group-cooperative memcpy
        for (int i = threadIdx.x; i * sizeof(int4) < q_stride; i += warp_group_size) {
            ((int4*)q_send_buffer_token)[i] = query_token[i];
        }
        if constexpr (KEY_VALUE) {
            for (int i = threadIdx.x; i * sizeof(int4) < kv_stride; i += warp_group_size) {
                ((int4*)kv_send_buffer_token)[i] = key_value_token[i];
            }
        }
        // Synchronize the warps within this warp group.
        asm volatile("bar.sync 1, %0;" ::"r"(warp_group_size));

        // --- 1. Dispatch the query tensor ---
        // The first lane of the assigned warp is responsible for dispatching the query.
        if (warp_id == 0) {
            int query_dest_rank = query_dst_id[token_idx];
            int query_dest_offset = query_dst_offset[token_idx];
            std::byte* q_recv_buffer_token = q_recv_buffer + query_dest_offset * q_stride;

            // TODO(yonghao): use a nvshmemx_putmem_signal_nbi_warp to add a signal on the receiver.
            nvshmem_putmem_nbi(
                q_recv_buffer_token,
                q_send_buffer_token,
                q_stride,
                query_dest_rank
            );
        }

        // --- 2. Dispatch the key_value tensor, if any ---
        // attn_out -> mlp only uses the query_tensor's part: each token is sent to only one rank.
        if constexpr (KEY_VALUE) {
            for (int j = warp_id; j < max_cp_degree; j += NUM_WARPS) {
                int kv_idx = token_idx * max_cp_degree + j;

                int kv_dest_rank = key_value_dst_id[kv_idx];
                if (kv_dest_rank == -1) {
                    continue;
                }
                int kv_dest_offset = key_value_dst_offset[kv_idx];
                std::byte* kv_out_buffer = key_value_out + kv_dest_offset * kv_stride;

                // TODO(yonghao): use a nvshmemx_putmem_signal_nbi_warp to add a signal on the receiver.
                nvshmem_putmem_nbi(kv_out_buffer, kv_send_buffer_token, kv_stride, kv_dest_rank);
            }
        }
    }

    // Sync before moving on to the receiver-side logic.
    cooperative_groups::this_grid().sync();

    // --- RECEIVER-SIDE SYNCHRONIZATION ---

    // FIXME(yonghao): add a sync signal to ensure that send is done. Only recv after that.

    // memcpy to the dst tensor. No need to have a metadata handler warp.
    for (int token_idx = blockIdx.x; token_idx < num_q_to_recv; token_idx += gridDim.x) {
        std::byte* q_recv_buffer_token = q_recv_buffer + token_idx * Q_BUFFER_STRIDE;
        int4* query_token = (int4*)(query_out + token_idx * q_stride);

        // Perform warp-cooperative memcpy
        for (int i = threadIdx.x; i * sizeof(int4) < q_stride; i += warp_group_size) {
            query_token[i] = ((int4*)q_recv_buffer_token)[i];
        }
    }
    if constexpr (KEY_VALUE) {
        for (int token_idx = blockIdx.x; token_idx < num_kv_to_recv; token_idx += gridDim.x) {
            int4* key_value_token = (int4*)(key_value_out + token_idx * kv_stride);
            std::byte* kv_recv_buffer_token = kv_recv_buffer + token_idx * KV_BUFFER_STRIDE;

            for (int i = threadIdx.x; i * sizeof(int4) < kv_stride; i += warp_group_size) {
                key_value_token[i] = ((int4*)kv_recv_buffer_token)[i];
            }
        }
    }
}
};  // namespace

namespace attn {

DispatchHelper::DispatchHelper(
    size_t q_stride,
    size_t kv_stride,
    size_t max_tokens_query,
    size_t max_tokens_key_value
) : q_stride(q_stride), kv_stride(kv_stride), max_tokens_query(max_tokens_query), max_tokens_key_value(max_tokens_key_value) {

    q_send_buffer = (std::byte *)nvshmem_malloc(max_tokens_query * q_stride);
    q_recv_buffer = (std::byte *)nvshmem_malloc(max_tokens_query * q_stride);
    kv_send_buffer = (std::byte *)nvshmem_malloc(max_tokens_key_value * kv_stride);
    kv_recv_buffer = (std::byte *)nvshmem_malloc(max_tokens_key_value * kv_stride);
}

DispatchHelper::~DispatchHelper() {
    nvshmem_free(q_send_buffer);
    nvshmem_free(q_recv_buffer);
    nvshmem_free(kv_send_buffer);
    nvshmem_free(kv_recv_buffer);
}

void DispatchHelper::dispatch(
    std::byte* query_out,
    std::byte* key_value_out,
    const std::byte* query_in,
    const std::byte* key_value_in,
    const int32_t* query_dst_id,
    const int32_t* query_dst_offset,
    const int32_t* key_value_dst_id,
    const int32_t* key_value_dst_offset,
    size_t num_send_tokens,
    size_t num_recv_tokens_query,
    size_t num_recv_tokens_key_value,
    size_t max_cp_degree,
    cudaStream_t stream
) {
    int numSMs = get_sm_count();

    const bool has_key_value = key_value_out != nullptr;

    constexpr unsigned NUM_WARPS = 10;
    const unsigned numBlocks = std::min(
        std::max(
            ceil_div<unsigned>(num_send_tokens, NUM_WARPS), (unsigned)(num_send_tokens * max_cp_degree)
        ),
        static_cast<unsigned>(numSMs)
    );
    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(NUM_WARPS * 32, 1, 1);

    // FIXME(yonghao): shared memory.
    const size_t sharedMemory = 0;
    // CudaLaunchCooperativeKernel
    void* args[] = {
        &query_out,
        &key_value_out,
        &query_in,
        &key_value_in,
        &query_dst_id,
        &query_dst_offset,
        &key_value_dst_id,
        &key_value_dst_offset,
        &num_send_tokens,
        &q_stride,
        &kv_stride,
        &max_cp_degree,
        &q_send_buffer,
        &q_recv_buffer,
        &kv_send_buffer,
        &kv_recv_buffer,
        &num_recv_tokens_query,
        &num_recv_tokens_key_value
    };
    CUDACHECK(cudaLaunchCooperativeKernel(
        (void *)&qkv_dispatch_kernel<has_key_value>,
        dimGrid,
        dimBlock,
        args,
        sharedMemory,
        stream
    ));
}

};  // namespace attn

