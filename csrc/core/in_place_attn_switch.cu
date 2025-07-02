#include <cooperative_groups.h>
#include <cuda.h>
#include <nvshmem.h>

#include "core/common_utils.h"
#include "core/cuda_utils.h"
#include "core/in_place_attn_switch.h"

#include <iostream>
#include <cassert>

namespace {
template <bool KEY_VALUE>
__global__ void qkv_dispatch_kernel(
    std::byte* query_out,
    std::byte* key_value_out,
    const std::byte* query_in,
    const std::byte* key_value_in,
    const int32_t* query_dst_id,
    const uint32_t* query_dst_offset,
    const int32_t* key_value_dst_id,
    const uint32_t* key_value_dst_offset,
    const size_t num_tokens,
    const size_t q_stride,
    const size_t kv_stride,
    const size_t max_cp_degree,
    // nvshmem buffers
    std::byte* q_send_buffer,
    std::byte* q_recv_buffer,
    std::byte* kv_send_buffer,
    std::byte* kv_recv_buffer,
    uint64_t* q_signal_buffer,
    uint64_t* kv_signal_buffer,
    // receive info
    const uint64_t* num_q_to_recv,   // array of size world_size. The number of tokens to recv from each rank. The last value is the sum.
    const uint64_t* num_kv_to_recv,  // Same as above.
    const unsigned rank,
    const unsigned world_size
) {
    // --- Calculate thread/warp IDs based on the new launch grid ---
    const unsigned WARP_SIZE = 32;
    const unsigned NUM_WARPS = blockDim.x / WARP_SIZE;

    // NOTE(yonghao): a warp is the minimum unit of token-level communication.
    // const unsigned lane_id = threadIdx.x % WARP_SIZE;
    const unsigned warp_id = threadIdx.x / WARP_SIZE;
    // NOTE(yonghao): a warp group is responsible for one token. (potentially multiple destinations)
    const unsigned warp_group_id = blockIdx.x;
    // NOTE(yonghao): We may later use a warp for metadata, and then this is different from blockIdx.x
    const unsigned warp_group_size = NUM_WARPS * WARP_SIZE;
    const unsigned num_warp_groups = gridDim.x;
    // NOTE(yonghao): we may put some metadata for each token's send buffer.
    const unsigned Q_BUFFER_STRIDE = attn::round_up<unsigned>(q_stride, sizeof(int4));
    const unsigned KV_BUFFER_STRIDE = attn::round_up<unsigned>(kv_stride, sizeof(int4));

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
            int32_t query_dest_rank = query_dst_id[token_idx];
            uint32_t query_dest_offset = query_dst_offset[token_idx];
            std::byte* q_recv_buffer_token = q_recv_buffer + query_dest_offset * Q_BUFFER_STRIDE;

            // TODO(yonghao): use a nvshmemx_putmem_signal_nbi_warp to add a signal on the receiver.
            nvshmemx_putmem_signal_nbi_warp(
                q_recv_buffer_token,
                q_send_buffer_token,
                Q_BUFFER_STRIDE,
                &q_signal_buffer[rank],
                1,
                NVSHMEM_SIGNAL_ADD,
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
                nvshmemx_putmem_signal_nbi_warp(
                    kv_out_buffer,
                    kv_send_buffer_token,
                    KV_BUFFER_STRIDE,
                    &kv_signal_buffer[rank],
                    1,
                    NVSHMEM_SIGNAL_ADD,
                    kv_dest_rank
                );
            }
        }
    }

    // Sync before moving on to the receiver-side logic.
    cooperative_groups::this_grid().sync();

    // --- RECEIVER-SIDE SYNCHRONIZATION ---

    // sync to ensure that all recv are done.
    for (size_t i = threadIdx.x; i < world_size; i += WARP_SIZE) {
        const size_t num_recv_from_rank = __ldg(&num_q_to_recv[i]);
        nvshmem_uint64_wait_until(&q_signal_buffer[i], NVSHMEM_CMP_EQ, num_recv_from_rank);
        if constexpr (KEY_VALUE) {
            const size_t num_recv_from_rank = __ldg(&num_kv_to_recv[i]);
            nvshmem_uint64_wait_until(&kv_signal_buffer[i], NVSHMEM_CMP_EQ, num_recv_from_rank);
        }
    }
    __syncthreads();

    // memcpy to the dst tensor. No need to have a metadata handler warp.
    for (size_t token_idx = blockIdx.x; token_idx < num_q_to_recv[world_size]; token_idx += gridDim.x) {
        std::byte* q_recv_buffer_token = q_recv_buffer + token_idx * Q_BUFFER_STRIDE;
        int4* query_token = (int4*)(query_out + token_idx * q_stride);

        // Perform warp-cooperative memcpy
        for (int i = threadIdx.x; i * sizeof(int4) < q_stride; i += warp_group_size) {
            query_token[i] = ((int4*)q_recv_buffer_token)[i];
        }
    }
    if constexpr (KEY_VALUE) {
        for (size_t token_idx = blockIdx.x; token_idx < num_kv_to_recv[world_size]; token_idx += gridDim.x) {
            int4* key_value_token = (int4*)(key_value_out + token_idx * kv_stride);
            std::byte* kv_recv_buffer_token = kv_recv_buffer + token_idx * KV_BUFFER_STRIDE;

            for (int i = threadIdx.x; i * sizeof(int4) < kv_stride; i += warp_group_size) {
                key_value_token[i] = ((int4*)kv_recv_buffer_token)[i];
            }
        }
    }
}


template <bool KEY_VALUE>
__global__ void dispatch_kernel_v1(
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
	const unsigned rank,
	const unsigned world_size,
	// nvshmem buffers
	std::byte *q_send_buffer,
	std::byte *q_recv_buffer,
	std::byte *kv_send_buffer,
	std::byte *kv_recv_buffer,
	uint64_t *q_signal_buffer,
	uint64_t *kv_signal_buffer
) {
	// --- Calculate thread/warp IDs based on the new launch grid ---
	const unsigned WARP_SIZE = 32;
	const unsigned NUM_WARPS = blockDim.x / WARP_SIZE;

	// NOTE(yonghao): a warp is the minimum unit of token-level communication.
	// const unsigned lane_id = threadIdx.x % WARP_SIZE;
	const unsigned warp_id = threadIdx.x / WARP_SIZE;
	// NOTE(yonghao): a warp group is responsible for one token. (potentially multiple destinations)
	const unsigned warp_group_id = blockIdx.x;
	// NOTE(yonghao): We may later use a warp for metadata, and then this is different from blockIdx.x
	const unsigned warp_group_size = NUM_WARPS * WARP_SIZE;
	const unsigned num_warp_groups = gridDim.x;
	// NOTE(yonghao): we may put some metadata for each token's send buffer.
	const unsigned Q_BUFFER_STRIDE = attn::round_up<unsigned>(stride, sizeof(int4));
	const unsigned KV_BUFFER_STRIDE = attn::round_up<unsigned>(kv_stride, sizeof(int4));

	// --- SENDER-SIDE LOGIC with Warp-Level Grid-Stride Loop ---

	// Each warp group processes one token at a time and strides through the entire sequence.
	// This allows a grid of any size to process all tokens.
	int32_t sequence_id = -1;
	size_t sequence_end = 0;
	int32_t recv_rank = 0;
	uint32_t recv_offset = 0;
	int32_t kv_recv_rank = 0;
	uint32_t kv_recv_offset = 0;

	for (int token_idx = warp_group_id; token_idx < num_tokens; token_idx += num_warp_groups) {

		// Copying the token to the send buffer.
		const int4* send_token = (int4*)(send_tensor + token_idx * stride);
		std::byte* send_buffer_token = q_send_buffer + token_idx * Q_BUFFER_STRIDE;

		const int4* kv_send_token = KEY_VALUE ? (int4*)(kv_send_tensor + token_idx * kv_stride) : nullptr;
		std::byte* kv_send_buffer_token = nullptr;
		if constexpr (KEY_VALUE) {
				kv_send_buffer_token = kv_send_buffer + token_idx * KV_BUFFER_STRIDE;
		}

		// Perform warp group-cooperative memcpy
		for (int i = threadIdx.x; i * sizeof(int4) < stride; i += warp_group_size) {
				((int4*)send_buffer_token)[i] = send_token[i];
		}
		if constexpr (KEY_VALUE) {
				for (int i = threadIdx.x; i * sizeof(int4) < kv_stride; i += warp_group_size) {
						((int4*)kv_send_buffer_token)[i] = kv_send_token[i];
				}
		}
		// Synchronize the warps within this warp group.
		asm volatile("bar.sync 1, %0;" ::"r"(warp_group_size));

		// --- 1. Dispatch the query tensor ---
		// The first lane of the assigned warp is responsible for dispatching the query.
		if (warp_id == 0) {
			// move to the next sequence.
			while (token_idx >= sequence_end) {
				sequence_id += 1;
				const size_t sequence_len = __ldg(&seq_lens[sequence_id]);
				sequence_end += sequence_len;
				recv_rank = __ldg(&dst_ranks[sequence_id]);
				recv_offset = __ldg(&dst_offsets[sequence_id]);
			}
			// dispatch the query.
			std::byte* recv_buffer_token = q_recv_buffer + recv_offset * Q_BUFFER_STRIDE;
			nvshmemx_putmem_signal_nbi_warp(
				recv_buffer_token,
				send_buffer_token,
				Q_BUFFER_STRIDE,
				&q_signal_buffer[rank],
				1,
				NVSHMEM_SIGNAL_ADD,
				recv_rank
			);
			recv_offset += 1;
		} else {
			if constexpr (KEY_VALUE) {
				// warp 1...max_cp_degree dispatches to its own rank and recv_offset
				if (warp_id <= max_cp_degree) {
					// move to the next sequence.
					while (token_idx >= sequence_end) {
						sequence_id += 1;
						const size_t sequence_len = __ldg(&seq_lens[sequence_id]);
						sequence_end += sequence_len;
						kv_recv_rank = __ldg(&kv_dst_ranks[sequence_id * max_cp_degree + warp_id - 1]);
						kv_recv_offset = __ldg(&kv_dst_offsets[sequence_id * max_cp_degree + warp_id - 1]);
					}
					// dispatch the key_value.
					if (kv_recv_rank != -1) {
						std::byte* kv_recv_buffer_token = kv_recv_buffer + kv_recv_offset * KV_BUFFER_STRIDE;
						nvshmemx_putmem_signal_nbi_warp(
							kv_recv_buffer_token,
							kv_send_buffer_token,
							KV_BUFFER_STRIDE,
							&kv_signal_buffer[rank],
							1,
							NVSHMEM_SIGNAL_ADD,
							kv_recv_rank
						);
					}
					kv_recv_offset += 1;
				}
			}
		}
	}

	cooperative_groups::this_grid().sync();

	// --- RECEIVER-SIDE SYNCHRONIZATION ---
	// sync to ensure that all recv are done.
	for (size_t i = threadIdx.x; i < world_size; i += WARP_SIZE) {
		const uint64_t num_recv_from_rank = __ldg(&num_recv_tokens[i]);
		nvshmem_uint64_wait_until(&q_signal_buffer[i], NVSHMEM_CMP_EQ, num_recv_from_rank);
		if constexpr (KEY_VALUE) {
			const size_t num_recv_from_rank = __ldg(&kv_num_recv_tokens[i]);
			nvshmem_uint64_wait_until(&kv_signal_buffer[i], NVSHMEM_CMP_EQ, num_recv_from_rank);
		}
	}
	__syncthreads();

	// --- RECEIVER-SIDE MEMCPY ---
	for (size_t token_idx = blockIdx.x; token_idx < num_recv_tokens[world_size]; token_idx += gridDim.x) {
		std::byte* recv_buffer_token = q_recv_buffer + token_idx * Q_BUFFER_STRIDE;
		int4* recv_token = (int4*)(recv_tensor + token_idx * stride);
		for (int i = threadIdx.x; i * sizeof(int4) < stride; i += warp_group_size) {
			recv_token[i] = ((int4*)recv_buffer_token)[i];
		}
	}
	if constexpr (KEY_VALUE) {
		for (size_t token_idx = blockIdx.x; token_idx < kv_num_recv_tokens[world_size]; token_idx += gridDim.x) {
			std::byte* kv_recv_buffer_token = kv_recv_buffer + token_idx * KV_BUFFER_STRIDE;
			int4* kv_recv_token = (int4*)(kv_recv_tensor + token_idx * kv_stride);
			for (int i = threadIdx.x; i * sizeof(int4) < kv_stride; i += warp_group_size) {
				kv_recv_token[i] = ((int4*)kv_recv_buffer_token)[i];
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
    size_t max_tokens_key_value,
    unsigned rank,
    unsigned world_size
) : _rank(rank), _world_size(world_size) {
    q_send_buffer = (std::byte *)nvshmem_malloc(max_tokens_query * q_stride);
    q_recv_buffer = (std::byte *)nvshmem_malloc(max_tokens_query * q_stride);
    kv_send_buffer = (std::byte *)nvshmem_malloc(max_tokens_key_value * kv_stride);
    kv_recv_buffer = (std::byte *)nvshmem_malloc(max_tokens_key_value * kv_stride);
    q_signal_buffer = (uint64_t *)nvshmem_malloc(sizeof(uint64_t) * world_size);
    kv_signal_buffer = (uint64_t *)nvshmem_malloc(sizeof(uint64_t) * world_size);
    cudaMemset(q_signal_buffer, 0, sizeof(uint64_t) * world_size);
    cudaMemset(kv_signal_buffer, 0, sizeof(uint64_t) * world_size);
}

DispatchHelper::~DispatchHelper() {
    nvshmem_free(q_send_buffer);
    nvshmem_free(q_recv_buffer);
    nvshmem_free(kv_send_buffer);
    nvshmem_free(kv_recv_buffer);
    nvshmem_free(q_signal_buffer);
    nvshmem_free(kv_signal_buffer);
}

void DispatchHelper::dispatch(
    std::byte* query_out,
    std::byte* key_value_out,
    const std::byte* query_in,
    const std::byte* key_value_in,
    const int32_t* query_dst_id,
    const uint32_t* query_dst_offset,
    const int32_t* key_value_dst_id,
    const uint32_t* key_value_dst_offset,
    const size_t num_send_tokens,
    const uint64_t* num_recv_tokens_query,
    const uint64_t* num_recv_tokens_key_value,
    const size_t max_cp_degree,
    const size_t q_stride,
    const size_t kv_stride,
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
        const_cast<std::byte **>(&query_in),
        const_cast<std::byte **>(&key_value_in),
        const_cast<int32_t **>(&query_dst_id),
        const_cast<uint32_t **>(&query_dst_offset),
        const_cast<int32_t **>(&key_value_dst_id),
        const_cast<uint32_t **>(&key_value_dst_offset),
        const_cast<size_t *>(&num_send_tokens),
        const_cast<size_t *>(&q_stride),
        const_cast<size_t *>(&kv_stride),
        const_cast<size_t *>(&max_cp_degree),
        &q_send_buffer,
        &q_recv_buffer,
        &kv_send_buffer,
        &kv_recv_buffer,
        &q_signal_buffer,
        &kv_signal_buffer,
        const_cast<uint64_t **>(&num_recv_tokens_query),
        const_cast<uint64_t **>(&num_recv_tokens_key_value),
        const_cast<unsigned *>(&_rank),
        const_cast<unsigned *>(&_world_size)
    };

    if (_rank == 0) {
        std::cerr << "numSMs: " << numSMs << ", max_cp_degree: " << max_cp_degree
                  << ", num blocks: " << numBlocks << ", num warps: " << NUM_WARPS << std::endl;
    }

    if (has_key_value) {
        CUDACHECK(cudaLaunchCooperativeKernel(
            (void *)&qkv_dispatch_kernel<true>,
            dimGrid,
            dimBlock,
            args,
            sharedMemory,
            stream
        ));
    } else {
        CUDACHECK(cudaLaunchCooperativeKernel(
            (void *)&qkv_dispatch_kernel<false>,
            dimGrid,
            dimBlock,
            args,
            sharedMemory,
            stream
        ));
        cudaMemsetAsync(kv_signal_buffer, 0, sizeof(uint64_t) * _world_size, stream);
    }
    cudaMemsetAsync(q_signal_buffer, 0, sizeof(uint64_t) * _world_size, stream);
}

void DispatchHelper::dispatch_v1(
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
    cudaStream_t stream
) {
    int numSMs = get_sm_count();
    const bool has_key_value = kv_send_tensor != nullptr;
    constexpr unsigned NUM_WARPS = 10;
    const unsigned numBlocks = std::min(
        static_cast<unsigned>(numSMs),
        (unsigned)(num_tokens)
    );

    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(NUM_WARPS * 32, 1, 1);

    const size_t sharedMemory = 0;
    // CudaLaunchCooperativeKernel
    void *args[] = {
        // Input and output tensors
        const_cast<std::byte **>(&send_tensor),
        &recv_tensor,
        const_cast<std::byte **>(&kv_send_tensor),
        &kv_recv_tensor,
        // Metadata tensors
        const_cast<int32_t **>(&dst_ranks),
        const_cast<uint32_t **>(&dst_offsets),
        const_cast<uint64_t **>(&num_recv_tokens),
        const_cast<uint32_t **>(&seq_lens),
        //
        const_cast<int32_t **>(&kv_dst_ranks),
        const_cast<uint32_t **>(&kv_dst_offsets),
        const_cast<uint64_t **>(&kv_num_recv_tokens),
        // Metadata
        const_cast<size_t *>(&num_tokens),
        const_cast<size_t *>(&num_sequence),
        const_cast<size_t *>(&max_cp_degree),
        const_cast<size_t *>(&stride),
        const_cast<size_t *>(&kv_stride),
        const_cast<unsigned *>(&_rank),
        const_cast<unsigned *>(&_world_size),
        // nvshmem buffers
        &q_send_buffer,
        &q_recv_buffer,
        &kv_send_buffer,
        &kv_recv_buffer,
        &q_signal_buffer,
        &kv_signal_buffer
    };

    if (has_key_value) {
        CUDACHECK(cudaLaunchCooperativeKernel(
            (void *)&dispatch_kernel_v1<true>,
            dimGrid,
            dimBlock,
            args,
            sharedMemory,
            stream
        ));
    } else {
        CUDACHECK(cudaLaunchCooperativeKernel(
            (void *)&dispatch_kernel_v1<false>,
            dimGrid,
            dimBlock,
            args,
            sharedMemory,
            stream
        ));
        cudaMemsetAsync(kv_signal_buffer, 0, sizeof(uint64_t) * _world_size, stream);
    }
    cudaMemsetAsync(q_signal_buffer, 0, sizeof(uint64_t) * _world_size, stream);
}

};  // namespace attn

