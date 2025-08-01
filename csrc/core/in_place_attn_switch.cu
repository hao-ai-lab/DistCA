#include <cooperative_groups.h>
#include <cuda.h>
#include <nvshmem.h>

#include "core/common_utils.h"
#include "core/cuda_utils.h"
#include "core/in_place_attn_switch.h"

#include <iostream>
#include <cassert>

namespace {

__forceinline__ __device__ void _dispatch_recv_impl_kv_backward(
  std::byte *recv_tensor,
  // Metadata Tensors
  const uint32_t *seq_lens,
  // Metadata
  const size_t stride,
  const unsigned world_size,
  const unsigned BUFFER_STRIDE,
  // nvshmem buffer
  std::byte *recv_buffer,
  // recv kv special metadata
  // TODO(yonghao): the seq_recv_mask is of layout (num_sequence, max_cp_degree), while tokens are stored of layout (max_cp_degree, num_sequence).
  // We need to transpose the token receive layout in compute metadata.
  const uint32_t *seq_recv_mask,
  const size_t max_cp_degree,
  const size_t num_tokens
) {
  size_t current_seq = 0;
  size_t current_cp = 0;
  // size_t seq_is_pad_idx = 0;
  size_t current_seq_end = __ldg(&seq_lens[0]);
  size_t current_seq_len = __ldg(&seq_lens[0]);
  bool skip_current_seq = __ldg(&seq_recv_mask[0]) == 0;

  for (size_t token_idx = blockIdx.x; token_idx < num_tokens * max_cp_degree; token_idx += gridDim.x) {
    // Move to a new sequence if necessary.
    while (token_idx >= current_seq_end) {
      if (current_seq_end % num_tokens == 0) {
        current_cp += 1;
        current_seq = 0;
      } else {
        current_seq += 1;
      }
      current_seq_len = __ldg(&seq_lens[current_seq]);
      current_seq_end += current_seq_len;
      if (token_idx < current_seq_end) {
        skip_current_seq = __ldg(&seq_recv_mask[current_seq * max_cp_degree + current_cp]) == 0;
      }
    }

    // This sequence should be skipped.
    if (skip_current_seq) {
      continue;
    }

    std::byte* recv_buffer_token = recv_buffer + token_idx * BUFFER_STRIDE;
    int4* recv_token = (int4*)(recv_tensor + token_idx * stride);
    for (int i = threadIdx.x; i * sizeof(int4) < stride; i += blockDim.x) {
      recv_token[i] = ((int4*)recv_buffer_token)[i];
    }
  }
}


template <bool KEY_VALUE>
__forceinline__ __device__ void _dispatch_recv_impl(
  // Input and output tensors
  std::byte *recv_tensor,
  std::byte *kv_recv_tensor,
  // Metadata tensors
  const uint64_t *num_recv_tokens,
  const uint32_t *seq_lens,
  //
  const uint64_t *kv_num_recv_tokens,
  // Metadata
  const size_t stride,
  const size_t kv_stride,
  const unsigned world_size,
  const unsigned Q_BUFFER_STRIDE,
  const unsigned KV_BUFFER_STRIDE,
  // nvshmem buffers
  std::byte *q_recv_buffer,
  std::byte *kv_recv_buffer
) {
  uint64_t tot_num_recv_tokens = __ldg(&num_recv_tokens[world_size]);
  for (size_t token_idx = blockIdx.x; token_idx < tot_num_recv_tokens; token_idx += gridDim.x) {
    std::byte* recv_buffer_token = q_recv_buffer + token_idx * Q_BUFFER_STRIDE;
    int4* recv_token = (int4*)(recv_tensor + token_idx * stride);
    for (int i = threadIdx.x; i * sizeof(int4) < stride; i += blockDim.x) {
      recv_token[i] = ((int4*)recv_buffer_token)[i];
    }
  }
  if constexpr (KEY_VALUE) {
    uint64_t tot_kv_num_recv_tokens = __ldg(&kv_num_recv_tokens[world_size]);
    for (size_t token_idx = blockIdx.x; token_idx < tot_kv_num_recv_tokens; token_idx += gridDim.x) {
      std::byte* kv_recv_buffer_token = kv_recv_buffer + token_idx * KV_BUFFER_STRIDE;
      int4* kv_recv_token = (int4*)(kv_recv_tensor + token_idx * kv_stride);
      for (int i = threadIdx.x; i * sizeof(int4) < kv_stride; i += blockDim.x) {
        kv_recv_token[i] = ((int4*)kv_recv_buffer_token)[i];
      }
    }
  }
}


template <bool KEY_VALUE, bool IS_KV_BACKWARD>
__global__ void dispatch_kernel(
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
  uint64_t *kv_signal_buffer,
  // recv kv special metadata
  const uint32_t *seq_recv_mask,
  const uint32_t *recv_seq_lens,
  const size_t kv_backward_num_tokens
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
  uint32_t recv_sequence_begin_token_id = 0;
  int32_t kv_recv_rank = 0;
  uint32_t kv_recv_offset = 0;
  uint32_t kv_recv_sequence_begin_token_id = 0;

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
        recv_sequence_begin_token_id = sequence_end;
        sequence_end += sequence_len;
        recv_rank = __ldg(&dst_ranks[sequence_id]);
        recv_offset = __ldg(&dst_offsets[sequence_id]);
      }
      // dispatch the query.
      const uint32_t recv_token_offset = recv_offset + (token_idx - recv_sequence_begin_token_id);
      std::byte* recv_buffer_token = q_recv_buffer + recv_token_offset * Q_BUFFER_STRIDE;
      nvshmemx_putmem_signal_nbi_warp(
        recv_buffer_token,
        send_buffer_token,
        Q_BUFFER_STRIDE,
        &q_signal_buffer[rank],
        1,
        NVSHMEM_SIGNAL_ADD,
        recv_rank
      );
    } else {
      if constexpr (KEY_VALUE) {
        // warp 1...max_cp_degree dispatches to its own rank and recv_offset
        // FIXME(yonghao): we assume that max_cp_degree is small. Otherwise, we should do a round-robin
        if (warp_id <= max_cp_degree) {
          // move to the next sequence.
          while (token_idx >= sequence_end) {
            sequence_id += 1;
            const size_t sequence_len = __ldg(&seq_lens[sequence_id]);
            kv_recv_sequence_begin_token_id = sequence_end;
            sequence_end += sequence_len;
            kv_recv_rank = __ldg(&kv_dst_ranks[sequence_id * max_cp_degree + warp_id - 1]);
            kv_recv_offset = __ldg(&kv_dst_offsets[sequence_id * max_cp_degree + warp_id - 1]);
          }
          // dispatch the key_value.
          if (kv_recv_rank != -1) {
            const uint32_t kv_recv_token_offset = kv_recv_offset + (token_idx - kv_recv_sequence_begin_token_id);
            std::byte* kv_recv_buffer_token = kv_recv_buffer + kv_recv_token_offset * KV_BUFFER_STRIDE;
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
  if constexpr (IS_KV_BACKWARD) {
    _dispatch_recv_impl_kv_backward(
      recv_tensor,
      recv_seq_lens,
      stride,
      world_size,
      Q_BUFFER_STRIDE,
      q_recv_buffer,
      seq_recv_mask,
      max_cp_degree,
      kv_backward_num_tokens
    );
  } else {
    _dispatch_recv_impl<KEY_VALUE>(
      recv_tensor,
      kv_recv_tensor,
      num_recv_tokens,
      seq_lens,
      kv_num_recv_tokens,
      stride,
      kv_stride,
      world_size,
      Q_BUFFER_STRIDE,
      KV_BUFFER_STRIDE,
      q_recv_buffer,
      kv_recv_buffer
    );
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
  _numSMs = get_sm_count();
}

DispatchHelper::~DispatchHelper() {
  nvshmem_free(q_send_buffer);
  nvshmem_free(q_recv_buffer);
  nvshmem_free(kv_send_buffer);
  nvshmem_free(kv_recv_buffer);
  nvshmem_free(q_signal_buffer);
  nvshmem_free(kv_signal_buffer);
}

void DispatchHelper::set_num_sms(const size_t num_sms) {
  const size_t total_sms = get_sm_count();
  _numSMs = std::min(num_sms, total_sms);
}

void DispatchHelper::dispatch(
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
) {
  const int numSMs = (_numSMs > 0) ? _numSMs : get_sm_count();
  const bool has_key_value = kv_send_tensor != nullptr;
  const bool is_kv_backward = seq_recv_mask != nullptr;
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
    &kv_signal_buffer,
    // recv kv special metadata
    const_cast<uint32_t **>(&seq_recv_mask),
    const_cast<uint32_t **>(&recv_seq_lens),
    const_cast<size_t *>(&kv_backward_num_tokens)
  };

  if (is_kv_backward) {
    CUDACHECK(cudaLaunchCooperativeKernel(
      (void *)&dispatch_kernel<false, true>,
      dimGrid,
      dimBlock,
      args,
      sharedMemory,
      stream
    ));
  } else if (has_key_value) {
    CUDACHECK(cudaLaunchCooperativeKernel(
      (void *)&dispatch_kernel<true, false>,
      dimGrid,
      dimBlock,
      args,
      sharedMemory,
      stream
    ));
    cudaMemsetAsync(kv_signal_buffer, 0, sizeof(uint64_t) * _world_size, stream);
  } else {
    CUDACHECK(cudaLaunchCooperativeKernel(
      (void *)&dispatch_kernel<false, false>,
      dimGrid,
      dimBlock,
      args,
      sharedMemory,
      stream
    ));
  }
  cudaMemsetAsync(q_signal_buffer, 0, sizeof(uint64_t) * _world_size, stream);
}

};  // namespace attn

