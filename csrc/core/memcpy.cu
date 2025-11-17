#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "core/cuda_utils.h"

// TODO: support memcpy of strided tensor.
// This is because on the sender side, the model uses qkv_proj
// and split output to q,k,v.
namespace attn{
namespace {
template <bool USE_SEQ_MASK>
__forceinline__ __device__ bool copy_seq(const size_t idx, const int8_t *copy_seq_mask) {
  if constexpr (USE_SEQ_MASK) {
    return __ldg(&copy_seq_mask[idx]);
  } else {
    return true;
  }
}

template <bool TO_NVSHMEM, bool USE_SEQ_MASK>
__global__ void memcpy_nvshmem_non_cp(
  uint8_t *tensor,
  uint8_t *nvshmem_buffer,
  const int64_t *seq_nvshmem_offset,
  const int64_t *seq_tokens,
  const int8_t *copy_seq_mask,
  const int64_t token_bytes,
  const int32_t total_num_tokens
) {

  size_t cur_seq = 0;
  size_t cur_seq_end = __ldg(&seq_tokens[0]);
  size_t cur_seq_start_token_id = 0;
  size_t cur_seq_start_offset = __ldg(&seq_nvshmem_offset[0]);
  bool copy_cur_seq = copy_seq<USE_SEQ_MASK>(0, copy_seq_mask);

  for (size_t token_idx = blockIdx.x; token_idx < total_num_tokens;
       token_idx += gridDim.x) {
    while (token_idx >= cur_seq_end) {
      // Move to the next sequence
      cur_seq += 1;
      cur_seq_start_token_id = cur_seq_end;
      cur_seq_end += __ldg(&seq_tokens[cur_seq]);
      if (token_idx < cur_seq_end) {
        cur_seq_start_offset = __ldg(&seq_nvshmem_offset[cur_seq]);
        copy_cur_seq = copy_seq<USE_SEQ_MASK>(cur_seq, copy_seq_mask);
      }
    }
    if (!copy_cur_seq) {
      continue;
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


template <typename T>
__global__ void inplace_grad_sum_kernel(
  T* data,
  const int32_t* num_copies,
  const int64_t* copy_start_id,
  const int64_t* seq_lens,
  const int64_t num_tokens,
  // NOTE: unlike other kernels, hidden is not in terms of bytes,
  // but instead in terms of elements
  const int64_t hidden,
  const int64_t num_seq,
  const int64_t num_max_copies
) {
  size_t seq_id = 0;
  size_t seq_start_id = 0;
  size_t seq_end_id = __ldg(&seq_lens[seq_id]);
  bool skip_seq = __ldg(&num_copies[seq_id]) == 0;

  for (size_t glob_id = blockIdx.x; glob_id < num_tokens; glob_id += gridDim.x) {
    // update the seq_id.
    while (glob_id >= seq_end_id) {
      seq_id += 1;
      seq_start_id = seq_end_id;
      seq_end_id += __ldg(&seq_lens[seq_id]);
      if (glob_id < seq_end_id) {
        skip_seq = __ldg(&num_copies[seq_id]) == 0;
        break;
      }
    }

    if (skip_seq) {
      continue;
    }

    const int64_t local_tok_id = glob_id - seq_start_id;
    T* main_copy_token = data + glob_id * hidden;

    for (int32_t copy_idx = 0; copy_idx < num_copies[seq_id]; ++copy_idx) {
      const int64_t copy_glob_start_id = __ldg(&copy_start_id[seq_id * num_max_copies + copy_idx]);
      const int64_t copy_glob_id = copy_glob_start_id + local_tok_id;
      const T* copy_token = data + copy_glob_id * hidden;

      // --- Vectorization Logic ---
      constexpr int num_elements_per_int4 = sizeof(int4) / sizeof(T);
      const int hidden_vec_size = hidden / num_elements_per_int4;

      // 1. Vectorized main loop
      for (int h_vec = threadIdx.x; h_vec < hidden_vec_size; h_vec += blockDim.x) {
        // Cast pointers to int4 for wide memory access
        int4* main_p = reinterpret_cast<int4*>(main_copy_token);
        const int4* copy_p = reinterpret_cast<const int4*>(copy_token);

        // Load 128 bits (16 bytes) at once
        int4 main_val = main_p[h_vec];
        int4 copy_val = copy_p[h_vec];

        // Perform operation on components. This part is tricky for a generic
        // template. For float, you'd cast to float4. Let's assume a generic cast.
        // NOTE: This requires `T` to be compatible in size with int components.
        // A production kernel would often use template specialization for types like float/half.
        T* main_val_T = reinterpret_cast<T*>(&main_val);
        const T* copy_val_T = reinterpret_cast<const T*>(&copy_val);

        for (int i = 0; i < num_elements_per_int4; ++i) {
          main_val_T[i] += copy_val_T[i];
        }

        // Write 128 bits back at once
        main_p[h_vec] = main_val;
      }

      // 2. Scalar cleanup loop for remaining elements
      const int start_cleanup_idx = hidden_vec_size * num_elements_per_int4;
      for (int h = start_cleanup_idx + threadIdx.x; h < hidden; h += blockDim.x) {
        main_copy_token[h] += copy_token[h];
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
  const int8_t *copy_seq_mask,
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
    const_cast<int8_t **>(&copy_seq_mask),
    const_cast<int64_t *>(&token_bytes),
    const_cast<int32_t *>(&total_num_tokens)
  };
  void * kernel = to_nvshmem ? (
    copy_seq_mask != nullptr ?
      (void *)memcpy_nvshmem_non_cp<true, true> :
      (void *)memcpy_nvshmem_non_cp<true, false>
  ) : (
    (void *)memcpy_nvshmem_non_cp<false, false>
  );
  CUDACHECK(cudaLaunchKernel(
    kernel, dimGrid, dimBlock, args, sharedMemory, stream
  ));
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
) {
  const int num_blocks = std::max((int)(num_tokens / 32), 128);
  // As a kv token's hidden can be as small as 128 elements (256 Bytes)
  // we do not need that much of threads.
  // 256 / 16(int4 size) / 32 = 1
  constexpr int WARP_SIZE = 32;
  int NUM_WARPS = std::min(
    10,  // at most 10 warps
    std::max( // at least one warp. at most each thread only transfer one int4
      1, (int)(hidden * sizeof(T) / sizeof(int4) / WARP_SIZE)
    )
  );
  dim3 dimGrid(num_blocks, 1, 1);
  dim3 dimBlock(NUM_WARPS * WARP_SIZE, 1, 1);
  const size_t sharedMemory = 0;
  void *args[] = {
    &data,
    const_cast<int32_t **>(&num_copies),
    const_cast<int64_t **>(&copy_start_id),
    const_cast<int64_t **>(&seq_lens),
    const_cast<int64_t *>(&num_tokens),
    const_cast<int64_t *>(&hidden),
    const_cast<int64_t *>(&num_seq),
    const_cast<int64_t *>(&num_max_copies)
  };
  CUDACHECK(cudaLaunchKernel(
    (void *)inplace_grad_sum_kernel<T>,
    dimGrid,
    dimBlock,
    args,
    sharedMemory,
    stream
  ));
}

// Instantiate fp32, fp16, bf16 instances
#define INSTANTIATE_INPLACE_GRAD_SUM(T) \
  template void launch_inplace_grad_sum<T>( \
    T* data, \
    const int32_t* num_copies, \
    const int64_t* copy_start_id, \
    const int64_t* seq_lens, \
    const int64_t num_tokens, \
    const int64_t hidden, \
    const int64_t num_seq, \
    const int64_t num_max_copies, \
    cudaStream_t stream \
  );

INSTANTIATE_INPLACE_GRAD_SUM(float);
INSTANTIATE_INPLACE_GRAD_SUM(half);
INSTANTIATE_INPLACE_GRAD_SUM(nv_bfloat16);

};  // namespace attn
