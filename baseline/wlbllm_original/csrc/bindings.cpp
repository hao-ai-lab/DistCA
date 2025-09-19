/*
Code modified from https://github.com/ppl-ai/pplx-kernels and subject to the MIT License.
*/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include "wlb_memcpy.h"
#include "registration.h"

using namespace wlb;

namespace {
#define _CHECK_TENSOR(ndim, x) \
  TORCH_CHECK(x.ndimension() == ndim, "tensor " #x " must have " #ndim " dimensions"); \
  TORCH_CHECK(x.is_cuda(), "tensor " #x " must be on CUDA"); \
  TORCH_CHECK(x.is_contiguous(), "tensor " #x " must be contiguous");


void wlb_shuffle_memcpy(
  at::Tensor &gathered_tensor,
  at::Tensor &shuffled_tensor,
  const at::Tensor &shard_lens,
  const at::Tensor &shard_gathered_offsets,
  const bool is_forward
) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  // shape of (num_shard, num_token_per_shard, hidden)
  _CHECK_TENSOR(3, gathered_tensor)
  // shape of (num_total_token, hidden)
  _CHECK_TENSOR(2, shuffled_tensor)
  // shape of (num_doc, num_shard)
  _CHECK_TENSOR(2, shard_lens)
  _CHECK_TENSOR(2, shard_gathered_offsets)

  const size_t hidden_sizes = gathered_tensor.size(2);
  TORCH_CHECK(hidden_sizes == shuffled_tensor.size(1));
  const size_t hidden_bytes = hidden_sizes * gathered_tensor.element_size();

  const size_t num_docs = shard_lens.size(0);
  const size_t num_shards = shard_lens.size(1);
  const size_t num_total_tokens = shuffled_tensor.size(0);

  TORCH_CHECK(num_docs == shard_gathered_offsets.size(0));
  TORCH_CHECK(num_shards == shard_gathered_offsets.size(1));

  launch_wlb_shuffle_memcpy(
    (uint8_t *)gathered_tensor.data_ptr(),
    (uint8_t *)shuffled_tensor.data_ptr(),
    shard_lens.data_ptr<int64_t>(),
    shard_gathered_offsets.data_ptr<int64_t>(),
    num_docs,
    num_total_tokens,
    hidden_bytes,
    is_forward,
    stream
  );
}

};  // namespace

TORCH_LIBRARY(wlbmemcpy_kernels, m) {
  m.def("wlb_shuffle_memcpy", &wlb_shuffle_memcpy);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
