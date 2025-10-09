#include "bindings/all_to_all.h"
#include "core/fastalltoall.h"
#include "core/memcpy.h"

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

using namespace attn;

using fptr_t = int64_t;
namespace {

#define _CHECK_TENSOR(ndim, x) \
  TORCH_CHECK(x.ndimension() == ndim, "tensor " #x " must have " #ndim " dimensions"); \
  TORCH_CHECK(x.is_cuda(), "tensor " #x " must be on CUDA"); \
  TORCH_CHECK(x.is_contiguous(), "tensor " #x " must be contiguous");

/* ---------------- FastAlltoAll ----------------*/

fptr_t create_fast_a2a_dispatch_helper(
  int64_t rank, int64_t local_rank, int64_t world_size,
  int64_t buffer_size
) {
  auto *ptr = new FastA2aDispatchHelper(
    rank, local_rank, world_size, buffer_size);
  return (fptr_t)ptr;
}

void destroy_fast_a2a_dispatch_helper(fptr_t fptr) {
  delete (FastA2aDispatchHelper*)fptr;
}


void update_buffer_size(
  fptr_t fptr, int64_t target_size
) {
  auto* dispatch_helper = (FastA2aDispatchHelper*)fptr;
  dispatch_helper->update_buffer_size(target_size);
}


void _fast_a2a_memcpy_non_cp_core(
  at::Tensor &tensor,
  uint8_t *buffer_ptr,
  const at::Tensor &seq_nvshmem_offset,
  const at::Tensor &seq_tokens,
  const std::optional<at::Tensor> &copy_seq_mask,
  const bool to_nvshmem
) {
  _CHECK_TENSOR(2, tensor);
  _CHECK_TENSOR(1, seq_nvshmem_offset);

  const int64_t num_seq = seq_nvshmem_offset.size(0);
  if (copy_seq_mask.has_value()) {
    _CHECK_TENSOR(1, copy_seq_mask.value());
    TORCH_CHECK(copy_seq_mask.value().size(0) == num_seq,
                "copy_seq_mask must be of shape (num_sequence,)");
  }
  const int64_t token_bytes = tensor.size(1) * tensor.element_size();
  const int64_t num_tokens = tensor.size(0);
  uint8_t *tensor_ptr = (uint8_t *)tensor.data_ptr();

  at::cuda::OptionalCUDAGuard const device_guard(device_of(tensor));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  launch_memcpy_non_cp(
    tensor_ptr, buffer_ptr,
    seq_nvshmem_offset.const_data_ptr<int64_t>(),
    seq_tokens.const_data_ptr<int64_t>(),
    copy_seq_mask.has_value() ? copy_seq_mask.value().const_data_ptr<int8_t>() : nullptr,
    token_bytes, num_tokens, to_nvshmem, stream
  );
}


void fast_a2a_memcpy_non_cp(
  fptr_t fptr,
  at::Tensor &tensor,
  const at::Tensor &seq_nvshmem_offset,
  const at::Tensor &seq_tokens,
  const std::optional<at::Tensor> &copy_seq_mask,
  const bool to_nvshmem
) {
  // This is the non-buffer branch.
  uint8_t *buffer_ptr;
  if (to_nvshmem) {
    // If it goes to nvshmem, then it's a send side memcpy
    buffer_ptr = ((FastA2aDispatchHelper*)fptr)->buffer.send_buffer;
  } else {
    // Otherwise, it's a recv side memcpy
    buffer_ptr = ((FastA2aDispatchHelper*)fptr)->buffer.recv_buffer;
  }
  _fast_a2a_memcpy_non_cp_core(tensor, buffer_ptr, seq_nvshmem_offset,
    seq_tokens, copy_seq_mask, to_nvshmem);
}


void fast_a2a_memcpy_non_cp_debug(
  at::Tensor &tensor,
  const at::Tensor &seq_nvshmem_offset,
  const at::Tensor &seq_tokens,
  const std::optional<at::Tensor> &copy_seq_mask,
  const bool to_nvshmem,
  at::Tensor &buffer
) {
  uint8_t *buffer_ptr = (uint8_t *)buffer.data_ptr();
  _fast_a2a_memcpy_non_cp_core(tensor, buffer_ptr, seq_nvshmem_offset,
    seq_tokens, copy_seq_mask, to_nvshmem);
}


void _fast_a2a_memcpy_cp_core(
  at::Tensor &tensor,
  uint8_t *buffer_ptr,
  const at::Tensor &do_shard,
  const at::Tensor &seq_nvshmem_offset,
  const at::Tensor &seq_tokens,
  const bool to_nvshmem
) {
  if (to_nvshmem) {
    _CHECK_TENSOR(2, tensor);
  } else {
    _CHECK_TENSOR(3, tensor);
  }
  const int64_t token_bytes = (
    (to_nvshmem ? tensor.size(1) : tensor.size(2)) *
    tensor.element_size()
  );
  const int32_t num_tokens = to_nvshmem ? tensor.size(0) : tensor.size(1);
  const int64_t num_seq = seq_nvshmem_offset.size(1);
  const int32_t cp_degree = seq_nvshmem_offset.size(0);

  // Some shape check here
  _CHECK_TENSOR(1, seq_tokens);
  TORCH_CHECK(seq_tokens.size(0) == num_seq,
              "seq_tokens must be of dimension (num_sequence)");
  _CHECK_TENSOR(2, do_shard);
  TORCH_CHECK((do_shard.size(0) == num_seq) && (do_shard.size(1) == cp_degree),
              "do_shard must be of dimension (num_sequence, cp_degree)");

  uint8_t *tensor_ptr = (uint8_t *)tensor.data_ptr();

  at::cuda::OptionalCUDAGuard const device_guard(device_of(tensor));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  launch_memcpy_cp(
    tensor_ptr, buffer_ptr,
    do_shard.const_data_ptr<int8_t>(),
    seq_nvshmem_offset.const_data_ptr<int64_t>(),
    seq_tokens.const_data_ptr<int64_t>(),
    token_bytes, num_tokens, cp_degree, num_seq,
    to_nvshmem, stream
  );
}


void fast_a2a_memcpy_cp(
  fptr_t fptr,
  at::Tensor &tensor,
  const at::Tensor &do_shard,
  const at::Tensor &seq_nvshmem_offset,
  const at::Tensor &seq_tokens,
  const bool to_nvshmem
) {
  // This is the non-buffer branch.
  uint8_t *buffer_ptr;
  if (to_nvshmem) {
    // If it goes to nvshmem, then it's a send side memcpy
    buffer_ptr = ((FastA2aDispatchHelper*)fptr)->buffer.send_buffer;
  } else {
    // Otherwise, it's a recv side memcpy
    buffer_ptr = ((FastA2aDispatchHelper*)fptr)->buffer.recv_buffer;
  }
  _fast_a2a_memcpy_cp_core(tensor, buffer_ptr, do_shard, seq_nvshmem_offset,
                           seq_tokens, to_nvshmem);
}


void fast_a2a_grad_acc_pre_memcpy_cp(
  at::Tensor &tensor,
  const at::Tensor num_copies,
  const at::Tensor copy_start_id,
  const at::Tensor seq_lens
) {
  _CHECK_TENSOR(2, tensor);
  _CHECK_TENSOR(1, num_copies);
  _CHECK_TENSOR(1, seq_lens);
  _CHECK_TENSOR(2, copy_start_id);

  const int64_t num_tokens = tensor.size(0);
  const int64_t hidden = tensor.size(1);
  const int64_t num_seqs = num_copies.size(0);
  const int64_t num_max_copies = copy_start_id.size(1);
  TORCH_CHECK(seq_lens.size(0) == num_seqs,
              "seq_lens must be of shape (num_sequence,)");
  TORCH_CHECK(copy_start_id.size(0) == num_seqs,
              "copy_start_id must be of shape (num_sequence, max_num_copies)");
  if (tensor.dtype() == at::kFloat) {
    launch_inplace_grad_sum<float>(
      (float *)(tensor.data_ptr()),
      num_copies.data_ptr<int32_t>(),
      copy_start_id.data_ptr<int64_t>(),
      seq_lens.data_ptr<int64_t>(),
      num_tokens,
      hidden,
      num_seqs,
      num_max_copies,
      c10::cuda::getCurrentCUDAStream().stream()
    );
  } else if (tensor.dtype() == at::kHalf) {
    launch_inplace_grad_sum<half>(
      (half *)(tensor.data_ptr()),
      num_copies.data_ptr<int32_t>(),
      copy_start_id.data_ptr<int64_t>(),
      seq_lens.data_ptr<int64_t>(),
      num_tokens,
      hidden,
      num_seqs,
      num_max_copies,
      c10::cuda::getCurrentCUDAStream().stream()
    );
  } else if (tensor.dtype() == at::kBFloat16) {
    launch_inplace_grad_sum<nv_bfloat16>(
      (nv_bfloat16 *)(tensor.data_ptr()),
      num_copies.data_ptr<int32_t>(),
      copy_start_id.data_ptr<int64_t>(),
      seq_lens.data_ptr<int64_t>(),
      num_tokens,
      hidden,
      num_seqs,
      num_max_copies,
      c10::cuda::getCurrentCUDAStream().stream()
    );
  } else {
    TORCH_CHECK(false, "Only fp32, fp16 and bf16 inputs are supported");
  }
}


void fast_a2a_memcpy_cp_debug(
  at::Tensor &tensor,
  const at::Tensor &do_shard,
  const at::Tensor &seq_nvshmem_offset,
  const at::Tensor &seq_tokens,
  const bool to_nvshmem,
  at::Tensor &buffer
) {
  uint8_t *buffer_ptr = (uint8_t *)buffer.data_ptr();
  _fast_a2a_memcpy_cp_core(tensor, buffer_ptr, do_shard, seq_nvshmem_offset,
                           seq_tokens, to_nvshmem);
}


void fast_a2a(
  fptr_t fptr,
  const at::Tensor &sender_send_disp_tensor,
  const at::Tensor &sender_transfer_sz_tensor,
  const at::Tensor &sender_recv_disp_tensor,
  const at::Tensor &recver_transfer_sz_tensor,
  int64_t my_rank_send_offset,
  int64_t my_rank_recv_offset,
  int64_t my_rank_send_sz,
  bool separate_send_recv
) {
  auto dispatch_helper = (FastA2aDispatchHelper*)fptr;
  _CHECK_TENSOR(1, sender_send_disp_tensor);
  _CHECK_TENSOR(1, sender_transfer_sz_tensor);
  _CHECK_TENSOR(1, sender_recv_disp_tensor);
  _CHECK_TENSOR(1, recver_transfer_sz_tensor);
  const size_t world_size = dispatch_helper->_world_size;
  TORCH_CHECK(sender_send_disp_tensor.size(0) == world_size,
              "sender_send_disp must be of shape (world_size)");
  TORCH_CHECK(sender_transfer_sz_tensor.size(0) == world_size,
              "sender_transfer_sz must be of shape (world_size)");
  TORCH_CHECK(sender_recv_disp_tensor.size(0) == world_size,
              "sender_recv_disp must be of shape (world_size)");
  TORCH_CHECK(recver_transfer_sz_tensor.size(0) == world_size,
              "recver_transfer_sz must be of shape (world_size)");

  uint64_t *sender_send_disp = sender_send_disp_tensor.data_ptr<uint64_t>();
  uint64_t *sender_transfer_sz = sender_transfer_sz_tensor.data_ptr<uint64_t>();
  uint64_t *sender_recv_disp = sender_recv_disp_tensor.data_ptr<uint64_t>();
  uint64_t *recver_transfer_sz = recver_transfer_sz_tensor.data_ptr<uint64_t>();

  at::cuda::OptionalCUDAGuard const device_guard(device_of(sender_send_disp_tensor));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  internode_transfer_params_t param {
    .sender_send_disp = sender_send_disp,
    .sender_transfer_sz = sender_transfer_sz,
    .sender_recv_disp = sender_recv_disp,
    .recver_transfer_sz = recver_transfer_sz
  };

  launch_alltoallv(
    dispatch_helper->_rank,
    1,
    world_size,
    &dispatch_helper->buffer,
    &param,
    my_rank_send_offset,
    my_rank_recv_offset,
    my_rank_send_sz,
    stream,
    dispatch_helper->_buffer_size,
    separate_send_recv
  );
}

void release_buffer(
  fptr_t fptr
) {
  auto dispatch_helper = (FastA2aDispatchHelper*)fptr;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  launch_buffer_availability_kernel(
    dispatch_helper->_rank,
    1,
    dispatch_helper->_world_size,
    &dispatch_helper->buffer,
    true, // is_release
    stream
  );
}

void wait_and_consume_buffer(
  fptr_t fptr
) {
  auto dispatch_helper = (FastA2aDispatchHelper*)fptr;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  launch_buffer_availability_kernel(
    dispatch_helper->_rank,
    1,
    dispatch_helper->_world_size,
    &dispatch_helper->buffer,
    false, // is_release
    stream
  );
}

void _debug_nvshmem_buffer(
  fptr_t fptr,
  bool get_send,
  bool get_recv,
  bool get_signal,
  at::Tensor &target
) {
  if (get_send) {
    TORCH_CHECK(!get_recv, "can only probe one buffer");
    TORCH_CHECK(!get_signal, "can only probe one buffer");
  } else if (get_recv) {
    TORCH_CHECK(!get_send, "can only probe one buffer");
    TORCH_CHECK(!get_signal, "can only probe one buffer");
  } else if (get_signal) {
    TORCH_CHECK(!get_send, "can only probe one buffer");
    TORCH_CHECK(!get_recv, "can only probe one buffer");
  } else {
    TORCH_CHECK(false, "must get at least one buffer");
  }

  auto *dispatch_helper = (FastA2aDispatchHelper*)fptr;
  size_t size;

  if (get_send || get_recv) {
    size = dispatch_helper->_buffer_size;
    uint8_t *dst_ptr = (uint8_t *)(
      target.data_ptr()
    );
    uint8_t *src_ptr = get_send ?
      dispatch_helper->buffer.send_buffer :
      dispatch_helper->buffer.recv_buffer;
    TORCH_CHECK(target.nbytes() >= size, "dst tensor size < buffer size");
    cudaMemcpy(
      dst_ptr, src_ptr, size, cudaMemcpyDeviceToDevice
    );
  } else {
    size = dispatch_helper->_world_size;
    TORCH_CHECK(target.scalar_type() == at::ScalarType::UInt64,
                "dst tensor must be of dtype uint64");
    TORCH_CHECK(target.numel() >= size, "dst tensor size < world_size");
    uint64_t *dst_ptr = target.data_ptr<uint64_t>();
    cudaMemcpy(
      dst_ptr, dispatch_helper->buffer.sync_signal,
      size * sizeof(uint64_t), cudaMemcpyDeviceToDevice
    );
  }
}


}; // namespace


namespace attn {
void register_all_to_all_ops(torch::Library &m) {
  m.def("create_fast_a2a_dispatch_helper", &create_fast_a2a_dispatch_helper);
  m.def("destroy_fast_a2a_dispatch_helper", &destroy_fast_a2a_dispatch_helper);
  m.def("fast_a2a_update_buffer_size", &update_buffer_size);
  m.def("fast_a2a_memcpy_non_cp", &fast_a2a_memcpy_non_cp);
  m.def("fast_a2a_memcpy_non_cp_debug", &fast_a2a_memcpy_non_cp_debug);
  m.def("fast_a2a_memcpy_cp", &fast_a2a_memcpy_cp);
  m.def("fast_a2a_memcpy_cp_debug", &fast_a2a_memcpy_cp_debug);
  m.def("fast_a2a_grad_acc", &fast_a2a_grad_acc_pre_memcpy_cp);
  m.def("fast_a2a", &fast_a2a);
  m.def("release_buffer", &release_buffer);
  m.def("wait_and_consume_buffer", &wait_and_consume_buffer);
  m.def("_debug_nvshmem_buffer", &_debug_nvshmem_buffer);
}
}; // namespace attn
