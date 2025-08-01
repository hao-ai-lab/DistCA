#include "bindings/all_to_all.h"
#include "core/fastalltoall.h"
#include "core/memcpy.h"
#include "core/in_place_attn_switch.h"

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

fptr_t create_dispatch_helper(
  int64_t q_stride,
  int64_t kv_stride,
  int64_t max_tokens_query,
  int64_t max_tokens_key_value,
  int64_t rank,
  int64_t world_size
) {
  auto *ptr = new DispatchHelper(
    q_stride, kv_stride, max_tokens_query, max_tokens_key_value,
    rank, world_size
  );
  return (fptr_t)ptr;
}


void dispatch_core(
  fptr_t fptr,
  //
  at::Tensor &send_tensor,
  at::Tensor &recv_tensor,
  const at::Tensor &dst_rank,
  const at::Tensor &dst_offset,
  const at::Tensor &num_recv_tokens,
  const at::Tensor &seq_len,
  //
  const std::optional<at::Tensor> &kv_send_tensor,
  // FIXME: this "const" is wrong! this tensor will be write.
  const std::optional<at::Tensor> &kv_recv_tensor,
  const std::optional<at::Tensor> &kv_dst_rank,
  const std::optional<at::Tensor> &kv_dst_offset,
  const std::optional<at::Tensor> &kv_num_recv_tokens,
  //
  const std::optional<at::Tensor> &seq_recv_mask,
  const std::optional<at::Tensor> &recv_seq_lens
) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  auto* dispatch_helper = (DispatchHelper*)fptr;

  // Check shape:
  const size_t hidden_size = send_tensor.size(1);
  const size_t num_sequence = dst_rank.size(0);
  const size_t num_tokens = send_tensor.size(0);

  const unsigned world_size = dispatch_helper->world_size();
  TORCH_CHECK(send_tensor.ndimension() == 2, "Input tensor is of dimension (token, hidden_size)");
  TORCH_CHECK(recv_tensor.ndimension() == 2, "Output tensor is of dimension (recv_token, hidden_size)");
  TORCH_CHECK(dst_rank.ndimension() == 1, "Dst rank must be of dimension (num_sequence)");
  TORCH_CHECK(dst_offset.ndimension() == 1, "Dst offset must be of dimension (num_sequence)");
  TORCH_CHECK(num_recv_tokens.ndimension() == 1, "Num recv tokens must be of dimension (world_size + 1)");
  TORCH_CHECK(seq_len.ndimension() == 1, "Seq len must be of dimension (num_sequence)");

  TORCH_CHECK(recv_tensor.size(1) == hidden_size, "Hidden size must match");
  TORCH_CHECK(dst_offset.size(0) == num_sequence, "Dst offset must be of dimension (num_sequence)");
  TORCH_CHECK(num_recv_tokens.size(0) == world_size + 1, "Num recv tokens must be of dimension (world_size + 1)");
  TORCH_CHECK(seq_len.size(0) == num_sequence, "Seq len must be of dimension (num_sequence)");
  if (kv_send_tensor.has_value()) {
    const size_t kv_hidden_size = kv_send_tensor.value().size(1);

    TORCH_CHECK(kv_recv_tensor.has_value(), "KV tensor send and recv must be provided together");
    TORCH_CHECK(kv_dst_rank.has_value(), "KV dst rank must be provided.");
    TORCH_CHECK(kv_dst_offset.has_value(), "KV dst offset must be provided.");
    TORCH_CHECK(kv_num_recv_tokens.has_value(), "KV num recv tokens must be provided.");

    TORCH_CHECK(kv_recv_tensor.value().ndimension() == 2, "KV recv tensor must be of dimension (recv_token, kv_hidden_size)");
    TORCH_CHECK(kv_dst_rank.value().ndimension() == 2, "KV dst rank must be of dimension (num_sequence, cp_degree)");
    TORCH_CHECK(kv_dst_offset.value().ndimension() == 2, "KV dst offset must be of dimension (num_sequence, cp_degree)");
    TORCH_CHECK(kv_num_recv_tokens.value().ndimension() == 1, "KV num recv tokens must be of dimension (world_size + 1)");

    TORCH_CHECK(kv_send_tensor.value().size(0) == num_tokens, "KV send tensor must be of dimension (num_tokens, kv_hidden_size)");
    TORCH_CHECK(kv_recv_tensor.value().size(1) == kv_hidden_size, "KV hidden size must match");
    TORCH_CHECK(kv_dst_rank.value().size(0) == num_sequence, "KV dst rank dim 0 must be sequence length");
    TORCH_CHECK(kv_dst_offset.value().size(0) == num_sequence, "KV dst offset dim 0 must be sequence length");
    TORCH_CHECK(kv_num_recv_tokens.value().size(0) == world_size + 1, "KV num recv tokens must be of dimension (world_size + 1)");
  }

  if (seq_recv_mask.has_value()) {
    TORCH_CHECK(seq_recv_mask.value().ndimension() == 2, "seq_recv_mask is of dimension (num_sequence, cp_degree)");
    TORCH_CHECK(recv_seq_lens.has_value(), "recv_seq_lens must be provided.");
    TORCH_CHECK(recv_seq_lens.value().ndimension() == 1, "recv_seq_lens is of dimension (num_sequence)");
    TORCH_CHECK(recv_seq_lens.value().size(0) == seq_recv_mask.value().size(0), "recv_seq_lens dim 0 different from seq_recv_mask dim 0");
    TORCH_CHECK(!kv_send_tensor.has_value(), "seq_recv_mask is used to send kv backward using the query-only-forward pattern");
  }
  // Get max cp degree for KV communication
  size_t max_cp_degree;
  if (kv_send_tensor.has_value()) {
    max_cp_degree = kv_dst_rank.value().size(1);
    TORCH_CHECK(kv_dst_offset.value().size(1) == max_cp_degree, "KV cp degree must match");
  } else if (seq_recv_mask.has_value()) {
    max_cp_degree = seq_recv_mask.value().size(1);
  } else {
    max_cp_degree = 0;
  }
  // Get kv backward num receive tokens for receive KV backward gradient.
  size_t kv_backward_num_tokens;
  if (seq_recv_mask.has_value()) {
    kv_backward_num_tokens = recv_tensor.size(0) / max_cp_degree;
  } else {
    kv_backward_num_tokens = 0;
  }

  // Get dtype for tensors to send
  const c10::ScalarType dtype = send_tensor.scalar_type();
  if (kv_send_tensor.has_value()) {
    TORCH_CHECK(kv_send_tensor.value().scalar_type() == dtype, "KV must have the same dtype as Query.");
  }

  // Set device
  at::cuda::OptionalCUDAGuard const device_guard(device_of(send_tensor));

  const size_t stride = send_tensor.stride(0) * send_tensor.element_size();
  const size_t kv_stride = kv_send_tensor.has_value() ?
               kv_send_tensor.value().stride(0) * kv_send_tensor.value().element_size() :
               0;

  dispatch_helper->dispatch(
    // Input and output tensors
    (const std::byte *)send_tensor.data_ptr(),
    (std::byte *)recv_tensor.data_ptr(),
    kv_send_tensor.has_value() ? (const std::byte *)kv_send_tensor.value().data_ptr() : nullptr,
    kv_recv_tensor.has_value() ? (std::byte *)kv_recv_tensor.value().data_ptr() : nullptr,
    // Metadata tensors
    dst_rank.data_ptr<int32_t>(),
    dst_offset.data_ptr<uint32_t>(),
    num_recv_tokens.data_ptr<uint64_t>(),
    seq_len.data_ptr<uint32_t>(),
    //
    kv_dst_rank.has_value() ? kv_dst_rank.value().data_ptr<int32_t>() : nullptr,
    kv_dst_offset.has_value() ? kv_dst_offset.value().data_ptr<uint32_t>() : nullptr,
    kv_num_recv_tokens.has_value() ? kv_num_recv_tokens.value().data_ptr<uint64_t>() : nullptr,
    // Metadata
    num_tokens,
    num_sequence,
    max_cp_degree,
    stride,
    kv_stride,
    stream,
    // recv kv backward metadata
    seq_recv_mask.has_value() ? seq_recv_mask.value().data_ptr<uint32_t>() : nullptr,
    recv_seq_lens.has_value() ? recv_seq_lens.value().data_ptr<uint32_t>() : nullptr,
    kv_backward_num_tokens
  );

}

void destroy_dispatch_helper(fptr_t fptr) {
  delete (DispatchHelper*)fptr;
}

void set_num_sms(fptr_t fptr, const int64_t num_sms) {
  auto* dispatch_helper = (DispatchHelper*)fptr;
  dispatch_helper->set_num_sms(num_sms);
}

/* ---------------- FastAlltoAll ----------------*/
// TODO

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
  const bool to_nvshmem
) {
  TORCH_CHECK(tensor.ndimension() == 2, "Input tensor is of dimension (token, hidden_size)");
  const int64_t token_bytes = tensor.size(1) * tensor.element_size();
  const int64_t num_tokens = tensor.size(0);
  uint8_t *tensor_ptr = (uint8_t *)tensor.data_ptr();

  at::cuda::OptionalCUDAGuard const device_guard(device_of(tensor));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  launch_memcpy_non_cp(
    tensor_ptr, buffer_ptr,
    seq_nvshmem_offset.const_data_ptr<int64_t>(),
    seq_tokens.const_data_ptr<int64_t>(),
    token_bytes, num_tokens, to_nvshmem, stream
  );
}


void fast_a2a_memcpy_non_cp(
  fptr_t fptr,
  at::Tensor &tensor,
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
  _fast_a2a_memcpy_non_cp_core(tensor, buffer_ptr,
                        seq_nvshmem_offset, seq_tokens, to_nvshmem);
}


void fast_a2a_memcpy_non_cp_debug(
  at::Tensor &tensor,
  const at::Tensor &seq_nvshmem_offset,
  const at::Tensor &seq_tokens,
  const bool to_nvshmem,
  at::Tensor &buffer
) {
  uint8_t *buffer_ptr = (uint8_t *)buffer.data_ptr();
  _fast_a2a_memcpy_non_cp_core(tensor, buffer_ptr,
                        seq_nvshmem_offset, seq_tokens, to_nvshmem);
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
  launch_memcpy_cp_send(
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
  int64_t my_rank_send_sz
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
    stream
  );
}


}; // namespace


namespace attn {
void register_all_to_all_ops(torch::Library &m) {
  m.def("dispatch_core", &dispatch_core);
  m.def("create_dispatch_helper", &create_dispatch_helper);
  m.def("destroy_dispatch_helper", &destroy_dispatch_helper);
  m.def("set_num_sms", &set_num_sms);
  m.def("create_fast_a2a_dispatch_helper", &create_fast_a2a_dispatch_helper);
  m.def("destroy_fast_a2a_dispatch_helper", &destroy_fast_a2a_dispatch_helper);
  m.def("fast_a2a_update_buffer_size", &update_buffer_size);
  m.def("fast_a2a_memcpy_non_cp", &fast_a2a_memcpy_non_cp);
  m.def("fast_a2a_memcpy_non_cp_debug", &fast_a2a_memcpy_non_cp_debug);
  m.def("fast_a2a_memcpy_cp", &fast_a2a_memcpy_cp);
  m.def("fast_a2a_memcpy_cp_debug", &fast_a2a_memcpy_cp_debug);
  m.def("fast_a2a", &fast_a2a);
}
}; // namespace attn
