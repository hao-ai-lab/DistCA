#include "bindings/all_to_all.h"
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

void dispatch(
    fptr_t fptr,    // pointer to the dispatch helper function.
    at::Tensor &query_out,
    const std::optional<at::Tensor> &key_value_out,
    const at::Tensor &query_in,
    const std::optional<at::Tensor> &key_value_in,
    const at::Tensor &query_dst_id,
    const at::Tensor &query_dst_offset,
    const std::optional<at::Tensor> &key_value_dst_id,
    const std::optional<at::Tensor> &key_value_dst_offset,
    int64_t num_tokens,
    const at::Tensor &num_recv_tokens_query,
    const std::optional<at::Tensor> &num_recv_tokens_key_value
) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Get max cp degree
    size_t max_cp_degree;
    TORCH_CHECK(
        num_tokens == query_in.size(0), "Query tensor first dimension must be tokens"
    )
    TORCH_CHECK(
        query_in.ndimension() == 2, "Query tensor must have 2 dimensions"
    )
    TORCH_CHECK(
        query_in.size(1) == query_out.size(1), "Query input and output hidden size must match"
    )
    TORCH_CHECK(
        query_dst_id.ndimension() == 1, "Query dst_id must have 1 dimension"
    )
    if (key_value_out.has_value()) {
        TORCH_CHECK(key_value_dst_id.has_value(), 
            "key_value tensor, dst_id, dst_offset must be provided together");
        TORCH_CHECK(
            num_tokens == key_value_dst_id.value().size(0)
        )
        TORCH_CHECK(
            key_value_dst_id.value().ndimension() == 2, "Key value dst_id must have 2 dimensions"
        )
        TORCH_CHECK(
            key_value_dst_offset.has_value(), "Key value dst_offset must be provided when key value is sent"
        )
        TORCH_CHECK(
            key_value_dst_offset.value().ndimension() == 2, "Key value dst_offset must have 2 dimensions"
        )
        max_cp_degree = key_value_dst_id.value().size(1);
    } else {
        max_cp_degree = 0;
    }

    // Get dtype for tensors to send
    const c10::ScalarType query_dtype = query_in.scalar_type();
    const c10::ScalarType key_value_dtype = key_value_in.has_value() ? 
        key_value_in.value().scalar_type() : query_dtype;

    // Set device
    at::cuda::OptionalCUDAGuard const device_guard(device_of(query_out));

    auto* dispatch_helper = (DispatchHelper*)fptr;
    const size_t q_stride = query_out.stride(0) * query_out.element_size();
    const size_t kv_stride = key_value_out.has_value() ?
                             key_value_out.value().stride(0) * key_value_out.value().element_size() :
                             0;

    // TODO: use const_data_ptr?
    dispatch_helper->dispatch(
        (std::byte*)query_out.data_ptr(),
        key_value_out.has_value() ? (std::byte*)key_value_out.value().data_ptr() : nullptr,
        (const std::byte*)query_in.data_ptr(),
        key_value_in.has_value() ? (const std::byte*)key_value_in.value().data_ptr() : nullptr,
        query_dst_id.data_ptr<int32_t>(),
        query_dst_offset.data_ptr<uint32_t>(),
        key_value_dst_id.has_value() ? key_value_dst_id.value().data_ptr<int32_t>() : nullptr,
        key_value_dst_offset.has_value() ? key_value_dst_offset.value().data_ptr<uint32_t>() : nullptr,
        num_tokens,
        num_recv_tokens_query.data_ptr<uint64_t>(),
        num_recv_tokens_key_value.has_value() ? num_recv_tokens_key_value.value().data_ptr<uint64_t>() : nullptr,
        max_cp_degree,
        q_stride,
        kv_stride,
        stream
    );
}

void dispatch_v1(
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
    const std::optional<at::Tensor> &kv_recv_tensor,
    const std::optional<at::Tensor> &kv_dst_rank,
    const std::optional<at::Tensor> &kv_dst_offset,
    const std::optional<at::Tensor> &kv_num_recv_tokens
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

    // Get max cp degree for KV communication
    size_t max_cp_degree;
    if (kv_send_tensor.has_value()) {
        max_cp_degree = kv_dst_rank.value().size(1);
        TORCH_CHECK(kv_dst_offset.value().size(1) == max_cp_degree, "KV cp degree must match");
    } else {
        max_cp_degree = 0;
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

    dispatch_helper->dispatch_v1(
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
        stream
    );

}

void destroy_dispatch_helper(fptr_t fptr) {
    delete (DispatchHelper*)fptr;
}

}; // namespace


namespace attn {
void register_all_to_all_ops(torch::Library &m) {
    m.def("dispatch", &dispatch);
    m.def("dispatch_v1", &dispatch_v1);
    m.def("create_dispatch_helper", &create_dispatch_helper);
    m.def("destroy_dispatch_helper", &destroy_dispatch_helper);
}
}; // namespace attn
