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
    if (key_value_out.has_value()) {
        TORCH_CHECK(key_value_dst_id.has_value(), 
            "key_value tensor, dst_id, dst_offset must be provided together");
        TORCH_CHECK(
            num_tokens == key_value_out.value().size(0), "Key value tensor first dimension must be tokens"
        )
        max_cp_degree = key_value_dst_id.value().size(2);
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
    const size_t q_stride = query_out.stride(0);
    const size_t kv_stride = key_value_out.has_value() ? key_value_out.value().stride(0) : 0;

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

void destroy_dispatch_helper(fptr_t fptr) {
    delete (DispatchHelper*)fptr;
}

}; // namespace


namespace attn {
void register_all_to_all_ops(torch::Library &m) {
    m.def("dispatch", &dispatch);
    m.def("create_dispatch_helper", &create_dispatch_helper);
    m.def("destroy_dispatch_helper", &destroy_dispatch_helper);
}
}; // namespace attn
