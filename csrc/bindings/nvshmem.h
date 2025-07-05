/*
Code modified from https://github.com/ppl-ai/pplx-kernels and subject to the MIT License.
*/

#pragma once

#include <torch/library.h>

namespace attn {
void register_nvshmem_ops(torch::Library &m);
} // namespace attn
