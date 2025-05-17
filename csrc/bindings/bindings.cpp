/*
Code modified from https://github.com/ppl-ai/pplx-kernels and subject to the MIT License.
*/

#include <torch/library.h>

// #include "bindings/all_to_all_ops.h"
#include "bindings/nvshmem.h"
#include "bindings/registration.h"

using namespace attn;

TORCH_LIBRARY(attn_kernels, m) {
  register_nvshmem_ops(m);
//   register_all_to_all_ops(m);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
