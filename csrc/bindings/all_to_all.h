#pragma once

#include <torch/library.h>

namespace attn {
void register_all_to_all_ops(torch::Library &m);
} // namespace attn
