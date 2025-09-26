/*
Code modified from https://github.com/ppl-ai/pplx-kernels and subject to the MIT License.
*/

#pragma once

#include <nvshmem.h>

#include "core/cuda_utils.h"

#define NVSHMEMCHECK(stmt)                                                                         \
  do {                                                                                             \
    int result = (stmt);                                                                           \
    if (NVSHMEMX_SUCCESS != result) {                                                              \
      fprintf(stderr, "[%s:%d] nvshmem failed with error %d \n", __FILE__, __LINE__, result);      \
      exit(-1);                                                                                    \
    }                                                                                              \
  } while (0)
