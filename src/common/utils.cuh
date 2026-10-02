#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    const cudaError_t error__ = (call);                                      \
    if (error__ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,\
                   cudaGetErrorString(error__));                             \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

#define CUBLAS_CHECK(call)                                                   \
  do {                                                                       \
    const cublasStatus_t status__ = (call);                                  \
    if (status__ != CUBLAS_STATUS_SUCCESS) {                                 \
      std::fprintf(stderr, "cuBLAS error at %s:%d: status %d\n", __FILE__,   \
                   __LINE__, static_cast<int>(status__));                    \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

// PASS if kernel output is within 1% of the reference. this is more like a pass fail type. this is a normal thresholder until we actually add cuBLAS refrence.
constexpr float kErrorTolerance = 1.0e-2f;

static int ceil_div(int a, int b) { return (a - 1) / b + 1; }
