#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "common/utils.cuh"

static cublasHandle_t g_cublas_handle;

static void init_cublas() {
  CUBLAS_CHECK(cublasCreate(&g_cublas_handle));
}

static void shutdown_cublas() {
  CUBLAS_CHECK(cublasDestroy(g_cublas_handle));
}

static void launch_cublas(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
                          int M, int N, int K, cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(g_cublas_handle, stream));

  // cuBLAS is column-major; A, B, C here are row-major. A row-major (r x c)
  // matrix is exactly the same bytes as its transpose read column-major, so
  // instead of transposing anything we compute C^T = B^T * A^T by swapping
  // the A/B arguments and swapping M/N: the N x M column-major result cuBLAS
  // writes is precisely our M x N row-major C.
  const float cublas_alpha = 1.0f;
  const float cublas_beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(g_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K,
                            &cublas_alpha,
                            B, CUDA_R_16BF, N,
                            A, CUDA_R_16BF, K,
                            &cublas_beta,
                            C, CUDA_R_32F, N,
                            CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
}
