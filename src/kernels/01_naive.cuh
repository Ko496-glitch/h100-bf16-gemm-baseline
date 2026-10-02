#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "common/utils.cuh"

// One thread computes one C[row, col]. A 16x16 block covers a 16x16 output
// region: threadIdx.y/blockIdx.y select row, and threadIdx.x/blockIdx.x select
// column. Bounds checks allow dimensions that are not multiples of 16.
// The K loop independently reloads global A/B values for every output; A is
// redundantly fetched across columns and B is strided per thread. Scalar FP32
// arithmetic here does not use Tensor Cores, making this a useful baseline.
__global__ void naive_bf16_gemm(const __nv_bfloat16* A,
                               const __nv_bfloat16* B,
                               float* C,
                               int M, int N, int K) {
  const size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      // Each thread walks one row of A and one column of B. Neighboring
      // columns in a warp read contiguous B values at each k, while A values
      // are redundantly loaded by many columns; B's per-thread walk is strided.
      const float a = __bfloat162float(A[row * static_cast<size_t>(K) + k]);
      const float b = __bfloat162float(B[static_cast<size_t>(k) * N + col]);
      acc += a * b;
    }
    C[row * static_cast<size_t>(N) + col] = acc;
  }
}

static void launch_naive(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
                         int M, int N, int K, cudaStream_t stream) {
  const dim3 naive_block(16, 16);
  const dim3 naive_grid(ceil_div(N, 16), ceil_div(M, 16));
  naive_bf16_gemm<<<naive_grid, naive_block, 0, stream>>>(A, B, C, M, N, K);
}
