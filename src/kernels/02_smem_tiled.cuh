#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "common/utils.cuh"

// Compile-time tile dimensions. With the initial 32x32x32 tile, a 16x16
// thread block gives each thread a 2x2 output patch (four FP32 accumulators).
constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int TILE_K = 32;

template <int BM, int BN, int BK>
__global__ void shared_memory_bf16_gemm(const __nv_bfloat16* A,
                                        const __nv_bfloat16* B,
                                        float* C,
                                        int M, int N, int K) {
  static_assert(BM % 2 == 0 && BN % 2 == 0,
                "This kernel assigns a 2x2 output patch to each thread");

  // This is still a SIMT kernel: scalar BF16 conversion and FP32 arithmetic,
  // with no Tensor Core, WMMA/WGMMA, TMA, or asynchronous pipeline operations.
  // Dynamic shared memory is partitioned into row-major A[BM,BK] then B[BK,BN].
  extern __shared__ __nv_bfloat16 shared_tiles[];
  __nv_bfloat16* A_tile = shared_tiles;
  __nv_bfloat16* B_tile = A_tile + BM * BK;

  const size_t thread_id = static_cast<size_t>(threadIdx.y) * blockDim.x + threadIdx.x;
  const size_t thread_count = static_cast<size_t>(blockDim.x) * blockDim.y;
  const size_t a_tile_elements = static_cast<size_t>(BM) * BK;
  const size_t b_tile_elements = static_cast<size_t>(BK) * BN;
  // blockIdx.y selects a BM-row output tile; blockIdx.x selects a BN-column tile.
  const size_t block_row = static_cast<size_t>(blockIdx.y) * BM;
  const size_t block_col = static_cast<size_t>(blockIdx.x) * BN;

  const size_t row0 = block_row + 2 * static_cast<size_t>(threadIdx.y);
  const size_t row1 = row0 + 1;
  const size_t col0 = block_col + 2 * static_cast<size_t>(threadIdx.x);
  const size_t col1 = col0 + 1;

  float acc00 = 0.0f;
  float acc01 = 0.0f;
  float acc10 = 0.0f;
  float acc11 = 0.0f;

  for (size_t k_base = 0; k_base < static_cast<size_t>(K); k_base += BK) {
    // Threads cooperatively load contiguous row-major tiles. Out-of-range
    // elements at the matrix edges are zero-filled; no thread skips barriers.
    for (size_t i = thread_id; i < a_tile_elements; i += thread_count) {
      const size_t tile_row = i / BK;
      const size_t tile_k = i % BK;
      const size_t global_row = block_row + tile_row;
      const size_t global_k = k_base + tile_k;
      A_tile[i] = (global_row < static_cast<size_t>(M) && global_k < static_cast<size_t>(K))
                      ? A[global_row * static_cast<size_t>(K) + global_k]
                      : __float2bfloat16(0.0f);
    }
    for (size_t i = thread_id; i < b_tile_elements; i += thread_count) {
      const size_t tile_k = i / BN;
      const size_t tile_col = i % BN;
      const size_t global_k = k_base + tile_k;
      const size_t global_col = block_col + tile_col;
      B_tile[i] = (global_k < static_cast<size_t>(K) && global_col < static_cast<size_t>(N))
                      ? B[global_k * N + global_col]
                      : __float2bfloat16(0.0f);
    }

    // Every thread must see the fully loaded A/B tiles before reading them.
    __syncthreads();

    for (int k = 0; k < BK && k_base + k < static_cast<size_t>(K); ++k) {
      const float a0 = __bfloat162float(A_tile[(2 * threadIdx.y) * BK + k]);
      const float a1 = __bfloat162float(A_tile[(2 * threadIdx.y + 1) * BK + k]);
      const float b0 = __bfloat162float(B_tile[k * BN + 2 * threadIdx.x]);
      const float b1 = __bfloat162float(B_tile[k * BN + 2 * threadIdx.x + 1]);

      // Each shared A value is reused for two columns, and each shared B value
      // is reused for two rows; the tiles are also shared across the whole CTA.
      acc00 += a0 * b0;
      acc01 += a0 * b1;
      acc10 += a1 * b0;
      acc11 += a1 * b1;
    }

    // Do not overwrite a shared tile until all threads have finished consuming it.
    __syncthreads();
  }

  if (row0 < static_cast<size_t>(M) && col0 < static_cast<size_t>(N))
    C[row0 * static_cast<size_t>(N) + col0] = acc00;
  if (row0 < static_cast<size_t>(M) && col1 < static_cast<size_t>(N))
    C[row0 * static_cast<size_t>(N) + col1] = acc01;
  if (row1 < static_cast<size_t>(M) && col0 < static_cast<size_t>(N))
    C[row1 * static_cast<size_t>(N) + col0] = acc10;
  if (row1 < static_cast<size_t>(M) && col1 < static_cast<size_t>(N))
    C[row1 * static_cast<size_t>(N) + col1] = acc11;
}

// One BF16 A tile plus one BF16 B tile; large allocations can reduce occupancy.
// H100 has up to 228 KiB shared memory per SM and 227 KiB addressable per
// block; that is a hardware ceiling, not a target. Allocations above 48 KiB
// per block need an explicit opt-in. This initial pair of tiles uses only 4 KiB.
static void launch_smem_tiled(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
                              int M, int N, int K, cudaStream_t stream) {
  const dim3 tiled_block(TILE_N / 2, TILE_M / 2);
  const dim3 tiled_grid(ceil_div(N, TILE_N), ceil_div(M, TILE_M));
  const size_t tiled_smem_bytes = sizeof(__nv_bfloat16) *
      (static_cast<size_t>(TILE_M) * TILE_K +
       static_cast<size_t>(TILE_K) * TILE_N);
  shared_memory_bf16_gemm<TILE_M, TILE_N, TILE_K>
      <<<tiled_grid, tiled_block, tiled_smem_bytes, stream>>>(A, B, C, M, N, K);
}
