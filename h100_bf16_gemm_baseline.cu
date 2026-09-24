// Naive BF16-input, FP32-accumulation GEMM baseline for NVIDIA Hopper.
//
// Build (requires a CUDA toolkit with sm_90 support):
//   nvcc -O3 -std=c++17 -arch=sm_90 h100_bf16_gemm_baseline.cu -o gemm_baseline
// Run: ./gemm_baseline [M N K]
//
// This intentionally simple kernel is a performance baseline for later
// shared-memory, TMA, WGMMA, and pipelining experiments.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    const cudaError_t error__ = (call);                                      \
    if (error__ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,\
                   cudaGetErrorString(error__));                             \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

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

struct VerificationResult {
  float max_abs_error;
  bool correct;
};

static VerificationResult verify_output(const float* d_C,
                                        std::vector<float>& h_C,
                                        const std::vector<float>& reference) {
  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  constexpr float atol = 1.0e-3f;
  constexpr float rtol = 1.0e-3f;
  VerificationResult result{0.0f, true};
  for (size_t i = 0; i < h_C.size(); ++i) {
    const float abs_error = std::fabs(h_C[i] - reference[i]);
    result.max_abs_error = std::max(result.max_abs_error, abs_error);
    if (abs_error > atol + rtol * std::fabs(reference[i])) {
      result.correct = false;
    }
  }
  return result;
}

template <typename Launch>
static float benchmark_kernel_ms(Launch launch) {
  constexpr int kSamples = 5;
  constexpr int kLaunchesPerSample = 10;
  float sample_ms[kSamples];

  // Warm up before timing so context/module initialization is not measured.
  launch();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int sample = 0; sample < kSamples; ++sample) {
    CUDA_CHECK(cudaEventRecord(start));
    for (int launch_index = 0; launch_index < kLaunchesPerSample; ++launch_index) {
      launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    sample_ms[sample] = elapsed_ms / kLaunchesPerSample;
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  std::sort(sample_ms, sample_ms + kSamples);
  return sample_ms[kSamples / 2];
}

static double achieved_gflops(int M, int N, int K, float elapsed_ms) {
  const double seconds = static_cast<double>(elapsed_ms) * 1.0e-3;
  return seconds > 0.0
             ? (2.0 * M * N * K) / seconds / 1.0e9
             : 0.0;
}

static void print_kernel_result(const char* name, int M, int N, int K,
                                float elapsed_ms,
                                const VerificationResult& result) {
  std::printf("%s dimensions: M=%d N=%d K=%d\n", name, M, N, K);
  std::printf("%s runtime: %.6f ms\n", name, elapsed_ms);
  std::printf("%s achieved GFLOP/s: %.3f\n",
              name, achieved_gflops(M, N, K, elapsed_ms));
  std::printf("%s max absolute error: %.8g\n", name, result.max_abs_error);
  std::printf("%s correctness: %s\n", name,
              result.correct ? "PASS" : "FAIL");
}

static int parse_dimension(const char* text, const char* name) {
  errno = 0;
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value <= 0 ||
      value > std::numeric_limits<int>::max()) {
    std::fprintf(stderr, "Invalid %s dimension: %s\n", name, text);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(value);
}

static size_t checked_elements(size_t x, size_t y) {
  if (y != 0 && x > std::numeric_limits<size_t>::max() / y) {
    std::fprintf(stderr, "Matrix size overflow\n");
    std::exit(EXIT_FAILURE);
  }
  return x * y;
}

int main(int argc, char** argv) {
  if (argc != 1 && argc != 4) {
    std::fprintf(stderr, "Usage: %s [M N K]\n", argv[0]);
    return EXIT_FAILURE;
  }

  const int M = argc == 4 ? parse_dimension(argv[1], "M") : 512;
  const int N = argc == 4 ? parse_dimension(argv[2], "N") : 512;
  const int K = argc == 4 ? parse_dimension(argv[3], "K") : 512;
  const size_t a_count = checked_elements(static_cast<size_t>(M), K);
  const size_t b_count = checked_elements(static_cast<size_t>(K), N);
  const size_t c_count = checked_elements(static_cast<size_t>(M), N);

  std::vector<__nv_bfloat16> h_A(a_count);
  std::vector<__nv_bfloat16> h_B(b_count);
  std::vector<float> h_C(c_count);
  std::vector<float> reference(c_count, 0.0f);

  // Deterministic, bounded inputs make runs reproducible and verification clear.
  for (size_t i = 0; i < a_count; ++i) {
    const float value = static_cast<float>(static_cast<int>((i * 17 + 3) % 101) - 50) / 50.0f;
    h_A[i] = __float2bfloat16(value);
  }
  for (size_t i = 0; i < b_count; ++i) {
    const float value = static_cast<float>(static_cast<int>((i * 29 + 11) % 97) - 48) / 48.0f;
    h_B[i] = __float2bfloat16(value);
  }

  // CPU reference uses the same BF16-rounded inputs and accumulates in FP32.
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        const float a = __bfloat162float(h_A[static_cast<size_t>(row) * K + k]);
        const float b = __bfloat162float(h_B[static_cast<size_t>(k) * N + col]);
        acc += a * b;
      }
      reference[static_cast<size_t>(row) * N + col] = acc;
    }
  }

  __nv_bfloat16* d_A = nullptr;
  __nv_bfloat16* d_B = nullptr;
  float* d_C = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), a_count * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), b_count * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), c_count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

  std::printf("Timing: median of 5 samples, 10 launches per sample; one warm-up launch per kernel\n");

  const dim3 naive_block(16, 16);
  const dim3 naive_grid((N - 1) / naive_block.x + 1,
                        (M - 1) / naive_block.y + 1);
  const float naive_ms = benchmark_kernel_ms([&]() {
    naive_bf16_gemm<<<naive_grid, naive_block>>>(d_A, d_B, d_C, M, N, K);
  });
  const VerificationResult naive_result = verify_output(d_C, h_C, reference);
  print_kernel_result("Naive", M, N, K, naive_ms, naive_result);

  const dim3 tiled_block(TILE_N / 2, TILE_M / 2);
  const dim3 tiled_grid((N - 1) / TILE_N + 1,
                        (M - 1) / TILE_M + 1);
  // One BF16 A tile plus one BF16 B tile. Future multi-stage pipelines will
  // multiply this by their stage count; large allocations can reduce occupancy.
  const size_t tiled_smem_bytes = sizeof(__nv_bfloat16) *
      (static_cast<size_t>(TILE_M) * TILE_K +
       static_cast<size_t>(TILE_K) * TILE_N);
  // H100 has up to 228 KiB shared memory per SM and 227 KiB addressable per
  // block; that is a hardware ceiling, not a target. Allocations above 48 KiB
  // per block need an explicit opt-in. This initial pair of tiles uses only 4 KiB.
  const float tiled_ms = benchmark_kernel_ms([&]() {
    shared_memory_bf16_gemm<TILE_M, TILE_N, TILE_K>
        <<<tiled_grid, tiled_block, tiled_smem_bytes>>>(d_A, d_B, d_C, M, N, K);
  });
  const VerificationResult tiled_result = verify_output(d_C, h_C, reference);
  print_kernel_result("Shared-memory tiled", M, N, K, tiled_ms, tiled_result);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  return naive_result.correct && tiled_result.correct
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}

// ============================================================
// HOPPER OPTIMIZATION ROADMAP
// ============================================================

// TODO(Hopper-TMA): Replace thread-driven global-to-shared tile loads with
// Hopper Tensor Memory Accelerator (TMA) multidimensional asynchronous transfers
// using tensor/copy descriptors.

// TODO(Hopper-WGMMA): Replace scalar FP32 multiply/accumulate with Hopper
// wgmma.mma_async operations executed by 128-thread warpgroups. Keep BF16 inputs
// and FP32 accumulation.

// TODO(Hopper-Pipeline): Introduce double-buffered shared-memory staging so that
// while WGMMA consumes K tile i, TMA asynchronously loads K tile i+1. Later
// investigate triple buffering and tune pipeline depth.

// TODO(Hopper-MBarrier): Use Hopper asynchronous transaction barriers / mbarrier
// to coordinate TMA producer completion, WGMMA consumers, and safe stage reuse.

// TODO(Hopper-SMEM-Layout): Investigate shared-memory layouts/swizzling for
// WGMMA/TMA requirements and to minimize shared-memory bank conflicts.

// TODO(Tile-Autotuning): Maintain legal optimized tile configurations rather
// than arbitrary runtime tile sizes. Select based on M, N, K and hardware limits.

// TODO(Register-Pressure): Measure WGMMA accumulator register usage and tune
// tile shape, warpgroups, and pipeline stages to balance pressure and occupancy.

// TODO(L2-Locality): Investigate output-tile traversal/block scheduling that
// increases reuse of A or B panels in H100's 50 MB L2 cache.

// TODO(L2-Persisting): Experiment with CUDA L2 access-policy windows / persisting
// cache hints for selected reusable working sets. Do not persist entire large
// matrices; measure whether this helps.

// TODO(Thread-Block-Clusters): Investigate Hopper Thread Block Clusters for
// cooperative GEMM execution across multiple blocks/SMs.

// TODO(DSMEM): Investigate Distributed Shared Memory (DSMEM) so blocks in one
// cluster can share A or B panels instead of fetching identical data repeatedly.

// TODO(Cluster-TMA): Investigate cluster-level data movement / multicast-style
// reuse to reduce redundant tile traffic between cooperating blocks.

// TODO(Dynamic-SMEM-Tuning): Choose dynamic shared-memory allocation based on
// tile shape and pipeline stages while respecting H100 resource and occupancy limits.

// TODO(CuBLASLt-Benchmark): Benchmark optimized implementations against cuBLASLt.

// TODO(CUTLASS-Benchmark): Compare design and performance with CUTLASS 3.x
// Hopper GEMM kernels.

// TODO(Nsight-Compute): Profile each stage with Nsight Compute: Tensor Core use,
// DRAM throughput, L2 behavior, shared-memory throughput/bank conflicts, register
// pressure, occupancy, and warp stalls.
