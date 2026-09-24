// Naive BF16-input, FP32-accumulation GEMM baseline for NVIDIA Hopper.
//
// Build (requires a CUDA toolkit with sm_90 support):
//   nvcc -O2 -std=c++17 -arch=sm_90 h100_bf16_gemm_baseline.cu -o gemm_baseline
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

  const dim3 block(16, 16);
  const dim3 grid((N - 1) / block.x + 1,
                  (M - 1) / block.y + 1);
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  naive_bf16_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, c_count * sizeof(float), cudaMemcpyDeviceToHost));

  float max_abs_error = 0.0f;
  bool correct = true;
  constexpr float atol = 1.0e-3f;
  constexpr float rtol = 1.0e-3f;
  for (size_t i = 0; i < c_count; ++i) {
    const float abs_error = std::fabs(h_C[i] - reference[i]);
    max_abs_error = std::max(max_abs_error, abs_error);
    if (abs_error > atol + rtol * std::fabs(reference[i])) {
      correct = false;
    }
  }

  const double elapsed_seconds = static_cast<double>(elapsed_ms) * 1.0e-3;
  const double gflops = elapsed_seconds > 0.0
                            ? (2.0 * M * N * K) / elapsed_seconds / 1.0e9
                            : 0.0;
  std::printf("Dimensions: M=%d N=%d K=%d\n", M, N, K);
  std::printf("Kernel runtime: %.6f ms\n", elapsed_ms);
  std::printf("Achieved GFLOP/s: %.3f\n", gflops);
  std::printf("Max absolute error vs CPU reference: %.8g\n", max_abs_error);
  std::printf("Correctness: %s (atol=%.1e, rtol=%.1e)\n",
              correct ? "PASS" : "FAIL", atol, rtol);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  // This loop uses scalar BF16-to-FP32 conversions and ordinary FP32 arithmetic;
  // it has no Tensor Core instructions or shared-memory data reuse. Every output
  // thread independently rereads K values from global memory (including repeated
  // A/B loads), making it intentionally inefficient. That simple, correct design
  // provides a useful reference point for measuring future optimizations.
  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
