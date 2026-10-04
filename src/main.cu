// Naive BF16-input, FP32-accumulation GEMM baseline for NVIDIA Hopper.
//
// Build (requires a CUDA toolkit with sm_90a support and cuBLAS): `make`
//   (equivalent to: nvcc -O3 -std=c++17 -arch=sm_90a -lineinfo
//    -I src -I benchmarks -I tests src/main.cu -o gemm_baseline
//    -lcublas -lcuda)
// Run: ./gemm_baseline [M N K]

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include "common/utils.cuh"
#include "correctness.cuh"
#include "harness.cuh"
#include "kernels/00_cublas.cuh"
#include "kernels/01_naive.cuh"
#include "kernels/02_smem_tiled.cuh"

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

  const int M = argc == 4 ? parse_dimension(argv[1], "M") : 4096;
  const int N = argc == 4 ? parse_dimension(argv[2], "N") : 4096;
  const int K = argc == 4 ? parse_dimension(argv[3], "K") : 4096;
  const size_t a_count = checked_elements(static_cast<size_t>(M), K);
  const size_t b_count = checked_elements(static_cast<size_t>(K), N);
  const size_t c_count = checked_elements(static_cast<size_t>(M), N);

  std::vector<__nv_bfloat16> h_A(a_count);
  std::vector<__nv_bfloat16> h_B(b_count);
  std::vector<float> h_C(c_count);
  std::vector<float> reference(c_count, 0.0f);

  initialize_inputs(h_A, h_B);

  // The CPU triple loop is only fast enough to use as ground truth up to
  // 1024^3; above that, cuBLAS's own output (verified at smaller sizes)
  // stands in as the reference.
  const bool use_cpu_reference = M <= 1024 && N <= 1024 && K <= 1024;
  if (use_cpu_reference) {
    compute_cpu_reference(h_A, h_B, reference, M, N, K);
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

  const cudaStream_t stream = 0;

  init_cublas();
  const float cublas_ms = benchmark_kernel_ms([&]() {
    launch_cublas(d_A, d_B, d_C, M, N, K, stream);
  });
  if (!use_cpu_reference) {
    // No CPU reference at this size: cuBLAS's own output is the reference.
    CUDA_CHECK(cudaMemcpy(reference.data(), d_C, c_count * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  const VerificationResult cublas_result = verify_output(d_C, h_C, reference);
  print_kernel_result("cuBLAS", M, N, K, cublas_ms, cublas_result);
  const double cublas_gflops = achieved_gflops(M, N, K, cublas_ms);

  const float naive_ms = benchmark_kernel_ms([&]() {
    launch_naive(d_A, d_B, d_C, M, N, K, stream);
  });
  const VerificationResult naive_result = verify_output(d_C, h_C, reference);
  print_kernel_result("Naive", M, N, K, naive_ms, naive_result, cublas_gflops);

  const float tiled_ms = benchmark_kernel_ms([&]() {
    launch_smem_tiled(d_A, d_B, d_C, M, N, K, stream);
  });
  const VerificationResult tiled_result = verify_output(d_C, h_C, reference);
  print_kernel_result("Shared-memory tiled", M, N, K, tiled_ms, tiled_result, cublas_gflops);

  shutdown_cublas();
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  return cublas_result.correct && naive_result.correct && tiled_result.correct
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
