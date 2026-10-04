#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

#include "common/utils.cuh"
#include "correctness.cuh"

// Fixed seed makes runs reproducible; inputs are standard normal, rounded
// to BF16 for the GPU kernels and for the CPU reference below.
static void initialize_inputs(std::vector<__nv_bfloat16>& h_A,
                               std::vector<__nv_bfloat16>& h_B) {
  constexpr unsigned kRandomSeed = 42;
  std::mt19937 rng(kRandomSeed);
  std::normal_distribution<float> standard_normal(0.0f, 1.0f);
  for (size_t i = 0; i < h_A.size(); ++i) {
    h_A[i] = __float2bfloat16(standard_normal(rng));
  }
  for (size_t i = 0; i < h_B.size(); ++i) {
    h_B[i] = __float2bfloat16(standard_normal(rng));
  }
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

// reference_gflops is the cuBLAS kernel's achieved GFLOP/s; pass 0.0 to skip
// the "% of cuBLAS" line (used for the cuBLAS kernel's own result).
static void print_kernel_result(const char* name, int M, int N, int K,
                                float elapsed_ms,
                                const VerificationResult& result,
                                double reference_gflops = 0.0) {
  const double gflops = achieved_gflops(M, N, K, elapsed_ms);
  std::printf("%s dimensions: M=%d N=%d K=%d\n", name, M, N, K);
  std::printf("%s runtime: %.6f ms\n", name, elapsed_ms);
  std::printf("%s achieved GFLOP/s: %.3f\n", name, gflops);
  if (reference_gflops > 0.0) {
    std::printf("%s %% of cuBLAS: %.2f\n", name, 100.0 * gflops / reference_gflops);
  }
  std::printf("%s normwise relative error: %.8g\n", name, result.relative_error);
  std::printf("%s correctness: %s\n", name,
              result.correct ? "PASS" : "FAIL");
}
