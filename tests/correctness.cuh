#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#include "common/utils.cuh"

// PASS if kernel output is within 1% of the reference. this is more like a pass fail type. this is a normal thresholder until we actually add cuBLAS refrence.
constexpr float kErrorTolerance = 1.0e-2f;

// CPU reference uses the same BF16-rounded inputs and accumulates in FP32.
static void compute_cpu_reference(const std::vector<__nv_bfloat16>& h_A,
                                  const std::vector<__nv_bfloat16>& h_B,
                                  std::vector<float>& reference,
                                  int M, int N, int K) {
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
}

struct VerificationResult {
  float relative_error;
  bool correct;
};

static VerificationResult verify_output(const float* d_C,
                                        std::vector<float>& h_C,
                                        const std::vector<float>& reference) {
  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Normwise relative error over the whole output, accumulated in double so
  // summing millions of terms does not itself become the dominant error.
  double diff_norm_sq = 0.0;
  double ref_norm_sq = 0.0;
  for (size_t i = 0; i < h_C.size(); ++i) {
    const double diff = static_cast<double>(h_C[i]) - static_cast<double>(reference[i]);
    diff_norm_sq += diff * diff;
    ref_norm_sq += static_cast<double>(reference[i]) * static_cast<double>(reference[i]);
  }
  const double relative_error = ref_norm_sq > 0.0
      ? std::sqrt(diff_norm_sq) / std::sqrt(ref_norm_sq)
      : std::sqrt(diff_norm_sq);

  VerificationResult result;
  result.relative_error = static_cast<float>(relative_error);
  result.correct = relative_error <= static_cast<double>(kErrorTolerance);
  return result;
}
