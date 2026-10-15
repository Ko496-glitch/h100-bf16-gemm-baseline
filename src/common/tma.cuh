#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "common/barrier.cuh"

#define CU_CHECK(call)                                                       \
  do {                                                                       \
    const CUresult result__ = (call);                                        \
    if (result__ != CUDA_SUCCESS) {                                          \
      const char* name__ = nullptr;                                          \
      cuGetErrorName(result__, &name__);                                     \
      std::fprintf(stderr, "CUDA driver error at %s:%d: %s\n", __FILE__,     \
                   __LINE__, name__ != nullptr ? name__ : "unknown");        \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

#define TMA_REQUIRE(condition, ...)                                          \
  do {                                                                       \
    if (!(condition)) {                                                      \
      std::fprintf(stderr, "TMA descriptor error at %s:%d: ", __FILE__,      \
                   __LINE__);                                                \
      std::fprintf(stderr, __VA_ARGS__);                                     \
      std::fprintf(stderr, "\n");                                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

// Encodes a 2D tile-mode tensor map for a row-major rows x cols BF16 matrix,
// loaded in boxes of box_rows x box_cols. TMA lists dimensions and strides
// fastest-varying first, so the global dims are {cols, rows} and the box is
// {box_cols, box_rows}. Out-of-bounds box elements are zero-filled.
// The layout rules are checked here first so a bad box fails with a clear
// message instead of a bare CUDA_ERROR_INVALID_VALUE from the driver.
static void make_tma_2d_bf16(CUtensorMap* map, const __nv_bfloat16* ptr,
                             uint64_t rows, uint64_t cols,
                             uint32_t box_rows, uint32_t box_cols,
                             CUtensorMapSwizzle swizzle) {
  const uint64_t element_bytes = sizeof(__nv_bfloat16);
  const uint64_t inner_box_bytes = box_cols * element_bytes;
  const uint64_t row_stride_bytes = cols * element_bytes;

  uint64_t swizzle_span_bytes = 0;  // 0: no limit on the inner box from swizzling
  switch (swizzle) {
    case CU_TENSOR_MAP_SWIZZLE_NONE: swizzle_span_bytes = 0; break;
    case CU_TENSOR_MAP_SWIZZLE_32B: swizzle_span_bytes = 32; break;
    case CU_TENSOR_MAP_SWIZZLE_64B: swizzle_span_bytes = 64; break;
    case CU_TENSOR_MAP_SWIZZLE_128B: swizzle_span_bytes = 128; break;
    default:
      TMA_REQUIRE(false, "unsupported swizzle mode %d", static_cast<int>(swizzle));
  }

  TMA_REQUIRE(box_rows >= 1 && box_rows <= 256 && box_cols >= 1 && box_cols <= 256,
              "box dims must be in [1, 256], got %u rows x %u cols", box_rows, box_cols);
  TMA_REQUIRE(inner_box_bytes % 16 == 0,
              "inner box must be a multiple of 16 bytes, got %llu bytes (%u BF16 columns)",
              static_cast<unsigned long long>(inner_box_bytes), box_cols);
  TMA_REQUIRE(swizzle_span_bytes == 0 || inner_box_bytes <= swizzle_span_bytes,
              "inner box of %llu bytes (%u BF16 columns) exceeds the %llu-byte swizzle span; "
              "at most %llu BF16 columns for this swizzle mode",
              static_cast<unsigned long long>(inner_box_bytes), box_cols,
              static_cast<unsigned long long>(swizzle_span_bytes),
              static_cast<unsigned long long>(swizzle_span_bytes / element_bytes));
  TMA_REQUIRE(row_stride_bytes % 16 == 0,
              "row stride must be a multiple of 16 bytes, got %llu bytes (%llu BF16 columns)",
              static_cast<unsigned long long>(row_stride_bytes),
              static_cast<unsigned long long>(cols));
  TMA_REQUIRE(reinterpret_cast<uintptr_t>(ptr) % 16 == 0,
              "global address %p must be 16-byte aligned", static_cast<const void*>(ptr));

  const cuuint64_t global_dim[2] = {cols, rows};
  const cuuint64_t global_strides[1] = {row_stride_bytes};
  const cuuint32_t box_dim[2] = {box_cols, box_rows};
  const cuuint32_t element_strides[2] = {1, 1};
  CU_CHECK(cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
                                  const_cast<__nv_bfloat16*>(ptr),
                                  global_dim, global_strides, box_dim, element_strides,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                                  CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}


__device__ __forceinline__ void tma_load_2d(void* smem_dst, const CUtensorMap* map,
                                            int col, int row, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];"
      :
      : "r"(shared_address(smem_dst)),
        "l"(reinterpret_cast<uint64_t>(map)),
        "r"(col), "r"(row),
        "r"(shared_address(bar))
      : "memory");
}
