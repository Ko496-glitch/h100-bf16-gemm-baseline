#pragma once

#include <cuda_runtime.h>

#include <cstdint>

// Shared-memory matrix descriptors, synchronization, and the m64n64k16 BF16
// wgmma.mma_async wrapper for one warpgroup (128 threads, warps 4k..4k+3).

// Swizzle mode field, descriptor bits 62-63.
enum class WgmmaSwizzle : uint64_t {
  None = 0,
  B128 = 1,
  B64 = 2,
  B32 = 3,
};

// matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4
__host__ __device__ constexpr uint64_t wgmma_desc_encode(uint64_t x) {
  return (x & 0x3FFFFull) >> 4;
}

// Packs a 64-bit matrix descriptor:
//   bits  0-13  start address (shared state space), encoded
//   bits 16-29  leading dimension byte offset, encoded
//   bits 32-45  stride dimension byte offset, encoded
//   bits 49-51  base offset, always 0 here: tiles start on a 1024-byte
//               boundary, where the 128B pattern repeats
//   bits 62-63  swizzle mode
__host__ __device__ constexpr uint64_t make_wgmma_desc(uint32_t smem_addr,
                                                       uint32_t lbo_bytes,
                                                       uint32_t sbo_bytes,
                                                       WgmmaSwizzle swizzle) {
  return wgmma_desc_encode(smem_addr) |
         (wgmma_desc_encode(lbo_bytes) << 16) |
         (wgmma_desc_encode(sbo_bytes) << 32) |
         (static_cast<uint64_t>(swizzle) << 62);
}

// Descriptor for a K-major, 128B-swizzled BF16 tile with 64 elements (128
// bytes) per row, starting on a 1024-byte boundary.
//   LBO: not used for swizzled K-major layouts and assumed to be 1, i.e. 16
//   SBO: offset from one group of 8 rows to the next, 8 x 128 = 1024 bytes.
constexpr uint32_t kWgmmaKSw128Lbo = 16;
constexpr uint32_t kWgmmaKSw128Sbo = 1024;

__device__ __forceinline__ uint64_t make_wgmma_desc_k_sw128(const void* smem_tile) {
  const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_tile));
  return make_wgmma_desc(smem_addr, kWgmmaKSw128Lbo, kWgmmaKSw128Sbo, WgmmaSwizzle::B128);
}

// Moves the descriptor's start address by a multiple of 16 bytes. Like
// CUTLASS's DescriptorIterator, this adds to the low 32 bits only. Inside a
// K-major 128B-swizzled 64-wide BF16 tile, one k16 step is +32 bytes; the
// hardware applies the swizzle to the advanced address.
__host__ __device__ constexpr uint64_t wgmma_desc_advance(uint64_t desc, uint32_t bytes) {
  const uint32_t low = static_cast<uint32_t>(desc) + (bytes >> 4);
  return (desc & 0xFFFFFFFF00000000ull) | low;
}

// Orders prior register accesses before later wgmma.mma_async accesses to the
// same accumulator registers. Every warp of the warpgroup must execute it.
__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}

// Batches all uncommitted wgmma.mma_async operations into one wgmma-group.
__device__ __forceinline__ void wgmma_commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

// Waits until at most N of the most recent wgmma-groups are pending.
template <int N>
__device__ __forceinline__ void wgmma_wait_group() {
  static_assert(N >= 0 && N <= 7, "wgmma.wait_group: N must be in [0, 7]");
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}

// Makes this thread's prior generic-proxy shared-memory writes visible to the
// async proxy before wgmma.mma_async reads them. Needed when threads fill the
// tiles with ordinary stores; tiles written by TMA are already async proxy.
__device__ __forceinline__ void wgmma_fence_smem_writes() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// Compiler-only fence: each accumulator register is an input and output of an
// empty asm, so the compiler cannot move accumulator reads or writes across
// it. Same idea as CUTLASS's warpgroup_fence_operand. Emits no instructions.
__device__ __forceinline__ void wgmma_fence_accumulators(float (&d)[32]) {
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    asm volatile("" : "+f"(d[i])::"memory");
  }
}

// D (64x64, FP32) = A (64x16) * B (16x64) + (scale_d != 0 ? D : 0)
// A and B are BF16 in shared memory, described by desc_a and desc_b, both
// K-major (imm-trans-a = imm-trans-b = 0). imm-scale-a = imm-scale-b = 1.
// scale-d is a predicate, set here from the runtime value scale_d.
// All 128 threads of the warpgroup must call this together.
__device__ __forceinline__ void wgmma_m64n64k16_bf16_f32(float (&d)[32],
                                                         uint64_t desc_a,
                                                         uint64_t desc_b,
                                                         int scale_d) {
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %34, 0;\n"
      "  wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, "
      "%8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, "
      "%24, %25, %26, %27, %28, %29, %30, %31}, "
      "%32, %33, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
        "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
        "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(desc_a), "l"(desc_b), "r"(scale_d));
}

// Position of accumulator register reg (0..N/2-1) held by thread t (0..127)
// of the warpgroup, for wgmma.mma_async.m64nNk16 with FP32 D. Warp w = t/32
// owns rows 16w..16w+15; with lane l = t%32 and 8-column chunk i = reg/4:
//   d[4i+0] -> (16w + l/4,     8i + 2(l%4))
//   d[4i+1] -> (16w + l/4,     8i + 2(l%4) + 1)
//   d[4i+2] -> (16w + l/4 + 8, 8i + 2(l%4))
//   d[4i+3] -> (16w + l/4 + 8, 8i + 2(l%4) + 1)
struct WgmmaAccumCoord {
  int row;
  int col;
};

__host__ __device__ constexpr WgmmaAccumCoord wgmma_m64nNk16_accum_coord(int t, int reg) {
  const int warp = t / 32;
  const int lane = t % 32;
  const int chunk = reg / 4;
  const int within = reg % 4;
  return WgmmaAccumCoord{16 * warp + lane / 4 + 8 * (within / 2),
                         8 * chunk + 2 * (lane % 4) + (within % 2)};
}
