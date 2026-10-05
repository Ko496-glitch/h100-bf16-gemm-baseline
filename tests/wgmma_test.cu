// Standalone test for the WGMMA wrapper and shared-memory descriptor builder
// in src/common/wgmma.cuh.
//
// Build: make test-wgmma
// Run: ./test_wgmma
//
// Convention: D[m][n] = sum_k A[m][k] * Bt[n][k]. A is 64 x K and Bt is 64 x K,
// both row-major in global memory, i.e. both operands are K-major for wgmma.
// One block of 128 threads (one warpgroup) computes a 64x64 FP32 tile.
//
// Inputs are integers in [-4, 4], exact in BF16. Every partial sum is an
// integer of magnitude at most 64 * 16 = 1024, exact in FP32, so the result
// is bit-identical to the CPU reference in any summation order.
//
// Stages 1-3 fill shared memory with ordinary stores using the 128B swizzle
// formula. Stage 4 fills it with TMA (the PR 3 helpers), when present. If the
// hand-filled stages pass and Stage 4 fails, the problem is the TMA/descriptor
// interaction; if both fail, it is the descriptor or the wgmma wrapper.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "common/utils.cuh"
#include "common/wgmma.cuh"

#if __has_include("common/tma.cuh") && __has_include("common/barrier.cuh")
#define WGMMA_TEST_HAVE_TMA 1
#include "common/barrier.cuh"
#include "common/tma.cuh"
#else
#define WGMMA_TEST_HAVE_TMA 0
#endif

constexpr int kTileM = 64;   // rows of A, and of D
constexpr int kTileN = 64;   // rows of Bt, columns of D
constexpr int kTileK = 64;   // K extent stored per tile (128 bytes per row)
constexpr int kRowBytes = kTileK * sizeof(__nv_bfloat16);
constexpr int kTileElements = kTileM * kTileK;
constexpr uint32_t kTileBytes = kTileElements * sizeof(__nv_bfloat16);
constexpr int kSmemAlignment = 1024;
constexpr int kSmemBytes = 2 * kTileBytes + kSmemAlignment;
constexpr int kThreads = 128;
constexpr int kWgmmaK = 16;
constexpr uint32_t kKStepBytes = kWgmmaK * sizeof(__nv_bfloat16);  // 32 bytes per k16 step
constexpr float kAccumulatorPrefill = 12345.0f;  // must be discarded by scale-d = 0

static_assert(kTileM == kTileN && kTileM == kTileK, "tiles are 64x64");

// Descriptor bit-packing, worked out by hand against PTX ISA 9.4, Matrix
// Descriptor Format, sec. 9.7.17.5.1.2.2:
//   start (0x400 & 0x3FFFF) >> 4 = 0x40          bits  0-13
//   LBO   16 >> 4 = 1            -> 0x10000      bits 16-29
//   SBO   1024 >> 4 = 0x40       -> 0x40 << 32   bits 32-45
//   swizzle 128B = 1             -> 1 << 62      bits 62-63
static_assert(make_wgmma_desc(0x0400, 16, 1024, WgmmaSwizzle::B128) == 0x4000004000010040ull,
              "descriptor packing");
static_assert(make_wgmma_desc(0x40400, 16, 1024, WgmmaSwizzle::B128) == 0x4000004000010040ull,
              "start address is masked with 0x3FFFF before the shift");
static_assert(make_wgmma_desc(0x0400, 16, 1024, WgmmaSwizzle::None) == 0x0000004000010040ull,
              "swizzle none = 0");
static_assert(wgmma_desc_advance(0x4000004000010040ull, 1 * kKStepBytes) == 0x4000004000010042ull,
              "k16 step 1 adds 2 to the start field");
static_assert(wgmma_desc_advance(0x4000004000010040ull, 2 * kKStepBytes) == 0x4000004000010044ull,
              "k16 step 2");
static_assert(wgmma_desc_advance(0x4000004000010040ull, 3 * kKStepBytes) == 0x4000004000010046ull,
              "k16 step 3");

// Accumulator mapping spot checks (PTX ISA 9.4, Matrix Fragments for
// wgmma.mma_async.m64nNk16, sec. 9.7.17.5.1.1.1, Figure 152).
static_assert(wgmma_m64nNk16_accum_coord(0, 0).row == 0 && wgmma_m64nNk16_accum_coord(0, 0).col == 0, "T0 d0");
static_assert(wgmma_m64nNk16_accum_coord(0, 1).row == 0 && wgmma_m64nNk16_accum_coord(0, 1).col == 1, "T0 d1");
static_assert(wgmma_m64nNk16_accum_coord(0, 2).row == 8 && wgmma_m64nNk16_accum_coord(0, 2).col == 0, "T0 d2");
static_assert(wgmma_m64nNk16_accum_coord(0, 3).row == 8 && wgmma_m64nNk16_accum_coord(0, 3).col == 1, "T0 d3");
static_assert(wgmma_m64nNk16_accum_coord(0, 4).row == 0 && wgmma_m64nNk16_accum_coord(0, 4).col == 8, "T0 d4");
static_assert(wgmma_m64nNk16_accum_coord(5, 0).row == 1 && wgmma_m64nNk16_accum_coord(5, 0).col == 2, "T5 d0");
static_assert(wgmma_m64nNk16_accum_coord(32, 0).row == 16 && wgmma_m64nNk16_accum_coord(32, 0).col == 0, "T32 d0");
static_assert(wgmma_m64nNk16_accum_coord(127, 31).row == 63 && wgmma_m64nNk16_accum_coord(127, 31).col == 63, "T127 d31");

// 128B swizzle relative to a 1024-byte-aligned base: the 16-byte chunk index
// (bits 4-6) is XORed with the 128-byte row index (bits 7-9).
// PTX ISA 9.4, Swizzling Modes, sec. 5.5.7
__host__ __device__ constexpr uint32_t swizzle_128b(uint32_t offset) {
  return offset ^ ((offset >> 3) & 0x70);
}

// Dynamic shared memory base alignment is not guaranteed; align the tiles by
// hand so the 128B swizzle pattern starts on a 1024-byte boundary and the
// descriptor base offset is 0.
__device__ unsigned char* aligned_tiles(unsigned char* smem_raw) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
  return smem_raw + (kSmemAlignment - addr % kSmemAlignment) % kSmemAlignment;
}

// Runs KSteps chained m64n64k16 instructions on the tiles and writes D to
// out[row * 64 + col]. scale-d is 0 on the first instruction, so the prefill
// value must not appear in the result.
template <int KSteps>
__device__ void wgmma_compute_and_store(const unsigned char* tile_a,
                                        const unsigned char* tile_bt,
                                        float* out) {
  float d[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    d[i] = kAccumulatorPrefill;
  }

  const uint64_t desc_a = make_wgmma_desc_k_sw128(tile_a);
  const uint64_t desc_bt = make_wgmma_desc_k_sw128(tile_bt);

  wgmma_fence_accumulators(d);
  wgmma_fence();
#pragma unroll
  for (int k = 0; k < KSteps; ++k) {
    wgmma_m64n64k16_bf16_f32(d,
                             wgmma_desc_advance(desc_a, k * kKStepBytes),
                             wgmma_desc_advance(desc_bt, k * kKStepBytes),
                             k == 0 ? 0 : 1);
  }
  wgmma_commit_group();
  wgmma_wait_group<0>();
  wgmma_fence_accumulators(d);

#pragma unroll
  for (int i = 0; i < 32; ++i) {
    const WgmmaAccumCoord c = wgmma_m64nNk16_accum_coord(threadIdx.x, i);
    out[c.row * kTileN + c.col] = d[i];
  }
}

// Fill path 1: ordinary stores at the swizzled byte offsets, then an async
// proxy fence so wgmma sees them.
template <int KSteps>
__global__ void wgmma_smem_fill_kernel(const __nv_bfloat16* a,
                                       const __nv_bfloat16* bt,
                                       float* out) {
  extern __shared__ unsigned char smem_raw[];
  unsigned char* tile_a = aligned_tiles(smem_raw);
  unsigned char* tile_bt = tile_a + kTileBytes;

  for (int e = threadIdx.x; e < kTileElements; e += blockDim.x) {
    const int row = e / kTileK;
    const int k = e % kTileK;
    const uint32_t physical = swizzle_128b(row * kRowBytes + k * sizeof(__nv_bfloat16));
    *reinterpret_cast<__nv_bfloat16*>(tile_a + physical) = a[e];
    *reinterpret_cast<__nv_bfloat16*>(tile_bt + physical) = bt[e];
  }
  wgmma_fence_smem_writes();
  __syncthreads();

  wgmma_compute_and_store<KSteps>(tile_a, tile_bt, out);
}

#if WGMMA_TEST_HAVE_TMA
// Fill path 2: one TMA load per tile with SWIZZLE_128B, completed on one
// mbarrier. TMA writes through the async proxy, so no proxy fence is needed
// before wgmma.
template <int KSteps>
__global__ void wgmma_tma_fill_kernel(const __grid_constant__ CUtensorMap map_a,
                                      const __grid_constant__ CUtensorMap map_bt,
                                      float* out) {
  extern __shared__ unsigned char smem_raw[];
  unsigned char* tile_a = aligned_tiles(smem_raw);
  unsigned char* tile_bt = tile_a + kTileBytes;
  __shared__ uint64_t bar;

  if (threadIdx.x == 0) {
    mbarrier_init(&bar);
    fence_proxy_async_shared_cta();
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    mbarrier_arrive_expect_tx(&bar, 2 * kTileBytes);
    tma_load_2d(tile_a, &map_a, 0, 0, &bar);
    tma_load_2d(tile_bt, &map_bt, 0, 0, &bar);
  }
  mbarrier_wait(&bar, 0);
  // Thread 0 took a different path above; reconverge every warp before the
  // .aligned wgmma instructions, which all threads must execute together.
  __syncthreads();

  wgmma_compute_and_store<KSteps>(tile_a, tile_bt, out);
}
#endif

static __nv_bfloat16 to_bf16(int value) {
  return __float2bfloat16(static_cast<float>(value));
}

static std::vector<float> cpu_gemm(const std::vector<int>& a, const std::vector<int>& bt, int k_extent) {
  std::vector<float> d(kTileM * kTileN, 0.0f);
  for (int m = 0; m < kTileM; ++m) {
    for (int n = 0; n < kTileN; ++n) {
      int acc = 0;
      for (int k = 0; k < k_extent; ++k) {
        acc += a[m * kTileK + k] * bt[n * kTileK + k];
      }
      d[m * kTileN + n] = static_cast<float>(acc);
    }
  }
  return d;
}

static bool compare_d(const char* stage, const std::vector<float>& got, const std::vector<float>& want) {
  int mismatches = 0;
  double max_abs_error = 0.0;
  for (int i = 0; i < kTileM * kTileN; ++i) {
    uint32_t got_bits;
    uint32_t want_bits;
    std::memcpy(&got_bits, &got[i], sizeof(got_bits));
    std::memcpy(&want_bits, &want[i], sizeof(want_bits));
    if (got_bits != want_bits) {
      if (mismatches < 8) {
        std::printf("%s: mismatch at (row %d, col %d): expected %.1f got %.9g\n",
                    stage, i / kTileN, i % kTileN, want[i], got[i]);
      }
      ++mismatches;
      const double err = std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
      max_abs_error = std::isnan(err) || err > max_abs_error ? err : max_abs_error;
    }
  }
  if (mismatches > 0) {
    std::printf("%s: %d of %d elements mismatch, max abs error %.9g\n",
                stage, mismatches, kTileM * kTileN, max_abs_error);
  }
  return mismatches == 0;
}

static bool report(const char* stage, bool pass) {
  std::printf("%s: %s\n", stage, pass ? "PASS" : "FAIL");
  std::fflush(stdout);
  return pass;
}

template <typename Launch>
static std::vector<float> run_stage(float* d_out, Launch launch) {
  // 0xFF bytes are NaN, so any element the kernel fails to write mismatches.
  CUDA_CHECK(cudaMemset(d_out, 0xFF, kTileM * kTileN * sizeof(float)));
  launch();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> out(kTileM * kTileN);
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
  return out;
}

static void upload(__nv_bfloat16* d_dst, const std::vector<int>& values) {
  std::vector<__nv_bfloat16> host(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    host[i] = to_bf16(values[i]);
  }
  CUDA_CHECK(cudaMemcpy(d_dst, host.data(), host.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
}

int main() {
  // The mapping must cover every element of the 64x64 tile exactly once.
  {
    std::vector<int> hits(kTileM * kTileN, 0);
    for (int t = 0; t < kThreads; ++t) {
      for (int reg = 0; reg < 32; ++reg) {
        const WgmmaAccumCoord c = wgmma_m64nNk16_accum_coord(t, reg);
        ++hits[c.row * kTileN + c.col];
      }
    }
    bool pass = true;
    for (int h : hits) {
      pass &= h == 1;
    }
    if (!report("Accumulator mapping covers 64x64 once (host)", pass)) {
      return EXIT_FAILURE;
    }
  }

  constexpr unsigned kRandomSeed = 42;
  std::mt19937 rng(kRandomSeed);
  std::uniform_int_distribution<int> small_int(-4, 4);
  std::vector<int> a_random(kTileElements);
  std::vector<int> bt_random(kTileElements);
  for (int& v : a_random) v = small_int(rng);
  for (int& v : bt_random) v = small_int(rng);

  // Selection matrix: A[m][k] = 1 if k == m % 16 else 0, so D[m][n] = Bt[n][m % 16]
  // and rows 0-15, 16-31, 32-47, 48-63 of D repeat the same pattern.
  std::vector<int> a_select(kTileElements, 0);
  for (int m = 0; m < kTileM; ++m) {
    a_select[m * kTileK + m % kWgmmaK] = 1;
  }

  __nv_bfloat16* d_a = nullptr;
  __nv_bfloat16* d_bt = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), kTileBytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bt), kTileBytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out), kTileM * kTileN * sizeof(float)));
  upload(d_bt, bt_random);

  bool all_pass = true;

  // Stage 1: selection matrix, one m64n64k16.
  {
    upload(d_a, a_select);
    std::vector<float> want(kTileM * kTileN);
    for (int m = 0; m < kTileM; ++m) {
      for (int n = 0; n < kTileN; ++n) {
        want[m * kTileN + n] = static_cast<float>(bt_random[n * kTileK + m % kWgmmaK]);
      }
    }
    const std::vector<float> got = run_stage(d_out, [&]() {
      wgmma_smem_fill_kernel<1><<<1, kThreads, kSmemBytes>>>(d_a, d_bt, d_out);
    });
    all_pass &= report("Stage 1 (selection A, k16, thread-filled smem)",
                       compare_d("Stage 1", got, want));
  }

  // Stage 2: random A and Bt, one m64n64k16 (k < 16 only).
  upload(d_a, a_random);
  {
    const std::vector<float> got = run_stage(d_out, [&]() {
      wgmma_smem_fill_kernel<1><<<1, kThreads, kSmemBytes>>>(d_a, d_bt, d_out);
    });
    all_pass &= report("Stage 2 (random, k16, thread-filled smem)",
                       compare_d("Stage 2", got, cpu_gemm(a_random, bt_random, kWgmmaK)));
  }

  // Stage 3: four chained m64n64k16 (BK = 64), descriptors advanced 32 bytes
  // per k16 step, scale-d 0 then 1, one commit and wait.
  const std::vector<float> want_k64 = cpu_gemm(a_random, bt_random, kTileK);
  {
    const std::vector<float> got = run_stage(d_out, [&]() {
      wgmma_smem_fill_kernel<4><<<1, kThreads, kSmemBytes>>>(d_a, d_bt, d_out);
    });
    all_pass &= report("Stage 3 (random, k64 chain, thread-filled smem)",
                       compare_d("Stage 3", got, want_k64));
  }

  // Stage 4: same as Stage 3 with tiles loaded by TMA (SWIZZLE_128B, 64x64 box).
#if WGMMA_TEST_HAVE_TMA
  {
    CUtensorMap map_a;
    CUtensorMap map_bt;
    make_tma_2d_bf16(&map_a, d_a, kTileM, kTileK, kTileM, kTileK, CU_TENSOR_MAP_SWIZZLE_128B);
    make_tma_2d_bf16(&map_bt, d_bt, kTileN, kTileK, kTileN, kTileK, CU_TENSOR_MAP_SWIZZLE_128B);
    const std::vector<float> got = run_stage(d_out, [&]() {
      wgmma_tma_fill_kernel<4><<<1, kThreads, kSmemBytes>>>(map_a, map_bt, d_out);
    });
    all_pass &= report("Stage 4 (random, k64 chain, TMA-filled smem)",
                       compare_d("Stage 4", got, want_k64));
  }
#else
  std::printf("Stage 4 (random, k64 chain, TMA-filled smem): SKIPPED (PR 3 helpers not present)\n");
  std::fflush(stdout);
#endif

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_bt));
  CUDA_CHECK(cudaFree(d_out));

  std::printf("Overall: %s\n", all_pass ? "PASS" : "FAIL");
  return all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
