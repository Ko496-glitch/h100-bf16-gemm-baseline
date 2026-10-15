// Standalone test for the TMA load and mbarrier building blocks in
// src/common/tma.cuh and src/common/barrier.cuh.
//
// Build: make test-tma
// Run: ./test_tma
//
// One block loads 64x64 BF16 tiles of a 256x256 matrix with TMA into shared
// memory, waits on an mbarrier, and copies the raw shared-memory bytes out to
// global memory. The host checks the bytes against the source matrix.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "common/barrier.cuh"
#include "common/tma.cuh"
#include "common/utils.cuh"

constexpr int kRows = 256;
constexpr int kCols = 256;
constexpr int kTile = 64;
constexpr int kTileElements = kTile * kTile;
constexpr uint32_t kTileBytes = kTileElements * sizeof(__nv_bfloat16);
constexpr int kSmemAlignment = 1024;
constexpr int kSmemBytes = kTileBytes + kSmemAlignment;
constexpr int kThreads = 128;
constexpr int kMaxLoads = 8;

struct TileList {
  int count;
  int col[kMaxLoads];
  int row[kMaxLoads];
};

__global__ void tma_copy_kernel(const __grid_constant__ CUtensorMap map,
                                TileList tiles,
                                uint16_t* out) {
  // Dynamic shared memory base alignment is not guaranteed, so align the tile
  // by hand. The TMA docs require 128 bytes; 1024 is the safe choice for the
  // 128B swizzle pattern, which repeats every 1024 bytes.
  extern __shared__ unsigned char smem_raw[];
  const uint32_t pad = (kSmemAlignment - (shared_address(smem_raw) % kSmemAlignment)) % kSmemAlignment;
  unsigned char* tile = smem_raw + pad;

  __shared__ uint64_t bar;  // mbarrier objects need 8-byte alignment

  if (threadIdx.x == 0) {
    mbarrier_init(&bar);
    fence_proxy_async_shared_cta();
  }
  __syncthreads();

  uint32_t phase = 0;
  for (int i = 0; i < tiles.count; ++i) {
    if (threadIdx.x == 0) {
      mbarrier_arrive_expect_tx(&bar, kTileBytes);
      tma_load_2d(tile, &map, tiles.col[i], tiles.row[i], &bar);
    }
    mbarrier_wait(&bar, phase);
    phase ^= 1;

    // Raw shared-memory bytes, i.e. still swizzled for a 128B-swizzle map.
    const uint16_t* tile_elements = reinterpret_cast<const uint16_t*>(tile);
    for (int e = threadIdx.x; e < kTileElements; e += blockDim.x) {
      out[static_cast<size_t>(i) * kTileElements + e] = tile_elements[e];
    }

    // All threads finish reading the tile before the next TMA overwrites it.
    __syncthreads();
    if (threadIdx.x == 0) {
      // Test-only and conservative: orders the generic-proxy reads above
      // before the next async-proxy (TMA) write to the same buffer. Later
      // kernels release buffers through empty barriers instead.
      fence_proxy_async_shared_cta();
    }
  }
}

static uint16_t bits(__nv_bfloat16 value) {
  uint16_t b;
  std::memcpy(&b, &value, sizeof(b));
  return b;
}

// Source tile with top-left corner (row0, col0); elements outside the matrix
// are 0x0000, matching the tensor map's zero out-of-bounds fill.
static std::vector<uint16_t> expected_tile(const std::vector<uint16_t>& src, int row0, int col0) {
  std::vector<uint16_t> tile(kTileElements, 0);
  for (int r = 0; r < kTile; ++r) {
    for (int c = 0; c < kTile; ++c) {
      const int row = row0 + r;
      const int col = col0 + c;
      if (row < kRows && col < kCols) {
        tile[r * kTile + c] = src[static_cast<size_t>(row) * kCols + col];
      }
    }
  }
  return tile;
}

// 128B swizzle (PTX ISA 5.5.7, 16-byte atomicity): within each 1024-byte
// block, the 16-byte chunk index (bits 4-6) is XORed with the 128-byte row
// index (bits 7-9). The mapping is its own inverse.
static uint32_t swizzle_128b(uint32_t offset) {
  return offset ^ ((offset >> 3) & 0x70);
}

static std::vector<uint16_t> run_loads(const CUtensorMap& map, const TileList& tiles,
                                       uint16_t* d_out) {
  tma_copy_kernel<<<1, kThreads, kSmemBytes>>>(map, tiles, d_out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> out(static_cast<size_t>(tiles.count) * kTileElements);
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, out.size() * sizeof(uint16_t),
                        cudaMemcpyDeviceToHost));
  return out;
}

static bool compare_tile(const char* stage, int load, const uint16_t* got,
                         const std::vector<uint16_t>& want) {
  for (int e = 0; e < kTileElements; ++e) {
    if (got[e] != want[e]) {
      std::printf("%s: load %d mismatch at row %d col %d: got 0x%04x expected 0x%04x\n",
                  stage, load, e / kTile, e % kTile, got[e], want[e]);
      return false;
    }
  }
  return true;
}

static bool report(const char* stage, bool pass) {
  std::printf("%s: %s\n", stage, pass ? "PASS" : "FAIL");
  std::fflush(stdout);
  return pass;
}

int main() {
  std::vector<uint16_t> h_src(static_cast<size_t>(kRows) * kCols);
  constexpr unsigned kRandomSeed = 42;
  std::mt19937 rng(kRandomSeed);
  std::normal_distribution<float> standard_normal(0.0f, 1.0f);
  for (size_t i = 0; i < h_src.size(); ++i) {
    h_src[i] = bits(__float2bfloat16(standard_normal(rng)));
  }

  __nv_bfloat16* d_src = nullptr;
  uint16_t* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_src), h_src.size() * sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out),
                        static_cast<size_t>(kMaxLoads) * kTileElements * sizeof(uint16_t)));
  CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), h_src.size() * sizeof(uint16_t),
                        cudaMemcpyHostToDevice));

  CUtensorMap map_none;
  CUtensorMap map_128b;
  make_tma_2d_bf16(&map_none, d_src, kRows, kCols, kTile, kTile, CU_TENSOR_MAP_SWIZZLE_NONE);
  // 64 BF16 columns = 128 bytes, exactly the 128B swizzle span.
  make_tma_2d_bf16(&map_128b, d_src, kRows, kCols, kTile, kTile, CU_TENSOR_MAP_SWIZZLE_128B);

  bool all_pass = true;

  // Stage A: one load at row 64, column 128. Swapped coordinates would read
  // rows 128.. and columns 64.. instead.
  {
    const TileList tiles = {1, {128}, {64}};
    const std::vector<uint16_t> out = run_loads(map_none, tiles, d_out);
    const bool pass = compare_tile("Stage A", 0, out.data(), expected_tile(h_src, 64, 128));
    all_pass &= report("Stage A (SWIZZLE_NONE, row 64 col 128)", pass);
  }

  // Phase test: six different tiles through one barrier, parity 0,1,0,1,0,1.
  // A wait that passes on a stale phase would show up as a repeated tile.
  {
    const TileList tiles = {6, {0, 64, 128, 0, 192, 64}, {0, 0, 64, 192, 128, 64}};
    const std::vector<uint16_t> out = run_loads(map_none, tiles, d_out);
    bool pass = true;
    for (int i = 0; i < tiles.count; ++i) {
      pass &= compare_tile("Phase test", i, out.data() + static_cast<size_t>(i) * kTileElements,
                           expected_tile(h_src, tiles.row[i], tiles.col[i]));
    }
    all_pass &= report("Phase test (6 loads, one barrier)", pass);
  }

  // Stage B: 128B swizzle, two loads (also exercises a second phase).
  {
    const TileList tiles = {2, {128, 0}, {64, 192}};
    const std::vector<uint16_t> out = run_loads(map_128b, tiles, d_out);
    bool swizzled_pass = true;
    bool unswizzled_pass = true;
    for (int i = 0; i < tiles.count; ++i) {
      const std::vector<uint16_t> want = expected_tile(h_src, tiles.row[i], tiles.col[i]);
      std::vector<unsigned char> got_bytes(kTileBytes);
      std::vector<unsigned char> want_bytes(kTileBytes);
      std::memcpy(got_bytes.data(), out.data() + static_cast<size_t>(i) * kTileElements, kTileBytes);
      std::memcpy(want_bytes.data(), want.data(), kTileBytes);

      // Swizzled layout: logical byte L of the source tile sits at physical
      // byte swizzle_128b(L) in shared memory.
      for (uint32_t logical = 0; logical < kTileBytes; ++logical) {
        const uint32_t physical = swizzle_128b(logical);
        if (got_bytes[physical] != want_bytes[logical]) {
          std::printf("Stage B: load %d swizzled mismatch at logical byte %u (physical %u): "
                      "got 0x%02x expected 0x%02x\n",
                      i, logical, physical, got_bytes[physical], want_bytes[logical]);
          swizzled_pass = false;
          break;
        }
      }

      // Un-swizzle the whole output, then compare it to the source tile.
      std::vector<unsigned char> unswizzled(kTileBytes);
      for (uint32_t physical = 0; physical < kTileBytes; ++physical) {
        unswizzled[swizzle_128b(physical)] = got_bytes[physical];
      }
      if (std::memcmp(unswizzled.data(), want_bytes.data(), kTileBytes) != 0) {
        std::printf("Stage B: load %d un-swizzled output does not match the source tile\n", i);
        unswizzled_pass = false;
      }
    }
    all_pass &= report("Stage B (SWIZZLE_128B, swizzled layout)", swizzled_pass);
    all_pass &= report("Stage B (SWIZZLE_128B, un-swizzled == source)", unswizzled_pass);
  }

  // Stage C (last, since a wrong byte count would hang here): the box at
  // row 224, col 224 extends 32 elements past both matrix edges. In-bounds
  // elements must match; out-of-bounds elements must be zero.
  // Unverified: this expects complete_tx to count the full box (8192 bytes,
  // including the zero-filled part). The PTX ISA only says "amount of data
  // copied". If that is wrong this stage hangs instead of failing.
  {
    const TileList tiles = {1, {224}, {224}};
    const std::vector<uint16_t> out = run_loads(map_none, tiles, d_out);
    const bool pass = compare_tile("Stage C", 0, out.data(), expected_tile(h_src, 224, 224));
    all_pass &= report("Stage C (box past matrix edge, zero fill)", pass);
  }

  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFree(d_out));

  std::printf("Overall: %s\n", all_pass ? "PASS" : "FAIL");
  return all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
