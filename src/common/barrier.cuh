#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>


#ifdef MBARRIER_DEBUG_TIMEOUT
constexpr unsigned long long kMbarrierDebugMaxPolls = 4000000ull;
#endif

__device__ __forceinline__ uint32_t shared_address(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void mbarrier_init(uint64_t* bar) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :
               : "r"(shared_address(bar)), "r"(1u)
               : "memory");
}


__device__ __forceinline__ void fence_proxy_async_shared_cta() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :
               : "r"(shared_address(bar)), "r"(bytes)
               : "memory");
}


__device__ __forceinline__ bool mbarrier_try_wait(uint64_t* bar, uint32_t parity) {
  uint32_t complete;
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
      "  selp.u32 %0, 1, 0, p;\n"
      "}\n"
      : "=r"(complete)
      : "r"(shared_address(bar)), "r"(parity)
      : "memory");
  return complete != 0;
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t parity) {
#ifdef MBARRIER_DEBUG_TIMEOUT
  unsigned long long polls = 0;
#endif
  while (!mbarrier_try_wait(bar, parity)) {
#ifdef MBARRIER_DEBUG_TIMEOUT
    if (++polls == kMbarrierDebugMaxPolls) {
      printf("mbarrier_wait timeout: block (%u,%u,%u) thread (%u,%u,%u) "
             "barrier smem 0x%x phase parity %u\n",
             blockIdx.x, blockIdx.y, blockIdx.z,
             threadIdx.x, threadIdx.y, threadIdx.z,
             shared_address(bar), parity);
      __trap();
    }
#endif
  }
}
