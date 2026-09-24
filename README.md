# H100 BF16 GEMM Baseline

This single-file CUDA project compares two row-major GEMM implementations for BF16 inputs and FP32 accumulation/output:

1. **Naive baseline:** one thread computes one output element and walks K, reading A and B from global memory.
2. **Shared-memory tiled:** a block computes a 32x32 output tile with BK=32. Threads cooperatively stage BF16 A/B tiles in dynamic shared memory; each thread accumulates a 2x2 output patch in FP32. This remains a SIMT kernel and does not use Tensor Cores, WMMA/WGMMA, TMA, or pipelining.

## Build

On a machine with the CUDA Toolkit and an NVIDIA Hopper GPU:

```bash
nvcc -O3 -std=c++17 -arch=sm_90 h100_bf16_gemm_baseline.cu -o gemm_baseline
```

## Run

Run the default 512x512x512 case, or pass dimensions in M N K order:

```bash
./gemm_baseline
./gemm_baseline 1024 768 512
```

The program initializes deterministic BF16 inputs and computes a CPU FP32 reference. Each GPU kernel gets a warm-up launch, then five timed samples of ten launches each. CUDA events time the kernels; host/device copies are outside the timed regions. The program reports runtime, achieved GFLOP/s, max absolute error, and correctness for both kernels. Correctness uses atol=1e-3 and rtol=1e-3.

The tiled kernel's dynamic shared-memory allocation is derived from sizeof(BF16) * (BM*BK + BK*BN) and passed at launch. For the initial 32x32x32 tile this is 4 KiB per block. Increasing tile sizes or pipeline stages can reduce block residency, so shared-memory capacity is not treated as a target.

## Compiler Explorer

You can try the program on [Compiler Explorer's CUDA instance](https://nvcc.godbolt.org/) with NVCC options -O3 -std=c++17 -arch=sm_90. The remote execution GPU may not be an H100, so use an H100 for representative H100 performance measurements.
