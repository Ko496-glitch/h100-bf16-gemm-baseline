# H100 BF16 GEMM Baseline

This single-file CUDA project benchmarks two hand-written row-major GEMM kernels for BF16 inputs and FP32 accumulation/output against a cuBLAS reference:

0. **cuBLAS reference:** a `cublasGemmEx` call (BF16 in, FP32 out, `CUBLAS_COMPUTE_32F`) used both as the performance baseline and, for large sizes, as the correctness reference.
1. **Naive baseline:** one thread computes one output element and walks K, reading A and B from global memory.
2. **Shared-memory tiled:** a block computes a 32x32 output tile with BK=32. Threads cooperatively stage BF16 A/B tiles in dynamic shared memory; each thread accumulates a 2x2 output patch in FP32. This remains a SIMT kernel and does not use Tensor Cores, WMMA/WGMMA, TMA, or pipelining.

## Build

On a machine with the CUDA Toolkit, cuBLAS, and an NVIDIA Hopper GPU:

```bash
make
```

This builds with `nvcc -O3 -std=c++17 -arch=sm_90a -lineinfo`, linking `-lcublas -lcuda`. Use `make run` to build (if needed) and run with default dimensions, and `make clean` to remove the binary.

## Run

Run the default 4096x4096x4096 case, or pass dimensions in M N K order:

```bash
./gemm_baseline
./gemm_baseline 1024 768 512
```

Inputs are standard normal (N(0,1)) values with a fixed seed, rounded to BF16. For M, N, K all <= 1024 the reference is a CPU FP32 computation; above that size the CPU triple loop is too slow to use, so cuBLAS's own output (itself checked against the CPU reference at smaller sizes) stands in as the reference. Each GPU kernel, including cuBLAS, gets a warm-up launch, then five timed samples of ten launches each. CUDA events time the kernels; host/device copies are outside the timed regions. The program reports runtime, achieved GFLOP/s, each hand-written kernel's GFLOP/s as a percentage of cuBLAS's, normwise relative error (`||C - Cref|| / ||Cref||`), and correctness. The error tolerance is a single constant (`kErrorTolerance`, starting at 1e-2) near the top of the file.

The tiled kernel's dynamic shared-memory allocation is derived from sizeof(BF16) * (BM*BK + BK*BN) and passed at launch. For the initial 32x32x32 tile this is 4 KiB per block. Increasing tile sizes or pipeline stages can reduce block residency, so shared-memory capacity is not treated as a target.

## Compiler Explorer

You can try the program on [Compiler Explorer's CUDA instance](https://nvcc.godbolt.org/) with NVCC options -O3 -std=c++17 -arch=sm_90a, linking -lcublas -lcuda. The remote execution GPU may not be an H100, so use an H100 for representative H100 performance measurements.
