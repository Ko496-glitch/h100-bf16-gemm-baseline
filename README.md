# H100 BF16 GEMM Baseline

A deliberately naive CUDA matrix multiplication baseline for NVIDIA Hopper. It computes **C = A × B** with row-major BF16 inputs and FP32 accumulation/output.

Each CUDA thread computes one output element and loops over the full reduction dimension. The kernel uses a 16 × 16 block, bounds checks, and no shared memory, tiling, Tensor Cores, WMMA/WGMMA, TMA, or CUDA Graphs. Its repeated global-memory loads make it intentionally inefficient; that simplicity provides a reference point for later optimizations.

## Build

On a system with the CUDA Toolkit and an NVIDIA Hopper GPU:

```bash
nvcc -O2 -std=c++17 -arch=sm_90 h100_bf16_gemm_baseline.cu -o gemm_baseline
```

## Run

Run the default 512 × 512 × 512 case:

```bash
./gemm_baseline
```

Or pass dimensions in **M N K** order:

```bash
./gemm_baseline 1024 768 512
```

The program initializes deterministic BF16 inputs, runs the kernel, and compares the output with a CPU FP32 reference. It prints the dimensions, CUDA-event kernel time, achieved GFLOP/s, maximum absolute error, and pass/fail status.

## Compiler Explorer

You can try compiling and running the file on [Compiler Explorer's CUDA instance](https://nvcc.godbolt.org/). Use NVCC options `-O2 -std=c++17 -arch=sm_90`. The remote execution GPU may not be an H100, so use an H100 for representative H100 performance measurements.
