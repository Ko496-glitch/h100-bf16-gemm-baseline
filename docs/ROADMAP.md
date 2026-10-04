# Hopper Optimization Roadmap

The hand-written kernels are still plain SIMT code (no Tensor Cores, WMMA/WGMMA, TMA, or pipelining). This intentionally simple kernel is a performance baseline for later shared-memory, TMA, WGMMA, and pipelining experiments.

Planned follow-on work, in roughly dependency order:

- **Hopper-TMA:** Replace thread-driven global-to-shared tile loads with Hopper Tensor Memory Accelerator (TMA) multidimensional asynchronous transfers using tensor/copy descriptors.
- **Hopper-WGMMA:** Replace scalar FP32 multiply/accumulate with Hopper wgmma.mma_async operations executed by 128-thread warpgroups. Keep BF16 inputs and FP32 accumulation.
- **Hopper-Pipeline:** Introduce double-buffered shared-memory staging so that while WGMMA consumes K tile i, TMA asynchronously loads K tile i+1. Later investigate triple buffering and tune pipeline depth.
- **Hopper-MBarrier:** Use Hopper asynchronous transaction barriers / mbarrier to coordinate TMA producer completion, WGMMA consumers, and safe stage reuse.
- **Hopper-SMEM-Layout:** Investigate shared-memory layouts/swizzling for WGMMA/TMA requirements and to minimize shared-memory bank conflicts.
- **Tile-Autotuning:** Maintain legal optimized tile configurations rather than arbitrary runtime tile sizes. Select based on M, N, K and hardware limits.
- **Register-Pressure:** Measure WGMMA accumulator register usage and tune tile shape, warpgroups, and pipeline stages to balance pressure and occupancy.
- **L2-Locality:** Investigate output-tile traversal/block scheduling that increases reuse of A or B panels in H100's 50 MB L2 cache.
- **L2-Persisting:** Experiment with CUDA L2 access-policy windows / persisting cache hints for selected reusable working sets. Do not persist entire large matrices; measure whether this helps.
- **Thread-Block-Clusters:** Investigate Hopper Thread Block Clusters for cooperative GEMM execution across multiple blocks/SMs.
- **DSMEM:** Investigate Distributed Shared Memory (DSMEM) so blocks in one cluster can share A or B panels instead of fetching identical data repeatedly.
- **Cluster-TMA:** Investigate cluster-level data movement / multicast-style reuse to reduce redundant tile traffic between cooperating blocks.
- **Dynamic-SMEM-Tuning:** Choose dynamic shared-memory allocation based on tile shape and pipeline stages while respecting H100 resource and occupancy limits.
- **CuBLASLt-Benchmark:** Benchmark optimized implementations against cuBLASLt.
- **CUTLASS-Benchmark:** Compare design and performance with CUTLASS 3.x Hopper GEMM kernels.
- **Nsight-Compute:** Profile each stage with Nsight Compute: Tensor Core use, DRAM throughput, L2 behavior, shared-memory throughput/bank conflicts, register pressure, occupancy, and warp stalls.

## Notes from the source

- Tiled kernel shared-memory sizing (`src/kernels/02_smem_tiled.cuh`): Future multi-stage pipelines will multiply this by their stage count.
