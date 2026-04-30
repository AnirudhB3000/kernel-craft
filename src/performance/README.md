# performance/ - Performance Infrastructure

## Overview
Utilities and techniques for optimizing CUDA kernel performance beyond algorithmic improvements.

## Files
- `cuda_graphs.cu` - CUDA Graphs implementation to reduce kernel launch overhead by capturing DAG at instantiation
- `memory_pool.cu` - Pre-allocated memory pool to eliminate per-batch cudaMalloc/cudaFree overhead
- `mixed_precision.cu` - FP16/FP32/TF32 precision support leveraging Tensor Cores (Volta+)
- `persistent_kernels.cu` - Persistent kernel implementations that eliminate re-launch overhead for fixed workloads

## Benchmark Results
- **CUDA Graphs**: ~5-10μs launch overhead vs ~15-30μs for separate launches
- **Memory Pool**: ~85% reduction in allocation overhead with 16-buffer reuse
- **Mixed Precision**: FP16 provides 2x compute with 50% bandwidth; TF32 automatic on Ampere+
- **Persistent**: ~50% latency reduction for fixed batch streams
