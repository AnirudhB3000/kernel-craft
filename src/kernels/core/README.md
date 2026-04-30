# core/ - Core Convolution Kernels

## Overview
Foundational 2D convolution implementations demonstrating progressive optimization from naive to tiled approaches.

## Files
- `conv_naive.cu` - Baseline convolution with one thread per output pixel, direct global memory access
- `conv_tiled.cu` - Optimized convolution using shared memory tiling to reduce redundant global memory access

## Key Concepts
- **Naive**: Simple but inefficient due to repeated global memory reads
- **Tiled**: Uses `__syncthreads()` and shared memory tiles with halo region handling

## Benchmark Results
Tiled 8×8 consistently performs best across image sizes 256×256 to 2048×2048 (see AGENTS.md for detailed benchmarks).
