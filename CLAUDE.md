# CLAUDE.md

## Project: kernel-craft - CUDA for Machine Learning Systems Optimization

This document defines the goals, scope, and execution plan for developing deep expertise in GPU programming for ML workloads, with a focus on CUDA-based optimization.

---

## Development Guidelines

- All kernel and library logic must be covered by unit/integration tests in the `tests/` directory.
- When any logic is added or modified, corresponding tests must be added or updated.
- Design decisions, whether high‑level architecture or low‑level implementation choices, must be documented in this CLAUDE.md file.
- Updates to logic that affect interfaces or behavior require an entry in CLAUDE.md describing the change.
- The naive convolution implementation (`src/conv_naive.cu`) is verified by `tests/test_conv_naive.cpp`.

# 1. Core Objectives

## 1.1 Foundational Understanding

Develop a systems-level understanding of:

* How convolution maps to GPU execution
* Thread hierarchy (grid, block, warp)
* Memory hierarchy:

  * Global memory
  * Shared memory
  * Registers
* Memory bandwidth vs compute bottlenecks

### Key Outcomes

* Ability to reason about kernel performance
* Identify memory-bound vs compute-bound workloads
* Understand GPU occupancy and utilization

---

# 2. Technical Focus Areas

## 2.1 Convolution Mechanics on GPU

### Goals

* Implement naive convolution kernel
* Optimize using shared memory tiling
* Reduce redundant global memory access

### Tasks

1. Implement baseline 2D convolution:

   * One thread per output pixel
   * Direct global memory access

2. Optimize with tiling:

   * Load input tiles into shared memory
   * Handle halo regions
   * Synchronize threads using `__syncthreads()`

3. Analyze:

   * Memory access patterns
   * Coalescing efficiency
   * Warp divergence

### Deliverables

* `conv_naive.cu`
* `conv_tiled.cu`
* Benchmark comparison

---

## 2.2 Memory Optimization

### Goals

* Minimize global memory traffic
* Maximize shared memory reuse

### Techniques

* Coalesced memory access
* Shared memory tiling
* Register usage optimization

### Experiments

* Compare:

  * Global-only vs shared memory kernels
  * Different tile sizes

* Measure:

  * Bandwidth utilization
  * Execution time

---

## 2.3 Custom Operations

### Motivation

Standard libraries do not support all operations efficiently.

### Targets

* Sparse convolution
* Custom filter kernels
* Domain-specific transforms

### Tasks

* Define a non-standard convolution variant
* Implement CUDA kernel
* Compare against dense fallback

### Deliverables

* `custom_op.cu`
* Performance report

---

## 2.4 Fused Kernels

### Motivation

Reduce kernel launch overhead and memory transfers.

### Target Pipeline

Separate implementation:

```
conv -> batchnorm -> relu
```

Fused implementation:

```
single CUDA kernel performing all steps
```

### Tasks

1. Implement individual kernels
2. Implement fused kernel
3. Compare:

   * Kernel launch overhead
   * Memory traffic
   * Latency

### Key Insight

Memory movement dominates cost more than arithmetic.

### Deliverables

* `pipeline_separate.cu`
* `pipeline_fused.cu`
* Benchmark report

---

## 2.5 Small / Irregular Workloads

### Motivation

Highly optimized libraries underperform on:

* Small tensors
* Irregular shapes
* Non-uniform workloads

### Tasks

* Benchmark small tensor convolutions
* Design kernel optimized for:

  * Low overhead
  * Minimal launch latency

### Experiments

* Vary tensor sizes
* Vary batch sizes

### Deliverables

* `small_conv_opt.cu`
* Benchmark comparison

---

## 2.6 End-to-End Pipeline Optimization

### Problem

CPU preprocessing becomes bottleneck in ML pipelines.

### Pipeline

```
CPU:
- resize
- normalize
- augment

GPU:
- model inference
```

### Goal

Move entire pipeline to GPU.

### Tasks

1. Implement GPU kernels for:

   * Resize (bilinear interpolation)
   * Normalization
   * Simple augmentation (flip, crop)

2. Build pipeline:

```
input -> GPU preprocess -> GPU conv -> output
```

3. Compare:

   * CPU + GPU hybrid
   * Full GPU pipeline

### Deliverables

* `preprocess_gpu.cu`
* `pipeline_full_gpu.cu`
* Throughput analysis

---

# 3. Performance Engineering

## 3.1 Metrics

Track:

* Execution time (ms)
* Throughput (images/sec)
* Memory bandwidth (GB/s)
* GPU utilization

## 3.2 Tools

* CUDA events for timing
* Profiling tools (Nsight Systems / Nsight Compute)

## 3.3 Methodology

* Always compare against baseline
* Change one variable at a time
* Document results rigorously

---

# 4. Project Structure

```
/project-root
  /src
    conv_naive.cu
    conv_tiled.cu
    custom_op.cu
    pipeline_fused.cu
    preprocess_gpu.cu

  /benchmarks
    benchmark_conv.cpp
    benchmark_pipeline.cpp

  /reports
    conv_analysis.md
    fusion_analysis.md
    pipeline_analysis.md

  /data
    sample_images/

  CMakeLists.txt
  README.md
  AGENTS.md
```

---

# 5. Execution Plan

## Phase 1: Foundations (Week 1-2)

* Implement naive convolution
* Learn CUDA basics
* Benchmark CPU vs GPU

## Phase 2: Optimization (Week 3-4)

* Implement tiled convolution
* Optimize memory usage
* Profile kernels

## Phase 3: Advanced Kernels (Week 5-6)

* Build fused kernels
* Implement custom operations

## Phase 4: Systems Design (Week 7-8)

* Build full GPU pipeline
* Optimize preprocessing
* Run end-to-end benchmarks

---

# 6. Success Criteria

You should be able to:

* Explain GPU execution at warp level
* Write efficient CUDA kernels
* Identify bottlenecks in ML pipelines
* Optimize memory-bound workloads
* Design fused kernels for performance

---

# 7. Non-Goals

* Beating highly optimized vendor libraries in general cases
* Building production-grade ML frameworks

Focus is on:

* Understanding
* Experimentation
* Measurable optimization

---

# 8. Notes

* Prioritize clarity over premature optimization
* Always validate correctness before optimizing
* Profile before making assumptions

---

End of AGENTS.md

