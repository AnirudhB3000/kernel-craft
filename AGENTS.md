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
- The tiled convolution implementation (`src/conv_tiled.cu`) is verified by `tests/test_conv_tiled.cpp`.

### Documentation Style

All source files must use Doxygen-style inline documentation:

- Use `\file` for file-level documentation
- Use `\brief` for function summaries
- Use `\param[in]` / `\param[out]` for parameters
- Use `\par` for paragraph sections
- Use `\code` / `\endcode` for code examples

Example:
```c
/**
 * \file example.cu
 * \brief Brief description.
 *
 * \par Detailed description
 * More details here.
 *
 * \param[in] input Input description.
 * \param[out] output Output description.
 */
```

---

## Design Decisions & Results

### Tiled vs Naive Convolution Benchmark Results (3×3 kernel)

| Image Size | Naive GPU (ms) | Tiled 8×8 (ms) | Tiled 16×16 (ms) | Tiled 32×32 (ms) | Best Tile |
|-----------|----------------|----------------|------------------|------------------|-----------|
| 256×256   | ~0.45          | 0.36           | 0.73             | 0.36             | 8×8 / 32×32 |
| 1024×1024 | ~0.70          | 0.60           | 0.60             | 0.95             | 8×8 / 16×16 |
| 2048×2048 | ~1.40          | 1.34           | 1.64             | 1.39             | 8×8 / 32×32 |

### Fused vs Separate Pipeline Results (1024×1024 image)

| Pipeline | Kernel Launches | Time (ms) | Speedup |
|----------|----------------|-----------|---------|
| Fused    | 1              | ~0.42     | ~2.3x   |
| Separate | 3              | ~0.96     | baseline|

**Key Insight**: Memory movement dominates; fused saves ~40% global memory traffic.

### Tile Size Analysis

| Tile Size | Shared Memory | Best For |
|-----------|---------------|----------|
| 8×8       | 10×10 = 100 floats | All sizes, consistently fast |
| 16×16    | 18×18 = 324 floats | Medium images (1024×1024) |
| 32×32    | 34×34 = 1156 floats | Large images, but overkill for small |

**Key Observations:**
- **8×8 tile**: Most consistent performer across all image sizes
- **16×16 tile**: Good balance, slight overhead at small sizes
- **32×32 tile**: Best for large images but overhead hurts small/medium

**Conclusion**: 8×8 tile provides the best overall performance for this 3×3 kernel workload.

### Phase 7: Performance Infrastructure Results

#### CUDA Graphs vs Separate Kernel Launches

| Method | Kernel Launches | Launch Overhead | Best For |
|--------|----------------|----------------|----------|
| Graphs | 1 (captured) | ~5-10μs | Batch processing |
| Separate | 3 | ~15-30μs | Debugging |

**Key Insight**: CUDA graphs reduce launch overhead by capturing kernel DAG at instantiation time.

#### Memory Pool Results

| Pool Size | Allocations | Overhead Reduction |
|-----------|------------|-----------------|
| 16 buffers | 16x reuse | ~85% |

**Key Insight**: Pre-allocated buffers eliminate per-batch cudaMalloc/cudaFree overhead.

#### Mixed Precision (FP16/FP32)

| Precision | Bandwidth | Compute | Accuracy |
|----------|----------|---------|----------|
| FP32 | baseline | baseline | 100% |
| FP16 | 50% | 2x (Tensor Cores) | ~99.9% |
| TF32 | baseline | ~1.1x | ~99.99% |

**Key Insight**: FP16 provides best throughput but requires Tensor Cores (Volta+). TF32 is automatic on Ampere+.

#### Persistent Kernels

| Mode | Launch Overhead | Latency | Best For |
|------|---------------|--------|----------|
| Standard | ~15-30μs/batch | baseline | Variable batch sizes |
| Persistent | ~0μs | ~50% lower | Fixed batch streams |

**Key Insight**: Persistent kernels eliminate re-launch overhead but require constant workload.

---

## Phase 6: Feature Extensions Results

### 3D Convolution

| Volume Size | Kernel | Naive GPU (ms) | Tiled (ms) |
|------------|--------|----------------|------------|
| 64×64×64   | 3×3×3  | ~8.5           | ~7.2       |
| 128×128×128| 3×3×3  | ~65            | ~52        |

**Key Insight**: 3D convolution scales cubically with volume dimensions; tiling provides modest gains due to shared memory constraints in 3D.

### Dilated Convolution

| Dilation Rate | Effective Receptive Field | Extra Overhead |
|---------------|--------------------------|----------------|
| 1 (baseline) | 3×3                     | 0%             |
| 2             | 5×5                     | ~15%           |
| 4             | 9×9                     | ~40%           |
| 8             | 17×17                   | ~120%          |

**Key Insight**: Dilated conv expands receptive field without additional parameters, but memory access pattern becomes less efficient.

### Transposed Convolution

| Input Size | Stride | Output Size | Notes |
|------------|--------|-------------|-------|
| 3×3        | 2      | 5×5         | +padding=1 |
| 4×4        | 2      | 7×7         | +padding=1 |
| 3×3        | 2      | 6×6         | +padding=0 |

**Key Insight**: Transposed convolution upsamples by strided expansion; atomic additions ensure correct accumulation from multiple input pixels.

### Grouped Convolution

| Groups | Parameters | Memory | Notes |
|--------|------------|--------|-------|
| 1 (dense) | C_in × C_out × K² | baseline | Standard conv |
| 2 | (C_in/2)×(C_out/2)×K² × 2 | ~50% | ResNeXt-style |
| 4 | (C_in/4)×(C_out/4)×K² × 4 | ~25% | More groups, fewer params |

**Key Insight**: Grouped convolution reduces parameters and memory by factor of groups²; each group processes independent channel subsets.

---

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

# 9. Python Extension (numpy / PyTorch)

## 9.1 Motivation

Enable real-world use cases by exposing CUDA kernels to Python ML workflows.

## 9.2 Goals

1. Add pybind11 via CMake FetchContent (auto-download, no pip install)
2. Expose `conv_naive` and `conv_tiled` to Python
3. Support numpy array input/output
4. Support PyTorch tensor input/output on GPU
5. Expose `tile_w` and `tile_h` as runtime parameters

## 9.3 API Design

```python
import kernel_craft

# numpy arrays
out = kernel_craft.conv_naive(input, kernel)      # -> np.ndarray
out = kernel_craft.conv_tiled(input, kernel, tile_w=8, tile_h=8)  # -> np.ndarray

# PyTorch tensors (CUDA)
out = kernel_craft.conv_naive(tensor, kernel)    # -> torch.Tensor on GPU
out = kernel_craft.conv_tiled(tensor, kernel, tile_w=16, tile_h=16)  # -> torch.Tensor on GPU
```

## 9.4 Implementation Tasks

### Task 1: Update CMakeLists.txt

Add pybind11 FetchContent and Python extension target:

```cmake
include(FetchContent)
FetchContent_Declare(pybind11 GIT_REPOSITORY https://github.com/pybind/pybind11.git GIT_TAG v2.11.1)
FetchContent_MakeAvailable(pybind11)
pybind11_add_module(kernel_craft src/pybind_cuda.cpp src/conv_naive.cu src/conv_tiled.cu)
```

### Task 2: Create src/pybind_cuda.cpp

Bind functions:

- `conv_naive(input: np.ndarray, kernel: np.ndarray) -> np.ndarray`
- `conv_tiled(input: np.ndarray, kernel: np.ndarray, tile_w: int, tile_h: int) -> np.ndarray`
- `conv_naive(input: Tensor, kernel: Tensor) -> Tensor`  (PyTorch overload)
- `conv_tiled(input: Tensor, kernel: Tensor, tile_w: int, tile_h: int) -> Tensor`  (PyTorch overload)

Handle memory transfer:

- numpy: copy host->device, run kernel, copy device->host, return numpy array
- torch: extract data pointer, run kernel, wrap result in new tensor on GPU

### Task 3: Support tile_w/tile_h runtime params

The tiled kernel requires compile-time tile size for shared memory. Two approaches:

1. Compile multiple kernel variants (8x8, 16x16, 32x32) and dispatch at runtime
2. Use runtime tile size with dynamic shared memory allocation

## 9.5 Deliverables

* `src/python/pybind_cuda.cpp` - pybind11 module (~480 lines)
* CMakeLists.txt updated with Python extension target
* `src/python/pyproject.toml` - package configuration
* `src/python/README.md` - user-facing documentation
* `src/python/tests/test_bindings.py` - 13 Python tests

## 9.6 Build & Test

```bash
cd src/python
python -m build

# Or with CMake
mkdir build && cd build
cmake ..
make kernel_craft_python
```

## 9.7 Verification

- [x] Module imports without error
- [x] numpy arrays produce correct output
- [x] PyTorch tensors produce correct output on GPU
- [x] `tile_w`/`tile_h` parameters work via runtime dispatch

---

## 10. Python Package Distribution

### Motivation

Enable easy installation via pip/PyPI for real-world use cases.

### Tasks

1. Create `src/python/pyproject.toml` for pip packaging
2. Configure Python package to include pre-built `.so` file
3. Add Python test suite in `src/python/tests/test_bindings.py`
4. Create Python examples in `examples/python/`
5. Document publishing steps

### Results

- Created `pyproject.toml` with build system and dependencies
- Python package includes pre-compiled `.so` for Python 3.12
- 13 Python tests pass (numpy + PyTorch)
- 4 example scripts demonstrating numpy/PyTorch usage

### Publishing

```bash
# Build
cd src/python
python -m build

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ kernel-craft

# Upload to PyPI
twine upload dist/*
```

---

## 11. PyPI Release Workflow

### Motivation

Manual release workflow for PyPI distribution with configurable version bumps.

### Workflow Design

| Trigger | Input | Runner |
|---------|-------|--------|
| `workflow_dispatch` (manual) | patch / minor / major | self-hosted (GPU) |

### Features

- Clears old distributions from `src/python/dist/` before build
- Builds and runs C++ tests (ctest)
- Builds and runs Python tests (pytest)
- Configurable version bump via workflow input
- Auto-upload to TestPyPI after successful build

### Usage

```bash
# Trigger via GitHub UI:
# 1. Go to Actions → Release
# 2. Select "Run workflow"
# 3. Choose version type: patch/minor/major
# 4. Click "Run workflow"
```

```bash
# Or via GitHub CLI:
gh workflow run release.yml -f version_type=patch
```

### Configuration

Required secrets:
- `TWINE_PASSWORD`: PyPI/TestPyPI API token

### Publishing to Production PyPI

After testing on TestPyPI, upload to production:

```bash
cd src/python
twine upload dist/*
```

---

End of AGENTS.md

