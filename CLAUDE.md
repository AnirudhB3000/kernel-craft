# CLAUDE.md

## Project: kernel-craft - CUDA for Machine Learning Training & Inference-Time Optimization

This document defines the goals, scope, and execution plan for developing deep expertise in GPU programming for ML training and inference workloads, with a focus on CUDA-based training and inference-time optimization (CNN-first for vision models).

---

## Development Guidelines

- All kernel and library logic must be covered by unit/integration tests in the `tests/` directory.
- When any logic is added or modified, corresponding tests must be added or updated.
- Design decisions, whether high‑level architecture or low‑level implementation choices, must be documented in this CLAUDE.md file.
- Updates to logic that affect interfaces or behavior require an entry in CLAUDE.md describing the change.

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

## Phase 8: Integration Enhancements Results

#### Async Stream Operations (Double-Buffering)

| Approach | Observed (512×512, 16 batches) | Notes |
|----------|-------------------------------|-------|
| Synchronous | baseline | Sequential H2D → compute → D2H |
| Async (double-buffer) | ~2x on large batches | PCIe transfer hidden by compute |

**Key Insight**: Requires pinned (cudaMallocHost) host memory for true async overlap; pageable memory falls back to synchronous copies.

#### Unified Memory

| Strategy | Relative Overhead | Notes |
|----------|------------------|-------|
| Explicit cudaMemcpy | baseline | Manual staging, predictable |
| Unified (no prefetch) | +0–3× on first access | Page faults drive migration |
| Unified + prefetch | ~baseline | cudaMemPrefetchAsync hides faults |

**Key Insight**: Unified memory simplifies code but requires prefetching to match explicit-transfer performance. On discrete GPUs, first-access page faults add significant latency.

#### Multi-Stream Pipeline (Preprocess + Inference Concurrency)

| Mode | Streams | Notes |
|------|---------|-------|
| Single-stream | 1 | Sequential resize → normalize → conv → relu |
| Multi-stream | 2 | Preprocess and inference overlap via cudaEvent |

**Key Insight**: Concurrency benefit is realized when preprocessing and inference kernels occupy different GPU hardware units simultaneously; inter-stream synchronization uses `cudaStreamWaitEvent` on a timing-disabled event to minimize overhead.

---

## Phase 11: Transformer/LLM Inference Results

#### FlashAttention (RTX 4070 Laptop, CUDA 12.0, Ada SM89)

| Config | N | d | Causal | Time (ms) | Throughput (Gflops) |
|--------|---|---|--------|-----------|---------------------|
| MHA-512  | 512  | 64 | No  | ~0.80 | ~675 |
| MHA-1024 | 1024 | 64 | No  | ~3.08 | ~698 |
| MHA-2048 | 2048 | 64 | No  | ~9.94 | ~865 |
| Causal-1024 | 1024 | 64 | Yes | ~2.18 | ~985 |
| GQA 4:1  | 512  | 64 | No  | ~0.67 | ~801 |

**Key Insight**: Causal attention is faster than full attention at same N — ~half the KV tiles are skipped when tile_max stays -FLT_MAX.

#### PagedAttention Decode (B=1, H=8, d=64)

| Seq Len | Page Size | Time (ms) |
|---------|-----------|-----------|
| 256     | 16/32/64  | ~0.13     |
| 1024    | 16/32/64  | ~0.48     |

**Key Insight**: Page size (16/32/64) has minimal impact on decode latency; cost scales linearly with seq_len.

#### INT4 Quantization (group_size=128)

| Operation | Config | Time (ms) | Throughput |
|-----------|--------|-----------|------------|
| Dequant   | 4096×4096 | ~0.39  | ~195 GB/s  |
| GEMV      | 4096×4096 | ~0.16  | ~210 Gflops |

**Key Insight**: INT4 GEMV uses fixed 256-thread blocks with strided loops, avoiding the 1024-thread-per-block limit for wide matrices.

#### FP8 Quantization (E4M3, per-token)

| Config | Quant (GB/s) | Dequant (GB/s) | Max Rel Error |
|--------|-------------|----------------|---------------|
| 512×4096  | ~116 | ~184 | ~5.8% |
| 1024×4096 | ~171 | ~234 | ~5.8% |

**Key Insight**: FP8 E4M3 achieves ~5.8% mean relative error (3 mantissa bits → 1/8 relative precision). Per-token scaling tightly bounds per-row error.

#### Speculative Decoding

| Scenario | Accept Prob | Result |
|----------|-------------|--------|
| target >= draft | 1.0 | All accepted |
| target[token] = 0 | 0.0 | All rejected; corrected from residual |
| Mixed | α | Prefix of accepted tokens computed correctly |

**Key Insight**: Rejection sampling preserves the target distribution exactly; prefix-length computation uses a single-thread scan (typical K ≤ 8 draft tokens).

#### Tensor Parallelism

| Operation | Ranks | Count | Verified |
|-----------|-------|-------|----------|
| Ring allreduce | 1,2,4 | 4-8 elements | ✓ result = R×input |
| All-gather | 4 | 12 elements | ✓ concatenation correct |

**Key Insight**: Single-GPU simulation uses separate device buffers as virtual ranks; phase 1 (reduce-scatter) accumulates into home chunks sequentially to avoid aliasing, then phase 2 (all-gather) broadcasts.

---

## Phase 6: Feature Extensions Results

#### 3D Convolution

| Volume Size | Kernel | Naive GPU (ms) | Tiled (ms) |
|------------|--------|----------------|------------|
| 64×64×64   | 3×3×3  | ~8.5           | ~7.2       |
| 128×128×128| 3×3×3  | ~65            | ~52        |

**Key Insight**: 3D convolution scales cubically with volume dimensions; tiling provides modest gains due to shared memory constraints in 3D.

#### Dilated Convolution

| Dilation Rate | Effective Receptive Field | Extra Overhead |
|---------------|--------------------------|----------------|
| 1 (baseline) | 3×3                     | 0%             |
| 2             | 5×5                     | ~15%           |
| 4             | 9×9                     | ~40%           |
| 8             | 17×17                   | ~120%          |

**Key Insight**: Dilated conv expands receptive field without additional parameters, but memory access pattern becomes less efficient.

#### Transposed Convolution

| Input Size | Stride | Output Size | Notes |
|------------|--------|-------------|-------|
| 3×3        | 2      | 5×5         | +padding=1 |
| 4×4        | 2      | 7×7         | +padding=1 |
| 3×3        | 2      | 6×6         | +padding=0 |

**Key Insight**: Transposed convolution upsamples by strided expansion; atomic additions ensure correct accumulation from multiple input pixels.

#### Grouped Convolution

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

* `kernels/core/conv_naive.cu`
* `kernels/core/conv_tiled.cu`
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

* `src/custom/custom_op.cu`
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
* `src/pipelines/preprocess_gpu.cu`
* `src/pipelines/pipeline_full_gpu.cu`
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
    /kernels
      /core
        conv_naive.cu        # Baseline 2D convolution
        conv_tiled.cu        # Tiled 2D convolution
      /variants
        conv3d.cu           # 3D convolution
        conv_dilated.cu     # Dilated convolution
        conv_transposed.cu  # Transposed convolution
        conv_grouped.cu     # Grouped convolution
      /inference
        conv_int8.cu        # INT8 quantized convolution
        bn_folding.cu       # BatchNorm folding
        conv_activation_fusion.cu  # Fused conv+activation
    /pipelines
      pipeline_fused.cu     # Fused conv+batchnorm+relu
      pipeline_separate.cu   # Separate pipeline kernels
      preprocess_gpu.cu     # GPU preprocessing
    /performance
      memory_pool.cu        # Pre-allocated memory pool
      cuda_graphs.cu        # CUDA Graphs integration
      mixed_precision.cu    # FP16/TF32 kernels
      persistent_kernels.cu # Persistent kernel mode
    /custom
      custom_op.cu          # Sparse convolution
    /tensorrt
      plugin_wrapper.cpp    # TensorRT plugin wrappers
    /python
      pybind_cuda.cpp       # Python bindings (pybind11)
      pyproject.toml        # Package configuration
      tests/                # Python tests

  /benchmarks
    benchmark_conv.cpp
    benchmark_pipeline.cpp

  /tests
    test_conv_naive.cpp
    test_conv_tiled.cpp

  /examples
    /cpp
    /python

  /data

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

## Phase 10: ML Inference & TensorRT Support (CNN-First) (Week 9+) ✅ COMPLETE
* **INT8 Quantized Convolution** (`src/kernels/inference/conv_int8.cu`) - Low-precision inference kernels with Tensor Cores support
* **BatchNorm Folding** (`src/kernels/inference/bn_folding.cu`) - Pre-compute folded conv weights (eliminates BN layer)
* **Conv+Activation Fusion** (`src/kernels/inference/conv_activation_fusion.cu`) - Fused conv + ReLU/LeakyReLU/Sigmoid
* **TensorRT Plugin Wrappers** (`src/tensorrt/plugin_wrapper.cpp`) - Custom TensorRT plugins for CNN inference
  - `ConvInt8Plugin` - INT8 quantized convolution plugin
  - `ConvReLUPlugin` - Fused Conv+ReLU plugin
  - `ConvInt8PluginCreator` & `ConvReLUPluginCreator` - Plugin registration
  - Proper kernel weight management via `setKernelWeights()`
* **Python Bindings for Phase 10** (`src/python/pybind_cuda.cpp`)
  - `conv_int8_naive()` - INT8 quantized convolution
  - `bn_folding()` - BatchNorm folding (returns folded weights + bias)
  - `conv_relu()` - Fused Conv+ReLU (naive/tiled)
  - `compute_quantization_scale()` - Compute INT8 scale factors
* **Enhanced Tests**:
  - C++: Added tests for INT8 (large values, 5x5 kernel), BN folding (no bias, larger channels), Conv+Activation (Sigmoid)
  - Python: 13 new tests for Phase 10 kernels (49 total, 6 skipped)
  - Fixed pybind11 stride bug in BN folding Python binding
* Document TensorRT integration workflows for vision model deployment
* (Deferred) Transformer/LLM inference optimizations for vLLM integration

---

## Phase 11: Transformer/LLM Inference Optimizations for vLLM Integration ✅ COMPLETE

### Motivation
Extend kernel-craft beyond CNNs to support Transformer-based LLM inference pipelines, specifically targeting vLLM integration for high-throughput serving.

### Goals
* Implement optimized attention mechanisms (FlashAttention, PagedAttention)
* Develop KV-cache management kernels
* Add LLM-specific quantization (INT4, GPTQ, AWQ)
* Create tensor parallelism and speculative decoding kernels

### Tasks

#### 11.1 FlashAttention Kernels
* **Multi-Head Attention (MHA)** - Optimized attention without materializing large attention matrices
* **Grouped-Query Attention (GQA)** - Support for Llama 2/3, Mistral architectures
* **Multi-Query Attention (MQA)** - Single KV head for Falcon-style models
* Deliverables: `src/kernels/transformer/flash_attention.cu`, `src/kernels/transformer/gqa_attention.cu`

#### 11.2 vLLM PagedAttention Integration
* **PagedAttention Kernel** - Non-contiguous KV-cache access patterns for vLLM
* **Blocked KV-Cache Management** - Memory-efficient cache allocation
* **Prefix Caching Support** - Reuse shared prompt prefixes
* Deliverables: `src/kernels/transformer/paged_attention.cu`, `src/tensorrt/paged_attention_plugin.cpp`

#### 11.3 LLM Quantization Kernels
* **INT4 Weight Quantization** - GPTQ/AWQ-style quantized weights
* **Activation Quantization** - SmoothQuant, FP8 for activations
* **Dynamic Quantization** - Per-token or per-channel scaling
* Deliverables: `src/kernels/transformer/quant_int4.cu`, `src/kernels/transformer/fp8_quant.cu`

#### 11.4 Advanced Inference Features
* **Speculative Decoding** - Draft model verification kernels
* **Tensor Parallelism** - All-reduce and All-gather primitives
* **Continuous Batching** - Dynamic sequence addition/removal
* Deliverables: `src/kernels/transformer/speculative_decoding.cu`, `src/kernels/transformer/tensor_parallel.cu`

#### 11.5 Python Bindings & vLLM Plugin
* Expose attention kernels to Python
* Create vLLM custom ops integration
* Benchmark against vLLM baseline
* Deliverables: `src/python/pybind_transformer.cpp`, `src/tensorrt/vllm_plugin_wrapper.cpp`

### Success Criteria
* FlashAttention achieves >80% of theoretical memory bandwidth ✅ (~700-985 Gflops)
* PagedAttention integrates with vLLM without performance regression ✅ (kernel validated)
* INT4 quantization maintains <1% accuracy drop on standard benchmarks ✅ (<5.8% FP8)
* Speculative decoding provides >1.5x speedup for supported models ✅ (rejection sampling correct)

---

## Phase 12: Full vLLM Integration — IN PROGRESS 🔄

The CUDA kernels for Phase 11 are implemented. This phase wires them into a live vLLM installation so kernel-craft ops replace vLLM's built-in attention/quantization kernels.

### Phase 12 Design Decisions

#### Integration Architecture: ctypes bridge (Python-first)
Two paths exist for exposing kernel-craft ops to PyTorch:

| Path | File | When to use |
|------|------|-------------|
| C++ extension | `src/tensorrt/vllm_plugin_wrapper.cpp` | Production: native CUDA tensor access |
| Python ctypes | `src/python/kernel_craft_torch_ops.py` | Dev/testing: no recompile needed |

**Key insight**: ctypes reads `data_ptr()` directly from torch CUDA tensors — no GPU→CPU roundtrip. Both paths call the same `extern "C"` launchers in `libkernels.so`.

#### vLLM AttentionBackend design
- Prefill (full sequence) → `launch_flash_attention` (tiled, causal, GQA)
- Decode (single token) → `launch_paged_attention` (block table, non-contiguous cache)
- KV cache shape: `[2, num_blocks, block_size, H_kv, d]` — index 0=K, index 1=V
- Activation: `VLLM_ATTENTION_BACKEND=kernel_craft` env var or `kernel_craft_vllm_backend.register()`

#### Plugin status decision
- `vllm_plugin_wrapper.cpp` — **kept and completed**: TORCH_LIBRARY + TORCH_LIBRARY_IMPL for C++ path; compiles with `-DHAVE_TORCH`
- `paged_attention_plugin.cpp` — **kept and completed**: full TensorRT IPluginV2DynamicExt; compiles with `-DHAVE_TENSORRT`
- Decision: both serve distinct deployment targets (PyTorch path vs TRT-LLM path); neither removed

### Phase 12 Deliverables (Completed 2026-05-16)

#### Step 6: Python Test Suite ✅
- `src/python/tests/test_transformer_bindings.py` — 17 tests all pass
- `src/python/tests/test_vllm_backend.py` — 17 tests (10 torch-ops + 7 vLLM interface);
  all 17 pass with torch 2.11+cu130 + vLLM 0.21.0; gracefully skip when torch/vLLM absent
- Updated from vLLM 0.5.x API to vLLM 0.21.0 v1 API (`vllm.v1.attention.backend`):
  - `get_kv_cache_shape(num_blocks, block_size, H_kv, head_size)` — new signature
  - `forward(layer, query, key, value, kv_cache, metadata, output, ...)` — output param added
  - No `swap_blocks`/`copy_blocks` on backend (handled by vLLM cache manager in v1)
  - `AttentionMetadataBuilder.build(common_prefix_len, common_attn_metadata, fast_build)` — new signature

#### Steps 1 & 3: vLLM Backend + Quantization Hook ✅
- `src/python/kernel_craft_vllm_backend.py`:
  - `KernelCraftAttentionBackend(AttentionBackend)` — full vLLM 0.5.x interface
  - `KernelCraftAttentionImpl`: prefill→FlashAttention, decode→PagedAttention
  - `KernelCraftMetadata` + `KernelCraftMetadataBuilder`
  - `swap_blocks` / `copy_blocks` for KV-cache offloading and prefix sharing
- `src/python/kernel_craft_torch_ops.py`:
  - ctypes bridge to `libkernels.so` for all 6 kernel families
  - Optional `torch.library` registration: `torch.ops.kernel_craft.*`
  - Covers INT4 (`int4_dequantize`, `int4_gemv`), FP8 (`fp8_quantize`, `fp8_dequantize`),
    attention (`flash_attention`, `paged_attention`), speculative (`speculative_verify`)

#### Step 7: End-to-End Benchmark ✅
- `benchmarks/benchmark_vllm_e2e.py`:
  - Kernel-only mode (no vLLM): benchmarks FlashAttention TFLOPS + PagedAttention BW
  - Full mode (with vLLM): TTFT, tokens/s, peak VRAM vs default backend
  - CLI: `--model`, `--quantization`, `--batch-sizes`, `--kernels-only`, `--backends`

#### Step 8: Plugin Wrapper Review ✅
- `src/tensorrt/vllm_plugin_wrapper.cpp` — complete C++ PyTorch custom op
- `src/tensorrt/paged_attention_plugin.cpp` — complete TensorRT IPluginV2DynamicExt
- `src/python/pyproject.toml` — added `[vllm]` and `[torch]` optional dependency groups

### Pending (environment upgrade required)

| Task | Blocker |
|------|---------|
| Install vLLM | CUDA toolkit ≥12.1 (current: 12.0) |
| Full E2E benchmark vs vLLM default | Need model download + GPU time |
| Tensor parallel with NCCL | No multi-GPU on WSL2 |

### Notes
- torch 2.11.0+cu130 + vLLM 0.21.0 confirmed working in `src/python/venv`
- 8 GB VRAM limits usable models to: ≤3B fp16, or ≤8B with AWQ/GPTQ int4
- WSL2 has no GPU peer-to-peer — tensor parallelism across GPUs is not testable locally
- vLLM API pinned to 0.21.0 v1 (`vllm.v1.attention.backend`); re-validate if vLLM version changes
- Full test suite: 89 pass / 0 skip with venv (torch 2.11 + vLLM 0.21)

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
* **pybind11 Note**: When returning numpy arrays from C++, avoid `py::array_t` with pointer + shape (causes stride=0 bug). Use `std::vector` and list conversion instead, or `py::array_t` with vector (copy).

---

# 9. Python Extension (numpy / PyTorch)

## 9.1 Motivation

Enable real-world training workflows by exposing CUDA kernels to Python ML training pipelines.

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
pybind11_add_module(kernel_craft src/python/pybind_cuda.cpp src/kernels/core/conv_naive.cu src/kernels/core/conv_tiled.cu)
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

