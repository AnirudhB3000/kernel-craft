# CLAUDE.md

## Project: kernel-craft - CUDA for Machine Learning Training & Inference-Time Optimization

This document defines the goals, scope, and execution plan for developing deep expertise in GPU programming for ML training and inference workloads, with a focus on CUDA-based training and inference-time optimization (CNN-first for vision models).

---

## Development Guidelines

- All kernel and library logic must be covered by unit/integration tests in the `tests/` directory.
- When any logic is added or modified, corresponding tests must be added or updated.
- Design decisions, whether high‑level architecture or low‑level implementation choices, must be documented in this CLAUDE.md file.
- Updates to logic that affect interfaces or behavior require an entry in CLAUDE.md describing the change.
- **NEVER commit to git.** Do not run `git commit`, `git push`, or any destructive git command. The user manages all commits.

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

### Phase 8: Integration Enhancements Results

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

### Phase 11: Transformer/LLM Inference Results

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

### Phase 6: Feature Extensions Results

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

### Phase 12: Full vLLM Integration Design Decisions

#### Integration Architecture: ctypes bridge (Python-first)

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

#### vLLM 0.21.0 v1 API (`vllm.v1.attention.backend`)
- `get_kv_cache_shape(num_blocks, block_size, H_kv, head_size)` — new signature vs 0.5.x
- `forward(layer, query, key, value, kv_cache, metadata, output, ...)` — output param added
- `AttentionMetadataBuilder.build(common_prefix_len, common_attn_metadata, fast_build)` — new signature
- No `swap_blocks`/`copy_blocks` on backend (handled by vLLM cache manager in v1)

#### Plugin strategy
- `vllm_plugin_wrapper.cpp` — TORCH_LIBRARY + TORCH_LIBRARY_IMPL; compiles with `-DHAVE_TORCH`
- `paged_attention_plugin.cpp` — TensorRT IPluginV2DynamicExt; compiles with `-DHAVE_TENSORRT`
- Both serve distinct deployment targets (PyTorch path vs TRT-LLM path)

#### Environment notes
- torch 2.11.0+cu130 + vLLM 0.21.0 confirmed working in `src/python/venv`
- 8 GB VRAM limits usable models to: ≤3B fp16, or ≤8B with AWQ/GPTQ int4
- WSL2 has no GPU peer-to-peer — tensor parallelism across GPUs not testable locally
- Full test suite: 89 pass / 0 skip with venv (torch 2.11 + vLLM 0.21)

---

## Phase 14: SSM / Mamba Kernels ✅ COMPLETE

### Design Decisions

#### Selective scan parallelism (`selective_scan.cu`)
- One thread per (batch, channel) pair (b, d); B×D threads run concurrently
- L timesteps processed sequentially within each thread — the SSM recurrence is inherently sequential along L
- Hidden state h[N_state] held in local registers; the compiler keeps these in registers for N_state ≤ SS_MAX_N=32 (Mamba-1 default: 16)
- ZOH discretization: `A_bar = exp(delta * A_log)`, `B_bar = delta * B`, `h = A_bar*h + B_bar*u`, `y = C @ h`
- Max absolute error vs CPU sequential reference: < 1e-8 (both paths use `expf` identically)

#### Depthwise conv1d (`depthwise_conv1d.cu`)
- One thread per output element; grid covers B × D × L elements
- Causal padding implemented by index-gating (`if src >= 0`) rather than explicit zero-padding buffer — avoids extra memory allocation
- Input layout: channels-first [B, D, L] for coalesced L-dimension access across threads

#### RMSNorm (`rmsnorm.cu`)
- One thread block per row (B×T rows); parallel reduction over D using shared memory
- Two-pass: (1) parallel sum-of-squares → shared reduction → rms_inv via `rsqrtf`; (2) normalize and scale
- `RMS_BLOCK=256` handles D up to 65536 via grid-stride loop in each pass

### Benchmark Results (RTX 4070 Laptop, B=1, D=512, N=16)

#### Selective Scan Throughput
| L | ms/iter | Gflops | Mtokens/s |
|---|---------|--------|-----------|
| 64 | 0.105 | ~30 | 0.61 |
| 256 | 0.406 | ~31 | 0.63 |
| 1024 | 1.717 | ~29 | 0.60 |
| 4096 | 5.570 | ~36 | 0.74 |
| 16384 | 24.047 | ~33 | 0.68 |

**Key Insight**: Selective scan is heavily memory-bound at small L (launch overhead dominates); throughput stabilizes at ~33 Gflops for L ≥ 4096. Sequential recurrence per (b,d) thread limits GPU utilization — real Mamba implementations use chunked parallel scan for long sequences.

#### Depthwise Conv1d (B=1, D=2048, d_conv=4)
| L | ms/iter | GB/s |
|---|---------|------|
| 64 | 0.025 | ~126 |
| 1024 | 0.068 | ~741 |
| 4096 | 0.303 | ~664 |
| 16384 | 1.211 | ~665 |

**Key Insight**: Memory bandwidth saturates near the RTX 4070's ~192 GB/s peak for L ≥ 1024 (effective BW higher due to weight reuse via d_conv=4 taps).

#### RMSNorm (128 rows)
| D | ms/iter | GB/s |
|---|---------|------|
| 512 | 0.030 | ~18 |
| 1024 | 0.023 | ~46 |
| 2048 | 0.029 | ~72 |
| 4096 | 0.025 | ~171 |

**Key Insight**: RMSNorm bandwidth scales with D; at D=4096 it approaches ~170 GB/s. At D=512, per-block launch overhead is visible.

### Deliverables
* `src/kernels/transformer/selective_scan.cu` — ZOH selective scan, N_state ≤ 32
* `src/kernels/transformer/depthwise_conv1d.cu` — causal depthwise conv1d, d_conv configurable
* `src/kernels/transformer/rmsnorm.cu` — fused RMSNorm (normalize + scale)
* `tests/test_selective_scan.cpp` — 5 tests: small/typical/long/zero/max-state; all pass
* `tests/test_mamba_ops.cpp` — 8 tests: 4 conv1d variants + 4 rmsnorm variants; all pass
* `benchmarks/benchmark_selective_scan.cpp` — scan / conv1d / rmsnorm throughput
* `src/python/pybind_transformer.cpp` — `selective_scan()`, `depthwise_conv1d()`, `rmsnorm()` bindings
* `src/python/tests/test_mamba_bindings.py` — 17 pytest tests; all pass

---

# 1. Core Objectives

## 1.1 Foundational Understanding

Develop a systems-level understanding of:

* How convolution maps to GPU execution
* Thread hierarchy (grid, block, warp)
* Memory hierarchy (global, shared, registers)
* Memory bandwidth vs compute bottlenecks

### Key Outcomes

* Ability to reason about kernel performance
* Identify memory-bound vs compute-bound workloads
* Understand GPU occupancy and utilization

---

# 2. Technical Focus Areas

## 2.1 Convolution Mechanics on GPU

**Goals**: Implement naive convolution; optimize with shared memory tiling; reduce redundant global memory access.

**Key techniques**: Coalesced memory access, halo-region handling, `__syncthreads()`, tile size dispatch.

## 2.2 Memory Optimization

**Goals**: Minimize global memory traffic; maximize shared memory reuse.

**Techniques**: Coalesced access, shared memory tiling, register pressure management.

## 2.3 Custom Operations

**Goal**: Non-standard convolution variants (sparse, custom filters, domain-specific transforms) not served efficiently by vendor libraries.

## 2.4 Fused Kernels

**Goal**: Reduce kernel launch overhead and memory transfers by fusing conv → batchnorm → relu into a single kernel.

**Key Insight**: Memory movement dominates cost more than arithmetic.

---

# 3. Performance Engineering

## 3.1 Metrics

Track: execution time (ms), throughput (images/sec), memory bandwidth (GB/s), GPU utilization.

## 3.2 Tools

CUDA events for timing; Nsight Systems / Nsight Compute for profiling.

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
      /transformer
        flash_attention.cu  # MHA/GQA/MQA (Phase 11)
        paged_attention.cu  # vLLM paged KV-cache (Phase 11)
        quant_int4.cu       # INT4 dequant + GEMV (Phase 11)
        fp8_quant.cu        # FP8 E4M3 quantization (Phase 11)
        speculative_decoding.cu  # Draft token verification (Phase 11)
        tensor_parallel.cu  # All-reduce / all-gather (Phase 11)
    /pipelines
      pipeline_fused.cu     # Fused conv+batchnorm+relu
      pipeline_separate.cu  # Separate pipeline kernels
      preprocess_gpu.cu     # GPU preprocessing
    /performance
      memory_pool.cu        # Pre-allocated memory pool
      cuda_graphs.cu        # CUDA Graphs integration
      mixed_precision.cu    # FP16/TF32 kernels
      persistent_kernels.cu # Persistent kernel mode
      async_streams.cu      # Double-buffered async streams
      unified_memory.cu     # Unified memory with prefetch
      multi_stream_pipeline.cu  # Concurrent preprocess+inference
    /tensorrt
      plugin_wrapper.cpp          # TensorRT CNN plugins
      paged_attention_plugin.cpp  # TensorRT PagedAttention plugin
      vllm_plugin_wrapper.cpp     # PyTorch custom ops (C++ path)
    /python
      pybind_cuda.cpp       # Python bindings: conv kernels
      pybind_transformer.cpp # Python bindings: transformer kernels
      kernel_craft_torch_ops.py    # ctypes bridge to libkernels.so
      kernel_craft_vllm_backend.py # vLLM AttentionBackend
      pyproject.toml        # Package configuration
      tests/                # Python tests

  /benchmarks
    benchmark_conv.cpp
    benchmark_pipeline.cpp
    benchmark_flash_attention.cpp
    benchmark_paged_attention.cpp
    benchmark_quant.cpp
    benchmark_vllm_e2e.py

  /tests
    test_conv_naive.cpp
    test_conv_tiled.cpp
    test_flash_attention.cpp
    test_paged_attention.cpp
    test_fp8_quant.cpp
    test_quant_int4.cpp
    test_speculative_decoding.cpp
    test_tensor_parallel.cpp

  CMakeLists.txt
  README.md
  AGENTS.md
```

---

# 5. Execution Plan

## Phase 1: Foundations (Week 1-2) ✅
* Implement naive convolution, learn CUDA basics, benchmark CPU vs GPU.

## Phase 2: Optimization (Week 3-4) ✅
* Implement tiled convolution, optimize memory usage, profile kernels.

## Phase 3: Advanced Kernels (Week 5-6) ✅
* Build fused kernels, implement custom operations.

## Phase 4: Systems Design (Week 7-8) ✅
* Build full GPU pipeline, optimize preprocessing, run end-to-end benchmarks.

## Phase 5: Python Integration ✅
* PyPI release workflow, pybind11 bindings, numpy/PyTorch support.

## Phase 6: Feature Extensions ✅
* 3D, dilated, transposed, and grouped convolution variants.

## Phase 7: Performance Infrastructure ✅
* CUDA graphs, memory pool, mixed precision (FP16/TF32), persistent kernels.

## Phase 8: Integration Enhancements ✅
* Async stream operations, unified memory, multi-stream pipeline.

## Phase 9: Additional Framework Support (Optional) — Pending
* JAX, ONNX Runtime, or TensorFlow integration options.

## Phase 10: ML Inference & TensorRT Support (CNN-First) ✅ COMPLETE
* **INT8 Quantized Convolution** (`src/kernels/inference/conv_int8.cu`)
* **BatchNorm Folding** (`src/kernels/inference/bn_folding.cu`)
* **Conv+Activation Fusion** (`src/kernels/inference/conv_activation_fusion.cu`)
* **TensorRT Plugin Wrappers** (`src/tensorrt/plugin_wrapper.cpp`)
  - `ConvInt8Plugin`, `ConvReLUPlugin`, and their plugin creators
* **Python Bindings**: `conv_int8_naive()`, `bn_folding()`, `conv_relu()`, `compute_quantization_scale()`
* **Tests**: C++ (INT8 large values, 5x5 kernel, BN no-bias, Sigmoid fusion) + Python (13 tests, 49 total)

## Phase 11: Transformer/LLM Inference Optimizations ✅ COMPLETE

### 11.1 FlashAttention Kernels
* MHA/GQA/MQA in `src/kernels/transformer/flash_attention.cu` — online softmax, causal masking, H_kv parameter for GQA
* 5 tests; benchmark: ~675–985 Gflops on RTX 4070

### 11.2 vLLM PagedAttention
* `src/kernels/transformer/paged_attention.cu` — block table lookup, variable seq lengths
* `src/tensorrt/paged_attention_plugin.cpp` — TensorRT IPluginV2DynamicExt
* 3 tests; decode: 0.13–0.48ms for seq 256–1024

### 11.3 LLM Quantization Kernels
* `src/kernels/transformer/quant_int4.cu` — GPTQ/AWQ-style 2×INT4/byte, per-group scales
* `src/kernels/transformer/fp8_quant.cu` — E4M3, per-token/per-tensor scaling, SmoothQuant smoothing
* 7 tests total; INT4 dequant ~195 GB/s; FP8 ~5.8% mean rel error

### 11.4 Advanced Inference Features
* `src/kernels/transformer/speculative_decoding.cu` — rejection sampling verification
* `src/kernels/transformer/tensor_parallel.cu` — ring all-reduce + all-gather (single-GPU simulation)
* 9 tests total

### 11.5 Python Bindings & vLLM Plugin
* `src/python/pybind_transformer.cpp` — exposes flash_attention, paged_attention, quant_int4_dequant, fp8_quantize/dequantize, speculative_decode
* `src/tensorrt/vllm_plugin_wrapper.cpp` — TORCH_LIBRARY registration for C++ path
* 17 pytest tests pass

## Phase 12: Full vLLM Integration ✅ SUBSTANTIALLY COMPLETE

### Deliverables
* `src/python/kernel_craft_torch_ops.py` — ctypes bridge covering all 6 kernel families
* `src/python/kernel_craft_vllm_backend.py` — KernelCraftAttentionBackend (vLLM 0.21.0 v1 API)
* `benchmarks/benchmark_vllm_e2e.py` — kernel-only + full vLLM TTFT/throughput/VRAM benchmark
* `src/python/tests/test_vllm_backend.py` — 17 tests; all pass with torch 2.11+cu130 + vLLM 0.21.0
* Full Python test suite: **89 pass / 0 skip** with venv

### Remaining
* Run full E2E benchmark with a downloaded model (`facebook/opt-125m` or similar)
* Multi-GPU tensor parallelism — deferred (WSL2, no peer-to-peer)

---

## Phase 13: Real Multi-GPU Tensor Parallelism ✅ COMPLETE

### Design Decisions

#### SGEMM kernel for parallel linear shards (`tensor_parallel.cu`)
- `sgemm_nt_kernel`: tiled 16×16 SGEMM for C[M,N] = A[M,K] × B[N,K]^T
- Shared memory: `sA[16][17]` and `sB[16][17]` (padding +1 avoids bank conflicts)
- Load pattern: thread (tx,ty) loads sA[ty][tx] = A[row, t*T+tx] and sB[ty][tx] = B[(bx*T+ty), t*T+tx]; inner loop: `acc += sA[ty][k] * sB[tx][k]`
- Both `launch_col_parallel_linear` and `launch_row_parallel_linear` share the same kernel
- `extern "C"` launchers take the local shard dimensions, not total dimensions

#### Col-parallel linear
- Each rank holds W_rank [N/R, K]; input x [M, K] is replicated
- y_rank [M, N/R] = x @ W_rank^T; then all-gather → y [M, N]
- In the GEMM: M=batch, N=N_rank, K=in_features

#### Row-parallel linear
- Each rank holds W_rank [N, K/R] and x_rank [M, K/R] (pre-split input)
- partial_rank [M, N] = x_rank @ W_rank^T; then all-reduce sum → y [M, N]
- In the GEMM: M=batch, N=out_features, K=K_rank

#### NCCL file (`tensor_parallel_nccl.cu`)
- Always compiled; `#ifdef HAVE_NCCL` guards actual NCCL calls; stubs print to stderr
- `nccl_comm_init_all(comms, n, devs)` wraps `ncclCommInitAll` (void** → ncclComm_t*)
- `nccl_group_start/end()` required in single-process multi-rank usage
- In-place allreduce: `ncclAllReduce(d_buf, d_buf, ...)` with `ncclSum`

#### CMake NCCL detection
- `find_path(NCCL_INCLUDE_DIR nccl.h)` + `find_library(NCCL_LIBRARY nccl)`
- Propagates `HAVE_NCCL` definition and headers/lib to `kernels` target via `PUBLIC`
- All downstream targets (test_tensor_parallel, benchmark_tensor_parallel, kernel_craft_transformer) inherit automatically

#### Python bindings (`pybind_transformer.cpp`)
- `col_parallel_linear(x[M,K], W[N_rank,K])` → numpy [M, N_rank]
- `row_parallel_linear(x_rank[M,K_rank], W[N,K_rank])` → numpy [M, N]
- `nccl_comm_init(devs)` → list of uint64 handles (opaque void*)
- `nccl_allreduce(handle, arr)` — allocates device buffer, calls NCCL, copies back
- `HAVE_NCCL` module attribute (True/False) for Python-side feature detection

### Benchmark Results (RTX 4070 Laptop, single-GPU simulation, no NCCL)

#### Simulation All-Gather (per-rank bandwidth)
| Config | Chunk | GB/s |
|--------|-------|------|
| 2 ranks | 4 MB | ~117 |
| 4 ranks | 4 MB | ~117 |
| 2 ranks | 64 MB | ~87 |
| 4 ranks | 64 MB | ~100 |

**Key insight**: All-gather bandwidth saturates memory BW at ~117 GB/s (4 MB+). Small messages (< 256 KB) are latency-bound.

#### Simulation Ring All-Reduce (per-rank effective bandwidth)
| Config | Count | GB/s |
|--------|-------|------|
| 2 ranks | 4 MB | ~39 |
| 4 ranks | 4 MB | ~20 |
| 2 ranks | 64 MB | ~28 |

**Key insight**: Simulation all-reduce is 3–6× slower than all-gather because it requires both scatter and gather passes; real NCCL all-reduce would achieve close to memory BW on NVLink.

#### Col/Row Parallel Linear TFLOPS (tiled SGEMM, TP_TILE=16)
| Config | Col-parallel | Row-parallel |
|--------|-------------|-------------|
| M=1, N=4096, K=4096  | 0.055 TFLOPS | 0.034 TFLOPS |
| M=8, N=4096, K=4096  | 0.44 TFLOPS | 0.26 TFLOPS |
| M=32, N=4096, K=4096 | 0.94 TFLOPS | 0.64 TFLOPS |
| M=128, N=4096, K=4096 | 0.61 TFLOPS | 0.77 TFLOPS |
| M=32, N=11008, K=4096 | 0.55 TFLOPS | 0.83 TFLOPS |

**Key insight**: Decode (M=1) is heavily memory-bound; tiled SGEMM helps only when M≥32. cuBLAS / Tensor Core GEMM would be 5–10× faster here. The custom SGEMM demonstrates the pattern; production would use cuBLAS.

### Deliverables
* `src/kernels/transformer/tensor_parallel_nccl.cu` — NCCL collectives + stubs
* `src/kernels/transformer/tensor_parallel.cu` — col/row parallel linear (tiled SGEMM)
* `CMakeLists.txt` — NCCL detection + HAVE_NCCL propagation
* `tests/test_tensor_parallel.cpp` — 7 pass / 3 skip (NCCL, no libnccl)
* `tests/test_tensor_parallel_multiprocess.py` — torchrun harness (4 tests)
* `benchmarks/benchmark_tensor_parallel.cpp` — sim BW + TFLOPS; NCCL section guarded
* `src/python/pybind_transformer.cpp` — col/row parallel linear + NCCL Python API

---

## Phase 14: SSM / Mamba Kernels ✅ COMPLETE

### Deliverables
* `src/kernels/transformer/selective_scan.cu` — ZOH selective scan (N_state ≤ 32)
* `src/kernels/transformer/depthwise_conv1d.cu` — causal depthwise conv1d
* `src/kernels/transformer/rmsnorm.cu` — fused RMSNorm (normalize + scale)
* `tests/test_selective_scan.cpp` — 5 tests; all pass
* `tests/test_mamba_ops.cpp` — 8 tests (4 conv1d + 4 rmsnorm); all pass
* `benchmarks/benchmark_selective_scan.cpp` — selective scan / conv1d / rmsnorm throughput
* `src/python/pybind_transformer.cpp` — `selective_scan()`, `depthwise_conv1d()`, `rmsnorm()` added
* `src/python/tests/test_mamba_bindings.py` — 17 tests; all pass

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

Focus is on understanding, experimentation, and measurable optimization.

---

# 8. Notes

* Prioritize clarity over premature optimization
* Always validate correctness before optimizing
* Profile before making assumptions
* **pybind11**: When returning numpy arrays from C++, avoid `py::array_t` with pointer + shape (causes stride=0 bug). Use `std::vector` and list conversion, or `py::array_t` with vector (copy).
* **PyPI release**: `gh workflow run release.yml -f version_type=patch|minor|major` — workflow in `.github/workflows/release.yml`; requires `testpypi_token` and `pypi_token` secrets.
