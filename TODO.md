# TODO List for kernel-craft

## Phase 10: ML Inference & TensorRT Support (CNN-First) - MOSTLY COMPLETE ✅

### Completed ✅
- [x] **INT8 quantized convolution kernel** (`src/kernels/inference/conv_int8.cu`)
  - Naive and tiled implementations
  - Quantization/dequantization helpers
  - Scale factor computation
  - Tests: `tests/test_conv_int8.cpp`

- [x] **Batch normalization folding** (`src/kernels/inference/bn_folding.cu`)
  - Fold BN parameters into conv weights/bias
  - CPU reference implementation for verification
  - Tests: `tests/test_bn_folding.cpp`

- [x] **Conv + activation fusion** (`src/kernels/inference/conv_activation_fusion.cu`)
  - Fused conv+ReLU (naive and tiled)
  - Fused conv+LeakyReLU
  - Fused conv+Sigmoid
  - Tests: `tests/test_conv_activation_fusion.cpp`

- [x] **TensorRT plugin wrappers** (`src/tensorrt/`)
  - Plugin interface header (`plugin_wrapper.h`)
  - Plugin implementation (`plugin_wrapper.cpp`)
  - ConvInt8Plugin and ConvReluPlugin
  - Plugin creator classes

- [x] **TensorRT integration documentation** (`docs/tensorrt_integration.md`)
  - Build instructions
  - Usage examples
  - Deployment pipeline
  - Performance considerations

### Remaining Tasks
- [x] Add Python helper scripts to export folded weights for TensorRT engine building
- [x] Add C++ and Python tests for all new inference components (currently have C++ tests)
- [x] Create TensorRT integration examples in `examples/tensorrt/`

### Deferred → Phase 11 ✅ Planned

---

## Phase 11: Transformer/LLM Inference Optimizations (vLLM Integration) - COMPLETE ✅

### 11.1 FlashAttention Kernels
- [x] **Multi-Head Attention** (`src/kernels/transformer/flash_attention.cu`)
  - Tiled Q/K/V with online softmax (no full N×N matrix materialization)
  - Causal masking support; GQA/MQA via H_kv parameter
  - Launcher: `launch_flash_attention(Q, K, V, O, B, H, H_kv, N, d, causal, stream)`
  - Tests: `tests/test_flash_attention.cpp` — 5 tests (MHA, GQA, MQA, causal, d=128)
  - Benchmark: `benchmarks/benchmark_flash_attention.cpp`
- [x] **Grouped-Query / Multi-Query Attention** — unified in `flash_attention.cu`
  - GQA: H_kv < H; MQA: H_kv = 1 (Llama 2/3, Mistral, Falcon architectures)

### 11.2 vLLM PagedAttention
- [x] **PagedAttention kernel** (`src/kernels/transformer/paged_attention.cu`)
  - Non-contiguous KV-cache via block table [B, max_blocks]
  - Variable sequence lengths per batch item
  - Launcher: `launch_paged_attention(Q, block_table, K_pool, V_pool, O, seq_lens, ...)`
  - Tests: `tests/test_paged_attention.cpp` — 3 tests
  - Benchmark: `benchmarks/benchmark_paged_attention.cpp` (page sizes 16/32/64)
- [x] **TensorRT PagedAttention plugin** (`src/tensorrt/paged_attention_plugin.cpp`)
  - IPluginV2DynamicExt + PagedAttentionPluginCreator

### 11.3 LLM Quantization Kernels
- [x] **INT4 weight dequant + GEMV** (`src/kernels/transformer/quant_int4.cu`)
  - GPTQ/AWQ-style: 2×INT4 packed per byte, per-group FP32 scales
  - `launch_dequantize_int4` and `launch_gemv_int4`
  - Tests: `tests/test_quant_int4.cpp` — 3 tests
- [x] **FP8 activation quantization** (`src/kernels/transformer/fp8_quant.cu`)
  - E4M3 format (`__nv_fp8_e4m3`, CUDA 12.0), per-token and per-tensor scaling
  - SmoothQuant channel-wise smoothing kernel
  - Tests: `tests/test_fp8_quant.cpp` — 4 tests

### 11.4 Advanced Inference Features
- [x] **Speculative decoding** (`src/kernels/transformer/speculative_decoding.cu`)
  - Draft token verification via rejection sampling (Chen et al., 2023)
  - `launch_verify_draft_tokens` + `launch_compute_prefix_length`
  - Tests: `tests/test_speculative_decoding.cpp` — 5 tests
- [x] **Tensor parallelism primitives** (`src/kernels/transformer/tensor_parallel.cu`)
  - Ring all-reduce (reduce-scatter + all-gather) and all-gather
  - Single-GPU simulation with multiple device buffers as virtual ranks
  - Tests: `tests/test_tensor_parallel.cpp` — 4 tests

### 11.5 Python Bindings & vLLM Plugin
- [x] **pybind11 transformer module** (`src/python/pybind_transformer.cpp`)
  - Exposes: `flash_attention()`, `paged_attention()`, `quant_int4_dequant()`,
    `fp8_quantize()`, `fp8_dequantize()`, `speculative_decode()`
  - CMake target: `kernel_craft_transformer`
- [x] **vLLM custom ops integration** (`src/tensorrt/vllm_plugin_wrapper.cpp`)
  - `torch.ops.kernel_craft.flash_attention` and `paged_attention` when HAVE_TORCH
- [x] **Python tests** (`src/python/tests/test_transformer_bindings.py`)
  - 14 pytest tests for all Python-exposed transformer kernels

### CMake
- [x] 6 test executables added and registered with CTest
- [x] 3 benchmark executables added and registered with CTest
- [x] `kernel_craft_transformer` pybind11 extension added

### Results
- FlashAttention: ~700-980 Gflops on RTX 4070 (see CLAUDE.md)
- PagedAttention: 0.12-0.48ms for seq=256-1024, page sizes 16/32/64
- INT4 dequantization: ~195-210 GB/s; GEMV: ~210-226 Gflops
- FP8 quantization: ~116-210 GB/s (per-token), dequant ~183-278 GB/s
- Speculative decoding: correct rejection sampling with residual distribution
- Python bindings: 17 pytest tests all pass (test_transformer_bindings.py)

---

## Phase 12: Full vLLM Integration — IN PROGRESS 🔄

### Completed ✅
- [x] **Python transformer bindings test suite** (`src/python/tests/test_transformer_bindings.py`)
  - 17 pytest tests pass: FlashAttention (6), PagedAttention (2), INT4 (2), FP8 (4), speculative (3)
  - All tests confirmed passing 2026-05-16

- [x] **PyTorch custom ops bridge** (`src/python/kernel_craft_torch_ops.py`)
  - ctypes wrapper loading `libkernels.so` — zero GPU→CPU roundtrip
  - Exposes: `flash_attention`, `paged_attention`, `int4_dequantize`, `int4_gemv`,
    `fp8_quantize`, `fp8_dequantize`, `speculative_verify`
  - Optional `torch.library` registration under `torch.ops.kernel_craft.*` namespace
  - `KERNEL_CRAFT_LIB` env var overrides library search path

- [x] **vLLM AttentionBackend** (`src/python/kernel_craft_vllm_backend.py`)
  - `KernelCraftAttentionBackend(AttentionBackend)` implementing vLLM 0.5.x interface
  - `KernelCraftAttentionImpl(AttentionImpl)`:
    - Prefill → `launch_flash_attention` (causal, GQA/MQA support)
    - Decode → `launch_paged_attention` (paged KV-cache, block table)
  - `KernelCraftMetadata` + `KernelCraftMetadataBuilder` for vLLM scheduling
  - `swap_blocks` / `copy_blocks` for KV-cache management (offloading, beam search)
  - Activation: `VLLM_ATTENTION_BACKEND=kernel_craft` or `register()`
  - Guarded: no-op import when vLLM is not installed

- [x] **End-to-end benchmark** (`benchmarks/benchmark_vllm_e2e.py`)
  - Kernel-only mode (runs without vLLM): FlashAttention + PagedAttention TFLOPS/BW
  - Full mode (requires vLLM): TTFT, tokens/s, peak VRAM vs default backend
  - `--model`, `--quantization`, `--batch-sizes`, `--backends` CLI flags
  - Reports written to `reports/benchmark_vllm_e2e.txt`

- [x] **vLLM backend tests** (`src/python/tests/test_vllm_backend.py`)
  - 16 tests: torch ops (10) + vLLM interface (6); gracefully skip if torch/vLLM absent
  - TestTorchOpsFlashAttention, TestTorchOpsPagedAttention, TestTorchOpsFP8,
    TestTorchOpsSpeculative, TestVLLMBackendInterface

- [x] **vllm_plugin_wrapper.cpp** (`src/tensorrt/vllm_plugin_wrapper.cpp`)
  - Complete: `TORCH_LIBRARY` + `TORCH_LIBRARY_IMPL` registration for C++ path
  - Compiles with `-DHAVE_TORCH`; falls back to stub otherwise

- [x] **paged_attention_plugin.cpp** (`src/tensorrt/paged_attention_plugin.cpp`)
  - Complete TensorRT IPluginV2DynamicExt implementation
  - Compiles with `-DHAVE_TENSORRT`; stub otherwise

- [x] **pyproject.toml extras** (`src/python/pyproject.toml`)
  - Added `[vllm]` and `[torch]` optional dependency groups
  - Pin: `vllm>=0.5.0`, `torch>=2.1`

### Completed with venv (torch 2.11+cu130 + vLLM 0.21.0)
- [x] Install vLLM: `src/python/venv` has torch 2.11+cu130 + vLLM 0.21.0
- [x] Run `test_vllm_backend.py`: **17/17 pass** — all torch-ops + vLLM interface tests
- [x] Updated backend to vLLM 0.21.0 v1 API (`vllm.v1.attention.backend`)
- [x] Full Python test suite: **89 pass / 0 skip** with venv

### Remaining
- [ ] Run full vLLM E2E benchmark: `python benchmarks/benchmark_vllm_e2e.py`
  - Requires model download (e.g. `facebook/opt-125m`)
  - Run with venv: `src/python/venv/bin/python benchmarks/benchmark_vllm_e2e.py`

### Deferred (multi-GPU; WSL2 has no GPU peer-to-peer)
- [ ] Replace tensor_parallel.cu simulation with torch.distributed/NCCL
- [ ] Test `tensor_parallel_size=2` on multi-GPU system

---

## Phase 4: End‑to‑End GPU Pipeline (Weeks 7‑8) - COMPLETE ✅
- [x] **Profile complete pipeline**
  - Use Nsight Systems to visualize data movement and kernel overlaps.

---

## Phase 5: Python Integration (Weeks 9‑10) - COMPLETE ✅
- [x] **Create PyPI release workflow**
  - GitHub Actions workflow to build and publish
  - Support TestPyPI and PyPI deployments

---

## Phase 6: Feature Extensions - COMPLETE ✅

- [x] **3D Convolution**
  - Implement volumetric convolution kernel for 3D input tensors (D×H×W)
  - Handle 3D kernels (D×H×W)
  - Test with medical imaging data (MRI, CT)

- [x] **Dilated Convolution**
  - Implement atrous convolution with configurable dilation rate
  - Support dilation rates: 1, 2, 4, 8
  - Test receptive field expansion

- [x] **Transposed Convolution**
  - Implement deconvolution/upconvolution kernel
  - Support stride-based upsampling
  - Test with output_padding parameter

- [x] **Grouped Convolution**
  - Implement ResNeXt-style grouped convolution
  - Support configurable group count
  - Benchmark memory savings vs dense

---

## Phase 7: Performance Infrastructure - COMPLETE ✅

- [x] **CUDA Graphs Integration**
  - Wrap pipeline kernels in CUDA graphs
  - Capture and launch graph for batch processing
  - Benchmark vs separate kernel launches

- [x] **Memory Pool**
  - Implement custom allocator with reusable buffers
  - Profile allocation overhead reduction
  - Integrate with tiled convolution

- [x] **Mixed Precision Support**
  - Add FP16 kernel variants
  - Add TF32 support on Ampere+
  - Benchmark vs FP32 baseline

- [x] **Persistent Kernels**
  - Refactor for persistent kernel mode
  - Reuse kernel across batches
  - Measure latency reduction

- [x] **Comments**
  - Add more comments to all .cu files, all new tests and benchmarks within src/performance in Doxygen style.

---

## Phase 8: Integration Enhancements - COMPLETE ✅

- [x] **Async Stream Operations** (`src/performance/async_streams.cu`)
  - Double-buffered batch processing with ping-pong device buffers
  - Pinned host memory for true async H2D/D2H overlap
  - Tests: `tests/test_async_streams.cpp` (4 tests)
  - Benchmark: `benchmarks/benchmark_async_streams.cpp`

- [x] **Unified Memory** (`src/performance/unified_memory.cu`)
  - `unified_conv_naive` and `unified_conv_tiled` using cudaMallocManaged
  - `unified_prefetch_to_gpu` / `unified_prefetch_to_cpu` via cudaMemPrefetchAsync
  - Tests: `tests/test_unified_memory.cpp` (5 tests)
  - Benchmark: `benchmarks/benchmark_unified_memory.cpp`

- [x] **Multi-Stream Pipeline** (`src/performance/multi_stream_pipeline.cu`)
  - Concurrent preprocess (resize+normalize) and inference (conv+relu) streams
  - Inter-stream synchronization via cudaEvent
  - Tests: `tests/test_multi_stream_pipeline.cpp` (3 tests)
  - Benchmark: `benchmarks/benchmark_multi_stream_pipeline.cpp`

---

## Phase 9: Additional Framework Support (Optional) - PENDING

**Option1**: Add JAX support
- [ ] Research JAX array interop with pybind11
- [ ] Add `conv_naive_jax()` and `conv_tiled_jax()` overloads
- [ ] Test with JAX arrays on GPU

**Option2**: Add ONNX Runtime support
- [ ] Create ONNX execution provider custom kernel
- [ ] Integrate with ONNX Runtime CUDA EP
- [ ] Benchmark vs native implementation

**Option3**: Add TensorFlow support
- [ ] Add TensorFlow tensor overloads
- [ ] Use TF's memory management for GPU tensors
- [ ] Test with TF Keras models

---

## Directory Structure (After Reorganization)

```
src/
├── kernels/
│   ├── core/            # Core convolution implementations
│   │   ├── conv_naive.cu
│   │   └── conv_tiled.cu
│   ├── variants/        # Specialized convolution variants
│   │   ├── conv_grouped.cu
│   │   ├── conv_transposed.cu
│   │   ├── conv_dilated.cu
│   │   └── conv3d.cu
│   ├── inference/       # Inference optimization kernels
│   │   ├── conv_int8.cu
│   │   ├── bn_folding.cu
│   │   └── conv_activation_fusion.cu
│   └── transformer/     # Transformer/LLM inference kernels (Phase 11)
│       ├── flash_attention.cu
│       ├── gqa_attention.cu
│       ├── paged_attention.cu
│       ├── quant_int4.cu
│       ├── fp8_quant.cu
│       ├── speculative_decoding.cu
│       └── tensor_parallel.cu
```

---

**Note:** Follow the methodology from `AGENTS.md`: change one variable at a time, compare against baselines, and document everything rigorously.
