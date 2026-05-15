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

## Phase 11: Transformer/LLM Inference Optimizations (vLLM Integration) - PLANNED

### 11.1 FlashAttention Kernels
- [ ] **Multi-Head Attention** (`src/kernels/transformer/flash_attention.cu`)
  - Tiled Q/K/V with online softmax (no full N×N matrix materialization)
  - Causal masking support
  - Launcher: `launch_flash_attention(Q, K, V, O, B, H, N, d, causal, stream)`
  - Tests: `tests/test_flash_attention.cpp`
  - Benchmark: `benchmarks/benchmark_flash_attention.cpp`
  - Target: >80% theoretical memory bandwidth
- [ ] **Grouped-Query / Multi-Query Attention** (`src/kernels/transformer/gqa_attention.cu`)
  - Fewer KV heads than Q heads (Llama 2/3, Mistral, Falcon architectures)
  - Tests included in `tests/test_flash_attention.cpp`

### 11.2 vLLM PagedAttention
- [ ] **PagedAttention kernel** (`src/kernels/transformer/paged_attention.cu`)
  - Non-contiguous KV-cache via block table (`int* block_table[batch, max_blocks]`)
  - Variable sequence lengths; prefix caching (skip duplicate block pointers)
  - Launcher: `launch_paged_attention(Q, block_table, K_pool, V_pool, O, seq_lens, ...)`
  - Tests: `tests/test_paged_attention.cpp` (correctness vs contiguous FlashAttention)
  - Benchmark: `benchmarks/benchmark_paged_attention.cpp` (page sizes 16/32/64)
- [ ] **TensorRT PagedAttention plugin** (`src/tensorrt/paged_attention_plugin.cpp`)
  - Custom TensorRT plugin following `plugin_wrapper.cpp` pattern

### 11.3 LLM Quantization Kernels
- [ ] **INT4 weight dequant + GEMM** (`src/kernels/transformer/quant_int4.cu`)
  - GPTQ/AWQ-style: 2×INT4 packed per byte, per-group (128) FP16 scales
  - Unpacking via bit shifts; Tensor Core path on Ampere+
  - Tests: `tests/test_quant_int4.cpp` (dequant→FP32 max relative error <1%)
- [ ] **FP8 activation quantization** (`src/kernels/transformer/fp8_quant.cu`)
  - E4M3 format (`__nv_fp8_e4m3`, CUDA 11.8+), per-token and per-channel scaling
  - SmoothQuant-compatible; saturate out-of-range values
  - Tests: `tests/test_fp8_quant.cpp` (round-trip quantize→dequantize error bounds)

### 11.4 Advanced Inference Features
- [ ] **Speculative decoding** (`src/kernels/transformer/speculative_decoding.cu`)
  - Draft token verification via rejection sampling
  - Inputs: `[draft_probs, target_probs, draft_tokens]` → `accepted_mask`
  - Tests: `tests/test_speculative_decoding.cpp` (mask correctness at varying acceptance rates)
  - Target: >1.5x decode speedup
- [ ] **Tensor parallelism primitives** (`src/kernels/transformer/tensor_parallel.cu`)
  - All-reduce (ring-based, multi-GPU) and all-gather
  - NCCL wrapper when available; manual ring-reduce fallback
  - Tests: `tests/test_tensor_parallel.cpp` (single-GPU stream simulation + multi-GPU if available)
- [ ] **Continuous batching** — dynamic `seq_lens` array in PagedAttention launcher

### 11.5 Python Bindings & vLLM Plugin
- [ ] **pybind11 transformer module** (`src/python/pybind_transformer.cpp`)
  - Exposes: `flash_attention()`, `paged_attention()`, `quant_int4_dequant()`, `fp8_quantize()`, `speculative_decode()`
  - Follows pattern of `src/python/pybind_cuda.cpp`
- [ ] **vLLM custom ops integration** (`src/tensorrt/vllm_plugin_wrapper.cpp`)
  - vLLM-compatible custom op interface
- [ ] **Python tests** (`src/python/tests/test_transformer_bindings.py`)
  - pytest suite for all Python-exposed transformer kernels vs numpy reference

### CMake
- [ ] Add test executables for all 6 new test files
- [ ] Add benchmark executables for 3 new benchmark files
- [ ] Add `pybind_transformer.cpp` to `kernel_craft_python` extension target
- [ ] Optional NCCL detection (mirror existing optional TensorRT block)

### Success Criteria
- FlashAttention: >80% theoretical memory bandwidth on A100
- PagedAttention: integrates with vLLM without performance regression
- INT4 quantization: <1% accuracy drop on standard benchmarks
- Speculative decoding: >1.5x speedup for supported models

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
