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

### Deferred ⏸️
- [ ] **Transformer/LLM inference optimizations for vLLM integration**
  - Attention kernel optimization
  - KV-cache management
  - PagedAttention implementation
  - Continuous batching support

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

## Phase 8: Integration Enhancements - PENDING

- [ ] **Async Stream Operations**
  - Implement stream-based async execution
  - Overlap memory transfer with compute
  - Profile pipeline throughput

- [ ] **Unified Memory**
  - Add unified memory kernel variants
  - Simplify memory management
  - Test with memory oversubscription

- [ ] **Multi-Stream Pipeline**
  - Run preprocessing + inference concurrently
  - Use separate CUDA streams
  - Benchmark end-to-end latency

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
│   └── inference/       # Inference optimization kernels
│       ├── conv_int8.cu
│       ├── bn_folding.cu
│       └── conv_activation_fusion.cu
```

---

**Note:** Follow the methodology from `AGENTS.md`: change one variable at a time, compare against baselines, and document everything rigorously.
