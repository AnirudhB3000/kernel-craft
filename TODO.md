# TODO List for kernel-craft


## Phase 4: End‑to‑End GPU Pipeline (Weeks 7‑8)
- [ ] **Profile complete pipeline**
  - Use Nsight Systems to visualize data movement and kernel overlaps.

## Phase 5: Python Integration (Weeks 9‑10)
- [x] **Create PyPI release workflow**
  - GitHub Actions workflow to build and publish
  - Support TestPyPI and PyPI deployments

## Phase 6: Feature Extensions

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

## Phase 7: Performance Infrastructure

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

## Phase 8: Integration Enhancements

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

## Phase 9: Additional Framework Support (Optional)

**Option 1**: Add JAX support
- [ ] Research JAX array interop with pybind11
- [ ] Add `conv_naive_jax()` and `conv_tiled_jax()` overloads
- [ ] Test with JAX arrays on GPU

**Option 2**: Add ONNX Runtime support
- [ ] Create ONNX execution provider custom kernel
- [ ] Integrate with ONNX Runtime CUDA EP
- [ ] Benchmark vs native implementation

**Option 3**: Add TensorFlow support
- [ ] Add TensorFlow tensor overloads
- [ ] Use TF's memory management for GPU tensors
- [ ] Test with TF Keras models

---

**Note:** Follow the methodology from `AGENTS.md`: change one variable at a time, compare against baselines, and document everything rigorously.
