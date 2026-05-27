# TODO List for kernel-craft


---

### Future improvements and utilities:
**Utility 1**: Add JAX support
- [ ] Research JAX array interop with pybind11
- [ ] Add `conv_naive_jax()` and `conv_tiled_jax()` overloads
- [ ] Test with JAX arrays on GPU

**Utility 2**: Add ONNX Runtime support
- [ ] Create ONNX execution provider custom kernel
- [ ] Integrate with ONNX Runtime CUDA EP
- [ ] Benchmark vs native implementation

**Utility 3**: Add TensorFlow support
- [ ] Add TensorFlow tensor overloads
- [ ] Use TF's memory management for GPU tensors
- [ ] Test with TF Keras models
