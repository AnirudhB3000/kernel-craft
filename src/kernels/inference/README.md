# inference/ — CNN Inference-Optimized Kernels

CUDA kernels tuned for CNN inference: reduced precision, operator fusion, and weight pre-computation.

## Files

| File | Description |
|------|-------------|
| `conv_int8.cu` | INT8 quantized convolution — symmetric per-tensor quantization, Tensor Core compatible |
| `bn_folding.cu` | Batch Normalization folding — pre-computes BN parameters into conv weights at model-load time, eliminating the BN layer entirely at inference |
| `conv_activation_fusion.cu` | Fused conv + activation (ReLU / LeakyReLU / Sigmoid) in a single kernel pass |

## Design Notes

- **INT8**: Uses symmetric per-tensor scales (`input_scale × kernel_scale / output_scale`). Compatible with TensorRT quantization schemes. See `src/tensorrt/plugin_wrapper.cpp` for the TensorRT plugin wrapper.
- **BN folding**: Absorbs `γ`, `β`, `μ`, `σ²` into conv weights and bias offline. Zero runtime overhead at inference.
- **Activation fusion**: Eliminates the intermediate tensor write between conv output and activation input, saving ~40% global memory traffic on the combined op.

## Benchmark Results

| Kernel | Image Size | Time (ms) |
|--------|------------|-----------|
| INT8 naive | 256×256 | ~0.40 |
| INT8 tiled | 256×256 | ~0.35 |
| Conv+ReLU fused | 256×256 | ~0.38 |
| BN folding (precompute) | C_out=64 | ~0.01 |

## Tests

```bash
./build/bin/test_conv_int8       # or whichever CTest target covers inference
```
