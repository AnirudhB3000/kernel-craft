# kernels/ — CUDA Kernel Implementations

All CUDA kernels organised by domain and use case.

## Subdirectories

| Directory | Contents |
|-----------|----------|
| `core/` | Foundational 2D convolution: naive and tiled |
| `variants/` | Extended convolution types: 3D, dilated, transposed, grouped |
| `inference/` | CNN inference optimizations: INT8 quantization, BN folding, activation fusion |
| `transformer/` | Transformer/LLM kernels: FlashAttention, PagedAttention, INT4/FP8 quantization, speculative decoding, tensor parallelism |

## Naming Conventions

- `conv_*.cu` — 2D convolution variants
- `*_int8.cu` / `*_int4.cu` / `fp8_*.cu` — quantized precision kernels
- Every kernel file has a matching test in `tests/test_<name>.cpp`
