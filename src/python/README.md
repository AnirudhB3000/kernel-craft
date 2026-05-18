# python/ — Python Bindings & vLLM Integration

Python interface for kernel-craft: pybind11 extension module, ctypes bridge to `libkernels.so`, and the vLLM attention backend.

## Files

| File | Description |
|------|-------------|
| `pybind_cuda.cpp` | pybind11 bindings for CNN kernels: conv, INT8, BN folding, fused ops |
| `pybind_transformer.cpp` | pybind11 bindings for transformer kernels: flash attention, paged attention, INT4/FP8 quant, speculative decoding |
| `kernel_craft_torch_ops.py` | ctypes bridge — calls `libkernels.so` launchers directly from Python via `data_ptr()`, no GPU→CPU roundtrip |
| `kernel_craft_vllm_backend.py` | `KernelCraftAttentionBackend` implementing the vLLM 0.21.0 v1 API (`vllm.v1.attention.backend`) |
| `pyproject.toml` | Package metadata; setuptools build backend |
| `tests/` | pytest test suite (see `tests/README.md`) |

## Installation

### From PyPI (pre-built wheel)

```bash
pip install kernel-craft
```

Requires: Python 3.11–3.12, numpy ≥ 1.20, CUDA runtime.

### From CMake build tree

```bash
# Build the .so first
mkdir build && cd build && cmake .. && make kernel_craft_python

# Install in editable mode
pip install -e src/python
```

### vLLM backend (requires specific torch + vLLM versions)

```bash
cd src/python
python -m venv venv
source venv/bin/activate
pip install torch==2.11.0 --extra-index-url https://download.pytorch.org/whl/cu130
pip install vllm==0.21.0
pip install -e .
```

## Usage

### Convolution kernels

```python
import kernel_craft_python as kc
import numpy as np

img = np.random.randn(512, 512).astype(np.float32)
kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)

out = kc.conv_naive(img, kernel)
out = kc.conv_tiled(img, kernel, tile_w=8, tile_h=8)

# PyTorch CUDA tensors are accepted directly (no copy)
import torch
img_t = torch.rand(512, 512, device='cuda')
out_t = kc.conv_naive(img_t, kernel)
```

### Inference CNN kernels

```python
scale = kc.compute_quantization_scale(img)
out_int8 = kc.conv_int8_naive(img, kernel, scale, scale, 1.0)

# BN folding: returns (folded_weights, folded_bias)
folded_w, folded_b = kc.bn_folding(
    conv_weights, conv_bias, bn_mean, bn_var, bn_gamma, bn_beta
)

out = kc.conv_relu(img, kernel, tiled=True)
```

### Transformer / LLM kernels

```python
B, H, N, d = 1, 8, 512, 64
Q = np.random.randn(B, H, N, d).astype(np.float32)
K = np.random.randn(B, H, N, d).astype(np.float32)
V = np.random.randn(B, H, N, d).astype(np.float32)

out = kc.flash_attention(Q, K, V, causal=True)

# INT4 dequantization
weights_fp32 = kc.quant_int4_dequant(packed_weights, scales, group_size=128)

# FP8 round-trip
q_data, scale = kc.fp8_quantize(tensor, mode="per_token")
recovered = kc.fp8_dequantize(q_data, scale, mode="per_token")

# Speculative decoding verification
accepted_mask = kc.speculative_decode(draft_probs, target_probs, draft_tokens)
```

### vLLM backend

```python
import kernel_craft_python.kernel_craft_vllm_backend as kb
kb.register()  # registers KernelCraftAttentionBackend with vLLM's registry
```

Or set the environment variable before starting a vLLM server:

```bash
VLLM_ATTENTION_BACKEND=kernel_craft python -m vllm.entrypoints.openai.api_server ...
```

### ctypes bridge (torch ops, no recompile)

```python
from kernel_craft_torch_ops import KernelCraftOps
import torch

ops = KernelCraftOps()   # loads libkernels.so from CMake build dir
Q = torch.randn(1, 8, 512, 64, device='cuda')
out = ops.flash_attention(Q, K, V, causal=True)
```

## API Reference — CNN Kernels

| Function | Signature | Returns |
|----------|-----------|---------|
| `conv_naive` | `(input, kernel)` | ndarray or Tensor |
| `conv_tiled` | `(input, kernel, tile_w=8, tile_h=8)` | ndarray or Tensor |
| `conv_int8_naive` | `(input, kernel, in_scale, k_scale, out_scale)` | ndarray |
| `bn_folding` | `(w, b, mean, var, gamma, beta, eps=1e-5)` | `(ndarray, ndarray)` |
| `conv_relu` | `(input, kernel, tiled=False)` | ndarray |
| `compute_quantization_scale` | `(data)` | float |

## API Reference — Transformer Kernels

| Function | Signature | Returns |
|----------|-----------|---------|
| `flash_attention` | `(Q, K, V, causal=False)` | ndarray |
| `paged_attention` | `(Q, block_table, K_pool, V_pool, seq_lens)` | ndarray |
| `quant_int4_dequant` | `(packed, scales, group_size=128)` | ndarray |
| `fp8_quantize` | `(tensor, mode="per_token")` | `(ndarray, ndarray)` |
| `fp8_dequantize` | `(q_data, scale, mode="per_token")` | ndarray |
| `speculative_decode` | `(draft_probs, target_probs, draft_tokens)` | ndarray |

## Publishing to PyPI

Use the GitHub Actions workflow — do not run twine manually:

```bash
gh workflow run release.yml -f version_type=patch   # or minor / major
```

Requires `testpypi_token` and `pypi_token` repository secrets. The workflow builds the C++ extension, runs the full test suite, uploads to TestPyPI, then gates on the `pypi-production` environment before uploading to PyPI.
