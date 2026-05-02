# kernel-craft Python API

CUDA convolution kernels for ML training-time optimization, exposed to Python with numpy and PyTorch support.

## Installation

### Option 1: pip (recommended for users)

```bash
pip install kernel-craft
```

Requires:
- Python 3.11 - 3.12
- numpy >= 1.20
- CUDA runtime (for GPU execution)

### Option 2: Build with Python (recommended for distribution)

```bash
cd src/python
python -m build
```

The `.so` file will be at `src/python/build/kernel_craft_python.cpython-*.so`.

### Option 3: Build with CMake

```bash
cd /path/to/kernel-craft
mkdir build && cd build
cmake ..
make kernel_craft_python
```

The module will be at `src/python/build/kernel_craft_python.cpython-*.so`.

## Usage

```python
import sys
sys.path.insert(0, 'src/python/build")

import kernel_craft_python as kc
import numpy as np

# Input: 2D float32 numpy array
input = np.random.randn(256, 256).astype(np.float32)
kernel = np.random.randn(3, 3).astype(np.float32)

# Naive convolution
out = kc.conv_naive(input, kernel)  # -> np.ndarray

# Tiled convolution with configurable tile size
out = kc.conv_tiled(input, kernel, tile_w=8, tile_h=8)  # -> np.ndarray

# Phase 10: Inference kernels
# INT8 quantized convolution
input_scale = kc.compute_quantization_scale(input)
kernel_scale = kc.compute_quantization_scale(kernel)
out_int8 = kc.conv_int8_naive(input, kernel, input_scale, kernel_scale, 1.0)

# Batch Normalization folding (pre-compute for inference)
conv_weights = np.random.rand(64, 3, 3, 3).astype(np.float32)
conv_bias = np.random.rand(64).astype(np.float32)
bn_mean = np.random.rand(64).astype(np.float32)
bn_variance = np.random.rand(64).astype(np.float32) + 0.01
bn_gamma = np.random.rand(64).astype(np.float32) + 0.5
bn_beta = np.random.rand(64).astype(np.float32) - 0.05
folded_weights, folded_bias = kc.bn_folding(
    conv_weights, conv_bias, bn_mean, bn_variance, bn_gamma, bn_beta
)

# Fused Conv+ReLU
out_relu = kc.conv_relu(input, kernel, tiled=True)
```

## Version

```python
import kernel_craft
print(kernel_craft.__version__)  # "0.1.1"
```

Or via the module directly:

```python
import kernel_craft_python as kc
print(kc.__version__)  # "0.1.1"
```

## PyTorch Tensors

```python
import torch
import kernel_craft_python as kc

# Input: 2D float32 PyTorch tensor on CUDA
input = torch.rand(256, 256, dtype=torch.float32, device='cuda')
kernel = torch.rand(3, 3, dtype=torch.float32, device='cuda')

# Naive convolution
out = kc.conv_naive(input, kernel)  # -> torch.Tensor on GPU

# Tiled convolution
out = kc.conv_tiled(input, kernel, tile_w=16, tile_h=16)  # -> torch.Tensor on GPU
```

## API Reference

| Function | Input Type | Output Type | Description |
|----------|-----------|--------------|-------------|
| `conv_naive(input, kernel)` | np.ndarray or Tensor | np.ndarray or Tensor | Baseline convolution |
| `conv_tiled(input, kernel, tile_w, tile_h)` | np.ndarray or Tensor | np.ndarray or Tensor | Tiled convolution |
| `conv_int8_naive(input, kernel, input_scale, kernel_scale, output_scale)` | np.ndarray | np.ndarray | INT8 quantized |
| `bn_folding(conv_weights, conv_bias, bn_mean, bn_variance, bn_gamma, bn_beta, epsilon)` | np.ndarray | tuple(np.ndarray, np.ndarray) | BN folding |
| `conv_relu(input, kernel, tiled)` | np.ndarray | np.ndarray | Fused Conv+ReLU |
| `compute_quantization_scale(data)` | np.ndarray | float | INT8 scale |

### Parameters

- `input`: Input image (2D, float32)
- `kernel`: Convolution kernel (2D, float32, odd dimension)
- `tile_w`: Tile width for tiled convolution (default: 8)
- `tile_h`: Tile height for tiled convolution (default: 8)
- `input_scale`: Scale factor for INT8 quantization
- `kernel_scale`: Scale factor for INT8 quantization
- `output_scale`: Scale factor for INT8 dequantization
- `conv_weights`: 4D array [C_out, C_in, K_h, K_w]
- `conv_bias`: 1D array [C_out] or None
- `bn_mean`, `bn_variance`, `bn_gamma`, `bn_beta`: 1D arrays [C_out]
- `epsilon`: Small constant for numerical stability (default: 1e-5)
- `tiled`: Use tiled implementation for Conv+ReLU (default: False)

### Supported Tile Sizes

- 8x8 (default, best overall performance)
- 16x16
- 32x32

### Error Handling

All functions raise `RuntimeError` with descriptive messages for:
- Invalid input dimensions (must be 2D)
- Invalid kernel dimensions (must be 2D, square, odd-sized)
- Invalid dtype (must be float32)
- CUDA errors (kernel launch failures, memory errors)

### Quick Start with pip

```python
import kernel_craft
import numpy as np

input = np.random.randn(256, 256).astype(np.float32)
kernel = np.random.randn(3, 3).astype(np.float32)

# Works with both import styles
out = kernel_craft.conv_naive(input, kernel)  # Recommended

# Or with the internal module name
import kernel_craft_python as kc
out = kc.conv_tiled(input, kernel, tile_w=8, tile_h=8)
```

## Publishing to PyPI

```bash
# Build package
cd src/python
python -m build

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

## Requirements

- Python 3.11 - 3.12
- numpy >= 1.20
- CUDA Toolkit (for building, not for installed .so)