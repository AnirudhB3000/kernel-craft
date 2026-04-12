# kernel-craft Python API

CUDA convolution kernels exposed to Python with numpy and PyTorch support.

## Installation

### Option 1: Build with Python (recommended for distribution)

```bash
cd src/python
python -m build
```

The `.so` file will be at `src/python/build/kernel_craft_python.cpython-*.so`.

### Option 2: Build with CMake

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
sys.path.insert(0, 'src/python/build')

import kernel_craft_python as kc
import numpy as np

# Input: 2D float32 numpy array
input = np.random.randn(256, 256).astype(np.float32)
kernel = np.random.randn(3, 3).astype(np.float32)

# Naive convolution
out = kc.conv_naive(input, kernel)  # -> np.ndarray

# Tiled convolution with configurable tile size
out = kc.conv_tiled(input, kernel, tile_w=8, tile_h=8)  # -> np.ndarray
```

## Version

```python
import kernel_craft_python as kc
print(kc.__version__)  # "0.1.0"
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

| Function | Input Type | Output Type |
|----------|-----------|--------------|
| `conv_naive(input, kernel)` | np.ndarray or Tensor | np.ndarray or Tensor |
| `conv_tiled(input, kernel, tile_w, tile_h)` | np.ndarray or Tensor | np.ndarray or Tensor |

### Parameters

- `input`: Input image (2D, float32)
- `kernel`: Convolution kernel (2D, float32, odd dimension)
- `tile_w`: Tile width for tiled convolution (default: 8)
- `tile_h`: Tile height for tiled convolution (default: 8)

### Supported Tile Sizes

- 8x8 (default, best overall performance)
- 16x16
- 32x32

### Error Handling

All functions raise `RuntimeError` with descriptive messages for:
- Invalid input dimensions (must be 2D)
- Invalid kernel dimensions (must be 2D, square, odd-sized)
- Invalid dtype (must be float32)

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