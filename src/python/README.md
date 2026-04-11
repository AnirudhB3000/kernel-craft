# kernel-craft Python API

CUDA convolution kernels exposed to Python with numpy and PyTorch support.

## Installation

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

```python
import kernel_craft
import numpy as np

# Input: 2D float32 numpy array
input = np.random.randn(256, 256).astype(np.float32)
kernel = np.random.randn(3, 3).astype(np.float32)

# Naive convolution
out = kernel_craft.conv_naive(input, kernel)  # -> np.ndarray

# Tiled convolution with configurable tile size
out = kernel_craft.conv_tiled(input, kernel, tile_w=8, tile_h=8)  # -> np.ndarray
```

## PyTorch Tensors

```python
import torch
import kernel_craft

# Input: 2D float32 PyTorch tensor on CUDA
input = torch.rand(256, 256, dtype=torch.float32, device='cuda')
kernel = torch.rand(3, 3, dtype=torch.float32, device='cuda')

# Naive convolution
out = kernel_craft.conv_naive(input, kernel)  # -> torch.Tensor on GPU

# Tiled convolution
out = kernel_craft.conv_tiled(input, kernel, tile_w=16, tile_h=16)  # -> torch.Tensor on GPU
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