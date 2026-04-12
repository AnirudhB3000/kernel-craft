# kernel-craft

CUDA kernels for machine learning systems optimization.

## Features

- **Naive Convolution** (`src/conv_naive.cu`) - Baseline 2D convolution with one thread per output pixel
- **Tiled Convolution** (`src/conv_tiled.cu`) - Optimized using shared memory tiling for better memory bandwidth
- **Sparse Convolution** (`src/custom_op.cu`) - Coordinate-list format for sparse inputs
- **Fused Pipeline** (`src/pipeline_fused.cu`) - Conv + BatchNorm + ReLU in single kernel
- **GPU Preprocessing** (`src/preprocess_gpu.cu`) - Resize, normalize, flip operations on GPU
- **Python Bindings** (`src/python/`) - pybind11 module exposing kernels to numpy and PyTorch

## Directory Structure

```
kernel-craft/
├── src/                     # CUDA kernel implementations
│   ├── conv_naive.cu        # Naive 2D convolution
│   ├── conv_tiled.cu       # Tiled 2D convolution
│   ├── custom_op.cu        # Sparse convolution
│   ├── pipeline_fused.cu   # Fused conv+batchnorm+relu
│   ├── pipeline_separate.cu # Separate pipeline kernels
│   ├── preprocess_gpu.cu   # GPU preprocessing (resize, normalize, flip)
│   └── python/
│       ├── pybind_cuda.cpp # Python bindings
│       ├── build/          # Compiled .so module
│       └── tests/          # Python tests
├── benchmarks/              # Benchmark programs
├── tests/                   # C++ unit tests
├── examples/
│   ├── cpp/                 # C++ examples
│   └── python/              # Python examples
├── data/
│   ├── sample_images/       # Input images
│   └── outputs/             # Output images
├── CMakeLists.txt           # Build configuration
├── README.md                # This file
└── AGENTS.md                # Project guidelines
```

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## Testing

```bash
make run_tests          # Build and run all C++ tests
ctest --output-on-failure
```

Run Python tests:
```bash
cd src/python
python -m pytest tests/ -v
```

## Python Usage

### Install

Build the Python module:
```bash
cd src/python
python -m build
```

The `.so` file will be in `src/python/build/`.

### Usage

```python
import sys
sys.path.insert(0, 'src/python/build')
import kernel_craft_python as kc
import numpy as np

# numpy arrays
input = np.random.rand(512, 512).astype(np.float32)
kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
result = kc.conv_naive(input, kernel)

# PyTorch tensors on CUDA
import torch
input_t = torch.rand(512, 512, device='cuda')
result_t = kc.conv_naive(input_t, kernel)
```

## Performance Results

| Kernel | Image Size | Time (ms) |
|--------|------------|-----------|
| Naive | 512x512 | ~4.5 |
| Tiled (8x8) | 512x512 | ~4.0 |
| Tiled (16x16) | 512x512 | ~3.5 |
| Fused pipeline | 512x512 | ~2.0 |

Key insights:
- Tiled convolution reduces global memory traffic by ~40%
- Fusing conv+batchnorm+relu saves ~40% memory bandwidth vs separate kernels
- 8x8 tile provides best overall performance for 3x3 kernels