# kernel-craft

CUDA kernels for machine learning training & inference-time optimization (CNN-first for vision models, extensible to LLMs).

## Features

### Core Convolution Kernels

- **Naive Convolution** (`src/kernels/core/conv_naive.cu`) - Baseline 2D convolution with one thread per output pixel
- **Tiled Convolution** (`src/kernels/core/conv_tiled.cu`) - Optimized using shared memory tiling for better memory bandwidth

### Convolution Variants

- **3D Convolution** (`src/kernels/variants/conv3d.cu`) - Volumetric convolution for 3D data
- **Dilated Convolution** (`src/kernels/variants/conv_dilated.cu`) - Expanded receptive field without parameter increase
- **Transposed Convolution** (`src/kernels/variants/conv_transposed.cu`) - Upsampling via strided expansion
- **Grouped Convolution** (`src/kernels/variants/conv_grouped.cu`) - Channel-grouped convolutions (ResNeXt-style)
- **Sparse Convolution** (`src/custom/custom_op.cu`) - Coordinate-list format for sparse inputs

### Inference-Optimized Kernels

- **INT8 Quantized Convolution** (`src/kernels/inference/conv_int8.cu`) - Low-precision inference kernels with Tensor Cores support
- **BatchNorm Folding** (`src/kernels/inference/bn_folding.cu`) - Pre-compute folded conv weights for inference (eliminates BN layer)
- **Conv+Activation Fusion** (`src/kernels/inference/conv_activation_fusion.cu`) - Fused conv + ReLU/LeakyReLU/Sigmoid for inference

### Pipeline Kernels

- **Fused Pipeline** (`src/pipelines/pipeline_fused.cu`) - Conv + BatchNorm + ReLU in single kernel
- **Separate Pipeline** (`src/pipelines/pipeline_separate.cu`) - Individual pipeline kernels
- **GPU Preprocessing** (`src/pipelines/preprocess_gpu.cu`) - Resize, normalize, flip operations on GPU for training data pipelines

### Performance Optimization

- **Memory Pool** (`src/performance/memory_pool.cu`) - Pre-allocated buffers to eliminate cudaMalloc/cudaFree overhead
- **CUDA Graphs** (`src/performance/cuda_graphs.cu`) - Captured kernel DAG for reduced launch overhead
- **Mixed Precision** (`src/performance/mixed_precision.cu`) - FP16/TF32 kernels for Tensor Cores
- **Persistent Kernels** (`src/performance/persistent_kernels.cu`) - Kernel reuse across batches

### TensorRT Integration

- **Plugin Wrappers** (`src/tensorrt/plugin_wrapper.cpp`) - Custom TensorRT plugins for CNN inference

### Python Bindings

- **pybind11 module** (`src/python/`) - numpy and PyTorch support
- Install via `pip install kernel-craft` (PyPI)
- Type stubs (`.pyi`) for IDE support

## Directory Structure

```
kernel-craft/
├── src/
│   ├── kernels/
│   │   ├── core/             # Core convolution kernels
│   │   │   ├── conv_naive.cu
│   │   │   └── conv_tiled.cu
│   │   ├── variants/        # Convolution variants
│   │   │   ├── conv3d.cu
│   │   │   ├── conv_dilated.cu
│   │   │   ├── conv_transposed.cu
│   │   │   └── conv_grouped.cu
│   │   └── inference/       # Inference-optimized kernels
│   │       ├── conv_int8.cu
│   │       ├── bn_folding.cu
│   │       └── conv_activation_fusion.cu
│   ├── pipelines/           # Pipeline kernels
│   │   ├── pipeline_fused.cu
│   │   ├── pipeline_separate.cu
│   │   └── preprocess_gpu.cu
│   ├── performance/         # Performance optimization modules
│   │   ├── memory_pool.cu
│   │   ├── cuda_graphs.cu
│   │   ├── mixed_precision.cu
│   │   └── persistent_kernels.cu
│   ├── custom/              # Custom operations
│   │   └── custom_op.cu
│   ├── tensorrt/            # TensorRT integration
│   │   └── plugin_wrapper.cpp
│   └── python/              # Python bindings
│       ├── pybind_cuda.cpp
│       ├── pyproject.toml
│       └── tests/
├── benchmarks/              # Benchmark programs
├── tests/                  # C++ unit tests
├── examples/
│   ├── cpp/                # C++ examples
│   └── python/             # Python examples
├── data/
├── CMakeLists.txt           # Build configuration
├── README.md
└── AGENTS.md              # Project guidelines
```

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## Testing

Tests are split into three suites. All commands run from the `build/` directory.

### Release gate — runs everything in order, fails fast

```bash
cd build && make run_tests
# C++ unit → Python unit → C++ integration → Python integration → benchmarks
```

### Individual suites

```bash
# C++ unit tests (isolated single-kernel tests)
cd build && make run_unit_tests

# C++ integration tests (multi-component: pipelines, CUDA graphs, streams, TensorRT)
cd build && make run_integration_tests

# All Python tests (unit + integration)
cd build && make run_python_tests

# C++ benchmarks
cd build && make run_benchmarks
```

### Python tests directly (without the local venv)

```bash
cd src/python
source venv/bin/activate
python -m pytest tests/test_bindings.py tests/test_transformer_bindings.py -v  # unit
python -m pytest tests/test_vllm_backend.py -v                                  # integration
```

## Benchmarks

```bash
cd build && make run_benchmarks   # run all benchmarks via CTest
```

Or run individual binaries:
```bash
./build/bin/benchmark_conv
./build/bin/benchmark_flash_attention
./build/bin/benchmark_paged_attention
./build/bin/benchmark_quant
./build/bin/benchmark_memory_pool      [width] [height] [iterations]
./build/bin/benchmark_cuda_graphs      [width] [height] [iterations]
./build/bin/benchmark_mixed_precision  [width] [height] [iterations]
./build/bin/benchmark_persistent_kernels [width] [height] [iterations]
```

## Python Usage

### Install from PyPI (recommended)

```bash
pip install kernel-craft
```

Requires: Python 3.11-3.12, numpy, CUDA runtime

### Build from source

```bash
cd src/python
python -m build
```

The `.so` file will be in `src/python/build/` or `src/python/kernel_craft_python/`.

### Usage

```python
import kernel_craft  # or: import kernel_craft_python as kc
import numpy as np

# numpy arrays
input = np.random.rand(512, 512).astype(np.float32)
kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
result = kernel_craft.conv_naive(input, kernel)

# PyTorch tensors on CUDA
import torch
input_t = torch.rand(512, 512, device='cuda')
result_t = kernel_craft.conv_naive(input_t, kernel)

# Tiled convolution with custom tile size
result = kernel_craft.conv_tiled(input, kernel, tile_w=8, tile_h=8)
```

## Performance Results

### Convolution Kernels (3×3 kernel)

| Kernel | Image Size | Time (ms) | Notes |
|--------|------------|-----------|---------|
| Naive | 256×256 | ~0.45 | Baseline |
| Tiled 8×8 | 256×256 | ~0.36 | Best for small images |
| Tiled 16×16 | 1024×1024 | ~0.60 | Balanced |
| Tiled 32×32 | 2048×2048 | ~1.39 | Best for large images |

### Inference Kernels (Phase 10)

| Kernel | Precision | Image Size | Time (ms) | Notes |
|--------|-----------|------------|-----------|---------|
| INT8 Naive | INT8 | 256×256 | ~0.40 | ~4× bandwidth reduction |
| INT8 Tiled | INT8 | 256×256 | ~0.35 | Best for inference |
| Conv+ReLU | FP32 | 256×256 | ~0.38 | Eliminates intermediate memory |
| BN Folding | FP32 | C_out=64 | ~0.01 | One-time precomputation |

### Pipeline Fusion

| Pipeline | Kernel Launches | Time (ms) | Speedup |
|----------|----------------|-----------|---------|
| Separate | 3 | ~0.96 | Baseline |
| Fused | 1 | ~0.42 | ~2.3× |

### Performance Optimizations

| Optimization | Overhead Reduction |
|-------------|-------------------|
| Memory Pool | ~85% (buffer reuse) |
| CUDA Graphs | ~5-10μs per launch |
| FP16 (Tensor Cores) | ~2× throughput |
| Persistent Kernels | ~50% lower latency |

### Phase 10: TensorRT Integration

| Plugin | Purpose | Status |
|--------|---------|--------|
| ConvInt8Plugin | INT8 quantized convolution | ✅ Complete |
| ConvReLUPlugin | Fused Conv+ReLU | ✅ Complete |
| ConvInt8PluginCreator | Plugin registration | ✅ Complete |
| ConvReLUPluginCreator | Plugin registration | ✅ Complete |

## Key Insights

- Tiled convolution reduces global memory traffic by ~40%
- Fusing conv+batchnorm+relu saves ~40% memory bandwidth vs separate kernels
- 8×8 tile provides best overall performance for 3×3 kernels
- Memory movement dominates cost more than arithmetic

## Inference Engine Integration
kernel-craft's inference optimizations are designed to integrate with production inference engines:

### TensorRT (Vision/CNN Models)
- **Custom Plugins**: Wrap INT8 quantized convolution and BN folding logic as TensorRT `IPluginV3` plugins to add custom kernels to serialized TensorRT engines.
- **Pre-Build Weight Folding**: Use the BN folding utility to pre-compute folded conv weights before TensorRT engine building, eliminating separate BN layers for inference.
- See `examples/tensorrt/` for step-by-step integration demos.

### vLLM (LLM Models, Deferred)
Transformer-specific inference kernels (Flash Attention, LayerNorm, RMSNorm) will be added in future phases and integrated as vLLM custom PyTorch ops following [vLLM CustomOp guidelines](https://docs.vllm.ai/en/stable/design/custom_op/).