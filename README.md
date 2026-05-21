# kernel-craft

CUDA kernels for machine learning training and inference-time optimization — convolution (CNN), transformer attention (LLM), and vLLM integration.

## Hardware Requirements

| Requirement | Minimum | Tested On |
|-------------|---------|-----------|
| GPU architecture | SM 6.0 (Pascal) | SM 8.9 (Ada, RTX 4070 Laptop) |
| VRAM | 4 GB | 8 GB |
| CUDA Toolkit | 11.8 | 12.0 / 12.3 |
| OS | Linux x86-64 | Ubuntu 22.04 / WSL2 |

**Architecture notes:**
- SM 7.0+ (Volta) required for FP16 Tensor Cores (`mixed_precision.cu`)
- SM 8.0+ (Ampere) required for TF32 and INT4 Tensor Core paths (`quant_int4.cu`)
- SM 8.9+ (Ada) required for FP8 E4M3 (`fp8_quant.cu`) — falls back to FP32 simulation on older hardware
- Multi-GPU tensor parallelism requires peer-to-peer access (not available on WSL2)

## Prerequisites

```
cmake >= 3.18
CUDA Toolkit >= 11.8
g++ >= 9 (C++17)
Python 3.11 or 3.12   # for Python bindings and vLLM backend
```

TensorRT is optional — detected automatically if `TENSORRT_ROOT` is set.

## Setup

### 1. Clone and build (C++ + CUDA kernels)

```bash
git clone https://github.com/anomalyco/kernel-craft.git
cd kernel-craft
mkdir build && cd build
cmake ..
make -j$(nproc)
```

To build with TensorRT plugin support:

```bash
cmake .. -DTENSORRT_ROOT=/path/to/TensorRT
make -j$(nproc)
```

### 2. Python bindings

The pybind11 extension is built automatically by CMake and placed in
`src/python/kernel_craft_python/`. Install directly from the build tree:

```bash
pip install -e src/python
```

Or install from PyPI (pre-built wheel, no CUDA build step):

```bash
pip install kernel-craft
```

Requires: Python 3.11–3.12, numpy, CUDA runtime on PATH.

### 3. vLLM backend (optional)

The vLLM attention backend requires a specific torch + vLLM combination.
Use the provided venv to avoid conflicts with system packages:

```bash
cd src/python
python -m venv venv
source venv/bin/activate
pip install torch==2.11.0 --extra-index-url https://download.pytorch.org/whl/cu130
pip install vllm==0.21.0
pip install -e .
```

The ctypes bridge (`kernel_craft_torch_ops.py`) reads `libkernels.so` from the
CMake build directory. Make sure the C++ build is complete before activating the
backend.

## Features

### Core Convolution Kernels

- **Naive Convolution** (`src/kernels/core/conv_naive.cu`) — baseline 2D, one thread per output pixel
- **Tiled Convolution** (`src/kernels/core/conv_tiled.cu`) — shared memory tiling, 8×8/16×16/32×32 dispatch

### Convolution Variants

- **3D Convolution** (`src/kernels/variants/conv3d.cu`) — volumetric
- **Dilated Convolution** (`src/kernels/variants/conv_dilated.cu`) — expanded receptive field without parameter increase
- **Transposed Convolution** (`src/kernels/variants/conv_transposed.cu`) — upsampling via strided expansion
- **Grouped Convolution** (`src/kernels/variants/conv_grouped.cu`) — ResNeXt-style channel groups
- **Sparse Convolution** (`src/custom/custom_op.cu`) — coordinate-list format for sparse inputs

### Inference-Optimized CNN Kernels

- **INT8 Quantized Convolution** (`src/kernels/inference/conv_int8.cu`) — low-precision with Tensor Core support
- **BatchNorm Folding** (`src/kernels/inference/bn_folding.cu`) — folds BN weights into conv at inference time
- **Conv+Activation Fusion** (`src/kernels/inference/conv_activation_fusion.cu`) — fused conv + ReLU/LeakyReLU/Sigmoid

### Transformer / LLM Kernels

- **FlashAttention** (`src/kernels/transformer/flash_attention.cu`) — tiled MHA/GQA/MQA with online softmax and causal masking
- **PagedAttention** (`src/kernels/transformer/paged_attention.cu`) — attention over non-contiguous paged KV-cache (vLLM-style block tables)
- **INT4 Dequant + GEMV** (`src/kernels/transformer/quant_int4.cu`) — GPTQ/AWQ-style 2×INT4 packed per byte, per-group scales
- **FP8 Quantization** (`src/kernels/transformer/fp8_quant.cu`) — E4M3 format, per-token/per-channel scaling, SmoothQuant-compatible
- **Speculative Decoding** (`src/kernels/transformer/speculative_decoding.cu`) — rejection-sampling draft token verification
- **Tensor Parallelism** (`src/kernels/transformer/tensor_parallel.cu`) — ring all-reduce, all-gather, column-parallel and row-parallel linear (tiled SGEMM)
- **NCCL Collectives** (`src/kernels/transformer/tensor_parallel_nccl.cu`) — NCCL-backed all-reduce and all-gather; compiles stub fallback when NCCL is not installed

### Pipeline Kernels

- **Fused Pipeline** (`src/pipelines/pipeline_fused.cu`) — conv + batchnorm + relu in a single kernel
- **Separate Pipeline** (`src/pipelines/pipeline_separate.cu`) — individual pipeline stages
- **GPU Preprocessing** (`src/pipelines/preprocess_gpu.cu`) — resize, normalize, flip on GPU

### Performance Infrastructure

- **Memory Pool** (`src/performance/memory_pool.cu`) — pre-allocated buffers (~85% cudaMalloc overhead reduction)
- **CUDA Graphs** (`src/performance/cuda_graphs.cu`) — captured kernel DAG for reduced launch overhead
- **Mixed Precision** (`src/performance/mixed_precision.cu`) — FP16/TF32 kernels for Tensor Cores
- **Persistent Kernels** (`src/performance/persistent_kernels.cu`) — kernel reuse across batches
- **Async Streams** (`src/performance/async_streams.cu`) — double-buffered async H2D/compute/D2H overlap
- **Unified Memory** (`src/performance/unified_memory.cu`) — managed memory with async prefetch
- **Multi-Stream Pipeline** (`src/performance/multi_stream_pipeline.cu`) — concurrent preprocessing + inference

### TensorRT Integration

- **CNN Plugins** (`src/tensorrt/plugin_wrapper.cpp`) — `ConvInt8Plugin`, `ConvReLUPlugin` (IPluginV2DynamicExt)
- **PagedAttention Plugin** (`src/tensorrt/paged_attention_plugin.cpp`) — TensorRT custom op for paged KV-cache
- **vLLM PyTorch Plugin** (`src/tensorrt/vllm_plugin_wrapper.cpp`) — TORCH_LIBRARY registration for C++ path

### Python API

- **pybind11 extension** (`src/python/pybind_cuda.cpp`, `pybind_transformer.cpp`) — numpy and PyTorch tensor support
- **ctypes bridge** (`src/python/kernel_craft_torch_ops.py`) — zero-copy access to `libkernels.so` via `data_ptr()`
- **vLLM backend** (`src/python/kernel_craft_vllm_backend.py`) — `KernelCraftAttentionBackend` implementing the vLLM 0.21.0 v1 API

## Testing

### Run everything (CI gate)

```bash
cd build && make run_tests
# C++ unit → Python unit → C++ integration → Python integration → benchmarks
```

### Individual suites

```bash
cd build && make run_unit_tests         # C++ unit tests (single-kernel)
cd build && make run_integration_tests  # C++ integration (pipelines, CUDA graphs, streams)
cd build && make run_python_tests       # All Python tests (unit + integration)
cd build && make run_benchmarks         # C++ benchmarks
```

### Python tests directly (using the vLLM venv)

```bash
cd src/python
source venv/bin/activate
pytest tests/test_bindings.py tests/test_transformer_bindings.py -v   # unit (conv + transformer)
pytest tests/test_vllm_backend.py -v                                   # vLLM backend integration
```

Expected: **89 tests pass, 0 skip** (with torch 2.11.0+cu130 + vLLM 0.21.0).

### Running a single test binary

```bash
./build/bin/test_flash_attention
./build/bin/test_paged_attention
./build/bin/test_quant_int4
./build/bin/test_fp8_quant
./build/bin/test_speculative_decoding
./build/bin/test_tensor_parallel
./build/bin/test_conv_naive
./build/bin/test_conv_tiled
```

### Multi-GPU tensor parallelism (requires 2+ GPUs or single-GPU simulation)

```bash
# Single-GPU simulation (always works):
./build/bin/test_tensor_parallel

# Real multi-GPU (requires torchrun and libnccl):
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    tests/test_tensor_parallel_multiprocess.py
```

## Benchmarks

```bash
cd build && make run_benchmarks   # all benchmarks via CTest
```

Or run individual binaries:

```bash
./build/bin/benchmark_conv
./build/bin/benchmark_flash_attention
./build/bin/benchmark_paged_attention
./build/bin/benchmark_quant
./build/bin/benchmark_tensor_parallel     # sim BW + col/row parallel TFLOPS
./build/bin/benchmark_memory_pool         [width] [height] [iterations]
./build/bin/benchmark_cuda_graphs         [width] [height] [iterations]
./build/bin/benchmark_mixed_precision     [width] [height] [iterations]
./build/bin/benchmark_persistent_kernels  [width] [height] [iterations]
./build/bin/benchmark_async_streams       [width] [height] [batches]
./build/bin/benchmark_unified_memory      [width] [height] [iterations]
```

## Python Usage

### Convolution

```python
import kernel_craft_python as kc
import numpy as np

img = np.random.rand(512, 512).astype(np.float32)
kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)

result = kc.conv_naive(img, kernel)
result = kc.conv_tiled(img, kernel, tile_w=8, tile_h=8)

# PyTorch CUDA tensors are accepted directly
import torch
img_t = torch.rand(512, 512, device='cuda')
result_t = kc.conv_naive(img_t, kernel)
```

### FlashAttention

```python
import kernel_craft_python as kc
import numpy as np

B, H, N, d = 1, 8, 512, 64
Q = np.random.randn(B, H, N, d).astype(np.float32)
K = np.random.randn(B, H, N, d).astype(np.float32)
V = np.random.randn(B, H, N, d).astype(np.float32)

out = kc.flash_attention(Q, K, V, causal=True)
```

### INT4 Dequantization

```python
# packed_weights: [out_features, in_features // 2]  (2×INT4 per byte)
# scales: [out_features, in_features // group_size]
weights_fp32 = kc.quant_int4_dequant(packed_weights, scales, group_size=128)
```

### FP8 Quantization

```python
q_data, scale = kc.fp8_quantize(tensor, mode="per_token")
recovered = kc.fp8_dequantize(q_data, scale, mode="per_token")
```

### Tensor Parallelism

```python
import kernel_craft_python as kc
import numpy as np

# Column-parallel linear: each rank holds W_rank [N/R, K]; input x is replicated
x = np.random.randn(32, 4096).astype(np.float32)
W_rank = np.random.randn(2048, 4096).astype(np.float32)   # 2 ranks → N/R = 2048
y_rank = kc.col_parallel_linear(x, W_rank)                 # [32, 2048]

# Row-parallel linear: each rank holds W_rank [N, K/R] and x_rank [M, K/R]
x_rank = np.random.randn(32, 2048).astype(np.float32)
W_rank = np.random.randn(4096, 2048).astype(np.float32)
partial = kc.row_parallel_linear(x_rank, W_rank)            # [32, 4096]

# NCCL (requires libnccl and multiple GPUs)
if kc.HAVE_NCCL:
    comms = kc.nccl_comm_init([0, 1])
    kc.nccl_allreduce(comms[0], partial)
```

### vLLM Backend

```python
import kernel_craft_python.kernel_craft_vllm_backend as kb
kb.register()  # registers KernelCraftAttentionBackend with vLLM

# Or set environment variable before starting vLLM:
# VLLM_ATTENTION_BACKEND=kernel_craft
```

## Performance Results

All results on RTX 4070 Laptop (SM 8.9, 8 GB VRAM, CUDA 12.0).

### Convolution (3×3 kernel)

| Kernel | Image Size | Time (ms) |
|--------|------------|-----------|
| Naive | 256×256 | ~0.45 |
| Tiled 8×8 | 256×256 | ~0.36 |
| Tiled 16×16 | 1024×1024 | ~0.60 |
| Tiled 32×32 | 2048×2048 | ~1.39 |
| Fused conv+BN+ReLU | 1024×1024 | ~0.42 (~2.3× vs separate) |

### FlashAttention (B=1, H=8, d=64)

| Config | Seq Len | Causal | Time (ms) | Throughput |
|--------|---------|--------|-----------|------------|
| MHA | 512 | No | ~0.80 | ~675 Gflops |
| MHA | 1024 | No | ~3.08 | ~698 Gflops |
| MHA | 2048 | No | ~9.94 | ~865 Gflops |
| MHA | 1024 | Yes | ~2.18 | ~985 Gflops |
| GQA 4:1 | 512 | No | ~0.67 | ~801 Gflops |

### PagedAttention Decode (B=1, H=8, d=64)

| Seq Len | Page Size | Time (ms) |
|---------|-----------|-----------|
| 256 | 16 / 32 / 64 | ~0.13 |
| 1024 | 16 / 32 / 64 | ~0.48 |

### Quantization

| Kernel | Config | Throughput |
|--------|--------|------------|
| INT4 dequant | 4096×4096, group=128 | ~195 GB/s |
| INT4 GEMV | 4096×4096 | ~210 Gflops |
| FP8 quantize (per-token) | 1024×4096 | ~171 GB/s |
| FP8 dequantize | 1024×4096 | ~234 GB/s |

### Performance Optimizations

| Optimization | Result |
|-------------|--------|
| Memory Pool | ~85% cudaMalloc overhead reduction (16× buffer reuse) |
| CUDA Graphs | ~5–10 μs launch overhead vs ~15–30 μs separate |
| FP16 Tensor Cores | ~2× throughput vs FP32 |
| Persistent Kernels | ~50% lower latency for fixed-batch streams |
| Async double-buffer | ~2× on large batches (512×512, 16 batches) |

### Tensor Parallelism (single-GPU simulation, RTX 4070 Laptop)

| Operation | Config | Result |
|-----------|--------|--------|
| Sim all-gather | 2 ranks, 4 MB | ~117 GB/s |
| Sim all-gather | 4 ranks, 4 MB | ~117 GB/s |
| Sim ring all-reduce | 2 ranks, 4 MB | ~39 GB/s |
| Col-parallel linear | M=32, N=4096, K=4096 | ~0.94 TFLOPS |
| Row-parallel linear | M=32, N=11008, K=4096 | ~0.83 TFLOPS |

Note: simulation bandwidth uses single-GPU device-copy; real NCCL all-reduce over NVLink would approach memory bandwidth.

## Directory Structure

```
kernel-craft/
├── src/
│   ├── kernels/
│   │   ├── core/                    # Core 2D convolution
│   │   │   ├── conv_naive.cu
│   │   │   └── conv_tiled.cu
│   │   ├── variants/                # Convolution variants
│   │   │   ├── conv3d.cu
│   │   │   ├── conv_dilated.cu
│   │   │   ├── conv_transposed.cu
│   │   │   └── conv_grouped.cu
│   │   ├── inference/               # CNN inference optimization
│   │   │   ├── conv_int8.cu
│   │   │   ├── bn_folding.cu
│   │   │   └── conv_activation_fusion.cu
│   │   └── transformer/             # LLM / attention kernels
│   │       ├── flash_attention.cu
│   │       ├── paged_attention.cu
│   │       ├── quant_int4.cu
│   │       ├── fp8_quant.cu
│   │       ├── speculative_decoding.cu
│   │       ├── tensor_parallel.cu
│   │       └── tensor_parallel_nccl.cu
│   ├── pipelines/
│   │   ├── pipeline_fused.cu
│   │   ├── pipeline_separate.cu
│   │   └── preprocess_gpu.cu
│   ├── performance/
│   │   ├── memory_pool.cu
│   │   ├── cuda_graphs.cu
│   │   ├── mixed_precision.cu
│   │   ├── persistent_kernels.cu
│   │   ├── async_streams.cu
│   │   ├── unified_memory.cu
│   │   └── multi_stream_pipeline.cu
│   ├── custom/
│   │   └── custom_op.cu
│   ├── tensorrt/
│   │   ├── plugin_wrapper.cpp          # CNN TensorRT plugins
│   │   ├── paged_attention_plugin.cpp  # PagedAttention TensorRT plugin
│   │   └── vllm_plugin_wrapper.cpp     # TORCH_LIBRARY custom ops (C++ path)
│   └── python/
│       ├── pybind_cuda.cpp             # Bindings: conv kernels
│       ├── pybind_transformer.cpp      # Bindings: transformer kernels
│       ├── kernel_craft_torch_ops.py   # ctypes bridge to libkernels.so
│       ├── kernel_craft_vllm_backend.py
│       ├── pyproject.toml
│       └── tests/
│           ├── test_bindings.py
│           ├── test_transformer_bindings.py
│           └── test_vllm_backend.py
├── benchmarks/
│   ├── benchmark_conv.cpp
│   ├── benchmark_flash_attention.cpp
│   ├── benchmark_paged_attention.cpp
│   ├── benchmark_quant.cpp
│   ├── benchmark_tensor_parallel.cpp
│   ├── benchmark_pipeline.cpp
│   ├── benchmark_memory_pool.cpp
│   ├── benchmark_cuda_graphs.cpp
│   ├── benchmark_mixed_precision.cpp
│   ├── benchmark_persistent_kernels.cpp
│   ├── benchmark_async_streams.cpp
│   ├── benchmark_unified_memory.cpp
│   └── benchmark_vllm_e2e.py
├── tests/                             # C++ unit tests
│   ├── test_conv_naive.cpp
│   ├── test_conv_tiled.cpp
│   ├── test_flash_attention.cpp
│   ├── test_paged_attention.cpp
│   ├── test_quant_int4.cpp
│   ├── test_fp8_quant.cpp
│   ├── test_speculative_decoding.cpp
│   ├── test_tensor_parallel.cpp
│   ├── test_tensor_parallel_multiprocess.py
│   └── ... (conv variants, pipelines, performance infrastructure)
├── CMakeLists.txt
└── README.md
```

## Key Insights

- Tiled convolution reduces global memory traffic by ~40% via shared memory reuse
- Fusing conv+BN+ReLU saves ~40% memory bandwidth over separate kernel launches — memory movement dominates over arithmetic
- Causal FlashAttention is faster than full attention at the same sequence length — skipped KV tiles cost nothing
- FP8 E4M3 achieves ~5.8% mean relative error (3 mantissa bits); per-token scaling bounds per-row error tightly
- Page size (16/32/64 tokens) has minimal impact on PagedAttention decode latency
- The ctypes bridge reads `data_ptr()` directly from torch CUDA tensors — no GPU→CPU roundtrip
