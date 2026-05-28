# python/ — Python Bindings & vLLM Integration

Python interface for kernel-craft: pybind11 extension module, ctypes bridge to `libkernels.so`, and the vLLM attention backend.

## Files

| File | Description |
|------|-------------|
| `pybind_cuda.cpp` | pybind11 bindings for CNN kernels: conv, INT8, BN folding, fused ops |
| `pybind_transformer.cpp` | pybind11 bindings for transformer kernels: flash attention, paged attention, INT4/FP8 quant, speculative decoding |
| `kernel_craft_torch_ops.py` | ctypes bridge — calls `libkernels.so` launchers directly from Python via `data_ptr()`, no GPU→CPU roundtrip; all 5 kernel methods emit OpenTelemetry spans |
| `kernel_craft_vllm_backend.py` | `KernelCraftAttentionBackend` implementing the vLLM 0.21.0 v1 API (`vllm.v1.attention.backend`); emits request-level OTEL spans for prefill/decode |
| `kernel_craft_otel.py` | Central OTEL module: `setup_tracing()`, `kernel_span()` context manager with CUDA event timing, no-op fallbacks when `opentelemetry` is not installed |
| `pyproject.toml` | Package metadata; setuptools build backend; optional dep groups: `vllm`, `triton`, `observability` |
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

### Triton kernels (optional)

```bash
pip install -e ".[triton]"   # installs triton>=3.0
```

### OpenTelemetry observability (optional)

```bash
pip install -e ".[observability]"   # installs opentelemetry-api/sdk + OTLP exporter
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

### Mamba / SSM kernels

```python
import kernel_craft_python as kc
import numpy as np

B, L, D, N = 1, 1024, 512, 16
u     = np.random.randn(B, L, D).astype(np.float32)
A_log = np.random.randn(D, N).astype(np.float32)
B_ssm = np.random.randn(B, L, N).astype(np.float32)
C     = np.random.randn(B, L, N).astype(np.float32)
delta = np.abs(np.random.randn(B, L, D).astype(np.float32))

y = kc.selective_scan(u, A_log, B_ssm, C, delta)   # [B, L, D]

# Causal depthwise conv1d (channels-first: [B, D, L])
x = np.random.randn(B, D, L).astype(np.float32)
w = np.random.randn(D, 4).astype(np.float32)        # d_conv=4
b = np.zeros(D, dtype=np.float32)
y_conv = kc.depthwise_conv1d(x, w, b)               # [B, D, L]

# RMSNorm
x_norm = np.random.randn(B * L, D).astype(np.float32)
g      = np.ones(D, dtype=np.float32)
y_norm = kc.rmsnorm(x_norm, g)                      # [B*L, D]
```

### Tensor parallelism (col/row parallel linear + NCCL)

```python
import kernel_craft_python as kc
import numpy as np

# Column-parallel linear — each rank holds W_rank [N/R, K]; input x is replicated
x       = np.random.randn(32, 4096).astype(np.float32)
W_rank  = np.random.randn(2048, 4096).astype(np.float32)  # 2 ranks → N/R = 2048
y_rank  = kc.col_parallel_linear(x, W_rank)                # [32, 2048]

# Row-parallel linear — each rank holds x_rank [M, K/R] and W_rank [N, K/R]
x_rank  = np.random.randn(32, 2048).astype(np.float32)
W_rank  = np.random.randn(4096, 2048).astype(np.float32)
partial = kc.row_parallel_linear(x_rank, W_rank)           # [32, 4096]

# NCCL (real multi-GPU; requires libnccl and peer-to-peer access)
print(kc.HAVE_NCCL)           # True / False
if kc.HAVE_NCCL:
    comms = kc.nccl_comm_init([0, 1])   # list of opaque handles
    kc.nccl_allreduce(comms[0], partial)
    kc.nccl_comm_destroy(comms[0])
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

### Triton kernels

```python
# Import third-party triton before adding src/ to sys.path to avoid shadowing
import triton
import sys; sys.path.insert(0, "src")
from triton import flash_attention, selective_scan, int4_gemv

import torch
Q = torch.randn(1, 8, 1024, 64, device='cuda')
out = flash_attention(Q, Q, Q, causal=True)   # [1, 8, 1024, 64]
```

### OpenTelemetry observability

```python
from kernel_craft_otel import setup_tracing

# Export to Jaeger (or any OTLP-compatible backend):
setup_tracing(endpoint="http://localhost:4317", service_name="kernel-craft")

# For testing with in-memory span capture (no external collector):
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
exporter = InMemorySpanExporter()
setup_tracing(exporter=exporter)   # SimpleSpanProcessor used when exporter is provided

# All KernelCraftOps calls now emit spans automatically.
# Zero-overhead no-op mode: just don't call setup_tracing().
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
| `selective_scan` | `(u, A_log, B, C, delta)` | ndarray [B, L, D] |
| `depthwise_conv1d` | `(x[B,D,L], w[D,d_conv], bias[D])` | ndarray [B, D, L] |
| `rmsnorm` | `(x[rows,D], g[D], eps=1e-6)` | ndarray [rows, D] |
| `col_parallel_linear` | `(x[M,K], W[N_rank,K])` | ndarray [M, N_rank] |
| `row_parallel_linear` | `(x_rank[M,K_rank], W[N,K_rank])` | ndarray [M, N] |
| `ring_allreduce` | `(buf)` | None (in-place, simulation) |
| `allgather` | `(sendbuf, recvbuf, rank, nranks)` | None (simulation) |
| `nccl_comm_init` | `(devs: list[int])` | list of handles (uint64) |
| `nccl_comm_destroy` | `(handle)` | None |
| `nccl_allreduce` | `(handle, arr)` | None (in-place) |

## Publishing to PyPI

Use the GitHub Actions workflow — do not run twine manually:

```bash
gh workflow run release.yml -f version_type=patch   # or minor / major
```

Requires `testpypi_token` and `pypi_token` repository secrets. The workflow builds the C++ extension, runs the full test suite, uploads to TestPyPI, then gates on the `pypi-production` environment before uploading to PyPI.
