# src/ — Source Code Root

Root directory for all CUDA kernel implementations, Python bindings, and supporting infrastructure.

## Subdirectories

| Directory | Contents |
|-----------|----------|
| `custom/` | Non-standard convolution ops (sparse, domain-specific) |
| `kernels/` | CUDA kernels: core convolution, variants, inference, and transformer/LLM |
| `performance/` | Performance infrastructure: CUDA Graphs, memory pools, mixed precision, async streams, unified memory |
| `pipelines/` | End-to-end pipelines: fused conv+BN+ReLU, GPU preprocessing |
| `python/` | Python bindings (pybind11), ctypes bridge, vLLM backend |
| `tensorrt/` | TensorRT plugins for CNN and LLM inference deployment |

## Design Philosophy

- Every kernel has a corresponding test in the top-level `tests/` directory
- Doxygen-style documentation (`\file`, `\brief`, `\param[in]`, `\param[out]`) required in all source files
- Design decisions and benchmark results are recorded in `CLAUDE.md` at the project root
