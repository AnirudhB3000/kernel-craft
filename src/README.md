# src/ - Source Code Root

## Overview
Root directory for all CUDA kernel implementations, Python bindings, and supporting infrastructure for the kernel-craft project.

## Subdirectories
- `custom/` - Custom non-standard convolution operations
- `kernels/` - CUDA convolution kernel implementations (core, variants, inference)
- `performance/` - Performance optimization utilities (CUDA graphs, memory pools, mixed precision, persistent kernels)
- `pipelines/` - End-to-end ML pipelines (fused/separate, preprocessing)
- `python/` - Python bindings via pybind11, package distribution
- `tensorrt/` - TensorRT plugin wrappers for inference deployment

## Design Philosophy
All kernel logic must be covered by unit/integration tests in `tests/` directory. Doxygen-style documentation is required for all source files.
