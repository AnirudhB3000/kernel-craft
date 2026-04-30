# kernels/ - CUDA Convolution Kernels

## Overview
Collection of CUDA convolution kernel implementations organized by category: core implementations, variants, and inference-optimized kernels.

## Subdirectories
- `core/` - Foundational convolution implementations (naive and tiled)
- `variants/` - Extended convolution types (3D, dilated, transposed, grouped)
- `inference/` - Inference-optimized kernels (INT8 quantized, BN folding, activation fusion)

## Organization
Kernels are separated by use case: training-focused (core, variants) vs inference-focused (inference). All kernels follow Doxygen documentation standards and include corresponding tests in `tests/`.
