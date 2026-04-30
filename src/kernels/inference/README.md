# inference/ - Inference-Optimized Kernels

## Overview
CUDA kernels optimized for CNN inference workloads, focusing on quantization, operator fusion, and Batch Normalization folding.

## Files
- `conv_int8.cu` - INT8 quantized convolution for reduced memory bandwidth and faster inference
- `bn_folding.cu` - Batch Normalization folding into convolution weights for inference deployment
- `conv_activation_fusion.cu` - Fused convolution + activation (ReLU) kernel to reduce memory traffic

## Design Notes
- INT8 uses TensorRT-compatible quantization schemes
- BN folding pre-computes BN parameters into conv weights at model load time
- Activation fusion eliminates intermediate tensor writes between conv and activation

## Phase 10 Goal
Enable efficient CNN inference deployment with TensorRT integration (see AGENTS.md).
