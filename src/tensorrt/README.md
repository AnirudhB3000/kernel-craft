# tensorrt/ - TensorRT Integration

## Overview
TensorRT plugin wrappers for deploying custom CUDA kernels in TensorRT inference workflows.

## Files
- `plugin_wrapper.cpp` - TensorRT plugin implementation for custom CNN kernels
- `plugin_wrapper.h` - Header file defining plugin interface and registration

## Purpose
Enables integration of kernel-craft custom kernels (INT8 conv, fused ops) into TensorRT inference pipelines for production CNN deployment.

## Phase 10 Goal
Build TensorRT plugin wrappers for custom CNN kernels and document integration workflows for vision model deployment (see AGENTS.md).
