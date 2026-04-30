# TensorRT Integration Workflow for Vision Model Deployment

## Overview

This document describes how to integrate custom kernel-craft CUDA kernels
into TensorRT inference pipelines for optimized vision model deployment.

## Prerequisites

- TensorRT 8.x SDK
- CUDA 11.x or later
- CMake 3.20+
- kernel-craft kernels compiled as shared library

## Building TensorRT Plugins

### Step 1: Compile Custom Kernels

```bash
cd /home/aniru/kernel-craft
mkdir build && cd build
cmake .. -DTENSORRT_ROOT=/path/to/TensorRT
make kernel_craft_tensorrt_plugin
```

### Step 2: Plugin Structure

The TensorRT plugin wrapper is located in:
- `src/tensorrt/plugin_wrapper.h` - Plugin class declarations
- `src/tensorrt/plugin_wrapper.cpp` - Plugin implementations

Available plugins:
1. **ConvInt8Plugin** - INT8 quantized convolution
2. **ConvReluPlugin** - Fused convolution + ReLU

### Step 3: Register Plugins

Plugins are registered using the `REGISTER_TENSORRT_PLUGIN` macro:

```cpp
REGISTER_TENSORRT_PLUGIN(kernel_craft::tensorrt::ConvInt8PluginCreator);
REGISTER_TENSORRT_PLUGIN(kernel_craft::tensorrt::ConvReluPluginCreator);
```

## Using Plugins in TensorRT

### Example: Building Engine with Custom Plugin

```cpp
#include <NvInfer.h>
#include "plugin_wrapper.h"

// Initialize TensorRT
nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);

// Register plugins (if not using REGISTER_TENSORRT_PLUGIN macro)
auto pluginRegistry = getPluginRegistry();
pluginRegistry->registerCreator(convInt8PluginCreator, "kernel_craft");

// Build network
nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);

// Add input
auto input = network->addInput("input", nvinfer1::DataType::kFLOAT, Dims3{3, 224, 224});

// Add custom INT8 convolution plugin
auto plugin = pluginRegistry->createPlugin("KernelCraft_ConvInt8", pluginFieldCollection);
auto convLayer = network->addPluginV2(&input, 1, plugin);

// Mark output
convLayer->getOutput(0)->setName("output");
network->markOutput(*convLayer->getOutput(0));

// Build engine
nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
```

### Example: INT8 Quantization Workflow

```cpp
// 1. Compute quantization scales from calibration data
float input_scale = compute_quantization_scale(calibration_data, data_size);
float kernel_scale = compute_quantization_scale(kernel_weights, kernel_size);
float output_scale = 1.0f;  // Typically 1.0 for direct dequantization

// 2. Create plugin with quantization parameters
nvinfer1::PluginField fields[] = {
    {"kernel_size", &kernelSize, nvinfer1::PluginFieldType::kINT32, 1},
    {"input_channels", &inputChannels, nvinfer1::PluginFieldType::kINT32, 1},
    {"output_channels", &outputChannels, nvinfer1::PluginFieldType::kINT32, 1},
    {"input_scale", &input_scale, nvinfer1::PluginFieldType::kFLOAT32, 1},
    {"kernel_scale", &kernel_scale, nvinfer1::PluginFieldType::kFLOAT32, 1},
    {"output_scale", &output_scale, nvinfer1::PluginFieldType::kFLOAT32, 1}
};
```

## End-to-End Deployment Pipeline

### Step 1: Model Preparation

1. Train model with standard framework (PyTorch/TensorFlow)
2. Apply batch normalization folding (see `src/kernels/bn_folding.cu`)
3. Quantize weights to INT8 if using INT8 plugin

### Step 2: Build TensorRT Engine

```bash
# Using trtexec command-line tool
trtexec --onnx=model.onnx \
        --plugins=libkernel_craft_tensorrt_plugin.so \
        --int8 \
        --saveEngine=model.engine
```

### Step 3: Inference

```cpp
// Load engine
std::ifstream file("model.engine", std::ios::binary);
file.seekg(0, std::ifstream::end);
size_t size = file.tellg();
file.seekg(0, std::ifstream::beg);

std::vector<char> engineData(size);
file.read(engineData.data(), size);

nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(
    engineData.data(), size, nullptr);

// Create execution context
nvinfer1::IExecutionContext* context = engine->createExecutionContext();

// Run inference
context->executeV2(bindings);
```

## Performance Considerations

### When to Use Custom Plugins

- **INT8 Plugin**: Use when targeting Tensor Core acceleration on Turing+ GPUs
- **Fused Conv+ReLU**: Use when model has consecutive conv and ReLU layers

### Expected Speedups

| Configuration | Speedup vs Standard | Notes |
|--------------|-------------------|-------|
| INT8 Conv | 2-4x | Requires Tensor Cores |
| Fused Conv+ReLU | 1.2-1.5x | Reduces memory traffic |
| BN-folded Conv | 1.1-1.3x | Eliminates BN layer |

## Troubleshooting

### Plugin Not Found

Ensure the plugin library is in the library path:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/kernel-craft/build
```

### Quantization Accuracy

If INT8 accuracy drops significantly:
1. Use more calibration data
2. Apply per-channel quantization
3. Use mixed precision (some layers in FP16)

## References

- [TensorRT Plugin API Documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/index.html)
- [INT8 Quantization in TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#int8_quantization)
- [kernel-craft Phase 10 Documentation](AGENTS.md)
