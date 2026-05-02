# tensorrt/ - TensorRT Integration

## Overview
TensorRT plugin wrappers for deploying custom CUDA kernels in TensorRT inference workflows.

## Files
- `plugin_wrapper.cpp` - TensorRT plugin implementation for custom CNN kernels
- `plugin_wrapper.h` - Header file defining plugin interface and registration

## Plugin Types

### ConvInt8Plugin
INT8 quantized convolution with Tensor Cores support.

**Plugin Type**: `"KernelCraft_ConvInt8"`

**Parameters**:
- `input_scale`: Scale factor for input quantization
- `kernel_scale`: Scale factor for kernel quantization  
- `output_scale`: Scale factor for output dequantization

### ConvReLUPlugin
Fused convolution + ReLU activation in a single TensorRT layer.

**Plugin Type**: `"KernelCraft_ConvReLU"`

## Purpose
Enables integration of kernel-craft custom kernels (INT8 conv, fused ops) into TensorRT inference pipelines for production CNN deployment.

## Status
✅ **Phase 10 Complete**:
- ConvInt8Plugin - INT8 quantized convolution
- ConvReLUPlugin - Fused Conv+ReLU  
- ConvInt8PluginCreator - Plugin registration
- ConvReLUPluginCreator - Plugin registration
- Proper kernel weight management via `setKernelWeights()`
- Example usage in `examples/tensorrt/example_plugin_usage.cpp`

## Usage

### Registration
```cpp
extern "C" void registerConvInt8Plugin();
extern "C" void registerConvReLUPlugin();

registerConvInt8Plugin();
registerConvReLUPlugin();
```

### Creating Plugin Layer
```cpp
auto* creator = nvinfer1::getPluginRegistry()->getPluginCreator("KernelCraft_ConvReLU", "1");
nvinfer1::IPluginV2* plugin = creator->createPlugin("my_conv_relu", fc);
nvinfer1::IPluginV2Layer* layer = network->addPluginV2(&input, 1, *plugin);
```

See `examples/tensorrt/example_plugin_usage.cpp` for complete examples.
