# tensorrt/ — TensorRT & PyTorch Custom Op Integration

TensorRT plugin wrappers and PyTorch custom op registration for deploying kernel-craft kernels in production inference pipelines.

## Files

| File | Description |
|------|-------------|
| `plugin_wrapper.cpp` | TensorRT `IPluginV2DynamicExt` plugins for CNN kernels: `ConvInt8Plugin`, `ConvReLUPlugin` |
| `plugin_wrapper.h` | Plugin interface declarations and creator registration helpers |
| `paged_attention_plugin.cpp` | TensorRT `IPluginV2DynamicExt` plugin wrapping `launch_paged_attention` |
| `vllm_plugin_wrapper.cpp` | PyTorch `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL` registration for the C++ kernel path |

## TensorRT CNN Plugins (`plugin_wrapper.cpp`)

### ConvInt8Plugin

Wraps `launch_conv_int8_naive` as a TensorRT layer.

**Plugin type string**: `"KernelCraft_ConvInt8"`

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_scale` | float | Per-tensor input quantization scale |
| `kernel_scale` | float | Per-tensor kernel quantization scale |
| `output_scale` | float | Per-tensor output dequantization scale |

### ConvReLUPlugin

Wraps `launch_conv_relu` (fused conv + ReLU) as a TensorRT layer.

**Plugin type string**: `"KernelCraft_ConvReLU"`

### Registration

```cpp
extern "C" void registerConvInt8Plugin();
extern "C" void registerConvReLUPlugin();

registerConvInt8Plugin();
registerConvReLUPlugin();
```

### Creating a Plugin Layer

```cpp
auto* creator = nvinfer1::getPluginRegistry()
                    ->getPluginCreator("KernelCraft_ConvReLU", "1");
nvinfer1::IPluginV2* plugin = creator->createPlugin("conv_relu_0", &fc);
network->addPluginV2(&input_tensor, 1, *plugin);
```

## TensorRT PagedAttention Plugin (`paged_attention_plugin.cpp`)

Wraps `launch_paged_attention` for use in TRT-LLM style inference graphs.

- Input tensors: `Q`, `block_table`, `K_pool`, `V_pool`, `seq_lens`
- Output tensor: `O` (attended output)
- Built when `-DTENSORRT_ROOT` is provided to CMake

## PyTorch Custom Ops (`vllm_plugin_wrapper.cpp`)

Registers kernel-craft launchers as `torch.ops.kernel_craft.*` operations via `TORCH_LIBRARY`. This is the C++ path for zero-copy PyTorch tensor access.

The Python-side ctypes bridge (`src/python/kernel_craft_torch_ops.py`) is the alternative path — no recompile needed, reads `data_ptr()` directly from torch CUDA tensors.

| Op name | Underlying kernel |
|---------|------------------|
| `kernel_craft::flash_attention` | `launch_flash_attention` |
| `kernel_craft::paged_attention` | `launch_paged_attention` |
| `kernel_craft::int4_dequant` | `launch_int4_dequant` |
| `kernel_craft::fp8_quantize` | `launch_fp8_quantize` |

## Build

TensorRT support is optional. Pass `-DTENSORRT_ROOT` to enable:

```bash
cmake .. -DTENSORRT_ROOT=/path/to/TensorRT
make -j$(nproc)
```

The `vllm_plugin_wrapper.cpp` target requires `-DHAVE_TORCH` and links against `libtorch`.
