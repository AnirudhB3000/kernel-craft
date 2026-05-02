/**
 * \file plugin_wrapper.cpp
 * \brief Simplified TensorRT plugin wrapper.
 */

#include "plugin_wrapper.h"
#include <cuda_runtime.h>
#include <cstring>

extern "C" void launch_conv_int8_tiled(const float* d_input,
                                        const float* d_kernel,
                                        float* d_output,
                                        int width, int height, int ksize,
                                        float input_scale,
                                        float kernel_scale,
                                        float output_scale,
                                        dim3 block);

extern "C" void launch_conv_relu_tiled(const float* d_input,
                                        const float* d_kernel,
                                        float* d_output,
                                        int width, int height, int ksize,
                                        dim3 block);

namespace kernel_craft {
namespace tensorrt {

// ---------------------------------------------------------------------------
// KernelCraftBasePlugin implementation
// ---------------------------------------------------------------------------

KernelCraftBasePlugin::KernelCraftBasePlugin(const std::string& name,
                                               int kernelSize,
                                               int inputChannels,
                                               int outputChannels)
    : mName(name)
    , mDataType(nvinfer1::DataType::kFLOAT)
    , mFormat(nvinfer1::PluginFormat::kLINEAR)
    , mKernelSize(kernelSize)
    , mInputChannels(inputChannels)
    , mOutputChannels(outputChannels)
    , mDeviceKernel(nullptr)
    , mKernelSizeBytes(0) {}

KernelCraftBasePlugin::KernelCraftBasePlugin(const void* data, size_t length) {
    const char* d = static_cast<const char*>(data);
    mKernelSize = *reinterpret_cast<const int32_t*>(d); d += sizeof(int32_t);
    mInputChannels = *reinterpret_cast<const int32_t*>(d); d += sizeof(int32_t);
    mOutputChannels = *reinterpret_cast<const int32_t*>(d); d += sizeof(int32_t);
    mKernelSizeBytes = *reinterpret_cast<const size_t*>(d); d += sizeof(size_t);
    // Note: kernel data would follow in real implementation
    mDeviceKernel = nullptr;
}

KernelCraftBasePlugin::~KernelCraftBasePlugin() {
    if (mDeviceKernel) {
        cudaFree(mDeviceKernel);
    }
}

void KernelCraftBasePlugin::setKernelWeights(const float* hostKernel, size_t kernelBytes) {
    if (mDeviceKernel) {
        cudaFree(mDeviceKernel);
    }
    mKernelSizeBytes = kernelBytes;
    cudaMalloc(&mDeviceKernel, kernelBytes);
    cudaMemcpy(mDeviceKernel, hostKernel, kernelBytes, cudaMemcpyHostToDevice);
}

const char* KernelCraftBasePlugin::getPluginVersion() const noexcept {
    return "1";
}

int32_t KernelCraftBasePlugin::getNbOutputs() const noexcept {
    return 1;
}

void KernelCraftBasePlugin::destroy() noexcept {
    delete this;
}

void KernelCraftBasePlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    mPluginNamespace = pluginNamespace;
}

const char* KernelCraftBasePlugin::getPluginNamespace() const noexcept {
    return mPluginNamespace.c_str();
}

nvinfer1::DimsExprs KernelCraftBasePlugin::getOutputDimensions(
    int32_t outputIndex,
    const nvinfer1::DimsExprs* inputs,
    int32_t nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
    return inputs[0];
}

bool KernelCraftBasePlugin::supportsFormatCombination(
    int32_t pos,
    const nvinfer1::PluginTensorDesc* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept {
    if (inOut[pos].format != nvinfer1::PluginFormat::kLINEAR) return false;
    if (inOut[pos].type != nvinfer1::DataType::kFLOAT) return false;
    return true;
}

void KernelCraftBasePlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int32_t nbOutputs) noexcept {
}

size_t KernelCraftBasePlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int32_t nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int32_t nbOutputs) const noexcept {
    return 0;
}



// ---------------------------------------------------------------------------
// ConvInt8Plugin implementation
// ---------------------------------------------------------------------------

ConvInt8Plugin::ConvInt8Plugin(const std::string& name, int kernelSize,
                                int inputChannels, int outputChannels,
                                float inputScale, float kernelScale, float outputScale)
    : KernelCraftBasePlugin(name, kernelSize, inputChannels, outputChannels)
    , mInputScale(inputScale)
    , mKernelScale(kernelScale)
    , mOutputScale(outputScale) {}

ConvInt8Plugin::ConvInt8Plugin(const void* data, size_t length)
    : KernelCraftBasePlugin(data, length) {
    const char* d = static_cast<const char*>(data) + 3 * sizeof(int32_t);
    mInputScale = *reinterpret_cast<const float*>(d); d += sizeof(float);
    mKernelScale = *reinterpret_cast<const float*>(d); d += sizeof(float);
    mOutputScale = *reinterpret_cast<const float*>(d);
}

const char* ConvInt8Plugin::getPluginType() const noexcept {
    return "KernelCraft_ConvInt8";
}

nvinfer1::IPluginV2DynamicExt* ConvInt8Plugin::clone() const noexcept {
    try {
        return new ConvInt8Plugin(mName, mKernelSize, mInputChannels,
                                  mOutputChannels, mInputScale, mKernelScale,
                                  mOutputScale);
    } catch (...) {
        return nullptr;
    }
}

size_t ConvInt8Plugin::getSerializationSize() const noexcept {
    return 3 * sizeof(int32_t) + 3 * sizeof(float);
}

void ConvInt8Plugin::serialize(void* buffer) const noexcept {
    char* d = static_cast<char*>(buffer);
    *reinterpret_cast<int32_t*>(d) = mKernelSize; d += sizeof(int32_t);
    *reinterpret_cast<int32_t*>(d) = mInputChannels; d += sizeof(int32_t);
    *reinterpret_cast<int32_t*>(d) = mOutputChannels; d += sizeof(int32_t);
    *reinterpret_cast<float*>(d) = mInputScale; d += sizeof(float);
    *reinterpret_cast<float*>(d) = mKernelScale; d += sizeof(float);
    *reinterpret_cast<float*>(d) = mOutputScale;
}

int32_t ConvInt8Plugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                  const nvinfer1::PluginTensorDesc* outputDesc,
                                  const void* const* inputs,
                                  void* const* outputs,
                                  void* workspace,
                                  cudaStream_t stream) noexcept {
    const auto& dims = inputDesc[0].dims;
    int height = dims.d[2];
    int width = dims.d[3];

    float* d_input = const_cast<float*>(static_cast<const float*>(inputs[0]));
    float* d_output = static_cast<float*>(outputs[0]);

    dim3 block(16, 16);
    launch_conv_int8_tiled(d_input, mDeviceKernel, d_output,
                           width, height, mKernelSize,
                           mInputScale, mKernelScale, mOutputScale, block);

    return 0;
}

// ---------------------------------------------------------------------------
// ConvReluPlugin implementation
// ---------------------------------------------------------------------------

ConvReluPlugin::ConvReluPlugin(const std::string& name, int kernelSize,
                                int inputChannels, int outputChannels)
    : KernelCraftBasePlugin(name, kernelSize, inputChannels, outputChannels) {}

ConvReluPlugin::ConvReluPlugin(const void* data, size_t length)
    : KernelCraftBasePlugin(data, length) {}

const char* ConvReluPlugin::getPluginType() const noexcept {
    return "KernelCraft_ConvReLU";
}

nvinfer1::IPluginV2DynamicExt* ConvReluPlugin::clone() const noexcept {
    try {
        return new ConvReluPlugin(mName, mKernelSize, mInputChannels, mOutputChannels);
    } catch (...) {
        return nullptr;
    }
}

size_t ConvReluPlugin::getSerializationSize() const noexcept {
    return 3 * sizeof(int32_t);
}

void ConvReluPlugin::serialize(void* buffer) const noexcept {
    char* d = static_cast<char*>(buffer);
    *reinterpret_cast<int32_t*>(d) = mKernelSize; d += sizeof(int32_t);
    *reinterpret_cast<int32_t*>(d) = mInputChannels; d += sizeof(int32_t);
    *reinterpret_cast<int32_t*>(d) = mOutputChannels;
}

int32_t ConvReluPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                  const nvinfer1::PluginTensorDesc* outputDesc,
                                  const void* const* inputs,
                                  void* const* outputs,
                                  void* workspace,
                                  cudaStream_t stream) noexcept {
    const auto& dims = inputDesc[0].dims;
    int height = dims.d[2];
    int width = dims.d[3];

    float* d_input = const_cast<float*>(static_cast<const float*>(inputs[0]));
    float* d_output = static_cast<float*>(outputs[0]);

    dim3 block(16, 16);
    launch_conv_relu_tiled(d_input, mDeviceKernel, d_output,
                             width, height, mKernelSize, block);

    return 0;
}

// ---------------------------------------------------------------------------
// ConvInt8PluginCreator implementation
// ---------------------------------------------------------------------------

ConvInt8PluginCreator::ConvInt8PluginCreator() : mNamespace("") {}

const char* ConvInt8PluginCreator::getPluginName() const noexcept {
    return "KernelCraft_ConvInt8";
}

const char* ConvInt8PluginCreator::getPluginVersion() const noexcept {
    return "1";
}

const nvinfer1::PluginFieldCollection* ConvInt8PluginCreator::getFieldNames() noexcept {
    return &mFieldCollection;
}

nvinfer1::IPluginV2* ConvInt8PluginCreator::createPlugin(
    const char* name,
    const nvinfer1::PluginFieldCollection* fc) noexcept {
    int kernelSize = 3;
    int inputChannels = 3;
    int outputChannels = 64;
    float inputScale = 1.0f;
    float kernelScale = 1.0f;
    float outputScale = 1.0f;

    return new ConvInt8Plugin(name, kernelSize, inputChannels,
                              outputChannels, inputScale, kernelScale, outputScale);
}

nvinfer1::IPluginV2* ConvInt8PluginCreator::deserializePlugin(
    const char* name,
    const void* serialData,
    size_t serialLength) noexcept {
    return new ConvInt8Plugin(serialData, serialLength);
}

void ConvInt8PluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

const char* ConvInt8PluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

} // namespace tensorrt
} // namespace kernel_craft

// Register plugins
extern "C" void __attribute__((constructor)) registerConvInt8Plugin() {
    static nvinfer1::PluginRegistrar<kernel_craft::tensorrt::ConvInt8PluginCreator> registrar;
    (void)registrar;
}

extern "C" void __attribute__((constructor)) registerConvReluPlugin() {
    static nvinfer1::PluginRegistrar<kernel_craft::tensorrt::ConvReluPluginCreator> registrar;
    (void)registrar;
}

// ---------------------------------------------------------------------------
// ConvReluPluginCreator implementation
// ---------------------------------------------------------------------------

ConvReluPluginCreator::ConvReluPluginCreator() : mNamespace("") {}

const char* ConvReluPluginCreator::getPluginName() const noexcept {
    return "KernelCraft_ConvReLU";
}

const char* ConvReluPluginCreator::getPluginVersion() const noexcept {
    return "1";
}

const nvinfer1::PluginFieldCollection* ConvReluPluginCreator::getFieldNames() noexcept {
    return &mFieldCollection;
}

nvinfer1::IPluginV2* ConvReluPluginCreator::createPlugin(
    const char* name,
    const nvinfer1::PluginFieldCollection* fc) noexcept {
    int kernelSize = 3;
    int inputChannels = 3;
    int outputChannels = 64;

    return new ConvReluPlugin(name, kernelSize, inputChannels, outputChannels);
}

nvinfer1::IPluginV2* ConvReluPluginCreator::deserializePlugin(
    const char* name,
    const void* serialData,
    size_t serialLength) noexcept {
    return new ConvReluPlugin(serialData, serialLength);
}

void ConvReluPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

const char* ConvReluPluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}
