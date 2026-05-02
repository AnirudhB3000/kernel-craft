#ifndef KERNEL_CRAFT_TENSORRT_PLUGIN_WRAPPER_H
#define KERNEL_CRAFT_TENSORRT_PLUGIN_WRAPPER_H

#include <NvInfer.h>
#include <string>
#include <vector>
#include <cuda_runtime.h>

namespace kernel_craft {
namespace tensorrt {

class KernelCraftBasePlugin : public nvinfer1::IPluginV2DynamicExt {
protected:
    std::string mPluginNamespace;
    std::string mName;
    nvinfer1::DataType mDataType;
    nvinfer1::PluginFormat mFormat;
    std::vector<int32_t> mInputDims;
    std::vector<int32_t> mOutputDims;
    int mKernelSize;
    int mInputChannels;
    int mOutputChannels;
    float* mDeviceKernel;
    size_t mKernelSizeBytes;

public:
KernelCraftBasePlugin(const std::string& name, int kernelSize,
                           int inputChannels, int outputChannels);
    KernelCraftBasePlugin(const void* data, size_t length);
    ~KernelCraftBasePlugin() override;

    void setKernelWeights(const float* hostKernel, size_t kernelBytes);

    const char* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex,
        const nvinfer1::DimsExprs* inputs,
        int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int32_t pos,
        const nvinfer1::PluginTensorDesc* inOut,
        int32_t nbInputs,
        int32_t nbOutputs) noexcept override;

    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* in,
        int32_t nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out,
        int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* inputs,
        int32_t nbInputs,
        const nvinfer1::PluginTensorDesc* outputs,
        int32_t nbOutputs) const noexcept override;

    int32_t initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    nvinfer1::DataType getOutputDataType(
        int32_t index,
        const nvinfer1::DataType* inputTypes,
        int32_t nbInputs) const noexcept override {
        return inputTypes[0];
    }
};

class ConvInt8Plugin : public KernelCraftBasePlugin {
private:
    float mInputScale;
    float mKernelScale;
    float mOutputScale;

public:
    ConvInt8Plugin(const std::string& name, int kernelSize,
                   int inputChannels, int outputChannels,
                   float inputScale, float kernelScale, float outputScale);
    ConvInt8Plugin(const void* data, size_t length);
    ~ConvInt8Plugin() override = default;

    const char* getPluginType() const noexcept override;
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;

    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                    const nvinfer1::PluginTensorDesc* outputDesc,
                    const void* const* inputs,
                    void* const* outputs,
                    void* workspace,
                    cudaStream_t stream) noexcept override;
};

class ConvReluPlugin : public KernelCraftBasePlugin {
public:
    ConvReluPlugin(const std::string& name, int kernelSize,
                   int inputChannels, int outputChannels);
    ConvReluPlugin(const void* data, size_t length);
    ~ConvReluPlugin() override = default;

    const char* getPluginType() const noexcept override;
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;

    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                    const nvinfer1::PluginTensorDesc* outputDesc,
                    const void* const* inputs,
                    void* const* outputs,
                    void* workspace,
                    cudaStream_t stream) noexcept override;
};

class ConvInt8PluginCreator : public nvinfer1::IPluginCreator {
public:
    ConvInt8PluginCreator();
    ~ConvInt8PluginCreator() override = default;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(const char* name,
                                       const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                            const void* serialData,
                                            size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace;
    nvinfer1::PluginFieldCollection mFieldCollection;
};

class ConvReluPluginCreator : public nvinfer1::IPluginCreator {
public:
    ConvReluPluginCreator();
    ~ConvReluPluginCreator() override = default;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(const char* name,
                                       const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                            const void* serialData,
                                            size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace;
    nvinfer1::PluginFieldCollection mFieldCollection;
};

} // namespace tensorrt
} // namespace kernel_craft

#endif
