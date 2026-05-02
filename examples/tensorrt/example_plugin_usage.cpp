/**
 * \file example_plugin_usage.cpp
 * \brief Complete TensorRT plugin usage example for kernel-craft.
 *
 * This example demonstrates how to use kernel-craft's custom TensorRT plugins
 * for CNN inference optimization. The example shows:
 * 1. Plugin registration
 * 2. Creating network with plugin layers
 * 3. Setting kernel weights for plugins
 *
 * \note This requires TensorRT to be installed and linked.
 *       Compile with: -I${TENSORRT_ROOT}/include -L${TENSORRT_ROOT}/lib -lnvinfer
 */

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstring>

// Plugin registration functions
extern "C" void registerConvInt8Plugin();
extern "C" void registerConvReluPlugin();

// Plugin wrapper functions to set kernel weights
namespace kernel_craft {
namespace tensorrt {
    class KernelCraftBasePlugin;
    void setKernelWeights(KernelCraftBasePlugin* plugin, const float* hostKernel, size_t kernelBytes);
}
}

/**
 * \brief Example: Building a TensorRT engine with Conv+ReLU plugin.
 */
void build_engine_with_conv_relu() {
    std::cout << "=== Building TensorRT Engine with Conv+ReLU Plugin ===\n\n";

    // Step 1: Register plugins
    std::cout << "Step 1: Registering plugins...\n";
    registerConvInt8Plugin();
    registerConvReluPlugin();
    std::cout << "  Plugins registered successfully.\n\n";

    // Step 2: Create TensorRT builder and network
    std::cout << "Step 2: Creating TensorRT builder and network...\n";
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(nvinfer1::getLogger());
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);
    std::cout << "  Builder and network created.\n\n";

    // Step 3: Add input tensor
    std::cout << "Step 3: Adding input tensor (1x3x224x224)...\n";
    nvinfer1::ITensor* input = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, 3, 224, 224});
    std::cout << "  Input tensor added.\n\n";

    // Step 4: Get plugin creator and create Conv+ReLU plugin
    std::cout << "Step 4: Creating Conv+ReLU plugin...\n";
    auto* creator = nvinfer1::getPluginRegistry()->getPluginCreator("KernelCraft_ConvReLU", "1");
    if (!creator) {
        std::cerr << "  ERROR: ConvReLU plugin creator not found!\n";
        return;
    }

    const nvinfer1::PluginFieldCollection* fc = creator->getFieldNames();
    nvinfer1::IPluginV2* plugin = creator->createPlugin("conv_relu_1", fc);
    std::cout << "  Plugin created: " << plugin->getPluginType() << "\n\n";

    // Step 5: Add plugin layer to network
    std::cout << "Step 5: Adding plugin layer to network...\n";
    nvinfer1::ITensor* inputTensor = input;
    nvinfer1::IPluginV2Layer* pluginLayer = network->addPluginV2(&inputTensor, 1, *plugin);
    pluginLayer->setName("ConvReLU");
    std::cout << "  Plugin layer added.\n\n";

    // Step 6: Set kernel weights (example - 3x3 kernel, 3 input channels, 64 output channels)
    std::cout << "Step 6: Setting kernel weights...\n";
    int kernelSize = 3;
    int inputChannels = 3;
    int outputChannels = 64;
    size_t kernelWeightsSize = inputChannels * outputChannels * kernelSize * kernelSize * sizeof(float);

    float* hostKernel = new float[kernelWeightsSize / sizeof(float)];
    // Initialize with random weights (in practice, load from trained model)
    for (size_t i = 0; i < kernelWeightsSize / sizeof(float); ++i) {
        hostKernel[i] = (rand() % 100) / 100.0f - 0.5f;
    }

    // Set weights on plugin (requires casting to KernelCraftBasePlugin)
    auto* basePlugin = static_cast<kernel_craft::tensorrt::KernelCraftBasePlugin*>(plugin);
    kernel_craft::tensorrt::setKernelWeights(basePlugin, hostKernel, kernelWeightsSize);
    std::cout << "  Kernel weights set (" << kernelWeightsSize << " bytes).\n\n";

    // Step 7: Mark output and build engine
    std::cout << "Step 7: Building engine...\n";
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    network->markOutput(*pluginLayer->getOutput(0));

    nvinfer1::IHostMemory* serializedEngine = builder->buildSerializedNetwork(*network, *config);
    if (serializedEngine) {
        std::cout << "  Engine built successfully (" << serializedEngine->size() << " bytes).\n";
        // Save to file in practice: std::ofstream("model.engine") << ...
    } else {
        std::cerr << "  ERROR: Failed to build engine!\n";
    }
    std::cout << "\n";

    // Cleanup
    delete[] hostKernel;
    if (serializedEngine) serializedEngine->destroy();
    config->destroy();
    network->destroy();
    builder->destroy();

    std::cout << "=== Example completed ===\n";
}

/**
 * \brief Example: INT8 quantized convolution plugin usage.
 */
void build_engine_with_int8_conv() {
    std::cout << "\n=== Building TensorRT Engine with INT8 Conv Plugin ===\n\n";

    // Similar structure for INT8 plugin
    std::cout << "Note: INT8 plugin requires calibration data for quantization.\n";
    std::cout << "      Scale factors must be provided: input_scale, kernel_scale, output_scale.\n\n";

    // Plugin creation (simplified)
    auto* creator = nvinfer1::getPluginRegistry()->getPluginCreator("KernelCraft_ConvInt8", "1");
    if (creator) {
        std::cout << "ConvInt8 plugin creator found.\n";
        // Create plugin with scales
        nvinfer1::PluginField fields[3];
        float input_scale = 1.0f, kernel_scale = 1.0f, output_scale = 1.0f;
        fields[0] = {"input_scale", &input_scale, nvinfer1::PluginFieldType::kFLOAT32, 1};
        fields[1] = {"kernel_scale", &kernel_scale, nvinfer1::PluginFieldType::kFLOAT32, 1};
        fields[2] = {"output_scale", &output_scale, nvinfer1::PluginFieldType::kFLOAT32, 1};

        nvinfer1::PluginFieldCollection fc{3, fields};
        nvinfer1::IPluginV2* plugin = creator->createPlugin("conv_int8_1", &fc);
        if (plugin) {
            std::cout << "ConvInt8 plugin created successfully.\n";
            plugin->destroy();
        }
    }

    std::cout << "\n=== INT8 Example completed ===\n";
}

int main() {
    std::cout << "\n============================================\n";
    std::cout << "  Kernel-Craft TensorRT Plugin Usage Example\n";
    std::cout << "============================================\n\n";

    try {
        build_engine_with_conv_relu();
        build_engine_with_int8_conv();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
