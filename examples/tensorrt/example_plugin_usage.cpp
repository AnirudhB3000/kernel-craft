/**
 * \file example_plugin_usage.cpp
 * \brief Simplified example of TensorRT plugin usage.
 */
#include <cstdio>
#include <cstdlib>

// Plugin registration function
extern "C" void registerConvInt8Plugin();

int main() {
    printf("=== TensorRT Plugin Usage Example ===\n");

    // Step 1: Register the plugin
    printf("Step 1: Registering kernel-craft plugin...\n");
    registerConvInt8Plugin();
    printf("  Plugin registered successfully.\n\n");

    // Step 2: Show how to use in TensorRT
    printf("Step 2: Using plugin in TensorRT code...\n");
    printf("  // In your TensorRT code:\n");
    printf("  #include <NvInfer.h>\n");
    printf("  \n");
    printf("  // Register plugin (call once at startup)\n");
    printf("  registerConvInt8Plugin();\n");
    printf("  \n");
    printf("  // Get plugin creator\n");
    printf("  auto creator = nvinfer1::getPluginRegistry()->getPluginCreator(\"KernelCraft_ConvInt8\", \"1\");\n");
    printf("  \n");
    printf("  // Create plugin\n");
    printf("  const nvinfer1::PluginFieldCollection* fc = creator->getFieldNames();\n");
    printf("  nvinfer1::IPluginV2* plugin = creator->createPlugin(\"my_conv\", fc);\n");
    printf("  \n");
    printf("  // Add to network\n");
    printf("  nvinfer1::ITensor* input = network->addInput(...);\n");
    printf("  nvinfer1::IPluginV2Layer* layer = network->addPluginV2(&input, 1, *plugin);\n");
    printf("  \n");

    printf("\n=== Example completed ===\n");
    printf("\nNote: For full working example with engine building:\n");
    printf("  1. Implement proper weight loading for the plugin\n");
    printf("  2. Configure optimization profiles for your use case\n");
    printf("  3. Handle INT8 calibration if using INT8 mode\n");
    printf("  4. See TensorRT documentation for complete examples\n");

    return 0;
}
