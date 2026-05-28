/**
 * \file tensorrt/plugin_cnn.cpp
 * \brief TensorRT plugin usage: small CNN (Conv→ReLU→Pool) with kernel-craft.
 *
 * \par Purpose
 * Shows the complete workflow for deploying a small CNN using kernel-craft's
 * custom TensorRT IPluginV2DynamicExt plugins for convolution and activation,
 * demonstrating how the plugins slot into a real inference engine.
 *
 * \par Network architecture
 * \code
 *   Input:  [1, 3, 224, 224]  (float32)
 *     ├─ KernelCraft_ConvReLU   — 3×3 conv (3→64 ch) + ReLU
 *     ├─ KernelCraft_ConvReLU   — 3×3 conv (64→128 ch) + ReLU
 *     ├─ 2×2 max-pool stride 2  — [1, 128, 112, 112]
 *   Output: [1, 128, 112, 112]
 * \endcode
 *
 * \note Compile requirements:
 *   - TensorRT ≥ 10.x  (NvInfer.h, libnvinfer.so)
 *   - CUDA ≥ 12.0
 *   - cmake -DTENSORRT_ROOT=/path/to/TensorRT ..
 */

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

// Full class definitions needed for setKernelWeights() and safe downcasting
#include "plugin_wrapper.h"

extern "C" void registerConvInt8Plugin();
extern "C" void registerConvReluPlugin();

/* =========================================================================
 * Logger
 * ========================================================================= */

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRT] " << msg << "\n";
    }
} gLogger;

/* =========================================================================
 * Helpers
 * ========================================================================= */

static void init_weights(float* buf, size_t n, int fan_in) {
    float sigma = std::sqrt(2.f / static_cast<float>(fan_in));
    for (size_t i = 0; i < n; ++i) {
        float u1 = (rand() % 10000 + 1) / 10000.f;
        float u2 = (rand() % 10000 + 1) / 10000.f;
        buf[i] = sigma * std::sqrt(-2.f * std::log(u1)) * std::cos(6.2832f * u2);
    }
}

/* =========================================================================
 * Build: two ConvReLU plugin layers + built-in pool
 * ========================================================================= */

static nvinfer1::IHostMemory* build_conv_relu_engine(int batch = 1) {
    std::cout << "\n=== Building CNN Engine (ConvReLU plugin × 2) ===\n";

    registerConvReluPlugin();
    std::cout << "  Plugins registered.\n";

    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    // TRT 10: explicit-batch is always on; createNetworkV2(0) == createNetworkV2()
    nvinfer1::INetworkDefinition* net = builder->createNetworkV2(0);

    nvinfer1::ITensor* input = net->addInput(
        "input", nvinfer1::DataType::kFLOAT,
        nvinfer1::Dims4{batch, 3, 224, 224});
    std::cout << "  Input: [" << batch << ", 3, 224, 224]\n";

    /* ---- ConvReLU layer 1  (3 → 64 channels, 3×3) ---- */
    // TRT 10: getPluginRegistry() is a global free function, not nvinfer1::
    auto* cr1_creator = getPluginRegistry()
                            ->getPluginCreator("KernelCraft_ConvReLU", "1");
    assert(cr1_creator && "ConvReLU plugin not registered");

    nvinfer1::PluginFieldCollection empty_fc{0, nullptr};
    auto* cr1_plugin = cr1_creator->createPlugin("conv_relu_1", &empty_fc);

    const int C0 = 3, C1 = 64, K = 3;
    const size_t w1_n = (size_t)C1 * C0 * K * K;
    std::vector<float> h_w1(w1_n);
    init_weights(h_w1.data(), w1_n, C0 * K * K);
    {
        auto* base = dynamic_cast<kernel_craft::tensorrt::KernelCraftBasePlugin*>(cr1_plugin);
        if (base) base->setKernelWeights(h_w1.data(), w1_n * sizeof(float));
    }

    nvinfer1::ITensor* cr1_in = input;
    auto* cr1_layer = net->addPluginV2(&cr1_in, 1, *cr1_plugin);
    cr1_layer->setName("conv_relu_1");
    std::cout << "  conv_relu_1: [" << batch << ", 64, 224, 224]\n";

    /* ---- ConvReLU layer 2  (64 → 128 channels, 3×3) ---- */
    auto* cr2_plugin = cr1_creator->createPlugin("conv_relu_2", &empty_fc);

    const int C2 = 128;
    const size_t w2_n = (size_t)C2 * C1 * K * K;
    std::vector<float> h_w2(w2_n);
    init_weights(h_w2.data(), w2_n, C1 * K * K);
    {
        auto* base = dynamic_cast<kernel_craft::tensorrt::KernelCraftBasePlugin*>(cr2_plugin);
        if (base) base->setKernelWeights(h_w2.data(), w2_n * sizeof(float));
    }

    nvinfer1::ITensor* cr2_in = cr1_layer->getOutput(0);
    auto* cr2_layer = net->addPluginV2(&cr2_in, 1, *cr2_plugin);
    cr2_layer->setName("conv_relu_2");
    std::cout << "  conv_relu_2: [" << batch << ", 128, 224, 224]\n";

    /* ---- 2×2 max-pool, stride 2  → [batch, 128, 112, 112] ---- */
    auto* pool = net->addPoolingNd(*cr2_layer->getOutput(0),
                                   nvinfer1::PoolingType::kMAX,
                                   nvinfer1::DimsHW{2, 2});
    pool->setStrideNd(nvinfer1::DimsHW{2, 2});
    pool->setName("maxpool");
    pool->getOutput(0)->setName("output");
    std::cout << "  maxpool: [" << batch << ", 128, 112, 112]\n";

    net->markOutput(*pool->getOutput(0));

    nvinfer1::IBuilderConfig* cfg = builder->createBuilderConfig();
    cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 256ULL << 20);

    nvinfer1::IHostMemory* engine_blob =
        builder->buildSerializedNetwork(*net, *cfg);

    if (engine_blob)
        std::cout << "  Engine built: " << engine_blob->size() / 1024
                  << " KB serialised.\n";
    else
        std::cerr << "  ERROR: Engine build failed.\n";

    // TRT 10: destroy() removed; delete objects directly
    delete cfg;
    delete net;
    delete builder;
    return engine_blob;
}

/* =========================================================================
 * Build: INT8 quantized conv demo
 * ========================================================================= */

static void demo_int8_plugin(float input_scale, float kernel_scale,
                              float output_scale) {
    std::cout << "\n=== INT8 ConvPlugin with calibration scales ===\n";

    registerConvInt8Plugin();

    // TRT 10: global getPluginRegistry()
    auto* creator = getPluginRegistry()
                        ->getPluginCreator("KernelCraft_ConvInt8", "1");
    if (!creator) {
        std::cerr << "  ConvInt8 plugin not found in registry.\n";
        return;
    }

    nvinfer1::PluginField fields[3];
    fields[0] = {"input_scale",  &input_scale,  nvinfer1::PluginFieldType::kFLOAT32, 1};
    fields[1] = {"kernel_scale", &kernel_scale, nvinfer1::PluginFieldType::kFLOAT32, 1};
    fields[2] = {"output_scale", &output_scale, nvinfer1::PluginFieldType::kFLOAT32, 1};
    nvinfer1::PluginFieldCollection fc{3, fields};

    auto* plugin = creator->createPlugin("conv_int8_stem", &fc);
    if (!plugin) {
        std::cerr << "  Plugin creation failed.\n";
        return;
    }

    std::cout << "  Plugin type    : " << plugin->getPluginType()    << "\n";
    std::cout << "  Plugin version : " << plugin->getPluginVersion() << "\n";
    std::cout << "  input_scale    : " << input_scale                << "\n";
    std::cout << "  kernel_scale   : " << kernel_scale               << "\n";
    std::cout << "  output_scale   : " << output_scale               << "\n";

    // TRT 10: destroy() removed
    delete plugin;
    std::cout << "  INT8 plugin demo complete.\n";
}

/* =========================================================================
 * Run inference with a deserialised engine
 * ========================================================================= */

static void run_inference(const nvinfer1::IHostMemory* blob) {
    if (!blob) return;
    std::cout << "\n=== Running inference with deserialised engine ===\n";

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(
        blob->data(), blob->size());
    if (!engine) {
        std::cerr << "  Deserialisation failed.\n";
        delete runtime;
        return;
    }

    nvinfer1::IExecutionContext* ctx = engine->createExecutionContext();

    // TRT 10: use getNbIOTensors() + getIOTensorName() + getTensorShape()
    const int n_io = engine->getNbIOTensors();
    std::vector<void*> buffers(n_io);
    for (int i = 0; i < n_io; ++i) {
        const char* name = engine->getIOTensorName(i);
        auto dims = engine->getTensorShape(name);
        size_t vol = 1;
        for (int j = 0; j < dims.nbDims; ++j) vol *= static_cast<size_t>(dims.d[j]);
        cudaMalloc(&buffers[i], vol * sizeof(float));
        ctx->setTensorAddress(name, buffers[i]);
    }

    // Fill the input tensor (index 0) with a constant value
    {
        const char* in_name = engine->getIOTensorName(0);
        auto dims = engine->getTensorShape(in_name);
        size_t vol = 1;
        for (int j = 0; j < dims.nbDims; ++j) vol *= static_cast<size_t>(dims.d[j]);
        std::vector<float> h_input(vol, 0.5f);
        cudaMemcpy(buffers[0], h_input.data(), vol * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    ctx->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    // Read back output tensor (last I/O tensor)
    {
        const char* out_name = engine->getIOTensorName(n_io - 1);
        auto dims = engine->getTensorShape(out_name);
        size_t vol = 1;
        for (int j = 0; j < dims.nbDims; ++j) vol *= static_cast<size_t>(dims.d[j]);
        std::vector<float> h_out(vol);
        cudaMemcpy(h_out.data(), buffers[n_io - 1],
                   vol * sizeof(float), cudaMemcpyDeviceToHost);
        auto it = std::max_element(h_out.begin(), h_out.end());
        std::cout << "  Output shape: " << dims.nbDims << "D tensor, max activation = "
                  << *it << "\n";
    }

    for (void* b : buffers) cudaFree(b);
    cudaStreamDestroy(stream);
    // TRT 10: delete instead of destroy()
    delete ctx;
    delete engine;
    delete runtime;
}

/* =========================================================================
 * main
 * ========================================================================= */

int main() {
    std::cout << "\n================================================\n";
    std::cout << "  kernel-craft TensorRT Plugin Example\n";
    std::cout << "  CNN: Conv×2 → MaxPool\n";
    std::cout << "================================================\n";

    try {
        nvinfer1::IHostMemory* blob = build_conv_relu_engine(/*batch=*/1);
        run_inference(blob);
        // TRT 10: delete instead of destroy()
        delete blob;

        demo_int8_plugin(0.0784f, 0.00394f, 0.0157f);

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n=== Example complete ===\n";
    return 0;
}
