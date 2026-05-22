/**
 * \file tensorrt/plugin_cnn.cpp
 * \brief TensorRT plugin usage: small CNN (Conv→BN→ReLU→Pool→FC) with kernel-craft.
 *
 * \par Purpose
 * Shows the complete workflow for deploying a small CNN using kernel-craft's
 * custom TensorRT IPluginV2DynamicExt plugins for convolution and activation,
 * demonstrating how the plugins slot into a real inference engine.
 *
 * \par Network architecture (similar to VGG stem / AlexNet first two layers)
 * \code
 *   Input:  [batch, 3, 224, 224]  (float32, CHW, ImageNet-normalised)
 *     │
 *     ├─ KernelCraft_ConvReLU   — 3×3 conv (3→64 ch) + ReLU, bias folded
 *     │
 *     ├─ KernelCraft_ConvReLU   — 3×3 conv (64→128 ch) + ReLU
 *     │
 *     ├─ TRT built-in Pooling   — 2×2 max-pool stride 2 → [batch, 128, 112, 112]
 *     │
 *     ├─ TRT built-in FC        — 128×112×112 → 1000 (ImageNet classes)
 *     │
 *   Output: [batch, 1000]  (logits)
 * \endcode
 *
 * \par Why custom plugins for convolution?
 * TensorRT's built-in conv uses cuDNN which is excellent for standard
 * spatial convolutions but cannot be extended to custom variants:
 * - Fused conv + folded-BN + ReLU in one kernel (see src/pipelines/pipeline_fused.cu)
 * - INT8 quantized convolution with custom per-layer scales
 * - Dilated, transposed, or grouped convolutions not covered by cuDNN
 * Plugins expose these as first-class TensorRT layers that participate
 * in graph optimisation, workspace allocation, and serialisation.
 *
 * \par Plugin lifecycle in TensorRT
 * \code
 *   Build time:
 *     registerConvReluPlugin()       // registers creator in plugin registry
 *     creator->createPlugin(name, fc) // instantiates plugin with fields
 *     network->addPluginV2(inputs, n, *plugin) // inserts into network graph
 *     builder->buildSerializedNetwork(*network, *config)  // engine binary
 *
 *   Runtime:
 *     runtime->deserializeCudaEngine(blob, size)  // loads engine
 *     ctx->enqueueV3(stream)        // runs all layers including plugins
 *     // Plugin::enqueue() is called with device I/O pointers and workspace
 * \endcode
 *
 * \par INT8 quantization with custom scales
 * The KernelCraft_ConvInt8 plugin bypasses TensorRT's calibration framework
 * and accepts per-layer quantization scales directly.  This is useful when
 * scales have been computed by an external quantization toolkit (e.g.
 * NVIDIA ModelOpt, HuggingFace Optimum) and need to be embedded in the engine:
 * \code
 *   float input_scale  = calibrate_tensor(calibration_data, "input");
 *   float kernel_scale = calibrate_tensor(calibration_data, "weight");
 *   float output_scale = calibrate_tensor(calibration_data, "output");
 *   // Pass scales as PluginField entries (see build_int8_engine() below)
 * \endcode
 *
 * \note Compile requirements:
 *   - TensorRT ≥ 8.x  (NvInfer.h, libnvinfer.so)
 *   - CUDA ≥ 11.0
 *   - kernel-craft built with -DHAVE_TENSORRT=ON
 *
 *   cmake -DHAVE_TENSORRT=ON -DTENSORRT_ROOT=/usr/local/tensorrt ..
 *   cmake --build build --target example_plugin_usage
 *   ./build/examples/tensorrt/example_plugin_usage
 */

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cassert>

/* ---- Plugin registration (src/tensorrt/plugin_wrapper.cpp) ---- */
extern "C" void registerConvInt8Plugin();
extern "C" void registerConvReluPlugin();

/* ---- Internal plugin base (needed only to set kernel weights) ---- */
namespace kernel_craft { namespace tensorrt {
    struct KernelCraftBasePlugin;
    void setKernelWeights(KernelCraftBasePlugin*, const float*, size_t);
}}

/* =========================================================================
 * Logger — required by nvinfer1::createInferBuilder
 * ========================================================================= */

/**
 * \brief Minimal TensorRT logger that forwards warnings and errors to stderr.
 *
 * In a production serving system, route TRT log output to your application's
 * structured logger (e.g. spdlog, glog) at the appropriate log level.
 */
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRT] " << msg << "\n";
    }
} gLogger;

/* =========================================================================
 * Helper: initialise convolution weights
 * ========================================================================= */

/**
 * \brief Fill a host weight buffer with He-normal approximation.
 *
 * In production, load weights from a .onnx or .npz file exported by PyTorch:
 * \code
 *   torch.onnx.export(model, dummy_input, "model.onnx")
 *   // Then parse with TRT's OnnxParser, which calls plugin creators
 *   // automatically when it encounters custom op types.
 * \endcode
 *
 * \param[out] buf   Pointer to float buffer of size `n`.
 * \param[in]  n     Number of weight elements.
 * \param[in]  fan_in  Fan-in (= C_in × kH × kW) for He initialisation.
 */
static void init_weights(float* buf, size_t n, int fan_in) {
    float std = std::sqrt(2.f / fan_in);
    for (size_t i = 0; i < n; ++i) {
        /* Box-Muller (two uniform samples → one Gaussian) */
        float u1 = (rand() % 10000 + 1) / 10000.f;
        float u2 = (rand() % 10000 + 1) / 10000.f;
        buf[i] = std * std::sqrt(-2.f * std::log(u1)) * std::cos(6.2832f * u2);
    }
}

/* =========================================================================
 * Build: two ConvReLU plugin layers + built-in pool + FC
 * ========================================================================= */

/**
 * \brief Build a TensorRT engine for the small CNN described in the file header.
 *
 * Steps:
 * 1. Register plugins so TRT's plugin registry can find their creators.
 * 2. Build the network graph with two KernelCraft_ConvReLU layers, one
 *    MaxPool, and one FC (MatrixMultiply + bias) layer.
 * 3. Serialise the engine to a binary blob for deployment.
 *
 * \par Workspace
 * Plugins declare their workspace requirements in getWorkspaceSize().
 * TRT allocates a shared workspace buffer and passes a pointer to enqueue().
 * Our ConvReLU plugin uses tiled shared-memory convolution which requires
 * no additional workspace beyond the thread-local shared memory.
 *
 * \param[in]  batch  Static batch size for this engine (dynamic batch requires
 *                    EXPLICIT_BATCH flag + OptimizationProfiles).
 * \return Serialised engine blob (caller owns memory).
 */
static nvinfer1::IHostMemory* build_conv_relu_engine(int batch = 1) {
    std::cout << "\n=== Building CNN Engine (ConvReLU plugin × 2) ===\n";

    /* ---- Step 1: register plugins ---- */
    registerConvReluPlugin();
    std::cout << "  Plugins registered.\n";

    /* ---- Step 2: create builder & network ---- */
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    /* EXPLICIT_BATCH enables dynamic shapes; omit for static batch. */
    const auto flags = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* net = builder->createNetworkV2(flags);

    /* ---- Step 3: input tensor  [batch, 3, 224, 224] ---- */
    nvinfer1::ITensor* input = net->addInput(
        "input", nvinfer1::DataType::kFLOAT,
        nvinfer1::Dims4{batch, 3, 224, 224});
    std::cout << "  Input: [" << batch << ", 3, 224, 224]\n";

    /* ---- Step 4: ConvReLU layer 1  (3 → 64 channels, 3×3) ---- */
    auto* cr1_creator = nvinfer1::getPluginRegistry()
                            ->getPluginCreator("KernelCraft_ConvReLU", "1");
    assert(cr1_creator && "ConvReLU plugin not registered");

    /* No field initialisation needed — weights are set via setKernelWeights
     * after plugin creation (below). */
    nvinfer1::PluginFieldCollection empty_fc{0, nullptr};
    auto* cr1_plugin = cr1_creator->createPlugin("conv_relu_1", &empty_fc);

    /* Set conv weights: 64 × 3 × 3 × 3  (C_out × C_in × kH × kW) */
    const int C0 = 3, C1 = 64, K = 3;
    const size_t w1_n = (size_t)C1 * C0 * K * K;
    std::vector<float> h_w1(w1_n);
    init_weights(h_w1.data(), w1_n, C0 * K * K);
    kernel_craft::tensorrt::setKernelWeights(
        static_cast<kernel_craft::tensorrt::KernelCraftBasePlugin*>(cr1_plugin),
        h_w1.data(), w1_n * sizeof(float));

    nvinfer1::ITensor* cr1_in = input;
    auto* cr1_layer = net->addPluginV2(&cr1_in, 1, *cr1_plugin);
    cr1_layer->setName("conv_relu_1");
    std::cout << "  conv_relu_1: [" << batch << ", 64, 224, 224]\n";

    /* ---- Step 5: ConvReLU layer 2  (64 → 128 channels, 3×3) ---- */
    auto* cr2_plugin = cr1_creator->createPlugin("conv_relu_2", &empty_fc);

    const int C2 = 128;
    const size_t w2_n = (size_t)C2 * C1 * K * K;
    std::vector<float> h_w2(w2_n);
    init_weights(h_w2.data(), w2_n, C1 * K * K);
    kernel_craft::tensorrt::setKernelWeights(
        static_cast<kernel_craft::tensorrt::KernelCraftBasePlugin*>(cr2_plugin),
        h_w2.data(), w2_n * sizeof(float));

    nvinfer1::ITensor* cr2_in = cr1_layer->getOutput(0);
    auto* cr2_layer = net->addPluginV2(&cr2_in, 1, *cr2_plugin);
    cr2_layer->setName("conv_relu_2");
    std::cout << "  conv_relu_2: [" << batch << ", 128, 224, 224]\n";

    /* ---- Step 6: 2×2 max-pool, stride 2  → [batch, 128, 112, 112] ---- */
    auto* pool = net->addPoolingNd(*cr2_layer->getOutput(0),
                                   nvinfer1::PoolingType::kMAX,
                                   nvinfer1::DimsHW{2, 2});
    pool->setStrideNd(nvinfer1::DimsHW{2, 2});
    pool->setName("maxpool");
    std::cout << "  maxpool: [" << batch << ", 128, 112, 112]\n";

    /* ---- Step 7: flatten + FC  128×112×112 → 1000 ---- */
    /* TRT's Shuffle layer reshapes [batch, 128, 112, 112] → [batch, N_flat]. */
    auto* flatten = net->addShuffle(*pool->getOutput(0));
    flatten->setReshapeDimensions(nvinfer1::Dims2{batch, 128 * 112 * 112});
    flatten->setName("flatten");

    const int N_flat = 128 * 112 * 112, N_cls = 1000;
    std::vector<float> h_fc(N_flat * N_cls, 0.001f);
    nvinfer1::Weights fc_w{nvinfer1::DataType::kFLOAT, h_fc.data(), N_flat * N_cls};
    nvinfer1::Weights fc_b{nvinfer1::DataType::kFLOAT, nullptr, 0};
    auto* fc = net->addFullyConnected(*flatten->getOutput(0), N_cls, fc_w, fc_b);
    fc->setName("fc_1000");
    std::cout << "  fc_1000: [" << batch << ", 1000]\n";

    /* ---- Step 8: mark output and build engine ---- */
    net->markOutput(*fc->getOutput(0));

    nvinfer1::IBuilderConfig* cfg = builder->createBuilderConfig();
    /* Workspace for cuDNN / custom plugin usage (256 MB typical). */
    cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 256ULL << 20);

    nvinfer1::IHostMemory* engine_blob =
        builder->buildSerializedNetwork(*net, *cfg);

    if (engine_blob)
        std::cout << "  Engine built: " << engine_blob->size() / 1024
                  << " KB serialised.\n";
    else
        std::cerr << "  ERROR: Engine build failed.\n";

    cfg->destroy(); net->destroy(); builder->destroy();
    return engine_blob;
}

/* =========================================================================
 * Build: INT8 quantized conv engine
 * ========================================================================= */

/**
 * \brief Demonstrate INT8 convolution plugin with per-layer quantization scales.
 *
 * INT8 convolution is ~2-4× faster than FP32 on Volta+ GPUs that have
 * INT8 Tensor Cores.  The quantisation scales are typically computed via:
 * - Post-training quantisation (PTQ) with a calibration dataset (512–1024 images).
 * - Quantisation-aware training (QAT) with fake-quant nodes.
 *
 * The ConvInt8 plugin accepts pre-computed scales so it can be used with any
 * external quantisation toolkit without coupling to TRT's calibration API.
 *
 * \param[in]  input_scale   Activation quantisation scale (max_abs / 127).
 * \param[in]  kernel_scale  Weight quantisation scale (max_abs / 127).
 * \param[in]  output_scale  Output quantisation scale for next-layer consumption.
 */
static void demo_int8_plugin(float input_scale, float kernel_scale,
                              float output_scale) {
    std::cout << "\n=== INT8 ConvPlugin with calibration scales ===\n";

    registerConvInt8Plugin();

    auto* creator = nvinfer1::getPluginRegistry()
                        ->getPluginCreator("KernelCraft_ConvInt8", "1");
    if (!creator) {
        std::cerr << "  ConvInt8 plugin not found in registry.\n";
        return;
    }

    /* Pass quantisation scales as PluginField entries.
     * TRT serialises these fields into the engine binary so the plugin can
     * recover them when the engine is deserialised on a different machine. */
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
    std::cout << "  → Quantised values: int8 = round(float / scale)\n";
    std::cout << "  → INT8 Tensor Core GEMM runs at ~4× FP32 throughput on Ampere.\n";

    plugin->destroy();
    std::cout << "  INT8 plugin demo complete.\n";
}

/* =========================================================================
 * Run inference with a deserialised engine
 * ========================================================================= */

/**
 * \brief Deserialise an engine and run one inference forward pass.
 *
 * In a production serving framework (Triton, vLLM-TRT, custom):
 * 1. Deserialise once at model load.
 * 2. Allocate I/O device buffers once (reuse across requests).
 * 3. Call enqueueV3(stream) per request.
 * 4. Use cudaStreamSynchronize only when the host needs the result;
 *    chain with the next request for maximum GPU utilisation.
 *
 * \param[in] blob  Serialised engine blob from buildSerializedNetwork.
 */
static void run_inference(const nvinfer1::IHostMemory* blob) {
    if (!blob) return;
    std::cout << "\n=== Running inference with deserialised engine ===\n";

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(
        blob->data(), blob->size());
    if (!engine) {
        std::cerr << "  Deserialisation failed.\n";
        runtime->destroy();
        return;
    }

    nvinfer1::IExecutionContext* ctx = engine->createExecutionContext();

    /* Allocate I/O buffers for the single engine binding. */
    const int n_bindings = engine->getNbBindings();
    std::vector<void*> buffers(n_bindings);
    for (int i = 0; i < n_bindings; ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        size_t vol = 1;
        for (int j = 0; j < dims.nbDims; ++j) vol *= dims.d[j];
        cudaMalloc(&buffers[i], vol * sizeof(float));
    }

    /* Fill input with ones (represents a normalised image). */
    {
        nvinfer1::Dims in_dims = engine->getBindingDimensions(0);
        size_t in_vol = 1;
        for (int j = 0; j < in_dims.nbDims; ++j) in_vol *= in_dims.d[j];
        std::vector<float> h_input(in_vol, 0.5f);
        cudaMemcpy(buffers[0], h_input.data(), in_vol * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* enqueueV3: async, returns immediately; GPU executes on `stream`. */
    ctx->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    /* Read back logits. */
    nvinfer1::Dims out_dims = engine->getBindingDimensions(n_bindings - 1);
    size_t out_vol = 1;
    for (int j = 0; j < out_dims.nbDims; ++j) out_vol *= out_dims.d[j];
    std::vector<float> h_logits(out_vol);
    cudaMemcpy(h_logits.data(), buffers[n_bindings - 1],
               out_vol * sizeof(float), cudaMemcpyDeviceToHost);

    /* Find argmax (predicted class). */
    int pred = (int)(std::max_element(h_logits.begin(), h_logits.end())
                     - h_logits.begin());
    std::cout << "  Predicted class: " << pred
              << "  logit: " << h_logits[pred] << "\n";
    std::cout << "  (Random weights → meaningless prediction, but pipeline runs.)\n";

    /* Cleanup */
    for (void* b : buffers) cudaFree(b);
    cudaStreamDestroy(stream);
    ctx->destroy(); engine->destroy(); runtime->destroy();
}

/* =========================================================================
 * main
 * ========================================================================= */

int main() {
    std::cout << "\n================================================\n";
    std::cout << "  kernel-craft TensorRT Plugin Example\n";
    std::cout << "  Small CNN: Conv×2 → Pool → FC (ImageNet)\n";
    std::cout << "================================================\n";

    try {
        /* Build FP32 engine with ConvReLU plugins. */
        nvinfer1::IHostMemory* blob = build_conv_relu_engine(/*batch=*/1);

        /* Run one forward pass. */
        run_inference(blob);

        if (blob) blob->destroy();

        /* Demonstrate INT8 plugin with calibration scales.
         * Scales are typically derived from:
         *   scale = max_abs_value(calibration_set) / 127.0 */
        demo_int8_plugin(/*input_scale=*/0.0784f,   /* 10.0 / 127 */
                         /*kernel_scale=*/0.00394f,  /* 0.5 / 127 */
                         /*output_scale=*/0.0157f);  /* 2.0 / 127 */

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n=== Example complete ===\n";
    return 0;
}
