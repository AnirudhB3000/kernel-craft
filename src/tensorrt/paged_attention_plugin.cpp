/**
 * \file paged_attention_plugin.cpp
 * \brief TensorRT custom plugin for PagedAttention decode.
 *
 * Wraps `launch_paged_attention` as a TensorRT IPluginV2DynamicExt plugin,
 * following the same pattern as plugin_wrapper.cpp (ConvInt8Plugin / ConvReLUPlugin).
 *
 * \par Plugin I/O
 * Inputs:  [Q: float[B,H,d], block_table: int[B,max_blocks],
 *           K_pool: float[np,bs,H_kv,d], V_pool: float[np,bs,H_kv,d],
 *           seq_lens: int[B]]
 * Outputs: [O: float[B,H,d]]
 *
 * \par Usage
 * \code
 * PagedAttentionPlugin plugin(H, H_kv, d, block_size, max_blocks);
 * // Register via PagedAttentionPluginCreator::createPlugin(...)
 * \endcode
 *
 * \par Compilation note
 * Only compiled when HAVE_TENSORRT is defined (TensorRT detected by CMake).
 */

#ifdef HAVE_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>

// Forward declaration of the CUDA launcher
extern "C" void launch_paged_attention(
    const float* Q, const int* block_table,
    const float* K_pool, const float* V_pool,
    float* O, const int* seq_lens,
    int B, int H, int H_kv, int d,
    int block_size, int max_blocks,
    cudaStream_t stream);

static const char* PAGED_ATTN_PLUGIN_NAME    = "PagedAttentionPlugin";
static const char* PAGED_ATTN_PLUGIN_VERSION = "1";

// ---------------------------------------------------------------------------
// Plugin implementation
// ---------------------------------------------------------------------------

class PagedAttentionPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    /**
     * \brief Construct PagedAttentionPlugin with attention configuration.
     *
     * \param[in] H          Number of query heads.
     * \param[in] H_kv       Number of KV heads.
     * \param[in] d          Head dimension.
     * \param[in] block_size Tokens per physical KV page.
     * \param[in] max_blocks Max logical blocks per sequence.
     */
    PagedAttentionPlugin(int H, int H_kv, int d, int block_size, int max_blocks)
        : H_(H), H_kv_(H_kv), d_(d), block_size_(block_size), max_blocks_(max_blocks) {}

    PagedAttentionPlugin(const void* data, size_t length) {
        const int* p = static_cast<const int*>(data);
        H_          = p[0];
        H_kv_       = p[1];
        d_          = p[2];
        block_size_ = p[3];
        max_blocks_ = p[4];
    }

    // IPluginV2 interface
    const char* getPluginType()    const noexcept override { return PAGED_ATTN_PLUGIN_NAME; }
    const char* getPluginVersion() const noexcept override { return PAGED_ATTN_PLUGIN_VERSION; }
    int getNbOutputs()             const noexcept override { return 1; }

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override {
        return new PagedAttentionPlugin(H_, H_kv_, d_, block_size_, max_blocks_);
    }

    nvinfer1::DimsExprs getOutputDimensions(
        int /*outputIndex*/,
        const nvinfer1::DimsExprs* inputs,
        int /*nbInputs*/,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override
    {
        // Output: [B, H, d] matching Q layout
        nvinfer1::DimsExprs out;
        out.nbDims = 3;
        out.d[0]   = inputs[0].d[0];  // B
        out.d[1]   = exprBuilder.constant(H_);
        out.d[2]   = exprBuilder.constant(d_);
        return out;
    }

    bool supportsFormatCombination(
        int /*pos*/,
        const nvinfer1::PluginTensorDesc* inOut,
        int /*nbInputs*/, int /*nbOutputs*/) noexcept override
    {
        return inOut[0].format == nvinfer1::TensorFormat::kLINEAR &&
               (inOut[0].type == nvinfer1::DataType::kFLOAT ||
                inOut[0].type == nvinfer1::DataType::kINT32);
    }

    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* /*in*/,
        int /*nbInputs*/,
        const nvinfer1::DynamicPluginTensorDesc* /*out*/,
        int /*nbOutputs*/) noexcept override {}

    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* /*inputs*/, int /*nbInputs*/,
        const nvinfer1::PluginTensorDesc* /*outputs*/, int /*nbOutputs*/)
        const noexcept override { return 0; }

    int enqueue(
        const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* /*outputDesc*/,
        const void* const* inputs,
        void* const* outputs,
        void* /*workspace*/,
        cudaStream_t stream) noexcept override
    {
        int B = inputDesc[0].dims.d[0];
        const float* Q          = static_cast<const float*>(inputs[0]);
        const int*   block_table= static_cast<const int*>  (inputs[1]);
        const float* K_pool     = static_cast<const float*>(inputs[2]);
        const float* V_pool     = static_cast<const float*>(inputs[3]);
        const int*   seq_lens   = static_cast<const int*>  (inputs[4]);
        float*       O          = static_cast<float*>      (outputs[0]);

        launch_paged_attention(Q, block_table, K_pool, V_pool, O, seq_lens,
                               B, H_, H_kv_, d_, block_size_, max_blocks_, stream);
        return 0;
    }

    nvinfer1::DataType getOutputDataType(
        int /*index*/,
        const nvinfer1::DataType* /*inputTypes*/,
        int /*nbInputs*/) const noexcept override
    { return nvinfer1::DataType::kFLOAT; }

    void destroy() noexcept override { delete this; }

    void setPluginNamespace(const char* ns) noexcept override { namespace_ = ns; }
    const char* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

    size_t getSerializationSize() const noexcept override { return 5 * sizeof(int); }
    void serialize(void* buffer) const noexcept override {
        int* p = static_cast<int*>(buffer);
        p[0] = H_; p[1] = H_kv_; p[2] = d_; p[3] = block_size_; p[4] = max_blocks_;
    }
    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}

private:
    int H_, H_kv_, d_, block_size_, max_blocks_;
    std::string namespace_;
};

// ---------------------------------------------------------------------------
// Plugin creator
// ---------------------------------------------------------------------------

class PagedAttentionPluginCreator : public nvinfer1::IPluginCreator {
public:
    const char* getPluginName()    const noexcept override { return PAGED_ATTN_PLUGIN_NAME; }
    const char* getPluginVersion() const noexcept override { return PAGED_ATTN_PLUGIN_VERSION; }
    const char* getPluginNamespace() const noexcept override { return ""; }

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
        static nvinfer1::PluginFieldCollection fc{0, nullptr};
        return &fc;
    }

    nvinfer1::IPluginV2* createPlugin(
        const char* /*name*/,
        const nvinfer1::PluginFieldCollection* fc) noexcept override
    {
        int H = 8, H_kv = 8, d = 64, block_size = 16, max_blocks = 64;
        for (int i = 0; i < fc->nbFields; ++i) {
            const auto& f = fc->fields[i];
            if (!strcmp(f.name, "H"))          H          = *static_cast<const int*>(f.data);
            else if (!strcmp(f.name, "H_kv"))  H_kv       = *static_cast<const int*>(f.data);
            else if (!strcmp(f.name, "d"))     d           = *static_cast<const int*>(f.data);
            else if (!strcmp(f.name, "block_size")) block_size = *static_cast<const int*>(f.data);
            else if (!strcmp(f.name, "max_blocks")) max_blocks = *static_cast<const int*>(f.data);
        }
        return new PagedAttentionPlugin(H, H_kv, d, block_size, max_blocks);
    }

    nvinfer1::IPluginV2* deserializePlugin(
        const char* /*name*/, const void* data, size_t length) noexcept override
    { return new PagedAttentionPlugin(data, length); }

    void setPluginNamespace(const char* /*ns*/) noexcept override {}
};

// Register the creator (auto-registration at library load)
REGISTER_TENSORRT_PLUGIN(PagedAttentionPluginCreator);

#endif // HAVE_TENSORRT
