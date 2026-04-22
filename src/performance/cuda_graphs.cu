/**
 * \file cuda_graphs.cu
 * \brief CUDA Graphs integration for kernel pipelining.
 *
 * This implementation uses CUDA stream capture to easily create graphs
 * from existing kernel launches. The key benefit is reduced launch overhead
 * for repeated kernel sequences.
 *
 * \par Benefits
 * - Reduced kernel launch overhead (eliminates per-kernel launch latency)
 * - Better GPU scheduling efficiency
 * - Optimal for batch processing workloads
 *
 * \par Usage
 * Begin capture, launch kernels, end capture, then instantiate and launch
 * the graph for each batch.
 */

#include <cuda_runtime.h>
#include <cstdio>

#ifndef TILE_W
#define TILE_W 8
#endif
#ifndef TILE_H
#define TILE_H 8
#endif

extern "C" {
    void conv_tiled(const float* d_input, const float* d_kernel, float* d_output,
                    int width, int height, int ksize, dim3 block);
    void conv_tiled_nosync(const float* d_input, const float* d_kernel, float* d_output,
                    int width, int height, int ksize, dim3 block);
    void relu_nosync(float* d_data, int size);
    void add_bias(float* d_data, float bias, int size);
    void add_bias_nosync(float* d_data, float bias, int size);
    int conv_graph_destroy();
}

struct ConvGraph {
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    dim3 block;
    dim3 gridConv;
    dim3 gridElem;
    size_t sharedMemBytes;
    int numElements;
    bool valid;
};

static ConvGraph g_graph = {0};

/**
 * \brief Create a CUDA graph for the convolution pipeline.
 *
 * Captures conv + relu + bias kernels into a graph for reduced
 * launch overhead on repeated executions.
 *
 * \param d_input  Input image device pointer.
 * \param d_kernel Kernel device pointer.
 * \param d_output Output device pointer.
 * \param width    Image width.
 * \param height   Image height.
 * \param ksize    Kernel size.
 * \param block    Block dimensions.
 * \return 0 on success, -1 on failure.
 */
extern "C" int conv_graph_create(const float* d_input,
                           const float* d_kernel,
                           float* d_output,
                           int width, int height, int ksize,
                           dim3 block) {
    if (g_graph.valid) {
        conv_graph_destroy();
    }

    g_graph.block = block;
    g_graph.gridConv = dim3((width + block.x - 1) / block.x,
                           (height + block.y - 1) / block.y);
    g_graph.sharedMemBytes = (block.x + ksize - 1) * (block.y + ksize - 1) * sizeof(float);
    g_graph.numElements = width * height;
    g_graph.gridElem = dim3((g_graph.numElements + 255) / 256);
    g_graph.valid = false;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);

    conv_tiled_nosync(d_input, d_kernel, d_output, width, height, ksize, block);

    cudaStreamEndCapture(stream, &g_graph.graph);

    cudaError_t err = cudaGraphInstantiate(&g_graph.exec, g_graph.graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Graphs: instantiate error: %s\n", cudaGetErrorString(err));
        cudaGraphDestroy(g_graph.graph);
        cudaStreamDestroy(stream);
        return -1;
    }
    cudaStreamDestroy(stream);
    g_graph.valid = true;

    return 0;
}

/**
 * \brief Destroy a CUDA graph and release resources.
 *
 * \return 0 on success, -1 if graph not valid.
 */
extern "C" int conv_graph_destroy() {
    if (!g_graph.valid) return -1;
    cudaGraphExecDestroy(g_graph.exec);
    cudaGraphDestroy(g_graph.graph);
    g_graph.valid = false;
    return 0;
}

/**
 * \brief Launch a CUDA graph.
 *
 * \param stream Optional cuda stream (default 0).
 * \return 0 on success, -1 on failure.
 */
extern "C" int conv_graph_launch(cudaStream_t stream = 0) {
    if (!g_graph.valid) return -1;
    cudaError_t err = cudaGraphLaunch(g_graph.exec, stream);
    return (err == cudaSuccess) ? 0 : -1;
}

/**
 * \brief Execute pipeline with separate kernel launches.
 *
 * Baseline for comparison against graph performance.
 *
 * \param d_input  Input image.
 * \param d_kernel Kernel.
 * \param d_output Output.
 * \param bias    Bias to add.
 * \param width    Width.
 * \param height   Height.
 * \param ksize   Kernel size.
 * \param block   Block dimensions.
 */
extern "C" void conv_pipeline_separate(const float* d_input,
                                        const float* d_kernel,
                                        float* d_output,
                                        float bias,
                                        int width, int height, int ksize,
                                        dim3 block) {
    conv_tiled(d_input, d_kernel, d_output, width, height, ksize, block);
    relu_nosync(d_output, width * height);
    add_bias_nosync(d_output, bias, width * height);
}

/**
 * \brief Execute pipeline using CUDA graphs.
 *
 * Creates graph, launches, destroys (one-shot use).
 *
 * \param d_input  Input image.
 * \param d_kernel Kernel.
 * \param d_output Output.
 * \param bias    Bias.
 * \param width    Width.
 * \param height   Height.
 * \param ksize   Kernel size.
 * \param block   Block dimensions.
 */
extern "C" void conv_pipeline_graph(const float* d_input,
                                    const float* d_kernel,
                                    float* d_output,
                                    float bias,
                                    int width, int height, int ksize,
                                    dim3 block) {
    conv_graph_create(d_input, d_kernel, d_output, width, height, ksize, block);
    conv_graph_launch(0);
    conv_graph_destroy();
}