/**
 * \file benchmark_cuda_graphs.cpp
 * \brief Benchmark for CUDA Graphs performance optimization.
 *
 * \par Description
 * Measures the performance improvement of CUDA Graphs
 * (captured kernel DAG) vs separate kernel launches.
 *
 * \par Benchmark Method
 * - Separate: Launch conv+relu+bias kernels individually
 * - Graph: Capture kernels in graph, launch graph repeatedly
 * - Compare total execution time over multiple iterations
 *
 * \par Usage
 * \code
 * ./benchmark_cuda_graphs [width] [height] [iterations]
 * \endcode
 * Defaults: 1024x1024, 100 iterations.
 */
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include <string>
#include <sys/stat.h>

/**
 * \brief Ensure reports directory exists.
 *
 * Creates the /home/aniru/kernel-craft/reports directory if it doesn't exist.
 * This is where benchmark results are written for later analysis.
 */
void ensure_reports_dir() {
    mkdir("/home/aniru/kernel-craft/reports", 0755);
}

/**
 * \brief Create a CUDA graph from the pipeline kernels.
 *
 * Captures conv+relu+bias kernels into a graph for efficient
 * repeated execution without per-kernel launch overhead.
 *
 * \param[in] d_input Input image on device.
 * \param[in] d_kernel Convolution kernel on device.
 * \param[out] d_output Output image on device.
 * \param[in] width Image width.
 * \param[in] height Image height.
 * \param[in] ksize Kernel size.
 * \param[in] block CUDA block dimensions.
 * \return 0 on success, -1 on failure.
 */
extern "C" {
    int conv_graph_create(const float* d_input, const float* d_kernel, float* d_output,
                         int width, int height, int ksize, dim3 block);

    /**
     * \brief Launch the captured CUDA graph.
     *
     * Executes the previously captured graph (conv+relu+bias pipeline).
     *
     * \param[in] stream CUDA stream (0 for default stream).
     * \return 0 on success, -1 on failure.
     */
    int conv_graph_launch(cudaStream_t stream = 0);

    /**
     * \brief Destroy the CUDA graph and release resources.
     * \return 0 on success, -1 on failure.
     */
    int conv_graph_destroy();

    /**
     * \brief Execute pipeline with separate kernel launches.
     * \param[in] d_input Input image on device.
     * \param[in] d_kernel Convolution kernel on device.
     * \param[out] d_output Output image on device.
     * \param[in] bias Bias value to add.
     * \param[in] width Image width.
     * \param[in] height Image height.
     * \param[in] ksize Kernel size.
     * \param[in] block CUDA block dimensions.
     */
    void conv_pipeline_separate(const float* d_input, const float* d_kernel, float* d_output,
                                 float bias, int width, int height, int ksize, dim3 block);

    /**
     * \brief Launch tiled convolution kernel.
     * \param[in] d_input Input image on device.
     * \param[in] d_kernel Convolution kernel on device.
     * \param[out] d_output Output image on device.
     * \param[in] width Image width.
     * \param[in] height Image height.
     * \param[in] ksize Kernel size.
     * \param[in] block CUDA block dimensions.
     */
    void conv_tiled(const float* d_input, const float* d_kernel, float* d_output,
                 int width, int height, int ksize, dim3 block);

    /**
     * \brief Apply ReLU activation in-place.
     * \param[in,out] d_data Data to apply ReLU to.
     * \param[in] size Number of elements.
     */
    void relu(float* d_data, int size);

    /**
     * \brief Add bias to each element in-place.
     * \param[in,out] d_data Data to modify.
     * \param[in] bias Bias value to add.
     * \param[in] size Number of elements.
     */
    void add_bias(float* d_data, float bias, int size);
}

/**
 * \brief Main entry point for CUDA graphs benchmark.
 *
 * \par Benchmark Overview
 * Measures the performance difference between:
 * - Separate: Launching conv+relu+bias kernels individually each iteration
 * - Graph: Capturing kernels once, launching graph each iteration
 *
 * \return EXIT_SUCCESS on completion.
 */
int main(int argc, char** argv) {
    // Parse command line arguments with defaults for 1024x1024 image, 100 iterations
    const int width = (argc > 1) ? std::atoi(argv[1]) : 1024;
    const int height = (argc > 2) ? std::atoi(argv[2]) : 1024;
    const int ksize = 3;  // 3x3 averaging kernel
    const int imgSize = width * height;
    const int kerSize = ksize * ksize;
    const int numIterations = (argc > 3) ? std::atoi(argv[3]) : 100;

    // Allocate host buffers for input image and kernel
    float *h_input = (float*)malloc(sizeof(float) * imgSize);
    float *h_kernel = (float*)malloc(sizeof(float) * kerSize);

    // Initialize with deterministic data
    for (int i = 0; i < imgSize; ++i) h_input[i] = static_cast<float>(i + 1);
    for (int i = 0; i < kerSize; ++i) h_kernel[i] = 1.0f / kerSize;

    // Allocate device buffers (persistent for all iterations)
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, sizeof(float) * imgSize);
    cudaMalloc(&d_kernel, sizeof(float) * kerSize);
    cudaMalloc(&d_output, sizeof(float) * imgSize);

    // Copy input data to device (done once, reused for all iterations)
    cudaMemcpy(d_input, h_input, sizeof(float) * imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float) * kerSize, cudaMemcpyHostToDevice);

    // Create CUDA events for high-precision timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Use 16x16 blocks for tiled convolution
    dim3 block(16, 16, 1);

    // ------------------------------------------------
    // Benchmark 1: Separate kernel launches
    // Launch each kernel independently for each iteration
    // This is the baseline approach without graph optimization
    // ------------------------------------------------
    cudaEventRecord(start);
    for (int i = 0; i < numIterations; ++i) {
        // Launch conv kernel
        conv_tiled(d_input, d_kernel, d_output, width, height, ksize, block);
        // Launch relu kernel
        relu(d_output, imgSize);
        // Launch bias kernel
        add_bias(d_output, 0.0f, imgSize);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float separate_ms = 0.0f;
    cudaEventElapsedTime(&separate_ms, start, stop);

    // ------------------------------------------------
    // Benchmark 2: CUDA Graph approach
    // Capture kernels into graph once, launch graph each iteration
    // Graph eliminates per-kernel launch overhead
    // ------------------------------------------------
    conv_graph_create(d_input, d_kernel, d_output, width, height, ksize, block);

    cudaEventRecord(start);
    for (int i = 0; i < numIterations; ++i) {
        // Launch captured graph (all 3 kernels execute)
        conv_graph_launch(0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float graph_ms = 0.0f;
    cudaEventElapsedTime(&graph_ms, start, stop);

    // Clean up graph resources
    conv_graph_destroy();

    // ------------------------------------------------
    // Compute throughput metrics and speedup
    // ------------------------------------------------
    double num_pixels = static_cast<double>(imgSize * numIterations);
    double separate_throughput = num_pixels / (separate_ms * 1e-3) / 1e6;  // MPixels/s
    double graph_throughput = num_pixels / (graph_ms * 1e-3) / 1e6;  // MPixels/s
    double speedup = separate_ms / graph_ms;

    // Print results to console
    printf("=== CUDA Graphs Benchmark ===\n");
    printf("Image size: %dx%d (%d iterations)\n", width, height, numIterations);
    printf("Separate kernels: %.3f ms (%.3f MPixels/s)\n", separate_ms, separate_throughput);
    printf("CUDA Graph: %.3f ms (%.3f MPixels/s)\n", graph_ms, graph_throughput);
    printf("Speed-up: %.2fx\n", speedup);

    // Write results to report file for later analysis
    ensure_reports_dir();
    FILE* f = fopen("/home/aniru/kernel-craft/reports/benchmark_cuda_graphs.txt", "w");
    if (f) {
        fprintf(f, "=== CUDA Graphs Benchmark ===\n");
        fprintf(f, "Image size: %dx%d (%d iterations)\n", width, height, numIterations);
        fprintf(f, "Separate kernels: %.3f ms (%.3f MPixels/s)\n", separate_ms, separate_throughput);
        fprintf(f, "CUDA Graph: %.3f ms (%.3f MPixels/s)\n", graph_ms, graph_throughput);
        fprintf(f, "Speed-up: %.2fx\n", speedup);
        fclose(f);
    }

    // Clean up all resources
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(h_input);
    free(h_kernel);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return EXIT_SUCCESS;
}