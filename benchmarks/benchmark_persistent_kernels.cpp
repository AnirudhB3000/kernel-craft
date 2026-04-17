/**
 * \file benchmark_persistent_kernels.cpp
 * \brief Benchmark for persistent kernel mode optimization.
 *
 * \par Description
 * Measures the performance improvement of persistent kernel modes
 * that reduce kernel launch overhead.
 *
 * \par Benchmark Method
 * - Separate: Standard kernel launch each iteration
 * - Persistent: Persistent kernel with work queue
 * - Streaming: Stream-based async execution
 * - Work Queue: Enqueue multiple, execute batch
 *
 * \par Usage
 * \code
 * ./benchmark_persistent_kernels [width] [height] [iterations]
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
 * \brief Launch persistent convolution kernel.
 *
 * Persistent kernel that stays resident on GPU for reduced launch overhead.
 *
 * \param[in] d_input Input image on device.
 * \param[in] d_kernel Convolution kernel on device.
 * \param[out] d_output Output image on device.
 * \param[in] width Image width.
 * \param[in] height Image height.
 * \param[in] ksize Kernel size.
 * \param[in] block CUDA block dimensions.
 */
extern "C" {
    void conv_persistent_launch(const float* d_input, const float* d_kernel, float* d_output,
                           int width, int height, int ksize, dim3 block);

    /**
     * \brief Launch streaming convolution kernel.
     *
     * Asynchronous execution on specified stream for overlapping
     * kernel execution with data transfers.
     *
     * \param[in] d_input Input image on device.
     * \param[in] d_kernel Convolution kernel on device.
     * \param[out] d_output Output image on device.
     * \param[in] width Image width.
     * \param[in] height Image height.
     * \param[in] ksize Kernel size.
     * \param[in] block CUDA block dimensions.
     * \param[in] stream CUDA stream for asynchronous execution.
     */
    void conv_streaming(const float* d_input, const float* d_kernel, float* d_output,
                       int width, int height, int ksize, dim3 block, cudaStream_t stream);

    /**
     * \brief Enqueue work item for batch execution.
     *
     * Adds work item to internal queue for later batch execution.
     *
     * \param[in] d_input Input image on device.
     * \param[in] d_kernel Convolution kernel on device.
     * \param[out] d_output Output image on device.
     * \param[in] width Image width.
     * \param[in] height Image height.
     * \param[in] ksize Kernel size.
     * \return 0 on success, -1 if queue is full.
     */
    int conv_work_enqueue(const float* d_input, const float* d_kernel, float* d_output,
                      int width, int height, int ksize);

    /**
     * \brief Execute all enqueued work items.
     *
     * Executes all work items that were previously enqueued.
     *
     * \param[in] block CUDA block dimensions.
     */
    void conv_work_execute(dim3 block);

    /**
     * \brief Clear the work queue.
     *
     * Removes all items from the work queue without executing.
     */
    void conv_work_clear();

    /**
     * \brief Launch standard tiled convolution kernel.
     *
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
}

/**
 * \brief Main entry point for persistent kernels benchmark.
 *
 * \par Benchmark Overview
 * Measures performance of different kernel execution modes:
 * - Separate: Standard kernel launch each iteration (baseline)
 * - Persistent: Kernel stays resident, reused across calls
 * - Work Queue: Batch multiple items, execute together
 * - Streaming: Async execution on CUDA streams
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

    // Use 8x8 blocks for tiled convolution
    dim3 block(8, 8, 1);

    // ------------------------------------------------
    // Benchmark 1: Separate kernel launches (baseline)
    // Standard approach with full launch overhead each iteration
    // ------------------------------------------------
    cudaEventRecord(start);
    for (int i = 0; i < numIterations; ++i) {
        conv_tiled(d_input, d_kernel, d_output, width, height, ksize, block);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float separate_ms = 0.0f;
    cudaEventElapsedTime(&separate_ms, start, stop);

    // ------------------------------------------------
    // Benchmark 2: Persistent kernel launch
    // Kernel stays resident on GPU, reducing launch overhead
    // ------------------------------------------------
    cudaEventRecord(start);
    for (int i = 0; i < numIterations; ++i) {
        conv_persistent_launch(d_input, d_kernel, d_output, width, height, ksize, block);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float persistent_ms = 0.0f;
    cudaEventElapsedTime(&persistent_ms, start, stop);

    // ------------------------------------------------
    // Benchmark 3: Work queue (batch execution)
    // Enqueue all items, execute in single batch call
    // ------------------------------------------------
    conv_work_clear();  // Ensure queue is empty
    for (int i = 0; i < numIterations; ++i) {
        conv_work_enqueue(d_input, d_kernel, d_output, width, height, ksize);
    }
    cudaEventRecord(start);
    conv_work_execute(block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float queue_ms = 0.0f;
    cudaEventElapsedTime(&queue_ms, start, stop);
    conv_work_clear();  // Clean up queue for next test

    // ------------------------------------------------
    // Benchmark 4: Streaming mode
    // Asynchronous execution on CUDA streams
    // Allows overlapping kernel execution with data transfers
    // ------------------------------------------------
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEventRecord(start);
    for (int i = 0; i < numIterations; ++i) {
        conv_streaming(d_input, d_kernel, d_output, width, height, ksize, block, stream);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float streaming_ms = 0.0f;
    cudaEventElapsedTime(&streaming_ms, start, stop);
    cudaStreamDestroy(stream);

    // ------------------------------------------------
    // Compute throughput metrics and speedup
    // ------------------------------------------------
    double num_pixels = static_cast<double>(imgSize * numIterations);
    double separate_throughput = num_pixels / (separate_ms * 1e-3) / 1e6;  // MPixels/s
    double persistent_throughput = num_pixels / (persistent_ms * 1e-3) / 1e6;  // MPixels/s
    double queue_throughput = num_pixels / (queue_ms * 1e-3) / 1e6;  // MPixels/s
    double streaming_throughput = num_pixels / (streaming_ms * 1e-3) / 1e6;  // MPixels/s

    // Print results to console
    printf("=== Persistent Kernels Benchmark ===\n");
    printf("Image size: %dx%d (%d iterations)\n", width, height, numIterations);
    printf("Separate: %.3f ms (%.3f MPixels/s)\n", separate_ms, separate_throughput);
    printf("Persistent: %.3f ms (%.3f MPixels/s)\n", persistent_ms, persistent_throughput);
    printf("Work queue: %.3f ms (%.3f MPixels/s)\n", queue_ms, queue_throughput);
    printf("Streaming: %.3f ms (%.3f MPixels/s)\n", streaming_ms, streaming_throughput);
    printf("Speed-up (persistent vs separate): %.2fx\n", separate_ms / persistent_ms);

    // Write results to report file for later analysis
    ensure_reports_dir();
    FILE* f = fopen("/home/aniru/kernel-craft/reports/benchmark_persistent_kernels.txt", "w");
    if (f) {
        fprintf(f, "=== Persistent Kernels Benchmark ===\n");
        fprintf(f, "Image size: %dx%d (%d iterations)\n", width, height, numIterations);
        fprintf(f, "Separate: %.3f ms (%.3f MPixels/s)\n", separate_ms, separate_throughput);
        fprintf(f, "Persistent: %.3f ms (%.3f MPixels/s)\n", persistent_ms, persistent_throughput);
        fprintf(f, "Work queue: %.3f ms (%.3f MPixels/s)\n", queue_ms, queue_throughput);
        fprintf(f, "Streaming: %.3f ms (%.3f MPixels/s)\n", streaming_ms, streaming_throughput);
        fprintf(f, "Speed-up (persistent vs separate): %.2fx\n", separate_ms / persistent_ms);
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