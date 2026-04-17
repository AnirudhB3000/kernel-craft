/**
 * \file benchmark_memory_pool.cpp
 * \brief Benchmark for memory pool performance optimization.
 *
 * \par Description
 * Measures the performance improvement of pre-allocated memory pools
 * vs per-iteration cudaMalloc/cudaFree overhead.
 *
 * \par Benchmark Method
 * - Naive: Allocate/free output buffer each iteration
 * - Pooled: Use pre-allocated buffer from pool
 * - Compare total execution time over multiple iterations
 *
 * \par Usage
 * \code
 * ./benchmark_memory_pool [width] [height] [iterations]
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
 * \brief Create a memory pool with pre-allocated buffers.
 * \param[out] out_pool Pointer to receive the pool handle.
 * \param[in] width Image width for buffer sizing.
 * \param[in] height Image height for buffer sizing.
 * \param[in] ksize Kernel size.
 * \param[in] numBuffers Number of buffers to pre-allocate.
 * \return 0 on success, -1 on failure.
 */
extern "C" {
    typedef struct MemPool MemPool;
    int mem_pool_create(MemPool** out_pool, int width, int height, int ksize, int numBuffers);

    /**
     * \brief Destroy a memory pool and free all buffers.
     * \param[in] pool Pool handle to destroy.
     * \return 0 on success, -1 on failure.
     */
    int mem_pool_destroy(MemPool* pool);

    /**
     * \brief Allocate a buffer from the pool.
     * \param[in] pool Pool handle.
     * \return Pointer to allocated buffer, or nullptr if none available.
     */
    void* mem_pool_alloc(MemPool* pool);

    /**
     * \brief Return a buffer to the pool.
     * \param[in] pool Pool handle.
     * \param[in] ptr Buffer pointer to return.
     * \return 0 on success, -1 on failure.
     */
    int mem_pool_free(MemPool* pool, void* ptr);

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
}

/**
 * \brief Main entry point for memory pool benchmark.
 *
 * \par Benchmark Overview
 * Measures the performance difference between:
 * - Naive: Allocating/dellocating buffer each iteration (simulates naive batching)
 * - Pooled: Using pre-allocated buffers (eliminates allocation overhead)
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

    // Allocate persistent device buffers (input, kernel, output)
    // These stay allocated for the entire benchmark
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

    // ------------------------------------------------
    // Benchmark 1: Naive approach (allocate/free each iteration)
    // This simulates the overhead seen in naive batch processing
    // ------------------------------------------------
    float *d_output_naive;
    cudaMalloc(&d_output_naive, sizeof(float) * imgSize);

    // Record start time, execute iterations, record stop time
    cudaEventRecord(start);
    for (int i = 0; i < numIterations; ++i) {
        // Allocate new buffer for each iteration
        cudaMalloc(&d_output_naive, sizeof(float) * imgSize);
        // Execute convolution kernel
        conv_tiled(d_input, d_kernel, d_output_naive, width, height, ksize, dim3(16, 16));
        // Free buffer after use
        cudaFree(d_output_naive);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float naive_ms = 0.0f;
    cudaEventElapsedTime(&naive_ms, start, stop);

    // ------------------------------------------------
    // Benchmark 2: Pooled approach (pre-allocated buffers)
    // Creates pool once, reuses buffers for all iterations
    // ------------------------------------------------
    MemPool* pool;
    mem_pool_create(&pool, width, height, ksize, 16);

    cudaEventRecord(start);
    for (int i = 0; i < numIterations; ++i) {
        // Get buffer from pool (no allocation overhead)
        void* buf = mem_pool_alloc(pool);
        // Execute convolution kernel
        conv_tiled(d_input, d_kernel, (float*)buf, width, height, ksize, dim3(16, 16));
        // Return buffer to pool (no free overhead)
        mem_pool_free(pool, buf);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float pool_ms = 0.0f;
    cudaEventElapsedTime(&pool_ms, start, stop);

    // Clean up pool
    mem_pool_destroy(pool);

    // ------------------------------------------------
    // Compute throughput metrics and speedup
    // ------------------------------------------------
    double num_pixels = static_cast<double>(imgSize * numIterations);
    double naive_throughput = num_pixels / (naive_ms * 1e-3) / 1e6;  // MPixels/s
    double pool_throughput = num_pixels / (pool_ms * 1e-3) / 1e6;  // MPixels/s
    double speedup = naive_ms / pool_ms;

    // Print results to console
    printf("=== Memory Pool Benchmark ===\n");
    printf("Image size: %dx%d (%d iterations)\n", width, height, numIterations);
    printf("Naive (malloc/free): %.3f ms (%.3f MPixels/s)\n", naive_ms, naive_throughput);
    printf("Pooled (pre-allocated): %.3f ms (%.3f MPixels/s)\n", pool_ms, pool_throughput);
    printf("Speed-up: %.2fx\n", speedup);

    // Write results to report file for later analysis
    ensure_reports_dir();
    FILE* f = fopen("/home/aniru/kernel-craft/reports/benchmark_memory_pool.txt", "w");
    if (f) {
        fprintf(f, "=== Memory Pool Benchmark ===\n");
        fprintf(f, "Image size: %dx%d (%d iterations)\n", width, height, numIterations);
        fprintf(f, "Naive (malloc/free): %.3f ms (%.3f MPixels/s)\n", naive_ms, naive_throughput);
        fprintf(f, "Pooled (pre-allocated): %.3f ms (%.3f MPixels/s)\n", pool_ms, pool_throughput);
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