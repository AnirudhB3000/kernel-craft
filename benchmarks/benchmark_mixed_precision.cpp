/**
 * \file benchmark_mixed_precision.cpp
 * \brief Benchmark for mixed precision (FP32, FP16, TF32) performance.
 *
 * \par Description
 * Measures the throughput of different precision modes:
 * - FP32: Standard single-precision (baseline)
 * - FP16: Half-precision with Tensor Cores (Volta+)
 * - TF32: TensorFloat-32 on Ampere+ GPUs
 *
 * \par Benchmark Method
 * Run identical tiled convolution in each precision mode,
 * measure total time over multiple iterations.
 *
 * \par Usage
 * \code
 * ./benchmark_mixed_precision [width] [height] [iterations]
 * \endcode
 * Defaults: 1024x1024, 100 iterations.
 */
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
 * \brief Launch FP32 tiled convolution kernel.
 * \param[in] d_input Input image (FP32) on device.
 * \param[in] d_kernel Convolution kernel (FP32) on device.
 * \param[out] d_output Output image (FP32) on device.
 * \param[in] width Image width.
 * \param[in] height Image height.
 * \param[in] ksize Kernel size.
 * \param[in] block CUDA block dimensions.
 */
extern "C" {
    void conv_tiled_fp32_launch(const float* d_input, const float* d_kernel, float* d_output,
                               int width, int height, int ksize, dim3 block);

    /**
     * \brief Launch FP16 tiled convolution kernel.
     * \param[in] d_input Input image (FP16) on device.
     * \param[in] d_kernel Convolution kernel (FP16) on device.
     * \param[out] d_output Output image (FP16) on device.
     * \param[in] width Image width.
     * \param[in] height Image height.
     * \param[in] ksize Kernel size.
     * \param[in] block CUDA block dimensions.
     */
    void conv_tiled_fp16_launch(const half* d_input, const half* d_kernel, half* d_output,
                                  int width, int height, int ksize, dim3 block);

    /**
     * \brief Launch TF32 tiled convolution kernel.
     * \param[in] d_input Input image on device.
     * \param[in] d_kernel Convolution kernel on device.
     * \param[out] d_output Output image on device.
     * \param[in] width Image width.
     * \param[in] height Image height.
     * \param[in] ksize Kernel size.
     * \param[in] block CUDA block dimensions.
     */
    void conv_tiled_tf32_launch(const float* d_input, const float* d_kernel, float* d_output,
                                int width, int height, int ksize, dim3 block);

    /**
     * \brief Convert FP32 array to FP16 (GPU batched).
     * \param[in] src Source FP32 array on device.
     * \param[out] dst Destination FP16 array on device.
     * \param[in] size Number of elements.
     */
    void convert_fp32_to_fp16_batch(const float* src, half* dst, int size);

    /**
     * \brief Convert FP16 array to FP32 (GPU batched).
     * \param[in] src Source FP16 array on device.
     * \param[out] dst Destination FP32 array on device.
     * \param[in] size Number of elements.
     */
    void convert_fp16_to_fp32_batch(const half* src, float* dst, int size);

    /**
     * \brief Query FP16/Tensor Core support.
     * \return 1 if supported (Volta+), 0 otherwise.
     */
    int get_fp16_support();

    /**
     * \brief Query TF32 support.
     * \return 1 if supported (Ampere+), 0 otherwise.
     */
    int get_tf32_support();
}

/**
 * \brief Main entry point for mixed precision benchmark.
 *
 * \par Benchmark Overview
 * Measures throughput of:
 * - FP32: Standard single-precision (baseline)
 * - FP16: Half-precision (requires Tensor Cores)
 * - TF32: TensorFloat-32 (automatic on Ampere+)
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

    // Allocate FP32 device buffers
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, sizeof(float) * imgSize);
    cudaMalloc(&d_kernel, sizeof(float) * kerSize);
    cudaMalloc(&d_output, sizeof(float) * imgSize);

    // Copy input data to device
    cudaMemcpy(d_input, h_input, sizeof(float) * imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float) * kerSize, cudaMemcpyHostToDevice);

    // Create CUDA events for high-precision timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Use 8x8 blocks for tiled convolution
    dim3 block(8, 8, 1);

    // ------------------------------------------------
    // Benchmark 1: FP32 (baseline)
    // Standard single-precision convolution
    // ------------------------------------------------
    cudaEventRecord(start);
    for (int i = 0; i < numIterations; ++i) {
        conv_tiled_fp32_launch(d_input, d_kernel, d_output, width, height, ksize, block);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float fp32_ms = 0.0f;
    cudaEventElapsedTime(&fp32_ms, start, stop);

    // Allocate FP16 device buffers
    half *d_input_fp16, *d_kernel_fp16, *d_output_fp16;
    cudaMalloc(&d_input_fp16, sizeof(half) * imgSize);
    cudaMalloc(&d_kernel_fp16, sizeof(half) * kerSize);
    cudaMalloc(&d_output_fp16, sizeof(half) * imgSize);

    // Convert FP32 input to FP16 once (reused for all iterations)
    convert_fp32_to_fp16_batch(d_input, d_input_fp16, imgSize);
    convert_fp32_to_fp16_batch(d_kernel, d_kernel_fp16, kerSize);

    // ------------------------------------------------
    // Benchmark 2: FP16 (half precision)
    // Uses Tensor Cores on Volta+ for 2x throughput
    // ------------------------------------------------
    if (get_fp16_support()) {
        cudaEventRecord(start);
        for (int i = 0; i < numIterations; ++i) {
            conv_tiled_fp16_launch(d_input_fp16, d_kernel_fp16, d_output_fp16, width, height, ksize, block);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float fp16_ms = 0.0f;
        cudaEventElapsedTime(&fp16_ms, start, stop);

        // Compute throughput and speedup
        double num_pixels = static_cast<double>(imgSize * numIterations);
        double fp32_throughput = num_pixels / (fp32_ms * 1e-3) / 1e6;  // MPixels/s
        double fp16_throughput = num_pixels / (fp16_ms * 1e-3) / 1e6;  // MPixels/s
        double speedup = fp32_ms / fp16_ms;

        // Print results
        printf("=== Mixed Precision Benchmark ===\n");
        printf("Image size: %dx%d (%d iterations)\n", width, height, numIterations);
        printf("FP32: %.3f ms (%.3f MPixels/s)\n", fp32_ms, fp32_throughput);
        printf("FP16: %.3f ms (%.3f MPixels/s)\n", fp16_ms, fp16_throughput);
        printf("Speed-up: %.2fx\n", speedup);

        // Write to report file
        ensure_reports_dir();
        FILE* f = fopen("/home/aniru/kernel-craft/reports/benchmark_mixed_precision.txt", "w");
        if (f) {
            fprintf(f, "=== Mixed Precision Benchmark ===\n");
            fprintf(f, "Image size: %dx%d (%d iterations)\n", width, height, numIterations);
            fprintf(f, "FP32: %.3f ms (%.3f MPixels/s)\n", fp32_ms, fp32_throughput);
            fprintf(f, "FP16: %.3f ms (%.3f MPixels/s)\n", fp16_ms, fp16_throughput);
            fprintf(f, "Speed-up: %.2fx\n", speedup);
            fclose(f);
        }

        // Clean up FP16 buffers
        cudaFree(d_input_fp16);
        cudaFree(d_kernel_fp16);
        cudaFree(d_output_fp16);
    }

    // ------------------------------------------------
    // Benchmark 3: TF32 (TensorFloat-32)
    // Automatic precision on Ampere+ with minimal overhead
    // ------------------------------------------------
    if (get_tf32_support()) {
        cudaEventRecord(start);
        for (int i = 0; i < numIterations; ++i) {
            conv_tiled_tf32_launch(d_input, d_kernel, d_output, width, height, ksize, block);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float tf32_ms = 0.0f;
        cudaEventElapsedTime(&tf32_ms, start, stop);

        // Compute throughput and speedup vs FP32
        double num_pixels = static_cast<double>(imgSize * numIterations);
        double tf32_throughput = num_pixels / (tf32_ms * 1e-3) / 1e6;  // MPixels/s
        double speedup = fp32_ms / tf32_ms;

        printf("TF32: %.3f ms (%.3f MPixels/s)\n", tf32_ms, tf32_throughput);
        printf("TF32 speed-up vs FP32: %.2fx\n", speedup);
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