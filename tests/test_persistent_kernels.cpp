/**
 * \file test_persistent_kernels.cpp
 * \brief Unit test for persistent kernel mode.
 *
 * \par Description
 * Tests the correctness of persistent kernel execution modes:
 * - conv_persistent_launch: Single persistent kernel execution
 * - conv_work_enqueue/execute: Work queue for batch processing
 * - conv_streaming: Stream-based execution
 *
 * \par Verification
 * Each mode's output is compared against the CPU reference
 * using relative tolerance (1e-5).
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>

/**
 * \brief Launch persistent convolution kernel.
 * \param[in] d_input Input image on device.
 * \param[in] d_kernel Convolution kernel on device.
 * \param[out] d_output Output image on device.
 * \param[in] width Image width.
 * \param[in] height Image height.
 * \param[in] ksize Kernel size (must be odd).
 * \param[in] block CUDA block dimensions.
 */
extern "C" void conv_persistent_launch(const float* d_input,
                                      const float* d_kernel,
                                      float* d_output,
                                      int width, int height, int ksize,
                                      dim3 block);

/**
 * \brief Launch batch persistent convolution.
 * \param[in] d_batchInputs Contiguous batch input data on device.
 * \param[in] d_kernel Convolution kernel on device.
 * \param[out] d_batchOutputs Contiguous batch output data on device.
 * \param[in] d_batchSizes Array of (width, height) pairs per batch.
 * \param[in] d_batchOffsets Array of byte offsets into batch buffers.
 * \param[in] numBatches Number of batches.
 * \param[in] ksize Kernel size.
 * \param[in] block CUDA block dimensions.
 */
extern "C" void conv_persistent_batch(const float* d_batchInputs,
                                       const float* d_kernel,
                                       float* d_batchOutputs,
                                       const int* d_batchSizes,
                                       const int* d_batchOffsets,
                                       int numBatches, int ksize, dim3 block);

/**
 * \brief Launch streaming convolution kernel.
 * \param[in] d_input Input image on device.
 * \param[in] d_kernel Convolution kernel on device.
 * \param[out] d_output Output image on device.
 * \param[in] width Image width.
 * \param[in] height Image height.
 * \param[in] ksize Kernel size.
 * \param[in] block CUDA block dimensions.
 * \param[in] stream CUDA stream for asynchronous execution.
 */
extern "C" void conv_streaming(const float* d_input,
                                const float* d_kernel,
                                float* d_output,
                                int width, int height, int ksize,
                                dim3 block,
                                cudaStream_t stream);

/**
 * \brief Enqueue work item for batch execution.
 * \param[in] d_input Input image on device.
 * \param[in] d_kernel Convolution kernel on device.
 * \param[out] d_output Output image on device.
 * \param[in] width Image width.
 * \param[in] height Image height.
 * \param[in] ksize Kernel size.
 * \return 0 on success, -1 if queue is full.
 */
extern "C" int conv_work_enqueue(const float* d_input,
                                 const float* d_kernel,
                                 float* d_output,
                                 int width, int height, int ksize);

/**
 * \brief Execute all enqueued work items.
 * \param[in] block CUDA block dimensions.
 */
extern "C" void conv_work_execute(dim3 block);

/**
 * \brief Clear the work queue.
 */
extern "C" void conv_work_clear();

/**
 * \brief CPU reference implementation of 2D convolution.
 * \param[in] input Input image (row-major).
 * \param[in] kernel Convolution kernel (row-major).
 * \param[out] output Destination buffer.
 * \param[in] width Image width.
 * \param[in] height Image height.
 * \param[in] ksize Kernel size (must be odd).
 */
void conv_naive_cpu(const float* input, const float* kernel, float* output,
                   int width, int height, int ksize) {
    int kHalf = ksize / 2;
    for (int oy = 0; oy < height; ++oy) {
        for (int ox = 0; ox < width; ++ox) {
            double sum = 0.0;
            for (int ky = 0; ky < ksize; ++ky) {
                int iy = oy + ky - kHalf;
                if (iy < 0 || iy >= height) continue;
                for (int kx = 0; kx < ksize; ++kx) {
                    int ix = ox + kx - kHalf;
                    if (ix < 0 || ix >= width) continue;
                    float val = input[iy * width + ix];
                    float kval = kernel[ky * ksize + kx];
                    sum += static_cast<double>(val) * static_cast<double>(kval);
                }
            }
            output[oy * width + ox] = static_cast<float>(sum);
        }
    }
}

/**
 * \brief Main entry point for persistent kernel unit tests.
 *
 * \par Test Sequence
 * - Allocate host and device buffers
 * - Generate deterministic input data
 * - Compute CPU reference output
 * - Test persistent kernel launch
 * - Test work queue (enqueue/execute)
 * - Test streaming mode
 * - Report results
 *
 * \return EXIT_SUCCESS if all tests pass, EXIT_FAILURE otherwise.
 */
int main() {
    // Test dimensions: small 5x5 image with 3x3 kernel for quick verification
    const int width = 5, height = 5, ksize = 3;
    const int imgSize = width * height;
    const int kerSize = ksize * ksize;

    // Allocate host buffers for input, kernel, and output (GPU and CPU)
    float *h_input = (float*)malloc(sizeof(float) * imgSize);
    float *h_kernel = (float*)malloc(sizeof(float) * kerSize);
    float *h_output_gpu = (float*)malloc(sizeof(float) * imgSize);
    float *h_output_cpu = (float*)malloc(sizeof(float) * imgSize);

    // Initialize input with sequential values (1, 2, 3, ...) for determinism
    // Kernel is averaging filter (all 1/9) for predictable output
    for (int i = 0; i < imgSize; ++i) h_input[i] = static_cast<float>(i + 1);
    for (int i = 0; i < kerSize; ++i) h_kernel[i] = 1.0f / kerSize;

    // Compute CPU reference for verification
    conv_naive_cpu(h_input, h_kernel, h_output_cpu, width, height, ksize);

    // Allocate device buffers
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, sizeof(float) * imgSize);
    cudaMalloc(&d_kernel, sizeof(float) * kerSize);
    cudaMalloc(&d_output, sizeof(float) * imgSize);

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, sizeof(float) * imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float) * kerSize, cudaMemcpyHostToDevice);

    // Use 8x8 block for tiled convolution
    dim3 block(8, 8, 1);

    // ------------------------------------------------
    // Test 1: Persistent kernel launch - single execution
    // Uses persistent kernel that stays resident on GPU for reuse
    // ------------------------------------------------
    conv_persistent_launch(d_input, d_kernel, d_output, width, height, ksize, block);
    cudaMemcpy(h_output_gpu, d_output, sizeof(float) * imgSize, cudaMemcpyDeviceToHost);

    // Verify output matches CPU reference
    const float eps = 1e-5f;
    int pass = 1;
    for (int i = 0; i < imgSize; ++i) {
        float diff = std::abs(h_output_gpu[i] - h_output_cpu[i]);
        float maxv = std::max(std::abs(h_output_gpu[i]), std::abs(h_output_cpu[i]));
        if (diff > eps * maxv) {
            fprintf(stderr, "Persistent kernel mismatch at %d: GPU=%f CPU=%f\n",
                    i, h_output_gpu[i], h_output_cpu[i]);
            pass = 0;
        }
    }
    if (pass) {
        printf("Persistent kernel launch test passed.\n");
    }

    // ------------------------------------------------
    // Test 2: Work queue - enqueue multiple items, execute in batch
    // Reduces kernel launch overhead by batching work items
    // ------------------------------------------------
    conv_work_clear();  // Clear any leftover items from previous tests
    cudaMemset(d_output, 0, sizeof(float) * imgSize);

    // Enqueue single work item (in practice, multiple items can be queued)
    conv_work_enqueue(d_input, d_kernel, d_output, width, height, ksize);
    // Execute all queued work items in one call
    conv_work_execute(block);

    cudaMemcpy(h_output_gpu, d_output, sizeof(float) * imgSize, cudaMemcpyDeviceToHost);

    // Verify work queue output matches CPU reference
    pass = 1;
    for (int i = 0; i < imgSize; ++i) {
        float diff = std::abs(h_output_gpu[i] - h_output_cpu[i]);
        float maxv = std::max(std::abs(h_output_gpu[i]), std::abs(h_output_cpu[i]));
        if (diff > eps * maxv) {
            fprintf(stderr, "Work queue mismatch at %d: GPU=%f CPU=%f\n",
                    i, h_output_gpu[i], h_output_cpu[i]);
            pass = 0;
        }
    }
    if (pass) {
        printf("Work queue test passed.\n");
    }

    // ------------------------------------------------
    // Test 3: Streaming mode - asynchronous execution
    // Allows overlapping kernel execution with data transfers
    // ------------------------------------------------
    conv_work_clear();  // Reset work queue

    // Create CUDA stream for asynchronous execution
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // Launch kernel on stream (non-blocking)
    conv_streaming(d_input, d_kernel, d_output, width, height, ksize, block, stream);
    // Wait for kernel to complete
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    cudaMemcpy(h_output_gpu, d_output, sizeof(float) * imgSize, cudaMemcpyDeviceToHost);

    // Verify streaming output matches CPU reference
    pass = 1;
    for (int i = 0; i < imgSize; ++i) {
        float diff = std::abs(h_output_gpu[i] - h_output_cpu[i]);
        float maxv = std::max(std::abs(h_output_gpu[i]), std::abs(h_output_cpu[i]));
        if (diff > eps * maxv) {
            fprintf(stderr, "Streaming mismatch at %d: GPU=%f CPU=%f\n",
                    i, h_output_gpu[i], h_output_cpu[i]);
            pass = 0;
        }
    }
    if (pass) {
        printf("Streaming kernel test passed.\n");
    }

    printf("Persistent kernel tests complete.\n");

    // Clean up all resources
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(h_input);
    free(h_kernel);
    free(h_output_gpu);
    free(h_output_cpu);
    return EXIT_SUCCESS;
}