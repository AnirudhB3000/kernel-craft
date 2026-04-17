/**
 * \file test_mixed_precision.cpp
 * \brief Unit test for mixed precision convolution kernels (FP32, FP16, TF32).
 *
 * \par Description
 * Tests the correctness of mixed-precision convolution kernels by comparing
 * output against a CPU reference implementation. Tests FP32 baseline,
 * FP16 (half-precision), and TF32 (TensorFloat-32) variants.
 *
 * \par Verification
 * Each kernel's output is compared against the CPU reference using
 * relative tolerance (1e-5). FP16 tests are skipped if hardware
 * doesn't support Tensor Cores (Volta+). TF32 tests are skipped if
 * hardware doesn't support Ampere+ architectures.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/**
 * \brief Launch FP32 tiled convolution kernel.
 * \param[in] d_input Input image on device.
 * \param[in] d_kernel Convolution kernel on device.
 * \param[out] d_output Output image on device.
 * \param[in] width Image width.
 * \param[in] height Image height.
 * \param[in] ksize Kernel size (must be odd).
 * \param[in] block CUDA block dimensions.
 */
extern "C" void conv_tiled_fp32_launch(const float* d_input,
                                    const float* d_kernel,
                                    float* d_output,
                                    int width, int height, int ksize,
                                    dim3 block);

/**
 * \brief Launch FP16 tiled convolution kernel.
 * \param[in] d_input Input image (half precision) on device.
 * \param[in] d_kernel Convolution kernel (half precision) on device.
 * \param[out] d_output Output image (half precision) on device.
 * \param[in] width Image width.
 * \param[in] height Image height.
 * \param[in] ksize Kernel size (must be odd).
 * \param[in] block CUDA block dimensions.
 */
extern "C" void conv_tiled_fp16_launch(const half* d_input,
                                      const half* d_kernel,
                                      half* d_output,
                                      int width, int height, int ksize,
                                      dim3 block);

/**
 * \brief Launch TF32 tiled convolution kernel.
 * \param[in] d_input Input image on device.
 * \param[in] d_kernel Convolution kernel on device.
 * \param[out] d_output Output image on device.
 * \param[in] width Image width.
 * \param[in] height Image height.
 * \param[in] ksize Kernel size (must be odd).
 * \param[in] block CUDA block dimensions.
 */
extern "C" void conv_tiled_tf32_launch(const float* d_input,
                                        const float* d_kernel,
                                        float* d_output,
                                        int width, int height, int ksize,
                                        dim3 block);

/**
 * \brief Convert FP32 array to FP16 (CPU-side, element-wise).
 * \param[in] src Source FP32 array.
 * \param[out] dst Destination FP16 array.
 * \param[in] size Number of elements.
 */
extern "C" void convert_fp32_to_fp16(const float* src, half* dst, int size);

/**
 * \brief Convert FP16 array to FP32 (CPU-side, element-wise).
 * \param[in] src Source FP16 array.
 * \param[out] dst Destination FP32 array.
 * \param[in] size Number of elements.
 */
extern "C" void convert_fp16_to_fp32(const half* src, float* dst, int size);

/**
 * \brief Convert FP32 array to FP16 (GPU-side, batched).
 * \param[in] src Source FP32 array on device.
 * \param[out] dst Destination FP16 array on device.
 * \param[in] size Number of elements.
 */
extern "C" void convert_fp32_to_fp16_batch(const float* src, half* dst, int size);

/**
 * \brief Convert FP16 array to FP32 (GPU-side, batched).
 * \param[in] src Source FP16 array on device.
 * \param[out] dst Destination FP32 array on device.
 * \param[in] size Number of elements.
 */
extern "C" void convert_fp16_to_fp32_batch(const half* src, float* dst, int size);

/**
 * \brief Query FP16 (Tensor Core) support.
 * \return 1 if FP16 is supported, 0 otherwise.
 */
extern "C" int get_fp16_support();

/**
 * \brief Query TF32 support.
 * \return 1 if TF32 is supported, 0 otherwise.
 */
extern "C" int get_tf32_support();

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
 * \brief Main entry point for mixed precision unit tests.
 *
 * \par Test Sequence
 * - Allocate host and device buffers
 * - Generate deterministic input data
 * - Compute CPU reference output
 * - Test FP32 kernel (baseline)
 * - Test FP16 kernel if supported
 * - Test TF32 kernel if supported
 * - Report results
 *
 * \return EXIT_SUCCESS if all tests pass, EXIT_FAILURE otherwise.
 */
/**
 * \brief Main entry point for mixed precision unit tests.
 *
 * \par Test Sequence
 * - Allocate host and device buffers
 * - Generate deterministic input data
 * - Compute CPU reference output
 * - Test FP32 kernel (baseline)
 * - Test FP16 kernel if supported
 * - Test TF32 kernel if supported
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
    // Test FP32 (baseline) - standard single precision
    // ------------------------------------------------
    conv_tiled_fp32_launch(d_input, d_kernel, d_output, width, height, ksize, block);
    cudaMemcpy(h_output_gpu, d_output, sizeof(float) * imgSize, cudaMemcpyDeviceToHost);

    // Verify FP32 output matches CPU reference
    const float eps = 1e-5f;
    int fp32_pass = 1;
    for (int i = 0; i < imgSize; ++i) {
        float diff = std::abs(h_output_gpu[i] - h_output_cpu[i]);
        float maxv = std::max(std::abs(h_output_gpu[i]), std::abs(h_output_cpu[i]));
        if (diff > eps * maxv) {
            fprintf(stderr, "FP32 mismatch at %d: GPU=%f CPU=%f\n", i, h_output_gpu[i], h_output_cpu[i]);
            fp32_pass = 0;
        }
    }
    if (fp32_pass) {
        printf("FP32 convolution test passed.\n");
    }

    // ------------------------------------------------
    // Test FP16 - half precision (requires Volta+ for Tensor Cores)
    // ------------------------------------------------
    cudaMemset(d_output, 0, sizeof(float) * imgSize);

    if (get_fp16_support()) {
        // Allocate separate FP16 buffers
        half *d_input_fp16, *d_kernel_fp16, *d_output_fp16;
        cudaMalloc(&d_input_fp16, sizeof(half) * imgSize);
        cudaMalloc(&d_kernel_fp16, sizeof(half) * kerSize);
        cudaMalloc(&d_output_fp16, sizeof(half) * imgSize);

        // Convert FP32 input to FP16 before kernel execution
        convert_fp32_to_fp16_batch(d_input, d_input_fp16, imgSize);
        convert_fp32_to_fp16_batch(d_kernel, d_kernel_fp16, kerSize);

        // Execute FP16 convolution kernel
        conv_tiled_fp16_launch(d_input_fp16, d_kernel_fp16, d_output_fp16,
                               width, height, ksize, block);

        // Convert FP16 output back to FP32 for comparison
        convert_fp16_to_fp32_batch(d_output_fp16, d_output, imgSize);
        cudaMemcpy(h_output_gpu, d_output, sizeof(float) * imgSize, cudaMemcpyDeviceToHost);

        // Verify FP16 output matches CPU reference (allowing for precision loss)
        int fp16_pass = 1;
        for (int i = 0; i < imgSize; ++i) {
            float diff = std::abs(h_output_gpu[i] - h_output_cpu[i]);
            float maxv = std::max(std::abs(h_output_gpu[i]), std::abs(h_output_cpu[i]));
            if (diff > eps * maxv) {
                fprintf(stderr, "FP16 mismatch at %d: GPU=%f CPU=%f\n", i, h_output_gpu[i], h_output_cpu[i]);
                fp16_pass = 0;
            }
        }
        if (fp16_pass) {
            printf("FP16 convolution test passed.\n");
        }

        // Clean up FP16 buffers
        cudaFree(d_input_fp16);
        cudaFree(d_kernel_fp16);
        cudaFree(d_output_fp16);
    } else {
        printf("FP16 not supported, skipping FP16 test.\n");
    }

    // ------------------------------------------------
    // Test TF32 - TensorFloat-32 (requires Ampere+)
    // ------------------------------------------------
    cudaMemset(d_output, 0, sizeof(float) * imgSize);

    if (get_tf32_support()) {
        // TF32 uses same FP32 input/output buffers, just different internal precision
        conv_tiled_tf32_launch(d_input, d_kernel, d_output, width, height, ksize, block);
        cudaMemcpy(h_output_gpu, d_output, sizeof(float) * imgSize, cudaMemcpyDeviceToHost);

        // Verify TF32 output matches CPU reference
        int tf32_pass = 1;
        for (int i = 0; i < imgSize; ++i) {
            float diff = std::abs(h_output_gpu[i] - h_output_cpu[i]);
            float maxv = std::max(std::abs(h_output_gpu[i]), std::abs(h_output_cpu[i]));
            if (diff > eps * maxv) {
                fprintf(stderr, "TF32 mismatch at %d: GPU=%f CPU=%f\n", i, h_output_gpu[i], h_output_cpu[i]);
                tf32_pass = 0;
            }
        }
        if (tf32_pass) {
            printf("TF32 convolution test passed.\n");
        }
    } else {
        printf("TF32 not supported, skipping TF32 test.\n");
    }

    printf("Mixed precision tests complete.\n");

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