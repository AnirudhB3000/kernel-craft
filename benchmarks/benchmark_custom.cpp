/**
 * \file benchmark_custom.cpp
 * \brief Benchmark for custom sparse convolution kernel.
 *
 * Compares sparse kernel (only non-zero values) vs dense fallback.
 */

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>
#include <string>

#define MAX_NONZERO 16

/**
 * \brief CPU reference implementation of naïve 2‑D convolution.
 *
 * \param[in] input  Pointer to the input image (row‑major).
 * \param[in] kernel Pointer to the convolution kernel (row‑major).
 * \param[out] output Destination buffer for the convolution result.
 * \param[in] width  Image width.
 * \param[in] height Image height.
 * \param[in] ksize Kernel dimension (must be odd).
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
                    sum += val * kval;
                }
            }
            output[oy * width + ox] = sum;
        }
    }
}

extern "C" void launch_conv_sparse(const float* d_input, const float* d_kernel_nonzero,
                                 const int* d_kernel_rows, const int* d_kernel_cols,
                                 int nnz, float* d_output,
                                 int width, int height, int ksize,
                                 dim3 block = dim3(16, 16, 1));

extern "C" void launch_conv_dense_fallback(const float* d_input, const float* d_kernel,
                                      float* d_output,
                                      int width, int height, int ksize,
                                      dim3 block = dim3(16, 16, 1));

/**
 * \brief Benchmark driver for sparse convolution.
 *
 * Compares sparse kernel (only non-zero values) vs dense fallback.
 * Use `--sparse` flag to enable sparse mode.
 *
 * \param argc Argument count.
 * \param argv Argument vector; optional `--sparse` enables sparse mode.
 * \return EXIT_SUCCESS if verification passes, EXIT_FAILURE otherwise.
 */
int main(int argc, char** argv) {
    bool use_sparse = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--sparse") use_sparse = true;
    }

    const int width = 1024, height = 1024, ksize = 3;
    const int imgSize = width * height;
    const int kerSize = ksize * ksize;

    float *h_input = (float*)malloc(sizeof(float) * imgSize);
    float *h_kernel = (float*)malloc(sizeof(float) * kerSize);
    float *h_output_gpu = (float*)malloc(sizeof(float) * imgSize);
    float *h_output_cpu = (float*)malloc(sizeof(float) * imgSize);

    for (int i = 0; i < imgSize; ++i) h_input[i] = static_cast<float>(i + 1);
    for (int i = 0; i < kerSize; ++i) h_kernel[i] = (i % 2 == 0) ? 0.0f : 1.0f;

    int nnz = 0;
    int h_kernel_rows[MAX_NONZERO];
    int h_kernel_cols[MAX_NONZERO];
    float h_kernel_nonzero[MAX_NONZERO];
    for (int ky = 0; ky < ksize && nnz < MAX_NONZERO; ++ky) {
        for (int kx = 0; kx < ksize && nnz < MAX_NONZERO; ++kx) {
            float val = h_kernel[ky * ksize + kx];
            if (val != 0.0f) {
                h_kernel_rows[nnz] = ky;
                h_kernel_cols[nnz] = kx;
                h_kernel_nonzero[nnz] = val;
                nnz++;
            }
        }
    }

    float *d_input, *d_kernel, *d_kernel_nonzero, *d_output;
    int *d_kernel_rows, *d_kernel_cols;
    cudaMalloc(&d_input, sizeof(float) * imgSize);
    cudaMalloc(&d_kernel, sizeof(float) * kerSize);
    cudaMalloc(&d_kernel_nonzero, sizeof(float) * MAX_NONZERO);
    cudaMalloc(&d_kernel_rows, sizeof(int) * MAX_NONZERO);
    cudaMalloc(&d_kernel_cols, sizeof(int) * MAX_NONZERO);
    cudaMalloc(&d_output, sizeof(float) * imgSize);

    cudaMemcpy(d_input, h_input, sizeof(float) * imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float) * kerSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_nonzero, h_kernel_nonzero, sizeof(float) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_rows, h_kernel_rows, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_cols, h_kernel_cols, sizeof(int) * nnz, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (use_sparse) {
        launch_conv_sparse(d_input, d_kernel_nonzero, d_kernel_rows, d_kernel_cols,
                          nnz, d_output, width, height, ksize);
    } else {
        launch_conv_dense_fallback(d_input, d_kernel, d_output, width, height, ksize);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    cudaMemcpy(h_output_gpu, d_output, sizeof(float) * imgSize, cudaMemcpyDeviceToHost);

    auto cpu_start = std::chrono::high_resolution_clock::now();
    conv_naive_cpu(h_input, h_kernel, h_output_cpu, width, height, ksize);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_ms = cpu_end - cpu_start;

    const float eps = 1e-4f;
    bool match = true;
    for (int i = 0; i < imgSize && match; ++i) {
        if (std::abs(h_output_gpu[i] - h_output_cpu[i]) > eps) {
            match = false;
        }
    }

    double num_pixels = static_cast<double>(imgSize);
    double gpu_throughput = num_pixels / (gpu_ms * 1e-3) / 1e6;
    double cpu_throughput = num_pixels / (cpu_ms.count() * 1e-3) / 1e6;
    double speedup = cpu_ms.count() / gpu_ms;

    printf("=== Custom Convolution Benchmark ===\n");
    printf("Mode: %s (nnz=%d/%d)\n", use_sparse ? "Sparse" : "Dense Fallback", nnz, kerSize);
    printf("Image size: %dx%d (%.2f MPixels)\n", width, height, num_pixels/1e6);
    printf("Kernel size: %dx%d\n", ksize, ksize);
    printf("GPU time: %.3f ms (%.3f MPixels/s)\n", gpu_ms, gpu_throughput);
    printf("CPU time: %.3f ms (%.3f MPixels/s)\n", cpu_ms.count(), cpu_throughput);
    printf("Speed-up: %.2fx\n", speedup);
    printf("Verification: %s\n", match ? "PASSED" : "FAILED");

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_kernel_nonzero);
    cudaFree(d_kernel_rows);
    cudaFree(d_kernel_cols);
    cudaFree(d_output);
    free(h_input);
    free(h_kernel);
    free(h_output_gpu);
    free(h_output_cpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return match ? EXIT_SUCCESS : EXIT_FAILURE;
}