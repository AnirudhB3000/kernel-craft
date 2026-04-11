/**
 * \file test_conv_tiled.cpp
 * \brief Unit test for the shared‑memory tiled convolution kernel.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>

// Declaration of the host launcher defined in conv_tiled.cu
extern "C" void launch_conv_tiled(const float* d_input,
                                 const float* d_kernel,
                                 float* d_output,
                                 int width, int height, int ksize,
                                 dim3 block = dim3(16, 16, 1));

// CPU reference implementation (same as naive, double accumulator)
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
 * \brief Entry point for the tiled convolution unit test.
 *
 * Allocates host and device buffers, launches the tiled kernel via
 * \p launch_conv_tiled, copies the result back, and verifies it against the
 * CPU reference implementation using a relative tolerance.
 *
 * \return EXIT_SUCCESS if all values match within tolerance, EXIT_FAILURE otherwise.
 */
int main() {
    const int width = 5, height = 5, ksize = 3;
    const int imgSize = width * height;
    const int kerSize = ksize * ksize;

    // Host buffers
    float *h_input = (float*)malloc(sizeof(float) * imgSize);
    float *h_kernel = (float*)malloc(sizeof(float) * kerSize);
    float *h_output_gpu = (float*)malloc(sizeof(float) * imgSize);
    float *h_output_cpu = (float*)malloc(sizeof(float) * imgSize);

    // Initialise deterministic data
    for (int i = 0; i < imgSize; ++i) h_input[i] = static_cast<float>(i + 1);
    for (int i = 0; i < kerSize; ++i) h_kernel[i] = 1.0f / kerSize;

    // Device allocation
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, sizeof(float) * imgSize);
    cudaMalloc(&d_kernel, sizeof(float) * kerSize);
    cudaMalloc(&d_output, sizeof(float) * imgSize);

    cudaMemcpy(d_input, h_input, sizeof(float) * imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float) * kerSize, cudaMemcpyHostToDevice);

    // Launch tiled kernel (default block size)
    launch_conv_tiled(d_input, d_kernel, d_output, width, height, ksize);

    cudaMemcpy(h_output_gpu, d_output, sizeof(float) * imgSize, cudaMemcpyDeviceToHost);

    // CPU reference
    conv_naive_cpu(h_input, h_kernel, h_output_cpu, width, height, ksize);

    // Verify with relative tolerance
    const float eps = 1e-5f;
    for (int i = 0; i < imgSize; ++i) {
        float diff = std::abs(h_output_gpu[i] - h_output_cpu[i]);
        float maxv = std::max(std::abs(h_output_gpu[i]), std::abs(h_output_cpu[i]));
        if (diff > eps * maxv) {
            fprintf(stderr, "Mismatch at %d: GPU=%f CPU=%f\n", i, h_output_gpu[i], h_output_cpu[i]);
            return EXIT_FAILURE;
        }
    }

    printf("Tiled convolution test passed.\n");

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(h_input);
    free(h_kernel);
    free(h_output_gpu);
    free(h_output_cpu);
    return EXIT_SUCCESS;
}
