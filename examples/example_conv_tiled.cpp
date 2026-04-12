/**
 * \file example_conv_tiled.cpp
 * \brief Example: Tiled convolution with shared memory optimization.
 *
 * Demonstrates the tiled convolution kernel which uses shared memory
 * to reduce global memory traffic.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

extern "C" void launch_conv_tiled(const float* d_input, const float* d_kernel, float* d_output,
                                   int width, int height, int ksize,
                                   dim3 block = dim3(8, 8, 1));

void conv_naive_cpu(const float* input, const float* kernel, float* output,
                    int width, int height, int ksize) {
    int kHalf = ksize / 2;
    for (int oy = 0; oy < height; ++oy) {
        for (int ox = 0; ox < width; ++ox) {
            float sum = 0.0f;
            for (int ky = 0; ky < ksize; ++ky) {
                int iy = oy + ky - kHalf;
                if (iy < 0 || iy >= height) continue;
                for (int kx = 0; kx < ksize; ++kx) {
                    int ix = ox + kx - kHalf;
                    if (ix < 0 || ix >= width) continue;
                    sum += input[iy * width + ix] * kernel[ky * ksize + kx];
                }
            }
            output[oy * width + ox] = sum;
        }
    }
}

int main() {
    const int width = 16, height = 16, ksize = 3;
    const int imgSize = width * height;
    const int kerSize = ksize * ksize;

    float h_input[256], h_kernel[9], h_output[256], h_cpu[256];
    for (int i = 0; i < imgSize; ++i) h_input[i] = static_cast<float>(i % 16);
    for (int i = 0; i < kerSize; ++i) h_kernel[i] = 1.0f / 9.0f;

    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, sizeof(float) * imgSize);
    cudaMalloc(&d_kernel, sizeof(float) * kerSize);
    cudaMalloc(&d_output, sizeof(float) * imgSize);

    cudaMemcpy(d_input, h_input, sizeof(float) * imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float) * kerSize, cudaMemcpyHostToDevice);

    launch_conv_tiled(d_input, d_kernel, d_output, width, height, ksize, dim3(8, 8, 1));

    cudaMemcpy(h_output, d_output, sizeof(float) * imgSize, cudaMemcpyDeviceToHost);

    conv_naive_cpu(h_input, h_kernel, h_cpu, width, height, ksize);

    bool ok = true;
    const float eps = 1e-4f;
    for (int i = 0; i < imgSize; ++i) {
        if (std::abs(h_output[i] - h_cpu[i]) > eps) { ok = false; break; }
    }

    printf("Tiled Convolution Example\n");
    printf("Input: %dx%d, Kernel: %dx%d, Tile: 8x8\n", width, height, ksize, ksize);
    printf("Result: %s\n", ok ? "MATCH" : "MISMATCH");
    printf("Sample output[0] = %.2f\n", h_output[0]);

    cudaFree(d_input); cudaFree(d_kernel); cudaFree(d_output);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}