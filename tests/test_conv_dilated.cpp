/**
 * \file test_conv_dilated.cpp
 * \brief Unit test for the dilated convolution kernel.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

extern "C" void launch_conv_dilated_naive(const float* d_input, const float* d_kernel, float* d_output,
                                          int width, int height, int ksize, int dilation,
                                          dim3 block = dim3(16,16,1));

void conv_dilated_cpu(const float* input, const float* kernel, float* output,
                      int width, int height, int ksize, int dilation) {
    int kHalf = ksize / 2;
    for (int oy = 0; oy < height; ++oy) {
        for (int ox = 0; ox < width; ++ox) {
            float sum = 0.0f;
            for (int ky = 0; ky < ksize; ++ky) {
                int iy = oy + (ky - kHalf) * dilation;
                if (iy < 0 || iy >= height) continue;
                for (int kx = 0; kx < ksize; ++kx) {
                    int ix = ox + (kx - kHalf) * dilation;
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

int main() {
    const int width = 5, height = 5, ksize = 3, dilation = 2;
    const int imgSize = width * height;
    const int kerSize = ksize * ksize;

    float *h_input = (float*)malloc(sizeof(float) * imgSize);
    float *h_kernel = (float*)malloc(sizeof(float) * kerSize);
    float *h_output_gpu = (float*)malloc(sizeof(float) * imgSize);
    float *h_output_cpu = (float*)malloc(sizeof(float) * imgSize);

    for (int i = 0; i < imgSize; ++i) h_input[i] = static_cast<float>(i + 1);
    for (int i = 0; i < kerSize; ++i) h_kernel[i] = 1.0f / kerSize;

    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, sizeof(float) * imgSize);
    cudaMalloc(&d_kernel, sizeof(float) * kerSize);
    cudaMalloc(&d_output, sizeof(float) * imgSize);

    cudaMemcpy(d_input, h_input, sizeof(float) * imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float) * kerSize, cudaMemcpyHostToDevice);

    launch_conv_dilated_naive(d_input, d_kernel, d_output, width, height, ksize, dilation);

    cudaMemcpy(h_output_gpu, d_output, sizeof(float) * imgSize, cudaMemcpyDeviceToHost);

    conv_dilated_cpu(h_input, h_kernel, h_output_cpu, width, height, ksize, dilation);

    const float eps = 1e-5f;
    for (int i = 0; i < imgSize; ++i) {
        if (std::abs(h_output_gpu[i] - h_output_cpu[i]) > eps) {
            fprintf(stderr, "Mismatch at index %d: GPU=%f CPU=%f\n", i, h_output_gpu[i], h_output_cpu[i]);
            return EXIT_FAILURE;
        }
    }
    printf("All values match. Dilated convolution test passed.\n");

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(h_input);
    free(h_kernel);
    free(h_output_gpu);
    free(h_output_cpu);
    return EXIT_SUCCESS;
}