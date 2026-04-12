/**
 * \file example_fused_pipeline.cpp
 * \brief Example: Fused conv + batchnorm + ReLU kernel.
 *
 * Demonstrates the fused kernel that combines convolution,
 * batch normalization, and ReLU activation in a single CUDA kernel.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

extern "C" void launch_conv_bn_relu_fused(const float* d_input, const float* d_kernel,
                                            float gamma, float beta,
                                            float* d_output, int width, int height, int ksize,
                                            dim3 block = dim3(8, 8, 1));

int main() {
    const int width = 16, height = 16, ksize = 3;
    const int imgSize = width * height;

    float h_input[256], h_kernel[9], h_output[256];
    for (int i = 0; i < imgSize; ++i) h_input[i] = static_cast<float>(i % 16);
    for (int i = 0; i < 9; ++i) h_kernel[i] = 1.0f / 9.0f;

    float gamma = 1.5f, beta = 0.2f;

    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, sizeof(float) * imgSize);
    cudaMalloc(&d_kernel, sizeof(float) * 9);
    cudaMalloc(&d_output, sizeof(float) * imgSize);

    cudaMemcpy(d_input, h_input, sizeof(float) * imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float) * 9, cudaMemcpyHostToDevice);

    launch_conv_bn_relu_fused(d_input, d_kernel, gamma, beta, d_output, width, height, ksize);

    cudaMemcpy(h_output, d_output, sizeof(float) * imgSize, cudaMemcpyDeviceToHost);

    printf("Fused Pipeline Example\n");
    printf("Input: %dx%d, Kernel: %dx%d\n", width, height, ksize, ksize);
    printf("Batchnorm: gamma=%.2f, beta=%.2f\n", gamma, beta);
    printf("Sample output[0] = %.2f (conv -> bn -> relu)\n", h_output[0]);

    cudaFree(d_input); cudaFree(d_kernel); cudaFree(d_output);
    return EXIT_SUCCESS;
}