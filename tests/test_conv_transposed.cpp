/**
 * \file test_conv_transposed.cpp
 * \brief Unit test for the transposed convolution kernel.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" void launch_conv_transposed_atomic(const float* d_input, const float* d_kernel, float* d_output,
                                              int inWidth, int inHeight, int outWidth, int outHeight,
                                              int ksize, int stride, int padding,
                                              dim3 block = dim3(16,16,1));

void conv_transposed_cpu(const float* input, const float* kernel, float* output,
                         int inWidth, int inHeight, int outWidth, int outHeight,
                         int ksize, int stride, int padding) {
    memset(output, 0, sizeof(float) * outWidth * outHeight);
    int kHalf = ksize / 2;
    for (int iy = 0; iy < inHeight; ++iy) {
        for (int ix = 0; ix < inWidth; ++ix) {
            float inputVal = input[iy * inWidth + ix];
            for (int ky = 0; ky < ksize; ++ky) {
                int oy = iy * stride + ky - kHalf + padding;
                if (oy < 0 || oy >= outHeight) continue;
                for (int kx = 0; kx < ksize; ++kx) {
                    int ox = ix * stride + kx - kHalf + padding;
                    if (ox < 0 || ox >= outWidth) continue;
                    float kval = kernel[ky * ksize + kx];
                    output[oy * outWidth + ox] += inputVal * kval;
                }
            }
        }
    }
}

int main() {
    const int inWidth = 3, inHeight = 3, ksize = 3, stride = 2, padding = 1;
    const int outWidth = (inWidth - 1) * stride - 2 * padding + ksize;
    const int outHeight = (inHeight - 1) * stride - 2 * padding + ksize;
    const int inSize = inWidth * inHeight;
    const int outSize = outWidth * outHeight;
    const int kerSize = ksize * ksize;

    float *h_input = (float*)malloc(sizeof(float) * inSize);
    float *h_kernel = (float*)malloc(sizeof(float) * kerSize);
    float *h_output_gpu = (float*)malloc(sizeof(float) * outSize);
    float *h_output_cpu = (float*)malloc(sizeof(float) * outSize);

    for (int i = 0; i < inSize; ++i) h_input[i] = static_cast<float>(i + 1);
    for (int i = 0; i < kerSize; ++i) h_kernel[i] = 1.0f / kerSize;

    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, sizeof(float) * inSize);
    cudaMalloc(&d_kernel, sizeof(float) * kerSize);
    cudaMalloc(&d_output, sizeof(float) * outSize);

    cudaMemcpy(d_input, h_input, sizeof(float) * inSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float) * kerSize, cudaMemcpyHostToDevice);

    launch_conv_transposed_atomic(d_input, d_kernel, d_output, inWidth, inHeight, outWidth, outHeight, ksize, stride, padding);

    cudaMemcpy(h_output_gpu, d_output, sizeof(float) * outSize, cudaMemcpyDeviceToHost);

    conv_transposed_cpu(h_input, h_kernel, h_output_cpu, inWidth, inHeight, outWidth, outHeight, ksize, stride, padding);

    const float eps = 1e-4f;
    for (int i = 0; i < outSize; ++i) {
        if (std::abs(h_output_gpu[i] - h_output_cpu[i]) > eps) {
            fprintf(stderr, "Mismatch at index %d: GPU=%f CPU=%f\n", i, h_output_gpu[i], h_output_cpu[i]);
            return EXIT_FAILURE;
        }
    }
    printf("All values match. Transposed convolution test passed.\n");

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(h_input);
    free(h_kernel);
    free(h_output_gpu);
    free(h_output_cpu);
    return EXIT_SUCCESS;
}