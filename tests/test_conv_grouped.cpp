/**
 * \file test_conv_grouped.cpp
 * \brief Unit test for the grouped convolution kernel.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" void launch_conv_grouped_opt(const float* d_input, const float* d_kernel, float* d_output,
                                        int width, int height,
                                        int inChannels, int outChannels,
                                        int groups, int ksize,
                                        dim3 block = dim3(8,8,4));

void conv_grouped_cpu(const float* input, const float* kernel, float* output,
                      int width, int height,
                      int inChannels, int outChannels,
                      int groups, int ksize) {
    memset(output, 0, sizeof(float) * outChannels * height * width);
    int channelsPerGroup = outChannels / groups;
    int inChannelsPerGroup = inChannels / groups;
    int kHalf = ksize / 2;

    for (int oc = 0; oc < outChannels; ++oc) {
        int group = oc / channelsPerGroup;
        int ocInGroup = oc % channelsPerGroup;
        for (int oy = 0; oy < height; ++oy) {
            for (int ox = 0; ox < width; ++ox) {
                float sum = 0.0f;
                for (int ic = 0; ic < inChannelsPerGroup; ++ic) {
                    int srcIc = group * inChannelsPerGroup + ic;
                    int kernelOffset = ((group * channelsPerGroup + ocInGroup) * inChannelsPerGroup + ic) * ksize * ksize;
                    for (int ky = 0; ky < ksize; ++ky) {
                        int iy = oy + ky - kHalf;
                        if (iy < 0 || iy >= height) continue;
                        for (int kx = 0; kx < ksize; ++kx) {
                            int ix = ox + kx - kHalf;
                            if (ix < 0 || ix >= width) continue;
                            float inVal = input[(srcIc * height + iy) * width + ix];
                            float kVal = kernel[kernelOffset + ky * ksize + kx];
                            sum += inVal * kVal;
                        }
                    }
                }
                output[(oc * height + oy) * width + ox] = sum;
            }
        }
    }
}

int main() {
    const int width = 4, height = 4, ksize = 3;
    const int inChannels = 4, outChannels = 4, groups = 2;
    const int inSize = inChannels * height * width;
    const int outSize = outChannels * height * width;
    const int kerSize = inChannels * outChannels / groups * ksize * ksize;

    float *h_input = (float*)malloc(sizeof(float) * inSize);
    float *h_kernel = (float*)malloc(sizeof(float) * kerSize);
    float *h_output_gpu = (float*)malloc(sizeof(float) * outSize);
    float *h_output_cpu = (float*)malloc(sizeof(float) * outSize);

    for (int i = 0; i < inSize; ++i) h_input[i] = static_cast<float>((i % 10) + 1);
    for (int i = 0; i < kerSize; ++i) h_kernel[i] = 0.1f;

    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, sizeof(float) * inSize);
    cudaMalloc(&d_kernel, sizeof(float) * kerSize);
    cudaMalloc(&d_output, sizeof(float) * outSize);

    cudaMemcpy(d_input, h_input, sizeof(float) * inSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float) * kerSize, cudaMemcpyHostToDevice);

    launch_conv_grouped_opt(d_input, d_kernel, d_output, width, height, inChannels, outChannels, groups, ksize);

    cudaMemcpy(h_output_gpu, d_output, sizeof(float) * outSize, cudaMemcpyDeviceToHost);

    conv_grouped_cpu(h_input, h_kernel, h_output_cpu, width, height, inChannels, outChannels, groups, ksize);

    const float eps = 1e-4f;
    for (int i = 0; i < outSize; ++i) {
        if (std::abs(h_output_gpu[i] - h_output_cpu[i]) > eps) {
            fprintf(stderr, "Mismatch at index %d: GPU=%f CPU=%f\n", i, h_output_gpu[i], h_output_cpu[i]);
            return EXIT_FAILURE;
        }
    }
    printf("All values match. Grouped convolution test passed.\n");

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(h_input);
    free(h_kernel);
    free(h_output_gpu);
    free(h_output_cpu);
    return EXIT_SUCCESS;
}