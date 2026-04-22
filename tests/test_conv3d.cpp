/**
 * \file test_conv3d.cpp
 * \brief Unit test for the 3‑D convolution kernels.
 *
 * The test allocates deterministic input volume and kernel data, copies it
 * to the device, launches \p launch_conv3d_naive, copies the result back,
 * and compares it against the CPU reference implementation.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>

extern "C" void launch_conv3d_naive(const float* d_input, const float* d_kernel, float* d_output,
                                    int depth, int height, int width,
                                    int kd, int kh, int kw,
                                    dim3 block = dim3(8,8,8));

void conv3d_cpu(const float* input, const float* kernel, float* output,
                int depth, int height, int width,
                int kd, int kh, int kw) {
    int kdHalf = kd / 2;
    int khHalf = kh / 2;
    int kwHalf = kw / 2;
    for (int oz = 0; oz < depth; ++oz) {
        for (int oy = 0; oy < height; ++oy) {
            for (int ox = 0; ox < width; ++ox) {
                float sum = 0.0f;
                for (int kz = 0; kz < kd; ++kz) {
                    int iz = oz + kz - kdHalf;
                    if (iz < 0 || iz >= depth) continue;
                    for (int ky = 0; ky < kh; ++ky) {
                        int iy = oy + ky - khHalf;
                        if (iy < 0 || iy >= height) continue;
                        for (int kx = 0; kx < kw; ++kx) {
                            int ix = ox + kx - kwHalf;
                            if (ix < 0 || ix >= width) continue;
                            float val = input[iz * height * width + iy * width + ix];
                            float kval = kernel[kz * kh * kw + ky * kw + kx];
                            sum += val * kval;
                        }
                    }
                }
                output[oz * height * width + oy * width + ox] = sum;
            }
        }
    }
}

int main() {
    const int depth = 4, height = 4, width = 4;
    const int kd = 3, kh = 3, kw = 3;
    const int volSize = depth * height * width;
    const int kerSize = kd * kh * kw;

    float *h_input = (float*)malloc(sizeof(float) * volSize);
    float *h_kernel = (float*)malloc(sizeof(float) * kerSize);
    float *h_output_gpu = (float*)malloc(sizeof(float) * volSize);
    float *h_output_cpu = (float*)malloc(sizeof(float) * volSize);

    for (int i = 0; i < volSize; ++i) h_input[i] = static_cast<float>(i + 1);
    for (int i = 0; i < kerSize; ++i) h_kernel[i] = 1.0f / kerSize;

    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, sizeof(float) * volSize);
    cudaMalloc(&d_kernel, sizeof(float) * kerSize);
    cudaMalloc(&d_output, sizeof(float) * volSize);

    cudaMemcpy(d_input, h_input, sizeof(float) * volSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float) * kerSize, cudaMemcpyHostToDevice);

    launch_conv3d_naive(d_input, d_kernel, d_output, depth, height, width, kd, kh, kw);

    cudaMemcpy(h_output_gpu, d_output, sizeof(float) * volSize, cudaMemcpyDeviceToHost);

    conv3d_cpu(h_input, h_kernel, h_output_cpu, depth, height, width, kd, kh, kw);

    const float eps = 1e-5f;
    for (int i = 0; i < volSize; ++i) {
        if (std::abs(h_output_gpu[i] - h_output_cpu[i]) > eps) {
            fprintf(stderr, "Mismatch at index %d: GPU=%f CPU=%f\n", i, h_output_gpu[i], h_output_cpu[i]);
            return EXIT_FAILURE;
        }
    }
    printf("All values match. 3D convolution test passed.\n");

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(h_input);
    free(h_kernel);
    free(h_output_gpu);
    free(h_output_cpu);
    return EXIT_SUCCESS;
}