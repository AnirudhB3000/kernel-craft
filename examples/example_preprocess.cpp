/**
 * \file example_preprocess.cpp
 * \brief Example: GPU preprocessing (resize, normalize, flip).
 *
 * Demonstrates the preprocessing kernels for ML inference pipelines:
 * - Bilinear resize
 * - Normalization (mean/std)
 * - Horizontal flip
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

extern "C" void launch_resize_bilinear(const float* d_input, float* d_output,
                                        int in_w, int in_h, int out_w, int out_h);
extern "C" void launch_normalize(const float* d_input, float* d_output,
                                  int width, int height, float mean, float std);
extern "C" void launch_flip_horizontal(const float* d_input, float* d_output,
                                        int width, int height);

int main() {
    const int in_w = 4, in_h = 4;
    const int out_w = 8, out_h = 8;
    float h_in[16], h_out[64], h_ref[64];

    for (int i = 0; i < 16; ++i) h_in[i] = static_cast<float>(i);

    float *d_in, *d_out;
    cudaMalloc(&d_in, sizeof(float) * 16);
    cudaMalloc(&d_out, sizeof(float) * 64);

    cudaMemcpy(d_in, h_in, sizeof(float) * 16, cudaMemcpyHostToDevice);

    // Run preprocessing pipeline
    launch_resize_bilinear(d_in, d_out, in_w, in_h, out_w, out_h);
    launch_normalize(d_out, d_out, out_w, out_h, 0.0f, 1.0f);
    launch_flip_horizontal(d_out, d_out, out_w, out_h);

    cudaMemcpy(h_out, d_out, sizeof(float) * 64, cudaMemcpyDeviceToHost);

    printf("Preprocessing Example\n");
    printf("Input: %dx%d -> Output: %dx%d\n", in_w, in_h, out_w, out_h);
    printf("Pipeline: resize -> normalize -> flip_h\n");
    printf("Sample output[0] = %.2f\n", h_out[0]);
    printf("Sample output[63] = %.2f (flipped from input[3])\n", h_out[63]);

    cudaFree(d_in); cudaFree(d_out);
    return EXIT_SUCCESS;
}