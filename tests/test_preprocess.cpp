/**
 * \file test_preprocess.cpp
 * \brief Unit tests for GPU preprocessing kernels.
 *
 * Tests: resize_bilinear, normalize, flip_horizontal, flip_vertical
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

extern "C" void launch_resize_bilinear(const float* d_input, float* d_output,
                                       int in_w, int in_h, int out_w, int out_h);
extern "C" void launch_normalize(const float* d_input, float* d_output,
                                 int width, int height, float mean, float std);
extern "C" void launch_flip_horizontal(const float* d_input, float* d_output,
                                       int width, int height);
extern "C" void launch_flip_vertical(const float* d_input, float* d_output,
                                     int width, int height);

void resize_cpu(const float* input, float* output, int in_w, int in_h, int out_w, int out_h) {
    for (int oy = 0; oy < out_h; ++oy) {
        for (int ox = 0; ox < out_w; ++ox) {
            float x = (ox + 0.5f) * static_cast<float>(in_w) / static_cast<float>(out_w) - 0.5f;
            float y = (oy + 0.5f) * static_cast<float>(in_h) / static_cast<float>(out_h) - 0.5f;
            int x0 = static_cast<int>(floorf(x));
            int y0 = static_cast<int>(floorf(y));
            int x1 = x0 + 1, y1 = y0 + 1;
            x0 = std::max(0, std::min(x0, in_w - 1));
            y0 = std::max(0, std::min(y0, in_h - 1));
            x1 = std::max(0, std::min(x1, in_w - 1));
            y1 = std::max(0, std::min(y1, in_h - 1));
            float fx = x - x0, fy = y - y0;
            float f00 = input[y0 * in_w + x0];
            float f10 = input[y0 * in_w + x1];
            float f01 = input[y1 * in_w + x0];
            float f11 = input[y1 * in_w + x1];
            output[oy * out_w + ox] = (1-fx)*(1-fy)*f00 + fx*(1-fy)*f10 + (1-fx)*fy*f01 + fx*fy*f11;
        }
    }
}

void normalize_cpu(const float* input, float* output, int n, float mean, float std) {
    for (int i = 0; i < n; ++i) output[i] = (input[i] - mean) / std;
}

void flip_h_cpu(const float* input, float* output, int width, int height) {
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            output[y * width + x] = input[y * width + (width - 1 - x)];
}

void flip_v_cpu(const float* input, float* output, int width, int height) {
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            output[y * width + x] = input[(height - 1 - y) * width + x];
}

bool verify(const float* gpu, const float* cpu, int n, float eps = 1e-4f) {
    for (int i = 0; i < n; ++i) {
        if (std::fabs(gpu[i] - cpu[i]) > eps) {
            fprintf(stderr, "Mismatch at %d: GPU=%f CPU=%f\n", i, gpu[i], cpu[i]);
            return false;
        }
    }
    return true;
}

int test_resize() {
    const int in_w = 4, in_h = 4, out_w = 8, out_h = 8;
    float h_in[] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
    float h_gpu[64], h_cpu[64];
    float *d_in, *d_out;
    cudaMalloc(&d_in, 16 * sizeof(float));
    cudaMalloc(&d_out, 64 * sizeof(float));
    cudaMemcpy(d_in, h_in, 16 * sizeof(float), cudaMemcpyHostToDevice);
    launch_resize_bilinear(d_in, d_out, in_w, in_h, out_w, out_h);
    cudaMemcpy(h_gpu, d_out, 64 * sizeof(float), cudaMemcpyDeviceToHost);
    resize_cpu(h_in, h_cpu, in_w, in_h, out_w, out_h);
    bool ok = verify(h_gpu, h_cpu, 64);
    cudaFree(d_in); cudaFree(d_out);
    printf("Resize test: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}

int test_normalize() {
    const int w = 5, h = 5;
    float h_in[25], h_gpu[25], h_cpu[25];
    for (int i = 0; i < 25; ++i) h_in[i] = static_cast<float>(i);
    float mean = 12.0f, std = 5.0f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, 25 * sizeof(float));
    cudaMalloc(&d_out, 25 * sizeof(float));
    cudaMemcpy(d_in, h_in, 25 * sizeof(float), cudaMemcpyHostToDevice);
    launch_normalize(d_in, d_out, w, h, mean, std);
    cudaMemcpy(h_gpu, d_out, 25 * sizeof(float), cudaMemcpyDeviceToHost);
    normalize_cpu(h_in, h_cpu, 25, mean, std);
    bool ok = verify(h_gpu, h_cpu, 25);
    cudaFree(d_in); cudaFree(d_out);
    printf("Normalize test: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}

int test_flip_horizontal() {
    const int w = 4, h = 3;
    float h_in[] = {1,2,3,4, 5,6,7,8, 9,10,11,12};
    float h_gpu[12], h_cpu[12];
    float *d_in, *d_out;
    cudaMalloc(&d_in, 12 * sizeof(float));
    cudaMalloc(&d_out, 12 * sizeof(float));
    cudaMemcpy(d_in, h_in, 12 * sizeof(float), cudaMemcpyHostToDevice);
    launch_flip_horizontal(d_in, d_out, w, h);
    cudaMemcpy(h_gpu, d_out, 12 * sizeof(float), cudaMemcpyDeviceToHost);
    flip_h_cpu(h_in, h_cpu, w, h);
    bool ok = verify(h_gpu, h_cpu, 12);
    cudaFree(d_in); cudaFree(d_out);
    printf("Flip horizontal test: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}

int test_flip_vertical() {
    const int w = 4, h = 3;
    float h_in[] = {1,2,3,4, 5,6,7,8, 9,10,11,12};
    float h_gpu[12], h_cpu[12];
    float *d_in, *d_out;
    cudaMalloc(&d_in, 12 * sizeof(float));
    cudaMalloc(&d_out, 12 * sizeof(float));
    cudaMemcpy(d_in, h_in, 12 * sizeof(float), cudaMemcpyHostToDevice);
    launch_flip_vertical(d_in, d_out, w, h);
    cudaMemcpy(h_gpu, d_out, 12 * sizeof(float), cudaMemcpyDeviceToHost);
    flip_v_cpu(h_in, h_cpu, w, h);
    bool ok = verify(h_gpu, h_cpu, 12);
    cudaFree(d_in); cudaFree(d_out);
    printf("Flip vertical test: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}

int main() {
    int failures = 0;
    failures += test_resize();
    failures += test_normalize();
    failures += test_flip_horizontal();
    failures += test_flip_vertical();
    printf("\nPreprocessing tests: %d failures\n", failures);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}