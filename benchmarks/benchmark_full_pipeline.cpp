/**
 * \file benchmark_full_pipeline.cpp
 * \brief End-to-end benchmark for full GPU preprocessing pipeline.
 *
 * Measures performance of the complete GPU pipeline:
 * resize (bilinear) → normalize → flip → convolution
 * Compares against CPU+GPU hybrid (CPU preprocess + GPU conv).
 */

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>
#include <string>
#include <cmath>
#include <sys/stat.h>

void ensure_reports_dir() {
    mkdir("/home/aniru/kernel-craft/reports", 0755);
}

// CPU reference for bilinear resize
void resize_bilinear_cpu(const float* input, float* output,
                         int in_w, int in_h, int out_w, int out_h) {
    for (int oy = 0; oy < out_h; ++oy) {
        for (int ox = 0; ox < out_w; ++ox) {
            float x = (ox + 0.5f) * static_cast<float>(in_w) / static_cast<float>(out_w) - 0.5f;
            float y = (oy + 0.5f) * static_cast<float>(in_h) / static_cast<float>(out_h) - 0.5f;

            int x0 = static_cast<int>(floorf(x));
            int y0 = static_cast<int>(floorf(y));
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            x0 = std::max(0, std::min(x0, in_w - 1));
            y0 = std::max(0, std::min(y0, in_h - 1));
            x1 = std::max(0, std::min(x1, in_w - 1));
            y1 = std::max(0, std::min(y1, in_h - 1));

            float fx = x - x0;
            float fy = y - y0;

            float f00 = input[y0 * in_w + x0];
            float f10 = input[y0 * in_w + x1];
            float f01 = input[y1 * in_w + x0];
            float f11 = input[y1 * in_w + x1];

            output[oy * out_w + ox] = (1 - fx) * (1 - fy) * f00 +
                                       fx * (1 - fy) * f10 +
                                       (1 - fx) * fy * f01 +
                                       fx * fy * f11;
        }
    }
}

void normalize_cpu(const float* input, float* output, int n, float mean, float std) {
    for (int i = 0; i < n; ++i) {
        output[i] = (input[i] - mean) / std;
    }
}

void flip_horizontal_cpu(const float* input, float* output, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            output[y * width + x] = input[y * width + (width - 1 - x)];
        }
    }
}

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
                    sum += input[iy * width + ix] * kernel[ky * ksize + kx];
                }
            }
            output[oy * width + ox] = sum;
        }
    }
}

// GPU kernels from preprocess_gpu.cu
extern "C" void launch_resize_bilinear(const float* d_input, float* d_output,
                                      int in_w, int in_h, int out_w, int out_h);
extern "C" void launch_normalize(const float* d_input, float* d_output,
                                  int width, int height, float mean, float std);
extern "C" void launch_flip_horizontal(const float* d_input, float* d_output,
                                       int width, int height);
extern "C" void launch_conv_tiled(const float* d_input, const float* d_kernel,
                                  float* d_output, int width, int height, int ksize,
                                  dim3 block = dim3(8, 8, 1));

int main(int argc, char** argv) {
    bool full_gpu = true;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--cpu-preprocess") {
            full_gpu = false;
        }
    }

    const int in_w = 640, in_h = 480;
    const int out_w = 1024, out_h = 1024;
    const int ksize = 3;
    const float mean = 0.0f, std = 1.0f;

    const int in_size = in_w * in_h;
    const int out_size = out_w * out_h;
    const int ker_size = ksize * ksize;

    float *h_input = (float*)malloc(sizeof(float) * in_size);
    float *h_kernel = (float*)malloc(sizeof(float) * ker_size);
    float *h_output = (float*)malloc(sizeof(float) * out_size);
    float *h_temp = (float*)malloc(sizeof(float) * out_size);

    for (int i = 0; i < in_size; ++i) h_input[i] = static_cast<float>(i % 256) / 255.0f;
    for (int i = 0; i < ker_size; ++i) h_kernel[i] = 1.0f / ker_size;

    // Allocate GPU memory
    float *d_input, *d_temp, *d_kernel, *d_output;
    cudaMalloc(&d_input, sizeof(float) * in_size);
    cudaMalloc(&d_temp, sizeof(float) * out_size);
    cudaMalloc(&d_kernel, sizeof(float) * ker_size);
    cudaMalloc(&d_output, sizeof(float) * out_size);

    cudaMemcpy(d_input, h_input, sizeof(float) * in_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float) * ker_size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up runs
    for (int i = 0; i < 3; ++i) {
        if (full_gpu) {
            launch_resize_bilinear(d_input, d_temp, in_w, in_h, out_w, out_h);
            launch_normalize(d_temp, d_temp, out_w, out_h, mean, std);
            launch_flip_horizontal(d_temp, d_temp, out_w, out_h);
            launch_conv_tiled(d_temp, d_kernel, d_output, out_w, out_h, ksize);
        } else {
            cudaMemcpy(d_temp, h_input, sizeof(float) * in_size, cudaMemcpyHostToDevice);
            launch_conv_tiled(d_temp, d_kernel, d_output, in_w, in_h, ksize);
        }
    }
    cudaDeviceSynchronize();

    // Timed run
    cudaEventRecord(start);
    if (full_gpu) {
        // Full GPU pipeline
        launch_resize_bilinear(d_input, d_temp, in_w, in_h, out_w, out_h);
        launch_normalize(d_temp, d_temp, out_w, out_h, mean, std);
        launch_flip_horizontal(d_temp, d_temp, out_w, out_h);
        launch_conv_tiled(d_temp, d_kernel, d_output, out_w, out_h, ksize);
    } else {
        // CPU preprocess + GPU convolution
        resize_bilinear_cpu(h_input, h_temp, in_w, in_h, out_w, out_h);
        normalize_cpu(h_temp, h_temp, out_size, mean, std);
        flip_horizontal_cpu(h_temp, h_temp, out_w, out_h);
        cudaMemcpy(d_temp, h_temp, sizeof(float) * out_size, cudaMemcpyHostToDevice);
        launch_conv_tiled(d_temp, d_kernel, d_output, out_w, out_h, ksize);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    cudaMemcpy(h_output, d_output, sizeof(float) * out_size, cudaMemcpyDeviceToHost);

    // CPU reference for verification
    float *h_cpu_out = (float*)malloc(sizeof(float) * out_size);
    resize_bilinear_cpu(h_input, h_temp, in_w, in_h, out_w, out_h);
    normalize_cpu(h_temp, h_temp, out_size, mean, std);
    flip_horizontal_cpu(h_temp, h_temp, out_w, out_h);
    conv_naive_cpu(h_temp, h_kernel, h_cpu_out, out_w, out_h, ksize);

    // Verify (sample check)
    bool match = true;
    for (int i = 0; i < std::min(100, out_size); ++i) {
        if (std::fabs(h_output[i] - h_cpu_out[i]) > 1e-4f) {
            match = false;
            break;
        }
    }

    double num_pixels = static_cast<double>(out_size);
    double throughput = num_pixels / (gpu_ms * 1e-3) / 1e6;

    printf("=== Full Pipeline Benchmark ===\n");
    printf("Input: %dx%d -> Output: %dx%d\n", in_w, in_h, out_w, out_h);
    printf("Pipeline: %s\n", full_gpu ? "Full GPU" : "CPU preprocess + GPU conv");
    printf("Time: %.3f ms (%.3f MPixels/s)\n", gpu_ms, throughput);
    printf("Verification: %s\n", match ? "PASSED" : "FAILED");

    ensure_reports_dir();
    FILE* f = fopen("/home/aniru/kernel-craft/reports/benchmark_full_pipeline.txt", "w");
    if (f) {
        fprintf(f, "=== Full Pipeline Benchmark ===\n");
        fprintf(f, "Input: %dx%d -> Output: %dx%d\n", in_w, in_h, out_w, out_h);
        fprintf(f, "Pipeline: %s\n", full_gpu ? "Full GPU" : "CPU preprocess + GPU conv");
        fprintf(f, "Time: %.3f ms (%.3f MPixels/s)\n", gpu_ms, throughput);
        fprintf(f, "Verification: %s\n", match ? "PASSED" : "FAILED");
        fclose(f);
    }

    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(h_input);
    free(h_kernel);
    free(h_output);
    free(h_temp);
    free(h_cpu_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return match ? EXIT_SUCCESS : EXIT_FAILURE;
}