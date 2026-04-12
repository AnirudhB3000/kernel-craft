/**
 * \file benchmark_pipeline.cpp
 * \brief Benchmark for fused vs separate conv → batchnorm → ReLU pipeline.
 *
 * Compares:
 * - Separate: 3 kernel launches (conv + batchnorm + relu)
 * - fused:    1 kernel launch (conv + batchnorm + relu fused)
 *
 * Measures kernel launch overhead, memory traffic, and total latency.
 */

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>
#include <sys/stat.h>

void ensure_reports_dir() {
    mkdir("/home/aniru/kernel-craft/reports", 0755);
}

extern "C" void launch_conv_layer(const float* d_input, const float* d_kernel,
                                 float* d_output,
                                 int width, int height, int ksize,
                                 dim3 block);
extern "C" void launch_batchnorm_layer(float* d_input, float gamma, float beta,
                                     float* d_output, int size, dim3 block);
extern "C" void launch_relu_layer(const float* d_input, float* d_output,
                                 int size, dim3 block);

extern "C" void launch_conv_bn_relu_fused(const float* d_input, const float* d_kernel,
                                        float gamma, float beta,
                                        float* d_output,
                                        int width, int height, int ksize,
                                        dim3 block);

#define TILE_W 8
#define TILE_H 8

/**
 * \brief CPU reference: conv → batchnorm → ReLU pipeline.
 */
void pipeline_cpu(const float* input, const float* kernel,
                float gamma, float beta,
                float* output,
                int width, int height, int ksize) {
    int size = width * height;
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
            float conv = static_cast<float>(sum);
            float bn = conv * gamma + beta;
            output[oy * width + ox] = bn > 0.0f ? bn : 0.0f;
        }
    }
}

/**
 * \brief Main benchmark driver.
 */
int main(int argc, char** argv) {
    const int width = 1024, height = 1024, ksize = 3;
    const int imgSize = width * height;
    const int kerSize = ksize * ksize;
    const float gamma = 1.5f;
    const float beta = 0.2f;

    float *h_input = (float*)malloc(sizeof(float) * imgSize);
    float *h_kernel = (float*)malloc(sizeof(float) * kerSize);
    float *h_output_fused = (float*)malloc(sizeof(float) * imgSize);
    float *h_output_separate = (float*)malloc(sizeof(float) * imgSize);
    float *h_output_cpu = (float*)malloc(sizeof(float) * imgSize);

    for (int i = 0; i < imgSize; ++i) h_input[i] = static_cast<float>((i % 256) + 1);
    for (int i = 0; i < kerSize; ++i) h_kernel[i] = 1.0f / static_cast<float>(kerSize);

    float *d_input, *d_kernel, *d_intermediate, *d_output;
    cudaMalloc(&d_input, sizeof(float) * imgSize);
    cudaMalloc(&d_kernel, sizeof(float) * kerSize);
    cudaMalloc(&d_intermediate, sizeof(float) * imgSize);
    cudaMalloc(&d_output, sizeof(float) * imgSize);

    cudaMemcpy(d_input, h_input, sizeof(float) * imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float) * kerSize, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block(TILE_W, TILE_H, 1);

    // Warm-up runs
    for (int i = 0; i < 3; ++i) {
        launch_conv_bn_relu_fused(d_input, d_kernel, gamma, beta,
                                d_output, width, height, ksize, block);
        launch_conv_layer(d_input, d_kernel, d_intermediate, width, height, ksize, block);
        launch_batchnorm_layer(d_intermediate, gamma, beta, d_intermediate, imgSize, dim3(256, 1, 1));
        launch_relu_layer(d_intermediate, d_output, imgSize, dim3(256, 1, 1));
    }
    cudaDeviceSynchronize();

    // --- Fused pipeline (single kernel) ---
    cudaEventRecord(start);
    launch_conv_bn_relu_fused(d_input, d_kernel, gamma, beta,
                            d_output, width, height, ksize, block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float fused_ms = 0.0f;
    cudaEventElapsedTime(&fused_ms, start, stop);

    cudaMemcpy(h_output_fused, d_output, sizeof(float) * imgSize, cudaMemcpyDeviceToHost);

    // --- Separate pipeline (3 kernels) ---
    cudaEventRecord(start);
    launch_conv_layer(d_input, d_kernel, d_intermediate, width, height, ksize, block);
    launch_batchnorm_layer(d_intermediate, gamma, beta, d_intermediate, imgSize, dim3(256, 1, 1));
    launch_relu_layer(d_intermediate, d_output, imgSize, dim3(256, 1, 1));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float separate_ms = 0.0f;
    cudaEventElapsedTime(&separate_ms, start, stop);

    cudaMemcpy(h_output_separate, d_output, sizeof(float) * imgSize, cudaMemcpyDeviceToHost);

    // --- CPU reference ---
    auto cpu_start = std::chrono::high_resolution_clock::now();
    pipeline_cpu(h_input, h_kernel, gamma, beta, h_output_cpu, width, height, ksize);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_ms = cpu_end - cpu_start;

    const float eps = 1e-3f;
    bool fused_match = true, separate_match = true;
    for (int i = 0; i < imgSize; ++i) {
        if (std::abs(h_output_fused[i] - h_output_cpu[i]) > eps) fused_match = false;
        if (std::abs(h_output_separate[i] - h_output_cpu[i]) > eps) separate_match = false;
    }

    double num_pixels = static_cast<double>(imgSize);
    double fused_throughput = num_pixels / (fused_ms * 1e-3) / 1e6;
    double separate_throughput = num_pixels / (separate_ms * 1e-3) / 1e6;
    double speedup = separate_ms / fused_ms;

    printf("=== Pipeline Benchmark ===\n");
    printf("Image size: %dx%d (%.2f MPixels)\n", width, height, num_pixels/1e6);
    printf("Kernel size: %dx%d\n", ksize, ksize);
    printf("Batchnorm: gamma=%.2f beta=%.2f\n", gamma, beta);
    printf("\n");
    printf("Fused (1 kernel):    %.3f ms (%.3f MPixels/s)\n", fused_ms, fused_throughput);
    printf("Separate (3 kernel): %.3f ms (%.3f MPixels/s)\n", separate_ms, separate_throughput);
    printf("CPU reference:       %.3f ms\n", cpu_ms.count());
    printf("\n");
    printf("Fused speedup vs separate: %.2fx\n", speedup);
    printf("Fused verification: %s\n", fused_match ? "PASSED" : "FAILED");
    printf("Separate verification: %s\n", separate_match ? "PASSED" : "FAILED");

    ensure_reports_dir();
    FILE* f = fopen("/home/aniru/kernel-craft/reports/benchmark_pipeline.txt", "w");
    if (f) {
        fprintf(f, "=== Pipeline Benchmark ===\n");
        fprintf(f, "Image size: %dx%d (%.2f MPixels)\n", width, height, num_pixels/1e6);
        fprintf(f, "Kernel size: %dx%d\n", ksize, ksize);
        fprintf(f, "Batchnorm: gamma=%.2f beta=%.2f\n", gamma, beta);
        fprintf(f, "\n");
        fprintf(f, "Fused (1 kernel):    %.3f ms (%.3f MPixels/s)\n", fused_ms, fused_throughput);
        fprintf(f, "Separate (3 kernel): %.3f ms (%.3f MPixels/s)\n", separate_ms, separate_throughput);
        fprintf(f, "CPU reference:       %.3f ms\n", cpu_ms.count());
        fprintf(f, "\n");
        fprintf(f, "Fused speedup vs separate: %.2fx\n", speedup);
        fprintf(f, "Fused verification: %s\n", fused_match ? "PASSED" : "FAILED");
        fprintf(f, "Separate verification: %s\n", separate_match ? "PASSED" : "FAILED");
        fclose(f);
    }

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_intermediate);
    cudaFree(d_output);
    free(h_input);
    free(h_kernel);
    free(h_output_fused);
    free(h_output_separate);
    free(h_output_cpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (fused_match && separate_match) ? EXIT_SUCCESS : EXIT_FAILURE;
}