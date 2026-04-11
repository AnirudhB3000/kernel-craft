/**
 * \file benchmark_conv.cpp
 * \brief Benchmark for the naïve 2‑D convolution kernel.
 *
 * This program generates a synthetic input image and a simple averaging
 * kernel, runs the convolution on both the CPU (reference implementation)
 * and the GPU (via the CUDA kernel), and reports execution time, throughput,
 * and the speed‑up factor. All timing is performed with high‑resolution
 * clocks (CUDA events for GPU and std::chrono for CPU).
 */

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>
#include <string>

// Host‑side reference implementation (identical to the one used in the unit test)
/**
 * \brief CPU reference implementation of naïve 2‑D convolution.
 *
 * This function mirrors the algorithm used in the CUDA kernel but runs
 * on the host for verification and benchmarking purposes.
 *
 * \param[in] input  Pointer to the input image (row‑major).
 * \param[in] kernel Pointer to the convolution kernel (row‑major).
 * \param[out] output Destination buffer for the convolution result.
 * \param[in] width  Image width.
 * \param[in] height Image height.
 * \param[in] ksize Kernel dimension (must be odd).
 */
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
                    float val = input[iy * width + ix];
                    float kval = kernel[ky * ksize + kx];
                    sum += val * kval;
                }
            }
            output[oy * width + ox] = sum;
        }
    }
}

// Declaration of the host launcher defined in conv_naive.cu
extern "C" void launch_conv_naive(const float* d_input,
                                 const float* d_kernel,
                                 float* d_output,
                                 int width, int height, int ksize,
                                 dim3 block = dim3(16,16,1));

// Declaration of the host launcher defined in conv_tiled.cu
extern "C" void launch_conv_tiled(const float* d_input,
                                  const float* d_kernel,
                                  float* d_output,
                                  int width, int height, int ksize,
                                  dim3 block = dim3(16,16,1));

/**
 * \brief Benchmark driver for convolution kernels.
 *
 * This program generates a synthetic input image and a simple averaging
 * kernel, runs either the naïve or tiled convolution kernel (selected
 * via the `--tiled` command‑line flag), verifies the result against a
 * CPU reference implementation, and reports timing and throughput.
 *
 * \param argc Argument count.
 * \param argv Argument vector; optional `--tiled` enables the tiled kernel.
 * \return EXIT_SUCCESS if verification passes, EXIT_FAILURE otherwise.
 */
int main(int argc, char** argv) {
    bool use_tiled = false;
    // Simple flag parsing: look for "--tiled"
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--tiled") {
            use_tiled = true;
            // Remove the flag from argument list for later parsing (if any)
            for (int j = i; j < argc - 1; ++j) argv[j] = argv[j+1];
            --argc;
            break;
        }
    }

    // Default dimensions – can be overridden via command line if desired
    const int width  = (argc > 1) ? std::atoi(argv[1]) : 1024;
    const int height = (argc > 2) ? std::atoi(argv[2]) : 1024;
    const int ksize  = 3; // fixed 3×3 averaging kernel for the benchmark

    const int imgSize = width * height;
    const int kerSize = ksize * ksize;

    // Allocate and initialise host buffers
    float *h_input  = (float*)malloc(sizeof(float) * imgSize);
    float *h_kernel = (float*)malloc(sizeof(float) * kerSize);
    float *h_output_gpu = (float*)malloc(sizeof(float) * imgSize);
    float *h_output_cpu = (float*)malloc(sizeof(float) * imgSize);

    // Fill input with deterministic values (e.g., sequential numbers)
    for (int i = 0; i < imgSize; ++i) h_input[i] = static_cast<float>(i + 1);
    // Simple averaging kernel
    for (int i = 0; i < kerSize; ++i) h_kernel[i] = 1.0f / kerSize;

    // ------------------------------------------------
    // GPU benchmark (timed using CUDA events)
    // ------------------------------------------------
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input,  sizeof(float) * imgSize);
    cudaMalloc(&d_kernel, sizeof(float) * kerSize);
    cudaMalloc(&d_output, sizeof(float) * imgSize);

    cudaMemcpy(d_input,  h_input,  sizeof(float) * imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float) * kerSize, cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    if (use_tiled) {
        launch_conv_tiled(d_input, d_kernel, d_output, width, height, ksize);
    } else {
        launch_conv_naive(d_input, d_kernel, d_output, width, height, ksize);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    // Copy result back to host
    cudaMemcpy(h_output_gpu, d_output, sizeof(float) * imgSize, cudaMemcpyDeviceToHost);

    // ------------------------------------------------
    // CPU benchmark
    // ------------------------------------------------
    auto cpu_start = std::chrono::high_resolution_clock::now();
    conv_naive_cpu(h_input, h_kernel, h_output_cpu, width, height, ksize);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_ms = cpu_end - cpu_start;

    // ------------------------------------------------
    // Verify correctness (quick check)
    // ------------------------------------------------
    const float eps = 1e-5f; // relative tolerance factor for GPU/CPU rounding differences
    bool match = true;
    for (int i = 0; i < imgSize; ++i) {
        float diff = std::abs(h_output_gpu[i] - h_output_cpu[i]);
        float max_val = std::max(std::abs(h_output_gpu[i]), std::abs(h_output_cpu[i]));
        if (diff > eps * max_val) {
            match = false;
            fprintf(stderr, "Mismatch at %d: GPU=%f CPU=%f (diff=%f)\n", i, h_output_gpu[i], h_output_cpu[i], diff);
            break;
        }
    }
    if (!match) {
        fprintf(stderr, "Result verification failed.\n");
        return EXIT_FAILURE;
    }

    // ------------------------------------------------
    // Report results
    // ------------------------------------------------
    double num_pixels = static_cast<double>(imgSize);
    double gpu_throughput = num_pixels / (gpu_ms * 1e-3) / 1e6; // MPixels/s
    double cpu_throughput = num_pixels / (cpu_ms.count() * 1e-3) / 1e6; // MPixels/s
    double speedup = cpu_ms.count() / gpu_ms;

    printf("=== Convolution Benchmark ===\n");
    printf("Image size: %dx%d (%.2f MPixels)\n", width, height, num_pixels/1e6);
    printf("Kernel size: %dx%d\n", ksize, ksize);
    printf("GPU time: %.3f ms (%.3f MPixels/s)\n", gpu_ms, gpu_throughput);
    printf("CPU time: %.3f ms (%.3f MPixels/s)\n", cpu_ms.count(), cpu_throughput);
    printf("Speed‑up: %.2fx\n", speedup);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(h_input);
    free(h_kernel);
    free(h_output_gpu);
    free(h_output_cpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return EXIT_SUCCESS;
}
