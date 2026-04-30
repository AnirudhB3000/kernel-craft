/**
 * \file test_conv_int8.cpp
 * \brief Unit tests for INT8 quantized convolution kernels.
 *
 * Verifies correctness of INT8 naive and tiled convolution
 * against reference float implementation.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>

// Declare device functions
extern "C" void launch_conv_int8_naive(const float* d_input,
                                        const float* d_kernel,
                                        float* d_output,
                                        int width, int height, int ksize,
                                        float input_scale,
                                        float kernel_scale,
                                        float output_scale,
                                        dim3 block);

extern "C" void launch_conv_int8_tiled(const float* d_input,
                                        const float* d_kernel,
                                        float* d_output,
                                        int width, int height, int ksize,
                                        float input_scale,
                                        float kernel_scale,
                                        float output_scale,
                                        dim3 block);

extern "C" float compute_quantization_scale(const float* h_data, int size);

// Reference CPU convolution
void conv_cpu_ref(const float* input, const float* kernel,
                 float* output, int width, int height, int ksize) {
    int kHalf = ksize / 2;
    for (int oy = 0; oy < height; ++oy) {
        for (int ox = 0; ox < width; ++ox) {
            float sum = 0.0f;
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

float max_error(const float* a, const float* b, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_err = fmaxf(max_err, fabsf(a[i] - b[i]));
    }
    return max_err;
}

bool test_int8_conv(int width, int height, int ksize, const char* name) {
    printf("Testing %s (%dx%d, kernel %dx%d)...\n", name, width, height, ksize, ksize);
    
    int img_size = width * height;
    int kernel_size = ksize * ksize;
    
    std::vector<float> h_input(img_size);
    std::vector<float> h_kernel(kernel_size);
    std::vector<float> h_output_gpu(img_size);
    std::vector<float> h_output_ref(img_size);
    
    // Initialize with small values (important for INT8 quantization)
    for (int i = 0; i < img_size; ++i) {
        h_input[i] = (rand() % 100) / 100.0f - 0.5f;  // [-0.5, 0.5]
    }
    for (int i = 0; i < kernel_size; ++i) {
        h_kernel[i] = (rand() % 100) / 100.0f - 0.5f;
    }
    
    // Compute quantization scales
    float input_scale = compute_quantization_scale(h_input.data(), img_size);
    float kernel_scale = compute_quantization_scale(h_kernel.data(), kernel_size);
    float output_scale = 1.0f;  // Direct dequantization
    
    if (input_scale == 0.0f) input_scale = 1.0f;
    if (kernel_scale == 0.0f) kernel_scale = 1.0f;
    
    // Allocate device memory
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, img_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMalloc(&d_output, img_size * sizeof(float));
    
    cudaMemcpy(d_input, h_input.data(), img_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel.data(), kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Run GPU kernel
    dim3 block(16, 16);
    if (strcmp(name, "INT8 Naive") == 0) {
        launch_conv_int8_naive(d_input, d_kernel, d_output, width, height, ksize,
                               input_scale, kernel_scale, output_scale, block);
    } else {
        launch_conv_int8_tiled(d_input, d_kernel, d_output, width, height, ksize,
                               input_scale, kernel_scale, output_scale, block);
    }
    
    cudaMemcpy(h_output_gpu.data(), d_output, img_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compute reference
    conv_cpu_ref(h_input.data(), h_kernel.data(), h_output_ref.data(), width, height, ksize);
    
    // Check accuracy (INT8 has lower precision)
    float err = max_error(h_output_gpu.data(), h_output_ref.data(), img_size);
    float tolerance = 0.1f;  // INT8 quantization error tolerance
    
    printf("  Max error: %f (tolerance: %f) -> %s\n", err, tolerance, err < tolerance ? "PASS" : "FAIL");
    
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    
    return err < tolerance;
}

int main() {
    printf("=== INT8 Quantized Convolution Tests ===\n\n");
    
    bool all_pass = true;
    
    // Test INT8 Naive
    all_pass &= test_int8_conv(64, 64, 3, "INT8 Naive");
    all_pass &= test_int8_conv(128, 128, 3, "INT8 Naive");
    
    // Test INT8 Tiled
    all_pass &= test_int8_conv(64, 64, 3, "INT8 Tiled");
    all_pass &= test_int8_conv(128, 128, 3, "INT8 Tiled");
    all_pass &= test_int8_conv(256, 256, 3, "INT8 Tiled");
    
    printf("\n=== %s ===\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
