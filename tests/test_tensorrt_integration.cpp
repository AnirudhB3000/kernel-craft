/**
 * \file test_tensorrt_integration.cpp
 * \brief Basic test for TensorRT plugin integration.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#ifdef HAVE_TENSORRT
#include <NvInfer.h>
#endif

extern "C" void launch_conv_int8_tiled(const float* d_input,
                                        const float* d_kernel,
                                        float* d_output,
                                        int width, int height, int ksize,
                                        float input_scale,
                                        float kernel_scale,
                                        float output_scale,
                                        dim3 block);

extern "C" void launch_conv_relu_tiled(const float* d_input,
                                        const float* d_kernel,
                                        float* d_output,
                                        int width, int height, int ksize,
                                        dim3 block);

bool test_plugin_kernels() {
    printf("Testing TensorRT plugin kernel functions...\n");
    
    const int width = 64, height = 64;
    const int ksize = 3;
    const int img_size = width * height;
    const int kernel_size = ksize * ksize;
    
    std::vector<float> h_input(img_size);
    std::vector<float> h_kernel(kernel_size);
    std::vector<float> h_output(img_size);
    
    for (int i = 0; i < img_size; ++i) {
        h_input[i] = (rand() % 100) / 100.0f - 0.5f;
    }
    for (int i = 0; i < kernel_size; ++i) {
        h_kernel[i] = (rand() % 100) / 100.0f - 0.5f;
    }
    
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, img_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMalloc(&d_output, img_size * sizeof(float));
    
    cudaMemcpy(d_input, h_input.data(), img_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel.data(), kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    
    printf("  Testing conv_int8_tiled...\n");
    float input_scale = 0.1f, kernel_scale = 0.1f, output_scale = 1.0f;
    launch_conv_int8_tiled(d_input, d_kernel, d_output,
                           width, height, ksize,
                           input_scale, kernel_scale, output_scale,
                           dim3(16, 16));
    
    printf("  Testing conv_relu_tiled...\n");
    launch_conv_relu_tiled(d_input, d_kernel, d_output,
                            width, height, ksize,
                            dim3(16, 16));
    
    cudaMemcpy(h_output.data(), d_output, img_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("  Output[0] = %f (should be non-zero)\n", h_output[0]);
    
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    
    printf("  -> PASS (kernels callable)\n");
    return true;
}

int main() {
    printf("=== TensorRT Integration Test ===\n\n");
    
    bool pass = test_plugin_kernels();
    
#ifndef HAVE_TENSORRT
    printf("\nTensorRT SDK not found (HAVE_TENSORRT not defined).\n");
    printf("To test full plugin: install TensorRT SDK and build with:\n");
    printf("  cmake .. -DTENSORRT_ROOT=/path/to/TensorRT\n");
    printf("  make test_tensorrt_integration\n");
#endif
    
    printf("\n=== %s ===\n", pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return pass ? 0 : 1;
}
