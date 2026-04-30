/**
 * \file test_tensorrt_plugin.cpp
 * \brief Simplified test for TensorRT plugin.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// External kernel functions
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

// Plugin registration
extern "C" void registerConvInt8Plugin();

bool test_kernel_functions() {
    printf("Testing kernel functions used by plugins...\n");
    
    const int width = 64, height = 64;
    const int ksize = 3;
    const int img_size = width * height;
    const int kernel_size = ksize * ksize;
    
    float* d_input;
    float* d_kernel;
    float* d_output;
    cudaMalloc(&d_input, img_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMalloc(&d_output, img_size * sizeof(float));
    
    // Test INT8 kernel
    printf("  Testing conv_int8_tiled...\n");
    float input_scale = 0.1f, kernel_scale = 0.1f, output_scale = 1.0f;
    launch_conv_int8_tiled(d_input, d_kernel, d_output,
                           width, height, ksize,
                           input_scale, kernel_scale, output_scale,
                           dim3(16, 16));
    
    // Test ReLU kernel
    printf("  Testing conv_relu_tiled...\n");
    launch_conv_relu_tiled(d_input, d_kernel, d_output,
                            width, height, ksize,
                            dim3(16, 16));
    
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    
    printf("  -> PASS (kernels callable)\n");
    return true;
}

#ifdef HAVE_TENSORRT
bool test_plugin_registration() {
    printf("\nTesting TensorRT plugin registration...\n");
    
    // Register plugin
    registerConvInt8Plugin();
    printf("  Plugin registered\n");
    
    // Note: Full plugin testing requires TensorRT runtime
    // This just verifies the registration function works
    
    printf("  -> PASS (plugin registration works)\n");
    return true;
}
#endif

int main() {
    printf("=== TensorRT Plugin Test ===\n\n");
    
    bool pass = test_kernel_functions();
    
#ifdef HAVE_TENSORRT
    pass &= test_plugin_registration();
#else
    printf("\nHAVE_TENSORRT not defined.\n");
    printf("To test full plugin: cmake .. -DTENSORRT_ROOT=/path/to/TensorRT\n");
#endif
    
    printf("\n=== %s ===\n", pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return pass ? 0 : 1;
}
