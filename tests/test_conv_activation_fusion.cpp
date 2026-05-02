/**
 * \file test_conv_activation_fusion.cpp
 * \brief Unit tests for fused convolution + activation kernels.
 *
 * Verifies correctness of fused kernels against separate conv + activation.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// Declare device functions
extern "C" void launch_conv_relu_naive(const float* d_input,
                                         const float* d_kernel,
                                         float* d_output,
                                         int width, int height, int ksize,
                                         dim3 block);

extern "C" void launch_conv_leaky_relu_naive(const float* d_input,
                                                const float* d_kernel,
                                                float* d_output,
                                                int width, int height, int ksize,
                                                float alpha,
                                                dim3 block);

extern "C" void launch_conv_relu_tiled(const float* d_input,
                                        const float* d_kernel,
                                        float* d_output,
                                        int width, int height, int ksize,
                                        dim3 block);

extern "C" void launch_conv_sigmoid_naive(const float* d_input,
                                           const float* d_kernel,
                                           float* d_output,
                                           int width, int height, int ksize,
                                           dim3 block);

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

float relu(float x) { return fmaxf(0.0f, x); }
float leaky_relu(float x, float alpha = 0.01f) { return x > 0.0f ? x : alpha * x; }
float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

float max_error(const float* a, const float* b, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_err = fmaxf(max_err, fabsf(a[i] - b[i]));
    }
    return max_err;
}

bool test_conv_relu(int width, int height, int ksize, const char* name, bool tiled) {
    printf("Testing Conv+ReLU (%s, %dx%d, kernel %dx%d)...\n", name, width, height, ksize, ksize);
    
    int img_size = width * height;
    int kernel_size = ksize * ksize;
    
    std::vector<float> h_input(img_size);
    std::vector<float> h_kernel(kernel_size);
    std::vector<float> h_output_gpu(img_size);
    std::vector<float> h_output_ref(img_size);
    std::vector<float> h_conv_ref(img_size);
    
    for (int i = 0; i < img_size; ++i) {
        h_input[i] = (rand() % 1000) / 100.0f - 5.0f;  // [-5, 5]
    }
    for (int i = 0; i < kernel_size; ++i) {
        h_kernel[i] = (rand() % 100) / 100.0f - 0.5f;
    }
    
    // Compute reference: conv + relu
    conv_cpu_ref(h_input.data(), h_kernel.data(), h_conv_ref.data(), width, height, ksize);
    for (int i = 0; i < img_size; ++i) {
        h_output_ref[i] = relu(h_conv_ref[i]);
    }
    
    // Allocate device memory
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, img_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMalloc(&d_output, img_size * sizeof(float));
    
    cudaMemcpy(d_input, h_input.data(), img_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel.data(), kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Run GPU kernel
    dim3 block(16, 16);
    if (tiled) {
        launch_conv_relu_tiled(d_input, d_kernel, d_output, width, height, ksize, block);
    } else {
        launch_conv_relu_naive(d_input, d_kernel, d_output, width, height, ksize, block);
    }
    
    cudaMemcpy(h_output_gpu.data(), d_output, img_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    float err = max_error(h_output_gpu.data(), h_output_ref.data(), img_size);
    float tolerance = 1e-4f;
    
    printf("  Max error: %f (tolerance: %f) -> %s\n", err, tolerance, err < tolerance ? "PASS" : "FAIL");
    
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    
    return err < tolerance;
}

bool test_conv_leaky_relu(int width, int height, int ksize) {
    printf("Testing Conv+LeakyReLU (%dx%d, kernel %dx%d)...\n", width, height, ksize, ksize);
    
    int img_size = width * height;
    int kernel_size = ksize * ksize;
    float alpha = 0.1f;
    
    std::vector<float> h_input(img_size);
    std::vector<float> h_kernel(kernel_size);
    std::vector<float> h_output_gpu(img_size);
    std::vector<float> h_output_ref(img_size);
    std::vector<float> h_conv_ref(img_size);
    
    for (int i = 0; i < img_size; ++i) {
        h_input[i] = (rand() % 1000) / 100.0f - 5.0f;
    }
    for (int i = 0; i < kernel_size; ++i) {
        h_kernel[i] = (rand() % 100) / 100.0f - 0.5f;
    }
    
    conv_cpu_ref(h_input.data(), h_kernel.data(), h_conv_ref.data(), width, height, ksize);
    for (int i = 0; i < img_size; ++i) {
        h_output_ref[i] = leaky_relu(h_conv_ref[i], alpha);
    }
    
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, img_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMalloc(&d_output, img_size * sizeof(float));
    
    cudaMemcpy(d_input, h_input.data(), img_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel.data(), kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 block(16, 16);
    launch_conv_leaky_relu_naive(d_input, d_kernel, d_output, width, height, ksize, alpha, block);
    
    cudaMemcpy(h_output_gpu.data(), d_output, img_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    float err = max_error(h_output_gpu.data(), h_output_ref.data(), img_size);
    float tolerance = 1e-4f;
    
    printf("  Max error: %f (tolerance: %f) -> %s\n", err, tolerance, err < tolerance ? "PASS" : "FAIL");
    
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    
    return err < tolerance;
}

bool test_conv_sigmoid(int width, int height, int ksize) {
    printf("Testing Conv+Sigmoid (%dx%d, kernel %dx%d)...\n", width, height, ksize, ksize);

    int img_size = width * height;
    int kernel_size = ksize * ksize;

    std::vector<float> h_input(img_size);
    std::vector<float> h_kernel(kernel_size);
    std::vector<float> h_output_gpu(img_size);
    std::vector<float> h_output_ref(img_size);
    std::vector<float> h_conv_ref(img_size);

    for (int i = 0; i < img_size; ++i) h_input[i] = (rand() % 1000) / 100.0f - 5.0f;
    for (int i = 0; i < kernel_size; ++i) h_kernel[i] = (rand() % 100) / 100.0f - 0.5f;

    conv_cpu_ref(h_input.data(), h_kernel.data(), h_conv_ref.data(), width, height, ksize);
    for (int i = 0; i < img_size; ++i) h_output_ref[i] = sigmoid(h_conv_ref[i]);

    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, img_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMalloc(&d_output, img_size * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), img_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel.data(), kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    launch_conv_sigmoid_naive(d_input, d_kernel, d_output, width, height, ksize, block);

    cudaMemcpy(h_output_gpu.data(), d_output, img_size * sizeof(float), cudaMemcpyDeviceToHost);

    float err = max_error(h_output_gpu.data(), h_output_ref.data(), img_size);
    float tolerance = 1e-4f;

    printf("  Max error: %f (tolerance: %f) -> %s\n", err, tolerance, err < tolerance ? "PASS" : "FAIL");

    cudaFree(d_input); cudaFree(d_kernel); cudaFree(d_output);

    return err < tolerance;
}

int main() {
    printf("=== Conv + Activation Fusion Tests ===\n\n");

    bool all_pass = true;

    // ReLU tests (naive and tiled)
    all_pass &= test_conv_relu(64, 64, 3, "Naive", false);
    all_pass &= test_conv_relu(64, 64, 3, "Tiled", true);
    all_pass &= test_conv_relu(128, 128, 3, "Tiled", true);
    all_pass &= test_conv_relu(256, 256, 3, "Tiled", true);

    // Leaky ReLU tests
    all_pass &= test_conv_leaky_relu(64, 64, 3);
    all_pass &= test_conv_leaky_relu(128, 128, 3);

    // Sigmoid tests
    all_pass &= test_conv_sigmoid(64, 64, 3);
    all_pass &= test_conv_sigmoid(128, 128, 3);

    // Test with 5x5 kernel
    all_pass &= test_conv_relu(64, 64, 5, "Tiled 5x5", true);

    // Test with edge values (zeros, negative values)
    printf("\nTesting with edge cases...\n");
    {
        int width = 32, height = 32, ksize = 3;
        int img_size = width * height;
        int kernel_size = ksize * ksize;

        std::vector<float> h_input(img_size, 0.0f); // All zeros
        std::vector<float> h_kernel(kernel_size, 1.0f); // All ones
        std::vector<float> h_output_gpu(img_size);

        float *d_input, *d_kernel, *d_output;
        cudaMalloc(&d_input, img_size * sizeof(float));
        cudaMalloc(&d_kernel, kernel_size * sizeof(float));
        cudaMalloc(&d_output, img_size * sizeof(float));

        cudaMemcpy(d_input, h_input.data(), img_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, h_kernel.data(), kernel_size * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        launch_conv_relu_tiled(d_input, d_kernel, d_output, width, height, ksize, block);

        cudaMemcpy(h_output_gpu.data(), d_output, img_size * sizeof(float), cudaMemcpyDeviceToHost);

        bool pass = true;
        for (int i = 0; i < img_size; ++i) {
            if (h_output_gpu[i] != 0.0f) { pass = false; break; }
        }
        printf("  All zeros input: -> %s\n", pass ? "PASS" : "FAIL");
        all_pass &= pass;

        cudaFree(d_input); cudaFree(d_kernel); cudaFree(d_output);
    }

    printf("\n=== %s ===\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
