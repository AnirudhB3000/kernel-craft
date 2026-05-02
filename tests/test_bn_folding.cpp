/**
 * \file test_bn_folding.cpp
 * \brief Unit tests for batch normalization folding.
 *
 * Verifies correctness of BN folding by comparing GPU output
 * against CPU reference implementation.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// Declare device functions
extern "C" void launch_bn_folding(const float* d_conv_weights,
                                   const float* d_conv_bias,
                                   float* d_folded_weights,
                                   float* d_folded_bias,
                                   const float* d_bn_mean,
                                   const float* d_bn_variance,
                                   const float* d_bn_gamma,
                                   const float* d_bn_beta,
                                   float epsilon,
                                   int C_out, int C_in, int K_h, int K_w);

extern "C" void bn_folding_cpu_ref(const float* conv_weights,
                                    const float* conv_bias,
                                    float* folded_weights,
                                    float* folded_bias,
                                    const float* bn_mean,
                                    const float* bn_variance,
                                    const float* bn_gamma,
                                    const float* bn_beta,
                                    float epsilon,
                                    int C_out, int C_in, int K_h, int K_w);

float max_error(const float* a, const float* b, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_err = fmaxf(max_err, fabsf(a[i] - b[i]));
    }
    return max_err;
}

bool test_bn_folding(int C_out, int C_in, int K_h, int K_w) {
    printf("Testing BN Folding (C_out=%d, C_in=%d, K=%dx%d)...\n", 
           C_out, C_in, K_h, K_w);
    
    int weight_size = C_out * C_in * K_h * K_w;
    int bias_size = C_out;
    
    std::vector<float> h_conv_weights(weight_size);
    std::vector<float> h_conv_bias(bias_size);
    std::vector<float> h_bn_mean(bias_size);
    std::vector<float> h_bn_variance(bias_size);
    std::vector<float> h_bn_gamma(bias_size);
    std::vector<float> h_bn_beta(bias_size);
    
    // Initialize with random values
    for (int i = 0; i < weight_size; ++i) {
        h_conv_weights[i] = (rand() % 1000) / 1000.0f - 0.5f;
    }
    for (int i = 0; i < bias_size; ++i) {
        h_conv_bias[i] = (rand() % 100) / 1000.0f;
        h_bn_mean[i] = (rand() % 100) / 1000.0f;
        h_bn_variance[i] = (rand() % 100) / 1000.0f + 0.01f;  // Ensure positive
        h_bn_gamma[i] = (rand() % 100) / 100.0f + 0.5f;
        h_bn_beta[i] = (rand() % 100) / 1000.0f - 0.05f;
    }
    
    float epsilon = 1e-5f;
    
    // Allocate device memory
    float *d_conv_weights, *d_conv_bias, *d_folded_weights, *d_folded_bias;
    float *d_bn_mean, *d_bn_variance, *d_bn_gamma, *d_bn_beta;
    
    cudaMalloc(&d_conv_weights, weight_size * sizeof(float));
    cudaMalloc(&d_conv_bias, bias_size * sizeof(float));
    cudaMalloc(&d_folded_weights, weight_size * sizeof(float));
    cudaMalloc(&d_folded_bias, bias_size * sizeof(float));
    cudaMalloc(&d_bn_mean, bias_size * sizeof(float));
    cudaMalloc(&d_bn_variance, bias_size * sizeof(float));
    cudaMalloc(&d_bn_gamma, bias_size * sizeof(float));
    cudaMalloc(&d_bn_beta, bias_size * sizeof(float));
    
    cudaMemcpy(d_conv_weights, h_conv_weights.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv_bias, h_conv_bias.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn_mean, h_bn_mean.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn_variance, h_bn_variance.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn_gamma, h_bn_gamma.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn_beta, h_bn_beta.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Run GPU kernel
    launch_bn_folding(d_conv_weights, d_conv_bias, d_folded_weights, d_folded_bias,
                      d_bn_mean, d_bn_variance, d_bn_gamma, d_bn_beta,
                      epsilon, C_out, C_in, K_h, K_w);
    
    std::vector<float> h_folded_weights_gpu(weight_size);
    std::vector<float> h_folded_bias_gpu(bias_size);
    cudaMemcpy(h_folded_weights_gpu.data(), d_folded_weights, weight_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_folded_bias_gpu.data(), d_folded_bias, bias_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compute CPU reference
    std::vector<float> h_folded_weights_cpu(weight_size);
    std::vector<float> h_folded_bias_cpu(bias_size);
    bn_folding_cpu_ref(h_conv_weights.data(), h_conv_bias.data(),
                       h_folded_weights_cpu.data(), h_folded_bias_cpu.data(),
                       h_bn_mean.data(), h_bn_variance.data(),
                       h_bn_gamma.data(), h_bn_beta.data(),
                       epsilon, C_out, C_in, K_h, K_w);
    
    // Check accuracy
    float weight_err = max_error(h_folded_weights_gpu.data(), h_folded_weights_cpu.data(), weight_size);
    float bias_err = max_error(h_folded_bias_gpu.data(), h_folded_bias_cpu.data(), bias_size);
    float tolerance = 1e-4f;
    
    printf("  Max weight error: %f, bias error: %f (tolerance: %f)\n", weight_err, bias_err, tolerance);
    printf("  -> %s\n", (weight_err < tolerance && bias_err < tolerance) ? "PASS" : "FAIL");
    
    cudaFree(d_conv_weights);
    cudaFree(d_conv_bias);
    cudaFree(d_folded_weights);
    cudaFree(d_folded_bias);
    cudaFree(d_bn_mean);
    cudaFree(d_bn_variance);
    cudaFree(d_bn_gamma);
    cudaFree(d_bn_beta);
    
    return (weight_err < tolerance && bias_err < tolerance);
}

int main() {
    printf("=== Batch Normalization Folding Tests ===\n\n");

    bool all_pass = true;

    // Basic tests with different configurations
    all_pass &= test_bn_folding(8, 3, 3, 3);
    all_pass &= test_bn_folding(16, 8, 3, 3);
    all_pass &= test_bn_folding(32, 16, 5, 5);

    // Test with no bias (nullptr case)
    printf("\nTesting BN Folding without bias (C_out=16, C_in=8, K=3x3)...\n");
    {
        int C_out = 16, C_in = 8, K_h = 3, K_w = 3;
        int weight_size = C_out * C_in * K_h * K_w;

        std::vector<float> h_conv_weights(weight_size);
        std::vector<float> h_bn_mean(C_out), h_bn_variance(C_out);
        std::vector<float> h_bn_gamma(C_out), h_bn_beta(C_out);

        for (int i = 0; i < weight_size; ++i) h_conv_weights[i] = (rand() % 1000) / 1000.0f - 0.5f;
        for (int i = 0; i < C_out; ++i) {
            h_bn_mean[i] = (rand() % 100) / 1000.0f;
            h_bn_variance[i] = (rand() % 100) / 1000.0f + 0.01f;
            h_bn_gamma[i] = (rand() % 100) / 100.0f + 0.5f;
            h_bn_beta[i] = (rand() % 100) / 1000.0f - 0.05f;
        }

        float *d_conv_weights, *d_folded_weights, *d_folded_bias;
        float *d_bn_mean, *d_bn_variance, *d_bn_gamma, *d_bn_beta;

        cudaMalloc(&d_conv_weights, weight_size * sizeof(float));
        cudaMalloc(&d_folded_weights, weight_size * sizeof(float));
        cudaMalloc(&d_folded_bias, C_out * sizeof(float));
        cudaMalloc(&d_bn_mean, C_out * sizeof(float));
        cudaMalloc(&d_bn_variance, C_out * sizeof(float));
        cudaMalloc(&d_bn_gamma, C_out * sizeof(float));
        cudaMalloc(&d_bn_beta, C_out * sizeof(float));

        cudaMemcpy(d_conv_weights, h_conv_weights.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bn_mean, h_bn_mean.data(), C_out * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bn_variance, h_bn_variance.data(), C_out * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bn_gamma, h_bn_gamma.data(), C_out * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bn_beta, h_bn_beta.data(), C_out * sizeof(float), cudaMemcpyHostToDevice);

        // Pass nullptr for bias
        launch_bn_folding(d_conv_weights, nullptr, d_folded_weights, d_folded_bias,
                          d_bn_mean, d_bn_variance, d_bn_gamma, d_bn_beta,
                          1e-5f, C_out, C_in, K_h, K_w);

        std::vector<float> h_folded_weights(weight_size), h_folded_bias(C_out);
        cudaMemcpy(h_folded_weights.data(), d_folded_weights, weight_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_folded_bias.data(), d_folded_bias, C_out * sizeof(float), cudaMemcpyDeviceToHost);

        // CPU reference (also without bias)
        std::vector<float> h_folded_weights_cpu(weight_size), h_folded_bias_cpu(C_out);
        bn_folding_cpu_ref(h_conv_weights.data(), nullptr,
                           h_folded_weights_cpu.data(), h_folded_bias_cpu.data(),
                           h_bn_mean.data(), h_bn_variance.data(),
                           h_bn_gamma.data(), h_bn_beta.data(),
                           1e-5f, C_out, C_in, K_h, K_w);

        float w_err = max_error(h_folded_weights.data(), h_folded_weights_cpu.data(), weight_size);
        float b_err = max_error(h_folded_bias.data(), h_folded_bias_cpu.data(), C_out);
        printf("  No bias: weight_err=%f, bias_err=%f -> %s\n", w_err, b_err, (w_err < 1e-4f && b_err < 1e-4f) ? "PASS" : "FAIL");
        all_pass &= (w_err < 1e-4f && b_err < 1e-4f);

        cudaFree(d_conv_weights); cudaFree(d_folded_weights); cudaFree(d_folded_bias);
        cudaFree(d_bn_mean); cudaFree(d_bn_variance); cudaFree(d_bn_gamma); cudaFree(d_bn_beta);
    }

    // Test with larger kernel (7x7)
    all_pass &= test_bn_folding(64, 32, 7, 7);

    printf("\n=== %s ===\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
