/**
 * \file test_pipeline.cpp
 * \brief Unit tests for pipeline kernels (separate and fused).
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

extern "C" void launch_conv_layer(const float* d_input, const float* d_kernel,
                                  float* d_output, int width, int height, int ksize,
                                  dim3 block = dim3(8, 8, 1));
extern "C" void launch_batchnorm_layer(float* d_input, float gamma, float beta,
                                        float* d_output, int size,
                                        dim3 block = dim3(256, 1, 1));
extern "C" void launch_relu_layer(const float* d_input, float* d_output, int size,
                                  dim3 block = dim3(256, 1, 1));

extern "C" void launch_conv_bn_relu_fused(const float* d_input, const float* d_kernel,
                                          float gamma, float beta,
                                          float* d_output, int width, int height, int ksize,
                                          dim3 block = dim3(8, 8, 1));

void conv_cpu(const float* input, const float* kernel, float* output,
              int width, int height, int ksize) {
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

void batchnorm_cpu(float* input, float gamma, float beta, float* output, int n) {
    for (int i = 0; i < n; ++i) output[i] = input[i] * gamma + beta;
}

void relu_cpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; ++i) output[i] = input[i] > 0.0f ? input[i] : 0.0f;
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

int test_conv_layer() {
    const int w = 5, h = 5, ksize = 3;
    float h_in[25], h_ker[9], h_gpu[25], h_cpu[25];
    for (int i = 0; i < 25; ++i) h_in[i] = static_cast<float>(i);
    for (int i = 0; i < 9; ++i) h_ker[i] = 1.0f / 9.0f;
    
    float *d_in, *d_ker, *d_out;
    cudaMalloc(&d_in, 25 * sizeof(float));
    cudaMalloc(&d_ker, 9 * sizeof(float));
    cudaMalloc(&d_out, 25 * sizeof(float));
    cudaMemcpy(d_in, h_in, 25 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ker, h_ker, 9 * sizeof(float), cudaMemcpyHostToDevice);
    
    launch_conv_layer(d_in, d_ker, d_out, w, h, ksize);
    cudaMemcpy(h_gpu, d_out, 25 * sizeof(float), cudaMemcpyDeviceToHost);
    
    conv_cpu(h_in, h_ker, h_cpu, w, h, ksize);
    bool ok = verify(h_gpu, h_cpu, 25);
    
    cudaFree(d_in); cudaFree(d_ker); cudaFree(d_out);
    printf("Conv layer test: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}

int test_batchnorm_layer() {
    const int n = 20;
    float h_in[20], h_gamma = 2.0f, h_beta = 0.5f, h_gpu[20], h_cpu[20];
    for (int i = 0; i < n; ++i) h_in[i] = static_cast<float>(i);
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    
    launch_batchnorm_layer(d_in, h_gamma, h_beta, d_out, n);
    cudaMemcpy(h_gpu, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    batchnorm_cpu(h_in, h_gamma, h_beta, h_cpu, n);
    bool ok = verify(h_gpu, h_cpu, n);
    
    cudaFree(d_in); cudaFree(d_out);
    printf("Batchnorm layer test: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}

int test_relu_layer() {
    const int n = 20;
    float h_in[20] = {-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,-1,-2,-3,-4};
    float h_gpu[20], h_cpu[20];
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    
    launch_relu_layer(d_in, d_out, n);
    cudaMemcpy(h_gpu, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    relu_cpu(h_in, h_cpu, n);
    bool ok = verify(h_gpu, h_cpu, n);
    
    cudaFree(d_in); cudaFree(d_out);
    printf("ReLU layer test: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}

int test_separate_pipeline() {
    const int w = 5, h = 5, ksize = 3, n = w * h;
    float h_in[25], h_ker[9], h_gamma = 1.0f, h_beta = 0.0f;
    float h_gpu[25], h_cpu_conv[25], h_cpu_bn[25], h_cpu_relu[25];
    
    for (int i = 0; i < 25; ++i) h_in[i] = static_cast<float>(i);
    for (int i = 0; i < 9; ++i) h_ker[i] = 1.0f / 9.0f;
    
    float *d_in, *d_ker, *d_temp, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_ker, 9 * sizeof(float));
    cudaMalloc(&d_temp, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ker, h_ker, 9 * sizeof(float), cudaMemcpyHostToDevice);
    
    launch_conv_layer(d_in, d_ker, d_temp, w, h, ksize);
    launch_batchnorm_layer(d_temp, h_gamma, h_beta, d_temp, n);
    launch_relu_layer(d_temp, d_out, n);
    
    cudaMemcpy(h_gpu, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    conv_cpu(h_in, h_ker, h_cpu_conv, w, h, ksize);
    batchnorm_cpu(h_cpu_conv, h_gamma, h_beta, h_cpu_bn, n);
    relu_cpu(h_cpu_bn, h_cpu_relu, n);
    
    bool ok = verify(h_gpu, h_cpu_relu, n);
    
    cudaFree(d_in); cudaFree(d_ker); cudaFree(d_temp); cudaFree(d_out);
    printf("Separate pipeline test: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}

int test_fused_pipeline() {
    const int w = 5, h = 5, ksize = 3, n = w * h;
    float h_in[25], h_ker[9], h_gamma = 1.0f, h_beta = 0.0f;
    float h_gpu[25], h_cpu_conv[25], h_cpu_bn[25], h_cpu_relu[25];
    
    for (int i = 0; i < 25; ++i) h_in[i] = static_cast<float>(i);
    for (int i = 0; i < 9; ++i) h_ker[i] = 1.0f / 9.0f;
    
    float *d_in, *d_ker, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_ker, 9 * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ker, h_ker, 9 * sizeof(float), cudaMemcpyHostToDevice);
    
    launch_conv_bn_relu_fused(d_in, d_ker, h_gamma, h_beta, d_out, w, h, ksize);
    
    cudaMemcpy(h_gpu, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    conv_cpu(h_in, h_ker, h_cpu_conv, w, h, ksize);
    batchnorm_cpu(h_cpu_conv, h_gamma, h_beta, h_cpu_bn, n);
    relu_cpu(h_cpu_bn, h_cpu_relu, n);
    
    bool ok = verify(h_gpu, h_cpu_relu, n);
    
    cudaFree(d_in); cudaFree(d_ker); cudaFree(d_out);
    printf("Fused pipeline test: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}

int main() {
    int failures = 0;
    failures += test_conv_layer();
    failures += test_batchnorm_layer();
    failures += test_relu_layer();
    failures += test_separate_pipeline();
    failures += test_fused_pipeline();
    printf("\nPipeline tests: %d failures\n", failures);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}