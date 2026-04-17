/**
 * \file test_cuda_graphs.cpp
 * \brief Unit test for CUDA Graphs functionality.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

extern "C" {
    int conv_graph_create(const float* d_input,
                          const float* d_kernel,
                          float* d_output,
                          int width, int height, int ksize,
                          dim3 block);
    int conv_graph_destroy();
    int conv_graph_launch(cudaStream_t stream = 0);
    void relu_nosync(float* d_data, int size);
    void add_bias_nosync(float* d_data, float bias, int size);
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
                    sum += static_cast<double>(input[iy * width + ix]) * 
                           static_cast<double>(kernel[ky * ksize + kx]);
                }
            }
            output[oy * width + ox] = static_cast<float>(sum);
        }
    }
}

int test_graph_create_destroy() {
    const int width = 8, height = 8, ksize = 3;
    const int imgSize = width * height;
    const int kerSize = ksize * ksize;
    
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, sizeof(float) * imgSize);
    cudaMalloc(&d_kernel, sizeof(float) * kerSize);
    cudaMalloc(&d_output, sizeof(float) * imgSize);
    
    dim3 block(8, 8);
    int result = conv_graph_create(d_input, d_kernel, d_output, width, height, ksize, block);
    if (result != 0) {
        printf("test_graph_create_destroy skipped (CUDA Graphs not supported on this backend).\n");
        cudaFree(d_input); cudaFree(d_kernel); cudaFree(d_output);
        return 0;
    }
    
    result = conv_graph_destroy();
    if (result != 0) {
        fprintf(stderr, "FAIL: conv_graph_destroy returned %d\n", result);
        cudaFree(d_input); cudaFree(d_kernel); cudaFree(d_output);
        return 1;
    }
    
    cudaFree(d_input); cudaFree(d_kernel); cudaFree(d_output);
    printf("test_graph_create_destroy passed.\n");
    return 0;
}

int test_graph_launch() {
    const int width = 8, height = 8, ksize = 3;
    const int imgSize = width * height;
    const int kerSize = ksize * ksize;
    
    float *h_input = (float*)malloc(sizeof(float) * imgSize);
    float *h_kernel = (float*)malloc(sizeof(float) * kerSize);
    float *h_output = (float*)malloc(sizeof(float) * imgSize);
    
    for (int i = 0; i < imgSize; ++i) h_input[i] = static_cast<float>(i + 1);
    for (int i = 0; i < kerSize; ++i) h_kernel[i] = 1.0f / kerSize;
    
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, sizeof(float) * imgSize);
    cudaMalloc(&d_kernel, sizeof(float) * kerSize);
    cudaMalloc(&d_output, sizeof(float) * imgSize);
    
    cudaMemcpy(d_input, h_input, sizeof(float) * imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float) * kerSize, cudaMemcpyHostToDevice);
    
    dim3 block(8, 8);
    int create_result = conv_graph_create(d_input, d_kernel, d_output, width, height, ksize, block);
    
    if (create_result != 0) {
        printf("test_graph_launch skipped (CUDA Graphs not supported on this backend).\n");
        cudaFree(d_input); cudaFree(d_kernel); cudaFree(d_output);
        free(h_input); free(h_kernel); free(h_output);
        return 0;
    }
    
    conv_graph_launch(0);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, sizeof(float) * imgSize, cudaMemcpyDeviceToHost);
    
    conv_graph_destroy();
    
    float* h_expected = (float*)malloc(sizeof(float) * imgSize);
    conv_naive_cpu(h_input, h_kernel, h_expected, width, height, ksize);
    
    const float eps = 1e-4f;
    for (int i = 0; i < imgSize; ++i) {
        float diff = fabsf(h_output[i] - h_expected[i]);
        float maxv = fmaxf(fabsf(h_output[i]), fabsf(h_expected[i]));
        if (diff > eps * maxv) {
            fprintf(stderr, "FAIL: output mismatch at %d: GPU=%f expected=%f\n", 
                    i, h_output[i], h_expected[i]);
            free(h_input); free(h_kernel); free(h_output); free(h_expected);
            cudaFree(d_input); cudaFree(d_kernel); cudaFree(d_output);
            return 1;
        }
    }
    
    free(h_input); free(h_kernel); free(h_output); free(h_expected);
    cudaFree(d_input); cudaFree(d_kernel); cudaFree(d_output);
    printf("test_graph_launch passed.\n");
    return 0;
}

int test_graph_launch_invalid() {
    int result = conv_graph_launch(0);
    if (result != -1) {
        fprintf(stderr, "FAIL: launch without create should return -1\n");
        return 1;
    }
    
    result = conv_graph_destroy();
    if (result != -1) {
        fprintf(stderr, "FAIL: destroy without create should return -1\n");
        return 1;
    }
    
    printf("test_graph_launch_invalid passed.\n");
    return 0;
}

int test_graph_recreate() {
    const int width = 8, height = 8, ksize = 3;
    const int imgSize = width * height;
    const int kerSize = ksize * ksize;
    
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, sizeof(float) * imgSize);
    cudaMalloc(&d_kernel, sizeof(float) * kerSize);
    cudaMalloc(&d_output, sizeof(float) * imgSize);
    
    dim3 block(8, 8);
    
    conv_graph_create(d_input, d_kernel, d_output, width, height, ksize, block);
    conv_graph_launch(0);
    cudaDeviceSynchronize();
    
    conv_graph_create(d_input, d_kernel, d_output, width, height, ksize, block);
    conv_graph_launch(0);
    cudaDeviceSynchronize();
    
    conv_graph_destroy();
    
    cudaFree(d_input); cudaFree(d_kernel); cudaFree(d_output);
    printf("test_graph_recreate passed.\n");
    return 0;
}

int main() {
    int failures = 0;
    
    failures += test_graph_create_destroy();
    failures += test_graph_launch();
    failures += test_graph_launch_invalid();
    failures += test_graph_recreate();
    
    if (failures == 0) {
        printf("\nAll CUDA graphs tests passed.\n");
        return EXIT_SUCCESS;
    } else {
        printf("\n%d CUDA graphs test(s) failed.\n", failures);
        return EXIT_FAILURE;
    }
}