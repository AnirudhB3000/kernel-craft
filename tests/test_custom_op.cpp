/**
 * \file test_custom_op.cpp
 * \brief Unit tests for custom operations (sparse convolution).
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

extern "C" void launch_conv_sparse(const float* d_input,
                                    const float* d_kernel_nonzero,
                                    const int* d_kernel_rows,
                                    const int* d_kernel_cols,
                                    int nnz,
                                    float* d_output,
                                    int width, int height, int ksize,
                                    dim3 block = dim3(16, 16, 1));

extern "C" void launch_conv_dense_fallback(const float* d_input,
                                           const float* d_kernel,
                                           float* d_output,
                                           int width, int height, int ksize,
                                           dim3 block = dim3(16, 16, 1));

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

bool verify(const float* gpu, const float* cpu, int n, float eps = 1e-4f) {
    for (int i = 0; i < n; ++i) {
        if (std::fabs(gpu[i] - cpu[i]) > eps) {
            fprintf(stderr, "Mismatch at %d: GPU=%f CPU=%f\n", i, gpu[i], cpu[i]);
            return false;
        }
    }
    return true;
}

int test_sparse_conv() {
    const int w = 5, h = 5, ksize = 3;
    float h_in[25], h_ker[9], h_gpu[25], h_cpu[25];
    for (int i = 0; i < 25; ++i) h_in[i] = static_cast<float>(i + 1);
    for (int i = 0; i < 9; ++i) h_ker[i] = 1.0f / 9.0f;

    float *d_in, *d_ker_nz, *d_out;
    int *d_rows, *d_cols;
    cudaMalloc(&d_in, 25 * sizeof(float));
    cudaMalloc(&d_ker_nz, 9 * sizeof(float));
    cudaMalloc(&d_rows, 9 * sizeof(int));
    cudaMalloc(&d_cols, 9 * sizeof(int));
    cudaMalloc(&d_out, 25 * sizeof(float));

    cudaMemcpy(d_in, h_in, 25 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ker_nz, h_ker, 9 * sizeof(float), cudaMemcpyHostToDevice);
    
    int h_rows[9] = {0,0,0, 1,1,1, 2,2,2};
    int h_cols[9] = {0,1,2, 0,1,2, 0,1,2};
    cudaMemcpy(d_rows, h_rows, 9 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, h_cols, 9 * sizeof(int), cudaMemcpyHostToDevice);

    launch_conv_sparse(d_in, d_ker_nz, d_rows, d_cols, 9, d_out, w, h, ksize);
    cudaMemcpy(h_gpu, d_out, 25 * sizeof(float), cudaMemcpyDeviceToHost);

    conv_cpu(h_in, h_ker, h_cpu, w, h, ksize);
    bool ok = verify(h_gpu, h_cpu, 25);

    cudaFree(d_in); cudaFree(d_ker_nz); cudaFree(d_rows); cudaFree(d_cols); cudaFree(d_out);
    printf("Sparse conv test: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}

int test_dense_fallback() {
    const int w = 5, h = 5, ksize = 3;
    float h_in[25], h_ker[9], h_gpu[25], h_cpu[25];
    for (int i = 0; i < 25; ++i) h_in[i] = static_cast<float>(i + 1);
    for (int i = 0; i < 9; ++i) h_ker[i] = 1.0f / 9.0f;

    float *d_in, *d_ker, *d_out;
    cudaMalloc(&d_in, 25 * sizeof(float));
    cudaMalloc(&d_ker, 9 * sizeof(float));
    cudaMalloc(&d_out, 25 * sizeof(float));

    cudaMemcpy(d_in, h_in, 25 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ker, h_ker, 9 * sizeof(float), cudaMemcpyHostToDevice);

    launch_conv_dense_fallback(d_in, d_ker, d_out, w, h, ksize);
    cudaMemcpy(h_gpu, d_out, 25 * sizeof(float), cudaMemcpyDeviceToHost);

    conv_cpu(h_in, h_ker, h_cpu, w, h, ksize);
    bool ok = verify(h_gpu, h_cpu, 25);

    cudaFree(d_in); cudaFree(d_ker); cudaFree(d_out);
    printf("Dense fallback test: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}

int test_sparse_correctness() {
    const int w = 8, h = 8, ksize = 3;
    float h_in[64], h_gpu[64], h_cpu[64];

    for (int i = 0; i < 64; ++i) h_in[i] = static_cast<float>(i);

    // Sparse kernel: center (1,1) only
    float h_nz[1] = {1.0f};
    int h_rows[1] = {1};
    int h_cols[1] = {1};

    float *d_in, *d_ker_nz, *d_out;
    int *d_rows, *d_cols;
    cudaMalloc(&d_in, 64 * sizeof(float));
    cudaMalloc(&d_ker_nz, 1 * sizeof(float));
    cudaMalloc(&d_rows, 1 * sizeof(int));
    cudaMalloc(&d_cols, 1 * sizeof(int));
    cudaMalloc(&d_out, 64 * sizeof(float));

    cudaMemcpy(d_in, h_in, 64 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ker_nz, h_nz, 1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rows, h_rows, 1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, h_cols, 1 * sizeof(int), cudaMemcpyHostToDevice);

    launch_conv_sparse(d_in, d_ker_nz, d_rows, d_cols, 1, d_out, w, h, ksize);
    cudaMemcpy(h_gpu, d_out, 64 * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU reference: single center element
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int iy = y, ix = x;
            h_cpu[y * w + x] = (iy >= 0 && iy < h && ix >= 0 && ix < w) ? h_in[iy * w + ix] : 0.0f;
        }
    }

    bool ok = verify(h_gpu, h_cpu, 64);

    cudaFree(d_in); cudaFree(d_ker_nz); cudaFree(d_rows); cudaFree(d_cols); cudaFree(d_out);
    printf("Sparse correctness test: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}

int main() {
    int failures = 0;
    failures += test_sparse_conv();
    failures += test_dense_fallback();
    failures += test_sparse_correctness();
    printf("\nCustom op tests: %d failures\n", failures);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}