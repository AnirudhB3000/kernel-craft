/**
 * \file test_unified_memory.cpp
 * \brief Unit tests for unified memory convolution variants.
 *
 * Verifies that naive and tiled convolutions using cudaMallocManaged produce
 * correct results matching the CPU reference, and that alloc/free lifecycle
 * functions behave correctly.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" {
    int  unified_alloc(float** ptr, int count);
    void unified_free(float* ptr);
    void unified_prefetch_to_gpu(float* ptr, int count, cudaStream_t stream);
    void unified_prefetch_to_cpu(float* ptr, int count, cudaStream_t stream);
    void unified_conv_naive(float* um_input, float* um_kernel, float* um_output,
                            int width, int height, int ksize, dim3 block);
    void unified_conv_tiled(float* um_input, float* um_kernel, float* um_output,
                            int width, int height, int ksize, dim3 block);
    void unified_benchmark(int width, int height, int ksize,
                           float* explicit_ms, float* um_fault_ms, float* um_prefetch_ms);
}

static void conv_naive_cpu(const float* input, const float* kernel, float* output,
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

int test_unified_alloc_free() {
    float* ptr = nullptr;
    int ret = unified_alloc(&ptr, 64);
    if (ret != 0 || ptr == nullptr) {
        fprintf(stderr, "FAIL: unified_alloc returned %d\n", ret);
        return 1;
    }
    unified_free(ptr);
    printf("test_unified_alloc_free passed.\n");
    return 0;
}

int test_unified_naive_correctness() {
    const int width = 16, height = 16, ksize = 3;
    const int imgSize = width * height;
    const int kerSize = ksize * ksize;

    float *um_input, *um_kernel, *um_output;
    unified_alloc(&um_input,  imgSize);
    unified_alloc(&um_kernel, kerSize);
    unified_alloc(&um_output, imgSize);

    for (int i = 0; i < imgSize; ++i) um_input[i]  = static_cast<float>(i + 1);
    for (int i = 0; i < kerSize; ++i) um_kernel[i] = 1.0f / kerSize;
    memset(um_output, 0, imgSize * sizeof(float));

    unified_conv_naive(um_input, um_kernel, um_output, width, height, ksize, dim3(8, 8));

    float* h_expected = (float*)malloc(imgSize * sizeof(float));
    conv_naive_cpu(um_input, um_kernel, h_expected, width, height, ksize);

    const float eps = 1e-3f;
    for (int i = 0; i < imgSize; ++i) {
        float diff = fabsf(um_output[i] - h_expected[i]);
        float maxv = fmaxf(fabsf(h_expected[i]), 1.0f);
        if (diff > eps * maxv) {
            fprintf(stderr, "FAIL: naive mismatch at %d: GPU=%f expected=%f\n",
                    i, um_output[i], h_expected[i]);
            unified_free(um_input); unified_free(um_kernel); unified_free(um_output);
            free(h_expected);
            return 1;
        }
    }

    unified_free(um_input); unified_free(um_kernel); unified_free(um_output);
    free(h_expected);
    printf("test_unified_naive_correctness passed.\n");
    return 0;
}

int test_unified_tiled_correctness() {
    const int width = 32, height = 32, ksize = 3;
    const int imgSize = width * height;
    const int kerSize = ksize * ksize;

    float *um_input, *um_kernel, *um_output;
    unified_alloc(&um_input,  imgSize);
    unified_alloc(&um_kernel, kerSize);
    unified_alloc(&um_output, imgSize);

    for (int i = 0; i < imgSize; ++i) um_input[i]  = static_cast<float>(i % 100 + 1);
    for (int i = 0; i < kerSize; ++i) um_kernel[i] = 1.0f / kerSize;
    memset(um_output, 0, imgSize * sizeof(float));

    unified_conv_tiled(um_input, um_kernel, um_output, width, height, ksize, dim3(8, 8));

    float* h_expected = (float*)malloc(imgSize * sizeof(float));
    conv_naive_cpu(um_input, um_kernel, h_expected, width, height, ksize);

    const float eps = 1e-3f;
    for (int i = 0; i < imgSize; ++i) {
        float diff = fabsf(um_output[i] - h_expected[i]);
        float maxv = fmaxf(fabsf(h_expected[i]), 1.0f);
        if (diff > eps * maxv) {
            fprintf(stderr, "FAIL: tiled mismatch at %d: GPU=%f expected=%f\n",
                    i, um_output[i], h_expected[i]);
            unified_free(um_input); unified_free(um_kernel); unified_free(um_output);
            free(h_expected);
            return 1;
        }
    }

    unified_free(um_input); unified_free(um_kernel); unified_free(um_output);
    free(h_expected);
    printf("test_unified_tiled_correctness passed.\n");
    return 0;
}

int test_unified_prefetch() {
    const int count = 256;
    float* ptr;
    unified_alloc(&ptr, count);
    for (int i = 0; i < count; ++i) ptr[i] = static_cast<float>(i);

    unified_prefetch_to_gpu(ptr, count, 0);
    cudaDeviceSynchronize();

    unified_prefetch_to_cpu(ptr, count, 0);
    cudaDeviceSynchronize();

    for (int i = 0; i < count; ++i) {
        if (ptr[i] != static_cast<float>(i)) {
            fprintf(stderr, "FAIL: data corrupted after prefetch at %d\n", i);
            unified_free(ptr);
            return 1;
        }
    }
    unified_free(ptr);
    printf("test_unified_prefetch passed.\n");
    return 0;
}

int test_unified_benchmark_runs() {
    float exp_ms = 0.0f, fault_ms = 0.0f, pf_ms = 0.0f;
    unified_benchmark(64, 64, 3, &exp_ms, &fault_ms, &pf_ms);
    if (exp_ms <= 0.0f || fault_ms <= 0.0f || pf_ms <= 0.0f) {
        fprintf(stderr, "FAIL: benchmark returned non-positive times\n");
        return 1;
    }
    printf("test_unified_benchmark_runs passed "
           "(explicit=%.3f ms, fault=%.3f ms, prefetch=%.3f ms).\n",
           exp_ms, fault_ms, pf_ms);
    return 0;
}

int main() {
    int failures = 0;
    failures += test_unified_alloc_free();
    failures += test_unified_naive_correctness();
    failures += test_unified_tiled_correctness();
    failures += test_unified_prefetch();
    failures += test_unified_benchmark_runs();

    if (failures == 0) {
        printf("\nAll unified memory tests passed.\n");
        return EXIT_SUCCESS;
    }
    printf("\n%d unified memory test(s) failed.\n", failures);
    return EXIT_FAILURE;
}
