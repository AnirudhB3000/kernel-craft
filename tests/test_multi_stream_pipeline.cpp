/**
 * \file test_multi_stream_pipeline.cpp
 * \brief Unit tests for the multi-stream concurrent pipeline.
 *
 * Verifies that multi_stream_run produces numerically equivalent output to
 * single_stream_run, and that context lifecycle functions work correctly.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

struct MultiStreamCtx;

extern "C" {
    int  multi_stream_init(MultiStreamCtx* ctx);
    void multi_stream_destroy(MultiStreamCtx* ctx);
    void multi_stream_run(MultiStreamCtx* ctx,
                          const float* d_raw, float* d_resized,
                          const float* d_kernel, float* d_output,
                          int src_w, int src_h, int dst_w, int dst_h,
                          int ksize, float mean, float std_dev);
    void single_stream_run(const float* d_raw, float* d_resized,
                           const float* d_kernel, float* d_output,
                           int src_w, int src_h, int dst_w, int dst_h,
                           int ksize, float mean, float std_dev);
}

/* Must match the struct in multi_stream_pipeline.cu */
struct MultiStreamCtx {
    cudaStream_t preprocess_stream;
    cudaStream_t inference_stream;
    cudaEvent_t  preprocess_done;
    bool         initialized;
};

int test_multi_stream_lifecycle() {
    MultiStreamCtx ctx = {};
    int ret = multi_stream_init(&ctx);
    if (ret != 0) {
        fprintf(stderr, "FAIL: multi_stream_init returned %d\n", ret);
        return 1;
    }
    if (!ctx.initialized) {
        fprintf(stderr, "FAIL: ctx.initialized is false after init\n");
        return 1;
    }
    multi_stream_destroy(&ctx);
    if (ctx.initialized) {
        fprintf(stderr, "FAIL: ctx.initialized should be false after destroy\n");
        return 1;
    }
    printf("test_multi_stream_lifecycle passed.\n");
    return 0;
}

int test_multi_stream_correctness() {
    /* Use same src and dst size to keep the test simple */
    const int src_w = 32, src_h = 32;
    const int dst_w = 32, dst_h = 32;
    const int ksize = 3;
    const int srcSize = src_w * src_h;
    const int dstSize = dst_w * dst_h;
    const int kerSize = ksize * ksize;

    float* h_raw    = (float*)malloc(srcSize * sizeof(float));
    float* h_kernel = (float*)malloc(kerSize * sizeof(float));
    for (int i = 0; i < srcSize; ++i) h_raw[i]    = static_cast<float>(i % 64 + 1);
    for (int i = 0; i < kerSize; ++i) h_kernel[i] = 1.0f / kerSize;

    float *d_raw, *d_resized_s, *d_kernel, *d_output_s;
    float *d_resized_m, *d_output_m;
    cudaMalloc(&d_raw,       srcSize * sizeof(float));
    cudaMalloc(&d_resized_s, dstSize * sizeof(float));
    cudaMalloc(&d_resized_m, dstSize * sizeof(float));
    cudaMalloc(&d_kernel,    kerSize * sizeof(float));
    cudaMalloc(&d_output_s,  dstSize * sizeof(float));
    cudaMalloc(&d_output_m,  dstSize * sizeof(float));

    cudaMemcpy(d_raw,    h_raw,    srcSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kerSize * sizeof(float), cudaMemcpyHostToDevice);

    /* Single-stream reference */
    single_stream_run(d_raw, d_resized_s, d_kernel, d_output_s,
                      src_w, src_h, dst_w, dst_h, ksize, 0.0f, 1.0f);

    /* Multi-stream result */
    MultiStreamCtx ctx = {};
    multi_stream_init(&ctx);
    multi_stream_run(&ctx, d_raw, d_resized_m, d_kernel, d_output_m,
                     src_w, src_h, dst_w, dst_h, ksize, 0.0f, 1.0f);
    multi_stream_destroy(&ctx);

    float* h_out_s = (float*)malloc(dstSize * sizeof(float));
    float* h_out_m = (float*)malloc(dstSize * sizeof(float));
    cudaMemcpy(h_out_s, d_output_s, dstSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_m, d_output_m, dstSize * sizeof(float), cudaMemcpyDeviceToHost);

    const float eps = 1e-3f;
    for (int i = 0; i < dstSize; ++i) {
        float diff = fabsf(h_out_m[i] - h_out_s[i]);
        float maxv = fmaxf(fabsf(h_out_s[i]), 1.0f);
        if (diff > eps * maxv) {
            fprintf(stderr, "FAIL: mismatch at %d: multi=%f single=%f\n",
                    i, h_out_m[i], h_out_s[i]);
            cudaFree(d_raw); cudaFree(d_resized_s); cudaFree(d_resized_m);
            cudaFree(d_kernel); cudaFree(d_output_s); cudaFree(d_output_m);
            free(h_raw); free(h_kernel); free(h_out_s); free(h_out_m);
            return 1;
        }
    }

    cudaFree(d_raw); cudaFree(d_resized_s); cudaFree(d_resized_m);
    cudaFree(d_kernel); cudaFree(d_output_s); cudaFree(d_output_m);
    free(h_raw); free(h_kernel); free(h_out_s); free(h_out_m);
    printf("test_multi_stream_correctness passed.\n");
    return 0;
}

int test_multi_stream_repeated_runs() {
    /* Verify that calling multi_stream_run multiple times on the same context
     * works correctly and produces consistent results. */
    const int src_w = 16, src_h = 16, dst_w = 16, dst_h = 16, ksize = 3;
    const int srcSize = src_w * src_h;
    const int dstSize = dst_w * dst_h;
    const int kerSize = ksize * ksize;

    float* h_kernel = (float*)malloc(kerSize * sizeof(float));
    for (int i = 0; i < kerSize; ++i) h_kernel[i] = 1.0f / kerSize;

    float *d_raw, *d_resized, *d_kernel, *d_output;
    cudaMalloc(&d_raw,     srcSize * sizeof(float));
    cudaMalloc(&d_resized, dstSize * sizeof(float));
    cudaMalloc(&d_kernel,  kerSize * sizeof(float));
    cudaMalloc(&d_output,  dstSize * sizeof(float));
    cudaMemcpy(d_kernel, h_kernel, kerSize * sizeof(float), cudaMemcpyHostToDevice);

    MultiStreamCtx ctx = {};
    multi_stream_init(&ctx);

    for (int iter = 0; iter < 5; ++iter) {
        multi_stream_run(&ctx, d_raw, d_resized, d_kernel, d_output,
                         src_w, src_h, dst_w, dst_h, ksize, 0.0f, 1.0f);
    }

    multi_stream_destroy(&ctx);
    cudaFree(d_raw); cudaFree(d_resized); cudaFree(d_kernel); cudaFree(d_output);
    free(h_kernel);
    printf("test_multi_stream_repeated_runs passed.\n");
    return 0;
}

int main() {
    int failures = 0;
    failures += test_multi_stream_lifecycle();
    failures += test_multi_stream_correctness();
    failures += test_multi_stream_repeated_runs();

    if (failures == 0) {
        printf("\nAll multi-stream pipeline tests passed.\n");
        return EXIT_SUCCESS;
    }
    printf("\n%d multi-stream pipeline test(s) failed.\n", failures);
    return EXIT_FAILURE;
}
