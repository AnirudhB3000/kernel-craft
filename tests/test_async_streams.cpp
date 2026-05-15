/**
 * \file test_async_streams.cpp
 * \brief Unit tests for async stream operations.
 *
 * Verifies that double-buffered async convolution produces the same numerical
 * results as the synchronous reference and that the context lifecycle
 * (init/destroy) works correctly.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

struct AsyncStreamCtx;

extern "C" {
    int  async_stream_init(AsyncStreamCtx* ctx, int width, int height, int ksize);
    void async_stream_destroy(AsyncStreamCtx* ctx);
    void async_conv_batch(AsyncStreamCtx* ctx,
                          float** h_inputs, float** h_outputs,
                          int num_batches, const float* h_kernel, dim3 block);
    void async_benchmark(int width, int height, int ksize, int num_batches,
                         float* sync_ms, float* async_ms);
}

/* AsyncStreamCtx must match the definition in async_streams.cu */
struct AsyncStreamCtx {
    cudaStream_t streams[2];
    float*       d_input[2];
    float*       d_output[2];
    float*       d_kernel;
    int          width;
    int          height;
    int          ksize;
    int          imgBytes;
    bool         initialized;
};

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

int test_async_ctx_lifecycle() {
    AsyncStreamCtx ctx = {};
    int ret = async_stream_init(&ctx, 16, 16, 3);
    if (ret != 0) {
        fprintf(stderr, "FAIL: async_stream_init returned %d\n", ret);
        return 1;
    }
    if (!ctx.initialized) {
        fprintf(stderr, "FAIL: ctx.initialized is false after init\n");
        return 1;
    }
    async_stream_destroy(&ctx);
    if (ctx.initialized) {
        fprintf(stderr, "FAIL: ctx.initialized should be false after destroy\n");
        return 1;
    }
    printf("test_async_ctx_lifecycle passed.\n");
    return 0;
}

int test_async_single_batch_correctness() {
    const int width = 16, height = 16, ksize = 3;
    const int imgSize = width * height;
    const int kerSize = ksize * ksize;

    float* h_input;
    float* h_output;
    cudaMallocHost(&h_input,  imgSize * sizeof(float));
    cudaMallocHost(&h_output, imgSize * sizeof(float));
    float* h_kernel = (float*)malloc(kerSize * sizeof(float));

    for (int i = 0; i < imgSize; ++i) h_input[i]  = static_cast<float>(i + 1);
    for (int i = 0; i < kerSize; ++i) h_kernel[i] = 1.0f / kerSize;
    memset(h_output, 0, imgSize * sizeof(float));

    AsyncStreamCtx ctx = {};
    async_stream_init(&ctx, width, height, ksize);

    float* inputs[1]  = { h_input };
    float* outputs[1] = { h_output };
    async_conv_batch(&ctx, inputs, outputs, 1, h_kernel, dim3(8, 8));

    async_stream_destroy(&ctx);

    float* h_expected = (float*)malloc(imgSize * sizeof(float));
    conv_naive_cpu(h_input, h_kernel, h_expected, width, height, ksize);

    const float eps = 1e-3f;
    for (int i = 0; i < imgSize; ++i) {
        float ref  = h_expected[i];
        float diff = fabsf(h_output[i] - ref);
        float maxv = fmaxf(fabsf(ref), 1.0f);
        if (diff > eps * maxv) {
            fprintf(stderr, "FAIL: mismatch at %d: GPU=%f expected=%f\n",
                    i, h_output[i], ref);
            cudaFreeHost(h_input); cudaFreeHost(h_output);
            free(h_kernel); free(h_expected);
            return 1;
        }
    }

    cudaFreeHost(h_input); cudaFreeHost(h_output);
    free(h_kernel); free(h_expected);
    printf("test_async_single_batch_correctness passed.\n");
    return 0;
}

int test_async_multi_batch_correctness() {
    const int width = 16, height = 16, ksize = 3;
    const int imgSize = width * height;
    const int kerSize = ksize * ksize;
    const int num_batches = 4;

    float* h_inputs[4];
    float* h_outputs[4];
    for (int b = 0; b < num_batches; ++b) {
        cudaMallocHost(&h_inputs[b],  imgSize * sizeof(float));
        cudaMallocHost(&h_outputs[b], imgSize * sizeof(float));
        for (int i = 0; i < imgSize; ++i)
            h_inputs[b][i] = static_cast<float>((b + 1) * (i + 1));
        memset(h_outputs[b], 0, imgSize * sizeof(float));
    }
    float* h_kernel = (float*)malloc(kerSize * sizeof(float));
    for (int i = 0; i < kerSize; ++i) h_kernel[i] = 1.0f / kerSize;

    AsyncStreamCtx ctx = {};
    async_stream_init(&ctx, width, height, ksize);
    async_conv_batch(&ctx, h_inputs, h_outputs, num_batches, h_kernel, dim3(8, 8));
    async_stream_destroy(&ctx);

    const float eps = 1e-3f;
    float* h_expected = (float*)malloc(imgSize * sizeof(float));
    for (int b = 0; b < num_batches; ++b) {
        conv_naive_cpu(h_inputs[b], h_kernel, h_expected, width, height, ksize);
        for (int i = 0; i < imgSize; ++i) {
            float diff = fabsf(h_outputs[b][i] - h_expected[i]);
            float maxv = fmaxf(fabsf(h_expected[i]), 1.0f);
            if (diff > eps * maxv) {
                fprintf(stderr, "FAIL: batch %d, pixel %d: GPU=%f expected=%f\n",
                        b, i, h_outputs[b][i], h_expected[i]);
                for (int bb = 0; bb < num_batches; ++bb) {
                    cudaFreeHost(h_inputs[bb]);
                    cudaFreeHost(h_outputs[bb]);
                }
                free(h_kernel); free(h_expected);
                return 1;
            }
        }
    }

    for (int b = 0; b < num_batches; ++b) {
        cudaFreeHost(h_inputs[b]);
        cudaFreeHost(h_outputs[b]);
    }
    free(h_kernel); free(h_expected);
    printf("test_async_multi_batch_correctness passed.\n");
    return 0;
}

int test_async_benchmark_runs() {
    float sync_ms = 0.0f, async_ms = 0.0f;
    async_benchmark(64, 64, 3, 4, &sync_ms, &async_ms);
    if (sync_ms <= 0.0f || async_ms <= 0.0f) {
        fprintf(stderr, "FAIL: benchmark returned non-positive times\n");
        return 1;
    }
    printf("test_async_benchmark_runs passed (sync=%.3f ms, async=%.3f ms).\n",
           sync_ms, async_ms);
    return 0;
}

int main() {
    int failures = 0;
    failures += test_async_ctx_lifecycle();
    failures += test_async_single_batch_correctness();
    failures += test_async_multi_batch_correctness();
    failures += test_async_benchmark_runs();

    if (failures == 0) {
        printf("\nAll async stream tests passed.\n");
        return EXIT_SUCCESS;
    }
    printf("\n%d async stream test(s) failed.\n", failures);
    return EXIT_FAILURE;
}
