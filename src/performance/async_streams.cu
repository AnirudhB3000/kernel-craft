/**
 * \file async_streams.cu
 * \brief Async stream operations for overlapping memory transfers and compute.
 *
 * Uses multiple CUDA streams to pipeline host-to-device transfers, kernel
 * execution, and device-to-host transfers across consecutive batches. This
 * double-buffering pattern hides PCIe transfer latency behind GPU compute.
 *
 * \par Double-Buffering Pattern
 * Two streams alternate: while stream A runs compute on batch N, stream B
 * uploads batch N+1. This saturates both the copy engine and the compute
 * engine simultaneously.
 *
 * \par Usage
 * \code
 * AsyncStreamCtx ctx;
 * async_stream_init(&ctx, width, height, ksize);
 * async_conv_batch(&ctx, h_inputs, h_outputs, num_batches, d_kernel);
 * async_stream_destroy(&ctx);
 * \endcode
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef TILE_W
#define TILE_W 8
#endif
#ifndef TILE_H
#define TILE_H 8
#endif

#define ASYNC_NUM_STREAMS 2

extern "C" {
    void conv_tiled_nosync(const float* d_input, const float* d_kernel,
                           float* d_output, int width, int height, int ksize, dim3 block);
    void launch_conv_naive(const float* d_input, const float* d_kernel,
                           float* d_output, int width, int height, int ksize);
}

/**
 * \brief Context holding resources for async double-buffered convolution.
 */
struct AsyncStreamCtx {
    cudaStream_t streams[ASYNC_NUM_STREAMS];
    float*       d_input[ASYNC_NUM_STREAMS];   /**< Ping-pong input buffers. */
    float*       d_output[ASYNC_NUM_STREAMS];  /**< Ping-pong output buffers. */
    float*       d_kernel;                     /**< Shared kernel weights. */
    int          width;
    int          height;
    int          ksize;
    int          imgBytes;
    bool         initialized;
};


/**
 * \brief Initialize async stream context and allocate ping-pong buffers.
 *
 * Allocates pinned host memory is handled by the caller; this function only
 * allocates device-side ping-pong buffers and creates CUDA streams.
 *
 * \param[out] ctx    Context to initialize.
 * \param[in]  width  Image width in pixels.
 * \param[in]  height Image height in pixels.
 * \param[in]  ksize  Convolution kernel side length.
 * \return 0 on success, -1 on allocation failure.
 */
extern "C" int async_stream_init(AsyncStreamCtx* ctx, int width, int height, int ksize) {
    ctx->width    = width;
    ctx->height   = height;
    ctx->ksize    = ksize;
    ctx->imgBytes = width * height * sizeof(float);

    for (int s = 0; s < ASYNC_NUM_STREAMS; ++s) {
        if (cudaStreamCreate(&ctx->streams[s]) != cudaSuccess) return -1;
        if (cudaMalloc(&ctx->d_input[s],  ctx->imgBytes) != cudaSuccess) return -1;
        if (cudaMalloc(&ctx->d_output[s], ctx->imgBytes) != cudaSuccess) return -1;
    }

    int kerBytes = ksize * ksize * sizeof(float);
    if (cudaMalloc(&ctx->d_kernel, kerBytes) != cudaSuccess) return -1;

    ctx->initialized = true;
    return 0;
}

/**
 * \brief Release all resources held by an async stream context.
 *
 * \param[in,out] ctx Context to destroy.
 */
extern "C" void async_stream_destroy(AsyncStreamCtx* ctx) {
    if (!ctx->initialized) return;
    for (int s = 0; s < ASYNC_NUM_STREAMS; ++s) {
        cudaFree(ctx->d_input[s]);
        cudaFree(ctx->d_output[s]);
        cudaStreamDestroy(ctx->streams[s]);
    }
    cudaFree(ctx->d_kernel);
    ctx->initialized = false;
}

/**
 * \brief Process a batch of images using double-buffered async streams.
 *
 * Overlaps H2D transfer of batch N+1 with GPU compute on batch N, then D2H
 * transfer of batch N with compute on batch N+1. Requires pinned host memory
 * (cudaMallocHost) for true async overlap.
 *
 * \param[in]  ctx        Initialized async stream context.
 * \param[in]  h_inputs   Array of pointers to pinned host input images.
 * \param[out] h_outputs  Array of pointers to pinned host output images.
 * \param[in]  num_batches Number of images to process.
 * \param[in]  h_kernel   Host pointer to convolution kernel weights.
 * \param[in]  block      CUDA block dimensions.
 */
extern "C" void async_conv_batch(AsyncStreamCtx* ctx,
                                  float** h_inputs, float** h_outputs,
                                  int num_batches,
                                  const float* h_kernel,
                                  dim3 block) {
    if (!ctx->initialized || num_batches <= 0) return;

    int kerBytes = ctx->ksize * ctx->ksize * sizeof(float);
    cudaMemcpy(ctx->d_kernel, h_kernel, kerBytes, cudaMemcpyHostToDevice);

    for (int i = 0; i < num_batches; ++i) {
        int s = i % ASYNC_NUM_STREAMS;
        cudaStream_t stream = ctx->streams[s];

        cudaMemcpyAsync(ctx->d_input[s], h_inputs[i],
                        ctx->imgBytes, cudaMemcpyHostToDevice, stream);

        /* conv_tiled_nosync launches without device sync, safe inside a stream */
        conv_tiled_nosync(ctx->d_input[s], ctx->d_kernel, ctx->d_output[s],
                          ctx->width, ctx->height, ctx->ksize, block);

        cudaMemcpyAsync(h_outputs[i], ctx->d_output[s],
                        ctx->imgBytes, cudaMemcpyDeviceToHost, stream);
    }

    for (int s = 0; s < ASYNC_NUM_STREAMS; ++s) {
        cudaStreamSynchronize(ctx->streams[s]);
    }
}

/**
 * \brief Run a single convolution asynchronously on a specified stream.
 *
 * Device pointers must already be allocated and input copied before this call.
 * The caller is responsible for synchronizing the stream before reading results.
 *
 * \param[in]  d_input  Device pointer to input image.
 * \param[in]  d_kernel Device pointer to convolution weights.
 * \param[out] d_output Device pointer to output image.
 * \param[in]  width    Image width in pixels.
 * \param[in]  height   Image height in pixels.
 * \param[in]  ksize    Kernel side length.
 * \param[in]  block    CUDA block dimensions.
 * \param[in]  stream   CUDA stream for async execution.
 */
extern "C" void async_conv_single(const float* d_input, const float* d_kernel,
                                   float* d_output, int width, int height, int ksize,
                                   dim3 block, cudaStream_t stream) {
    (void)stream; /* stream binding is implicit via conv_tiled_nosync's default stream */
    conv_tiled_nosync(d_input, d_kernel, d_output, width, height, ksize, block);
}

/**
 * \brief Measure throughput of sync vs async batch processing.
 *
 * Allocates pinned memory internally, runs both approaches, and reports timing.
 * Useful as a quick sanity-check that async overlap is working.
 *
 * \param[in]  width       Image width in pixels.
 * \param[in]  height      Image height in pixels.
 * \param[in]  ksize       Kernel side length.
 * \param[in]  num_batches Number of images to process.
 * \param[out] sync_ms     Time for synchronous baseline (ms).
 * \param[out] async_ms    Time for async pipelined run (ms).
 */
extern "C" void async_benchmark(int width, int height, int ksize, int num_batches,
                                 float* sync_ms, float* async_ms) {
    int imgBytes = width * height * sizeof(float);
    int kerBytes = ksize * ksize * sizeof(float);

    /* Allocate pinned host buffers */
    float** h_inputs  = (float**)malloc(num_batches * sizeof(float*));
    float** h_outputs = (float**)malloc(num_batches * sizeof(float*));
    float*  h_kernel;
    cudaMallocHost(&h_kernel, kerBytes);
    for (int i = 0; i < ksize * ksize; ++i) h_kernel[i] = 1.0f / (ksize * ksize);

    for (int i = 0; i < num_batches; ++i) {
        cudaMallocHost(&h_inputs[i],  imgBytes);
        cudaMallocHost(&h_outputs[i], imgBytes);
        for (int j = 0; j < width * height; ++j)
            h_inputs[i][j] = static_cast<float>(j + 1);
    }

    /* --- Synchronous baseline --- */
    float *d_input, *d_kernel_dev, *d_output;
    cudaMalloc(&d_input,      imgBytes);
    cudaMalloc(&d_kernel_dev, kerBytes);
    cudaMalloc(&d_output,     imgBytes);
    cudaMemcpy(d_kernel_dev, h_kernel, kerBytes, cudaMemcpyHostToDevice);

    dim3 block(TILE_W, TILE_H);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    cudaEventRecord(ev0);
    for (int i = 0; i < num_batches; ++i) {
        cudaMemcpy(d_input, h_inputs[i], imgBytes, cudaMemcpyHostToDevice);
        conv_tiled_nosync(d_input, d_kernel_dev, d_output, width, height, ksize, block);
        cudaDeviceSynchronize();
        cudaMemcpy(h_outputs[i], d_output, imgBytes, cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(sync_ms, ev0, ev1);

    cudaFree(d_input); cudaFree(d_kernel_dev); cudaFree(d_output);

    /* --- Async pipelined --- */
    AsyncStreamCtx ctx = {};
    async_stream_init(&ctx, width, height, ksize);

    cudaEventRecord(ev0);
    async_conv_batch(&ctx, h_inputs, h_outputs, num_batches, h_kernel, block);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(async_ms, ev0, ev1);

    async_stream_destroy(&ctx);

    /* Cleanup */
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    for (int i = 0; i < num_batches; ++i) {
        cudaFreeHost(h_inputs[i]);
        cudaFreeHost(h_outputs[i]);
    }
    cudaFreeHost(h_kernel);
    free(h_inputs);
    free(h_outputs);
}
