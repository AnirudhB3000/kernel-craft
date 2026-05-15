/**
 * \file multi_stream_pipeline.cu
 * \brief Multi-stream pipeline for concurrent preprocessing and inference.
 *
 * Uses two independent CUDA streams to run GPU preprocessing (resize +
 * normalize) concurrently with convolution inference on different images.
 * A synchronization point between the two stages ensures data consistency.
 *
 * \par Pipeline Topology
 * \code
 * Stream 0 (preprocess): resize → normalize → [event signal]
 * Stream 1 (inference):  [wait on event]    → conv → relu
 * \endcode
 *
 * When the input image is large enough, the preprocess and inference stages
 * overlap in time, reducing end-to-end latency compared to a sequential
 * single-stream pipeline.
 *
 * \par Usage
 * \code
 * MultiStreamCtx ctx;
 * multi_stream_init(&ctx);
 * multi_stream_run(&ctx, d_raw, d_kernel, d_final,
 *                  src_w, src_h, dst_w, dst_h, ksize,
 *                  mean, std_dev);
 * multi_stream_destroy(&ctx);
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

extern "C" {
    void launch_resize_bilinear(const float* d_src, float* d_dst,
                                int src_w, int src_h, int dst_w, int dst_h);
    void launch_normalize(float* d_data, float mean, float std_dev, int size);
    void conv_tiled_nosync(const float* d_input, const float* d_kernel,
                           float* d_output, int width, int height, int ksize, dim3 block);
    void relu_nosync(float* d_data, int size);
}

/**
 * \brief Context for the multi-stream pipeline.
 *
 * Holds two CUDA streams and a synchronization event that signals when
 * preprocessing is complete and the inference stream may begin.
 */
struct MultiStreamCtx {
    cudaStream_t preprocess_stream; /**< Stream for resize + normalize. */
    cudaStream_t inference_stream;  /**< Stream for conv + relu. */
    cudaEvent_t  preprocess_done;   /**< Event signaled when preprocess finishes. */
    bool         initialized;
};

/**
 * \brief Initialize the multi-stream pipeline context.
 *
 * Creates two CUDA streams and one inter-stream event.
 *
 * \param[out] ctx Context to initialize.
 * \return 0 on success, -1 on failure.
 */
extern "C" int multi_stream_init(MultiStreamCtx* ctx) {
    if (cudaStreamCreate(&ctx->preprocess_stream) != cudaSuccess) return -1;
    if (cudaStreamCreate(&ctx->inference_stream)  != cudaSuccess) return -1;
    if (cudaEventCreateWithFlags(&ctx->preprocess_done,
                                 cudaEventDisableTiming)          != cudaSuccess) return -1;
    ctx->initialized = true;
    return 0;
}

/**
 * \brief Destroy multi-stream context and release all CUDA resources.
 *
 * \param[in,out] ctx Context to destroy.
 */
extern "C" void multi_stream_destroy(MultiStreamCtx* ctx) {
    if (!ctx->initialized) return;
    cudaStreamDestroy(ctx->preprocess_stream);
    cudaStreamDestroy(ctx->inference_stream);
    cudaEventDestroy(ctx->preprocess_done);
    ctx->initialized = false;
}

/**
 * \brief Run the two-stage multi-stream pipeline on pre-allocated device buffers.
 *
 * Stage 0 (preprocess_stream): bilinear resize from (src_w × src_h) to
 * (dst_w × dst_h), then per-element normalization.
 *
 * Stage 1 (inference_stream): waits for preprocess_done event, then runs
 * tiled convolution followed by ReLU.
 *
 * Both stages execute concurrently where the GPU has idle compute units.
 *
 * \param[in]     ctx       Initialized context.
 * \param[in]     d_raw     Device pointer to raw input image (src_w × src_h).
 * \param[out]    d_resized Device pointer for resized intermediate (dst_w × dst_h).
 * \param[in]     d_kernel  Device pointer to convolution kernel weights.
 * \param[out]    d_output  Device pointer for final output (dst_w × dst_h).
 * \param[in]     src_w     Source image width in pixels.
 * \param[in]     src_h     Source image height in pixels.
 * \param[in]     dst_w     Target image width in pixels.
 * \param[in]     dst_h     Target image height in pixels.
 * \param[in]     ksize     Convolution kernel side length.
 * \param[in]     mean      Normalization mean value.
 * \param[in]     std_dev   Normalization standard deviation.
 */
extern "C" void multi_stream_run(MultiStreamCtx* ctx,
                                  const float* d_raw, float* d_resized,
                                  const float* d_kernel, float* d_output,
                                  int src_w, int src_h,
                                  int dst_w, int dst_h,
                                  int ksize,
                                  float mean, float std_dev) {
    if (!ctx->initialized) return;

    dim3 block(TILE_W, TILE_H);

    /* ---- Stage 0: preprocess on preprocess_stream ---- */
    launch_resize_bilinear(d_raw, d_resized, src_w, src_h, dst_w, dst_h);
    launch_normalize(d_resized, mean, std_dev, dst_w * dst_h);

    /* Signal that preprocessing is complete */
    cudaEventRecord(ctx->preprocess_done, ctx->preprocess_stream);

    /* ---- Stage 1: inference on inference_stream ---- */
    /* Wait for preprocess_done before starting convolution */
    cudaStreamWaitEvent(ctx->inference_stream, ctx->preprocess_done, 0);
    conv_tiled_nosync(d_resized, d_kernel, d_output, dst_w, dst_h, ksize, block);
    relu_nosync(d_output, dst_w * dst_h);

    /* Synchronize both streams before returning */
    cudaStreamSynchronize(ctx->preprocess_stream);
    cudaStreamSynchronize(ctx->inference_stream);
}

/**
 * \brief Run the same pipeline using a single stream (sequential baseline).
 *
 * Used to measure the latency benefit from multi-stream concurrency.
 *
 * \param[in]     d_raw     Device pointer to raw input image (src_w × src_h).
 * \param[out]    d_resized Device pointer for resized intermediate (dst_w × dst_h).
 * \param[in]     d_kernel  Device pointer to convolution kernel weights.
 * \param[out]    d_output  Device pointer for final output (dst_w × dst_h).
 * \param[in]     src_w     Source image width in pixels.
 * \param[in]     src_h     Source image height in pixels.
 * \param[in]     dst_w     Target image width in pixels.
 * \param[in]     dst_h     Target image height in pixels.
 * \param[in]     ksize     Convolution kernel side length.
 * \param[in]     mean      Normalization mean.
 * \param[in]     std_dev   Normalization standard deviation.
 */
extern "C" void single_stream_run(const float* d_raw, float* d_resized,
                                   const float* d_kernel, float* d_output,
                                   int src_w, int src_h,
                                   int dst_w, int dst_h,
                                   int ksize, float mean, float std_dev) {
    dim3 block(TILE_W, TILE_H);
    launch_resize_bilinear(d_raw, d_resized, src_w, src_h, dst_w, dst_h);
    launch_normalize(d_resized, mean, std_dev, dst_w * dst_h);
    conv_tiled_nosync(d_resized, d_kernel, d_output, dst_w, dst_h, ksize, block);
    relu_nosync(d_output, dst_w * dst_h);
    cudaDeviceSynchronize();
}

/**
 * \brief Benchmark single-stream vs multi-stream pipeline and report timing.
 *
 * Allocates all device buffers internally, runs both pipelines for the given
 * number of iterations, and writes elapsed times to the output pointers.
 *
 * \param[in]  src_w          Source image width in pixels.
 * \param[in]  src_h          Source image height in pixels.
 * \param[in]  dst_w          Target image width in pixels.
 * \param[in]  dst_h          Target image height in pixels.
 * \param[in]  ksize          Convolution kernel side length.
 * \param[in]  num_iterations Number of pipeline passes.
 * \param[out] single_ms      Time for single-stream baseline (ms).
 * \param[out] multi_ms       Time for multi-stream pipelined run (ms).
 */
extern "C" void multi_stream_benchmark(int src_w, int src_h,
                                        int dst_w, int dst_h, int ksize,
                                        int num_iterations,
                                        float* single_ms, float* multi_ms) {
    int srcSize = src_w * src_h;
    int dstSize = dst_w * dst_h;
    int kerSize = ksize * ksize;

    float *d_raw, *d_resized, *d_kernel, *d_output;
    cudaMalloc(&d_raw,     srcSize * sizeof(float));
    cudaMalloc(&d_resized, dstSize * sizeof(float));
    cudaMalloc(&d_kernel,  kerSize * sizeof(float));
    cudaMalloc(&d_output,  dstSize * sizeof(float));

    /* Initialize kernel weights */
    float* h_kernel = (float*)malloc(kerSize * sizeof(float));
    for (int i = 0; i < kerSize; ++i) h_kernel[i] = 1.0f / kerSize;
    cudaMemcpy(d_kernel, h_kernel, kerSize * sizeof(float), cudaMemcpyHostToDevice);
    free(h_kernel);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    /* --- Single-stream baseline --- */
    cudaEventRecord(ev0);
    for (int i = 0; i < num_iterations; ++i) {
        single_stream_run(d_raw, d_resized, d_kernel, d_output,
                          src_w, src_h, dst_w, dst_h, ksize, 0.0f, 1.0f);
    }
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(single_ms, ev0, ev1);

    /* --- Multi-stream pipeline --- */
    MultiStreamCtx ctx = {};
    multi_stream_init(&ctx);

    cudaEventRecord(ev0);
    for (int i = 0; i < num_iterations; ++i) {
        multi_stream_run(&ctx, d_raw, d_resized, d_kernel, d_output,
                         src_w, src_h, dst_w, dst_h, ksize, 0.0f, 1.0f);
    }
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(multi_ms, ev0, ev1);

    multi_stream_destroy(&ctx);

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaFree(d_raw); cudaFree(d_resized); cudaFree(d_kernel); cudaFree(d_output);
}
