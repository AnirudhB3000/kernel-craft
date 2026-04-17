/**
 * \file preprocess_gpu.cu
 * \brief GPU preprocessing kernels for ML pipelines.
 *
 * This module provides GPU-accelerated preprocessing operations commonly
 * needed in ML inference pipelines: resize (bilinear), normalization,
 * and augmentation (flip). These kernels eliminate CPU-GPU data transfers
 * by performing all preprocessing on the GPU.
 *
 * \par Operations
 * - Bilinear resize: upsample/downsampple with interpolation
 * - Normalization: per-channel mean/std normalization
 * - Flip: horizontal and vertical flips for data augmentation
 */

#include <cuda_runtime.h>

/**
 * \brief Bilinear interpolation resize kernel.
 *
 * Resizes input image to output dimensions using bilinear interpolation.
 * Each thread computes one output pixel.
 *
 * \param[in] input  Input image (row-major, float).
 * \param[out] output Output image buffer (row-major, float).
 * \param[in] in_w   Input image width.
 * \param[in] in_h   Input image height.
 * \param[in] out_w  Output image width.
 * \param[in] out_h  Output image height.
 */
extern "C" __global__ void resize_bilinear(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int in_w, int in_h,
                                           int out_w, int out_h) {
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ox >= out_w || oy >= out_h) return;

    float x = (ox + 0.5f) * static_cast<float>(in_w) / static_cast<float>(out_w) - 0.5f;
    float y = (oy + 0.5f) * static_cast<float>(in_h) / static_cast<float>(out_h) - 0.5f;

    int x0 = static_cast<int>(floorf(x));
    int y0 = static_cast<int>(floorf(y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    x0 = max(0, min(x0, in_w - 1));
    y0 = max(0, min(y0, in_h - 1));
    x1 = max(0, min(x1, in_w - 1));
    y1 = max(0, min(y1, in_h - 1));

    float fx = x - x0;
    float fy = y - y0;

    float f00 = input[y0 * in_w + x0];
    float f10 = input[y0 * in_w + x1];
    float f01 = input[y1 * in_w + x0];
    float f11 = input[y1 * in_w + x1];

    output[oy * out_w + ox] = (1 - fx) * (1 - fy) * f00 +
                               fx * (1 - fy) * f10 +
                               (1 - fx) * fy * f01 +
                               fx * fy * f11;
}

/**
 * \brief Launch resize kernel.
 */
extern "C" void launch_resize_bilinear(const float* d_input,
                                       float* d_output,
                                       int in_w, int in_h,
                                       int out_w, int out_h) {
    dim3 block(16, 16, 1);
    dim3 grid((out_w + block.x - 1) / block.x,
              (out_h + block.y - 1) / block.y);
    resize_bilinear<<<grid, block>>>(d_input, d_output, in_w, in_h, out_w, out_h);
    cudaDeviceSynchronize();
}

/**
 * \brief Normalization kernel.
 *
 * Applies per-channel normalization: (input - mean) / std.
 *
 * \param[in] input     Input image (row-major, float).
 * \param[out] output   Output image buffer (row-major, float).
 * \param[in] width     Image width.
 * \param[in] height    Image height.
 * \param[in] mean      Per-channel mean value.
 * \param[in] std       Per-channel std value.
 */
extern "C" __global__ void normalize(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int width, int height,
                                     float mean, float std) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    output[idx] = (input[idx] - mean) / std;
}

/**
 * \brief Launch normalization kernel.
 */
extern "C" void launch_normalize(const float* d_input,
                                 float* d_output,
                                 int width, int height,
                                 float mean, float std) {
    int n = width * height;
    dim3 block(256, 1, 1);
    dim3 grid((n + block.x - 1) / block.x);
    normalize<<<grid, block>>>(d_input, d_output, width, height, mean, std);
    cudaDeviceSynchronize();
}

/**
 * \brief Horizontal flip kernel.
 *
 * Flips image horizontally (left-right mirror).
 *
 * \param[in] input  Input image (row-major, float).
 * \param[out] output Output image buffer (row-major, float).
 * \param[in] width  Image width.
 * \param[in] height Image height.
 */
extern "C" __global__ void flip_horizontal(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    output[y * width + x] = input[y * width + (width - 1 - x)];
}

/**
 * \brief Vertical flip kernel.
 *
 * Flips image vertically (top-bottom mirror).
 *
 * \param[in] input  Input image (row-major, float).
 * \param[out] output Output image buffer (row-major, float).
 * \param[in] width  Image width.
 * \param[in] height Image height.
 */
extern "C" __global__ void flip_vertical(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    output[y * width + x] = input[(height - 1 - y) * width + x];
}

/**
 * \brief Launch horizontal flip kernel.
 */
extern "C" void launch_flip_horizontal(const float* d_input,
                                       float* d_output,
                                       int width, int height) {
    dim3 block(16, 16, 1);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    flip_horizontal<<<grid, block>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
}

/**
 * \brief Launch vertical flip kernel.
 */
extern "C" void launch_flip_vertical(const float* d_input,
                                     float* d_output,
                                     int width, int height) {
    dim3 block(16, 16, 1);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    flip_vertical<<<grid, block>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
}