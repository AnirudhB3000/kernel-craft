/**
 * \file pipeline_separate.cu
 * \brief Separate kernels for conv → batchnorm → ReLU pipeline.
 *
 * This file provides individual kernel implementations for each stage
 * of a typical convolution + batch normalization + ReLU pipeline.
 * Used as baseline when comparing against fused kernel performance.
 */

#include <cuda_runtime.h>

#ifndef TILE_W
#define TILE_W 8
#endif
#ifndef TILE_H
#define TILE_H 8
#endif

/**
 * \brief Separable convolution kernel (reuses conv_tiled implementation).
 *
 * \param[in] input  Input image (row-major, float).
 * \param[in] kernel Convolution kernel (row-major, float).
 * \param[out] output Output image buffer (row-major, float).
 * \param[in] width  Image width.
 * \param[in] height Image height.
 * \param[in] ksize Kernel size (must be odd).
 */
extern "C" __global__ void conv_layer(const float* __restrict__ input,
                                   const float* __restrict__ kernel,
                                   float* __restrict__ output,
                                   int width, int height, int ksize) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int ox = blockIdx.x * blockDim.x + tx;
    const int oy = blockIdx.y * blockDim.y + ty;
    const int kHalf = ksize / 2;
    const int sharedW = blockDim.x + ksize - 1;
    const int sharedH = blockDim.y + ksize - 1;
    extern __shared__ float tile[];

    for (int dy = ty; dy < sharedH; dy += blockDim.y) {
        for (int dx = tx; dx < sharedW; dx += blockDim.x) {
            int img_x = blockIdx.x * blockDim.x + dx - kHalf;
            int img_y = blockIdx.y * blockDim.y + dy - kHalf;
            float val = 0.0f;
            if (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height) {
                val = input[img_y * width + img_x];
            }
            tile[dy * sharedW + dx] = val;
        }
    }
    __syncthreads();

    if (ox < width && oy < height) {
        double sum = 0.0;
        for (int ky = 0; ky < ksize; ++ky) {
            for (int kx = 0; kx < ksize; ++kx) {
                float imgVal = tile[(ty + ky) * sharedW + (tx + kx)];
                float kVal   = kernel[ky * ksize + kx];
                sum += static_cast<double>(imgVal) * static_cast<double>(kVal);
            }
        }
        output[oy * width + ox] = static_cast<float>(sum);
    }
}

/**
 * \brief Batch normalization kernel (inference mode).
 *
 * Applies: y = x * gamma + beta
 * (skips mean/variance subtraction for simplicity;
 * in practice, use running mean/variance from training)
 *
 * \param[in] input  Input feature map (row-major, float).
 * \param[in] gamma Scale parameter.
 * \param[in] beta  Shift parameter.
 * \param[out] output Output feature map (row-major, float).
 * \param[in] size  Number of elements.
 */
extern "C" __global__ void batchnorm_layer(float* __restrict__ input,
                                           float gamma, float beta,
                                           float* __restrict__ output,
                                           int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x * gamma + beta;
    }
}

/**
 * \brief ReLU activation kernel.
 *
 * Applies: y = max(0, x)
 *
 * \param[in] input  Input activation (row-major, float).
 * \param[out] output Output activation (row-major, float).
 * \param[in] size  Number of elements.
 */
extern "C" __global__ void relu_layer(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x > 0.0f ? x : 0.0f;
    }
}

/**
 * \brief Host launcher for separable conv layer.
 */
extern "C" void launch_conv_layer(const float* d_input,
                                 const float* d_kernel,
                                 float* d_output,
                                 int width, int height, int ksize,
                                 dim3 block = dim3(TILE_W, TILE_H, 1)) {
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    size_t sharedMemBytes = (block.x + ksize - 1) * (block.y + ksize - 1) * sizeof(float);
    conv_layer<<<grid, block, sharedMemBytes>>>(d_input, d_kernel, d_output,
                                               width, height, ksize);
    cudaDeviceSynchronize();
}

/**
 * \brief Launch batch normalization layer.
 *
 * \param d_input  Input feature map.
 * \param gamma   Scale parameter.
 * \param beta    Shift parameter.
 * \param d_output Output feature map.
 * \param size    Number of elements.
 * \param block   Block dimensions.
 */
extern "C" void launch_batchnorm_layer(float* d_input,
                                        float gamma, float beta,
                                        float* d_output,
                                        int size,
                                        dim3 block = dim3(256, 1, 1)) {
    dim3 grid((size + block.x - 1) / block.x);
    batchnorm_layer<<<grid, block>>>(d_input, gamma, beta, d_output, size);
    cudaDeviceSynchronize();
}

/**
 * \brief Launch ReLU activation layer.
 *
 * \param d_input  Input activation.
 * \param d_output Output activation.
 * \param size    Number of elements.
 * \param block   Block dimensions.
 */
extern "C" void launch_relu_layer(const float* d_input,
                                   float* d_output,
                                   int size,
                                   dim3 block = dim3(256, 1, 1)) {
    dim3 grid((size + block.x - 1) / block.x);
    relu_layer<<<grid, block>>>(d_input, d_output, size);
    cudaDeviceSynchronize();
}

/**
 * \brief ReLU activation in-place (no sync).
 *
 * \param d_data Activation data.
 * \param size  Number of elements.
 */
extern "C" void relu_nosync(float* d_data, int size) {
    dim3 grid((size + 256 - 1) / 256);
    dim3 block(256, 1, 1);
    relu_layer<<<grid, block>>>(d_data, d_data, size);
}

/**
 * \brief Add bias to activation (in-place).
 *
 * \param d_data Activation data.
 * \param bias Bias value.
 * \param size  Number of elements.
 */
extern "C" void add_bias(float* d_data, float bias, int size) {
    dim3 grid((size + 256 - 1) / 256);
    dim3 block(256, 1, 1);
    batchnorm_layer<<<grid, block>>>(d_data, 1.0f, bias, d_data, size);
    cudaDeviceSynchronize();
}

/**
 * \brief Add bias to activation in-place (no sync).
 *
 * \param d_data Activation data.
 * \param bias Bias value.
 * \param size  Number of elements.
 */
extern "C" void add_bias_nosync(float* d_data, float bias, int size) {
    dim3 grid((size + 256 - 1) / 256);
    dim3 block(256, 1, 1);
    batchnorm_layer<<<grid, block>>>(d_data, 1.0f, bias, d_data, size);
}