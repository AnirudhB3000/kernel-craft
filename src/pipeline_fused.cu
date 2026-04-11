/**
 * \file pipeline_fused.cu
 * \brief Fused kernel for conv → batchnorm → ReLU pipeline.
 *
 * This implementation combines three separate operations into a single CUDA kernel:
 * 1. Convolution (tiled, shared memory)
 * 2. Batch normalization: y = x * gamma + beta
 * 3. ReLU activation: y = max(0, x)
 *
 * \par Benefits
 * - Eliminates two kernel launch overheads (2 extra launches → 0)
 * - Avoids intermediate global memory writes between stages
 * - All computation happens in registers before writing final result
 *
 * \par Notes
 * - gamma and beta are passed as kernel parameters (could be constants)
 * - Tile dimensions match compile-time TILE_W/TILE_H from CMakeLists.txt
 */

#include <cuda_runtime.h>

#ifndef TILE_W
#define TILE_W 8
#endif
#ifndef TILE_H
#define TILE_H 8
#endif

/**
 * \brief Fused conv + batchnorm + ReLU kernel.
 *
 * Each threadblock computes one output tile. Within each thread,
 * convolution is performed in shared memory, then batchnorm and ReLU
 * are applied in registers before writing the final result to global memory.
 *
 * \param[in] input  Input image (row-major, float).
 * \param[in] kernel Convolution kernel (row-major, float).
 * \param[in] gamma  Batchnorm scale parameter.
 * \param[in] beta   Batchnorm shift parameter.
 * \param[out] output Output feature map (row-major, float).
 * \param[in] width  Image width.
 * \param[in] height Image height.
 * \param[in] ksize Kernel size (must be odd).
 */
extern "C" __global__ void conv_bn_relu_fused(const float* __restrict__ input,
                                           const float* __restrict__ kernel,
                                           float gamma, float beta,
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

        float conv_result = static_cast<float>(sum);
        float bn_result = conv_result * gamma + beta;
        float relu_result = bn_result > 0.0f ? bn_result : 0.0f;

        output[oy * width + ox] = relu_result;
    }
}

/**
 * \brief Host launcher for fused conv + batchnorm + ReLU.
 *
 * \param[in] d_input  Device pointer to input image.
 * \param[in] d_kernel Device pointer to convolution kernel.
 * \param[in] gamma    Batchnorm gamma (scale) parameter.
 * \param[in] beta     Batchnorm beta (shift) parameter.
 * \param[out] d_output Device pointer to output buffer.
 * \param[in] width    Image width.
 * \param[in] height   Image height.
 * \param[in] ksize    Kernel size (must be odd).
 * \param[in] block    Block dimensions (default matches TILE_W×TILE_H).
 */
extern "C" void launch_conv_bn_relu_fused(const float* d_input,
                                           const float* d_kernel,
                                           float gamma, float beta,
                                           float* d_output,
                                           int width, int height, int ksize,
                                           dim3 block = dim3(TILE_W, TILE_H, 1)) {
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    size_t sharedMemBytes = (block.x + ksize - 1) * (block.y + ksize - 1) * sizeof(float);
    conv_bn_relu_fused<<<grid, block, sharedMemBytes>>>(d_input, d_kernel,
                                                         gamma, beta, d_output,
                                                         width, height, ksize);
    cudaDeviceSynchronize();
}