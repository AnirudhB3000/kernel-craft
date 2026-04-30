/**
 * \file conv3d.cu
 * \brief 3‑D volumetric convolution implementation.
 *
 * This file provides CUDA kernels for performing convolution on 3‑D
 * volumetric data (depth × height × width), useful for medical imaging
 * applications such as MRI and CT scan processing.
 */

#include <cuda_runtime.h>

/**
 * \brief Naïve 3‑D convolution kernel.
 *
 * Each thread computes one output voxel using direct global‑memory reads.
 * Zero‑padding is applied for out‑of‑bounds coordinates.
 *
 * \param[in] input  Pointer to the input volume (row‑major, size = depth×height×width).
 * \param[in] kernel Pointer to the 3‑D convolution kernel (size = kd×kh×kw).
 * \param[out] output Pointer to the output volume buffer (same size as input).
 * \param[in] depth  Depth of the input and output volumes.
 * \param[in] height Height of the input and output volumes.
 * \param[in] width  Width of the input and output volumes.
 * \param[in] kd Kernel depth (must be odd).
 * \param[in] kh Kernel height (must be odd).
 * \param[in] kw Kernel width (must be odd).
 */
extern "C" __global__ void conv3d_naive(const float* __restrict__ input,
                                       const float* __restrict__ kernel,
                                       float* __restrict__ output,
                                       int depth, int height, int width,
                                       int kd, int kh, int kw) {
    int oz = blockIdx.z * blockDim.z + threadIdx.z;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int ox = blockIdx.x * blockDim.x + threadIdx.x;

    int kdHalf = kd / 2;
    int khHalf = kh / 2;
    int kwHalf = kw / 2;

    if (oz >= depth || oy >= height || ox >= width) return;

    double sum = 0.0;
    for (int kz = 0; kz < kd; ++kz) {
        int iz = oz + kz - kdHalf;
        if (iz < 0 || iz >= depth) continue;
        for (int ky = 0; ky < kh; ++ky) {
            int iy = oy + ky - khHalf;
            if (iy < 0 || iy >= height) continue;
            for (int kx = 0; kx < kw; ++kx) {
                int ix = ox + kx - kwHalf;
                if (ix < 0 || ix >= width) continue;
                float val = input[iz * height * width + iy * width + ix];
                float kval = kernel[kz * kh * kw + ky * kw + kx];
                sum += static_cast<double>(val) * static_cast<double>(kval);
            }
        }
    }
    output[oz * height * width + oy * width + ox] = static_cast<float>(sum);
}

/**
 * \brief Tiled 3‑D convolution kernel using shared memory.
 *
 * Loads tiles of the input volume into shared memory to reduce
 * global memory traffic. Each block processes one output tile.
 *
 * \param[in] input  Pointer to the input volume.
 * \param[in] kernel Pointer to the 3‑D convolution kernel.
 * \param[out] output Pointer to the output volume buffer.
 * \param[in] depth  Depth of the input and output volumes.
 * \param[in] height Height of the input and output volumes.
 * \param[in] width  Width of the input and output volumes.
 * \param[in] kd Kernel depth (must be odd).
 * \param[in] kh Kernel height (must be odd).
 * \param[in] kw Kernel width (must be odd).
 */
extern "C" __global__ void conv3d_tiled(const float* __restrict__ input,
                                       const float* __restrict__ kernel,
                                       float* __restrict__ output,
                                       int depth, int height, int width,
                                       int kd, int kh, int kw) {
    extern __shared__ float s_data[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int tile_w = blockDim.x;
    int tile_h = blockDim.y;
    int tile_d = blockDim.z;

    int kwHalf = kw / 2;
    int khHalf = kh / 2;
    int kdHalf = kd / 2;

    int ox = blockIdx.x * tile_w + tx;
    int oy = blockIdx.y * tile_h + ty;
    int oz = blockIdx.z * tile_d + tz;

    int outIdx = oz * height * width + oy * width + ox;
    bool inBounds = (ox < width && oy < height && oz < depth);

    if (inBounds) {
        s_data[tz * tile_h * tile_w + ty * tile_w + tx] = input[outIdx];
    }
    __syncthreads();

    if (!inBounds) return;

    double sum = 0.0;
    for (int kz = 0; kz < kd; ++kz) {
        int sz = tz + kz - kdHalf;
        if (sz < 0 || sz >= tile_d) {
            int iz = oz + kz - kdHalf;
            if (iz >= 0 && iz < depth) {
                sum += static_cast<double>(input[iz * height * width + oy * width + ox]) *
                       static_cast<double>(kernel[kz * kh * kw + (kh/2) * kw + kw/2]);
            }
            continue;
        }
        for (int ky = 0; ky < kh; ++ky) {
            int sy = ty + ky - khHalf;
            if (sy < 0 || sy >= tile_h) {
                int iy = oy + ky - khHalf;
                if (iy >= 0 && iy < height) {
                    sum += static_cast<double>(input[oz * height * width + iy * width + ox]) *
                           static_cast<double>(kernel[kz * kh * kw + ky * kw + kw/2]);
                }
                continue;
            }
            for (int kx = 0; kx < kw; ++kx) {
                int sx = tx + kx - kwHalf;
                if (sx < 0 || sx >= tile_w) {
                    int ix = ox + kx - kwHalf;
                    if (ix >= 0 && ix < width) {
                        sum += static_cast<double>(input[oz * height * width + oy * width + ix]) *
                               static_cast<double>(kernel[kz * kh * kw + (kh/2) * kw + kx]);
                    }
                    continue;
                }
                float val = s_data[sz * tile_h * tile_w + sy * tile_w + sx];
                float kval = kernel[kz * kh * kw + ky * kw + kx];
                sum += static_cast<double>(val) * static_cast<double>(kval);
            }
        }
    }
    output[outIdx] = static_cast<float>(sum);
}

/**
 * \brief Host‑side launcher for the naïve 3‑D convolution kernel.
 *
 * \param[in] input  Device pointer to the input volume.
 * \param[in] kernel Device pointer to the 3‑D convolution kernel.
 * \param[out] output Device pointer to the output buffer.
 * \param[in] depth  Volume depth.
 * \param[in] height Volume height.
 * \param[in] width  Volume width.
 * \param[in] kd Kernel depth (must be odd).
 * \param[in] kh Kernel height (must be odd).
 * \param[in] kw Kernel width (must be odd).
 * \param[in] block Block dimensions (default 8×8×8 threads).
 */
extern "C" void launch_conv3d_naive(const float* input,
                                    const float* kernel,
                                    float* output,
                                    int depth, int height, int width,
                                    int kd, int kh, int kw,
                                    dim3 block = dim3(8,8,8)) {
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y,
              (depth + block.z - 1) / block.z);
    conv3d_naive<<<grid, block>>>(input, kernel, output, depth, height, width, kd, kh, kw);
    cudaDeviceSynchronize();
}

/**
 * \brief Host‑side launcher for the tiled 3‑D convolution kernel.
 *
 * \param[in] input  Device pointer to the input volume.
 * \param[in] kernel Device pointer to the 3‑D convolution kernel.
 * \param[out] output Device pointer to the output buffer.
 * \param[in] depth  Volume depth.
 * \param[in] height Volume height.
 * \param[in] width  Volume width.
 * \param[in] kd Kernel depth (must be odd).
 * \param[in] kh Kernel height (must be odd).
 * \param[in] kw Kernel width (must be odd).
 * \param[in] block Block dimensions (default 8×8×8 threads).
 */
extern "C" void launch_conv3d_tiled(const float* input,
                                    const float* kernel,
                                    float* output,
                                    int depth, int height, int width,
                                    int kd, int kh, int kw,
                                    dim3 block = dim3(8,8,8)) {
    int tile_w = block.x;
    int tile_h = block.y;
    int tile_d = block.z;
    int sharedSize = tile_d * tile_h * tile_w * sizeof(float);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y,
              (depth + block.z - 1) / block.z);
    conv3d_tiled<<<grid, block, sharedSize>>>(input, kernel, output, depth, height, width, kd, kh, kw);
    cudaDeviceSynchronize();
}