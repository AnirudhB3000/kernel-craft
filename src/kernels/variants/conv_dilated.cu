/**
 * \file conv_dilated.cu
 * \brief Dilated (atrous) convolution implementation.
 *
 * This file provides CUDA kernels for performing dilated convolution,
 * which expands the receptive field without increasing parameters.
 * Useful for semantic segmentation and dense prediction tasks.
 */

#include <cuda_runtime.h>

/**
 * \brief Naïve dilated convolution kernel.
 *
 * Each thread computes one output pixel. The dilation rate affects
 * the spacing between kernel samples (atrous sampling).
 *
 * \param[in] input  Pointer to the input image (row‑major, size = width×height).
 * \param[in] kernel Pointer to the convolution kernel (size = ksize×ksize).
 * \param[out] output Pointer to the output image buffer.
 * \param[in] width  Width of the input and output images.
 * \param[in] height Height of the input and output images.
 * \param[in] ksize Width and height of the square kernel (must be odd).
 * \param[in] dilation Dilation rate (1 = standard, 2 = skip every other sample).
 */
extern "C" __global__ void conv_dilated_naive(const float* __restrict__ input,
                                             const float* __restrict__ kernel,
                                             float* __restrict__ output,
                                             int width, int height,
                                             int ksize, int dilation) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int kHalf = ksize / 2;

    if (ox >= width || oy >= height) return;

    double sum = 0.0;

    for (int ky = 0; ky < ksize; ++ky) {
        int iy = oy + (ky - kHalf) * dilation;
        if (iy < 0 || iy >= height) continue;
        for (int kx = 0; kx < ksize; ++kx) {
            int ix = ox + (kx - kHalf) * dilation;
            if (ix < 0 || ix >= width) continue;
            float val = input[iy * width + ix];
            float kval = kernel[ky * ksize + kx];
            sum += static_cast<double>(val) * static_cast<double>(kval);
        }
    }
    output[oy * width + ox] = static_cast<float>(sum);
}

/**
 * \brief Host‑side launcher for the naïve dilated convolution kernel.
 *
 * \param[in] input  Device pointer to the input image.
 * \param[in] kernel Device pointer to the convolution kernel.
 * \param[out] output Device pointer to the output buffer.
 * \param[in] width  Image width.
 * \param[in] height Image height.
 * \param[in] ksize Kernel size (must be odd).
 * \param[in] dilation Dilation rate (1, 2, 4, 8 supported).
 * \param[in] block Block dimensions (default 16×16 threads).
 */
extern "C" void launch_conv_dilated_naive(const float* input,
                                          const float* kernel,
                                          float* output,
                                          int width, int height,
                                          int ksize, int dilation,
                                          dim3 block = dim3(16,16,1)) {
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    conv_dilated_naive<<<grid, block>>>(input, kernel, output, width, height, ksize, dilation);
    cudaDeviceSynchronize();
}