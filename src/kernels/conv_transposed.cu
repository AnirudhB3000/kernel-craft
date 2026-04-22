/**
 * \file conv_transposed.cu
 * \brief Transposed convolution (deconvolution) implementation.
 *
 * This file provides CUDA kernels for performing transposed convolution,
 * also known as deconvolution or upconvolution. It is used for upsampling
 * in decoder networks, autoencoders, and generative models.
 */

#include <cuda_runtime.h>

/**
 * \brief Naïve transposed convolution kernel.
 *
 * Each thread computes one output pixel. The transposed convolution
 * upsamples the input by inserting zeros between samples based on stride.
 *
 * \param[in] input  Pointer to the input image (row‑major).
 * \param[in] kernel Pointer to the convolution kernel (size = ksize×ksize).
 * \param[out] output Pointer to the output image buffer.
 * \param[in] inWidth  Width of the input image.
 * \param[in] inHeight Height of the input image.
 * \param[in] outWidth  Width of the output image.
 * \param[in] outHeight Height of the output image.
 * \param[in] ksize Width and height of the square kernel (must be odd).
 * \param[in] stride Stride for upsampling (output = input * stride).
 * \param[in] padding Padding applied to input before convolution.
 */
extern "C" __global__ void conv_transposed_naive(const float* __restrict__ input,
                                                const float* __restrict__ kernel,
                                                float* __restrict__ output,
                                                int inWidth, int inHeight,
                                                int outWidth, int outHeight,
                                                int ksize, int stride, int padding) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int kHalf = ksize / 2;

    if (ox >= outWidth || oy >= outHeight) return;

    double sum = 0.0;
    for (int ky = 0; ky < ksize; ++ky) {
        for (int kx = 0; kx < ksize; ++kx) {
            int ix = (ox + padding - kx + kHalf) / stride;
            int iy = (oy + padding - ky + kHalf) / stride;

            if (ix < 0 || ix >= inWidth || iy < 0 || iy >= inHeight) continue;
            if ((ox + padding - kx + kHalf) % stride != 0) continue;
            if ((oy + padding - ky + kHalf) % stride != 0) continue;

            float val = input[iy * inWidth + ix];
            float kval = kernel[ky * ksize + kx];
            sum += static_cast<double>(val) * static_cast<double>(kval);
        }
    }
    output[oy * outWidth + ox] = static_cast<float>(sum);
}

/**
 * \brief Optimized transposed convolution using atomic additions.
 *
 * Each input pixel contributes to multiple output pixels based on
 * stride. Uses atomicAdd for correct accumulation.
 *
 * \param[in] input  Pointer to the input image.
 * \param[in] kernel Pointer to the convolution kernel.
 * \param[out] output Pointer to the output image buffer (initialized to zero).
 * \param[in] inWidth  Width of the input image.
 * \param[in] inHeight Height of the input image.
 * \param[in] outWidth  Width of the output image.
 * \param[in] outHeight Height of the output image.
 * \param[in] ksize Kernel size (must be odd).
 * \param[in] stride Stride for upsampling.
 * \param[in] padding Padding applied to input.
 */
extern "C" __global__ void conv_transposed_atomic(const float* __restrict__ input,
                                                 const float* __restrict__ kernel,
                                                 float* __restrict__ output,
                                                 int inWidth, int inHeight,
                                                 int outWidth, int outHeight,
                                                 int ksize, int stride, int padding) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= inWidth || iy >= inHeight) return;

    float inputVal = input[iy * inWidth + ix];
    int kHalf = ksize / 2;

    for (int ky = 0; ky < ksize; ++ky) {
        int oy = iy * stride + ky - kHalf + padding;
        if (oy < 0 || oy >= outHeight) continue;
        for (int kx = 0; kx < ksize; ++kx) {
            int ox = ix * stride + kx - kHalf + padding;
            if (ox < 0 || ox >= outWidth) continue;
            float kval = kernel[ky * ksize + kx];
            atomicAdd(&output[oy * outWidth + ox], inputVal * kval);
        }
    }
}

/**
 * \brief Host‑side launcher for the naïve transposed convolution kernel.
 *
 * \param[in] input  Device pointer to the input image.
 * \param[in] kernel Device pointer to the convolution kernel.
 * \param[out] output Device pointer to the output buffer.
 * \param[in] inWidth  Input width.
 * \param[in] inHeight Input height.
 * \param[in] outWidth  Output width.
 * \param[in] outHeight Output height.
 * \param[in] ksize Kernel size (must be odd).
 * \param[in] stride Stride (output = input * stride).
 * \param[in] padding Padding (default 0).
 * \param[in] block Block dimensions (default 16×16 threads).
 */
extern "C" void launch_conv_transposed_naive(const float* input,
                                             const float* kernel,
                                             float* output,
                                             int inWidth, int inHeight,
                                             int outWidth, int outHeight,
                                             int ksize, int stride, int padding = 0,
                                             dim3 block = dim3(16,16,1)) {
    dim3 grid((outWidth + block.x - 1) / block.x,
              (outHeight + block.y - 1) / block.y);
    conv_transposed_naive<<<grid, block>>>(input, kernel, output, inWidth, inHeight, outWidth, outHeight, ksize, stride, padding);
    cudaDeviceSynchronize();
}

/**
 * \brief Host‑side launcher for the atomic transposed convolution kernel.
 *
 * \param[in] input  Device pointer to the input image.
 * \param[in] kernel Device pointer to the convolution kernel.
 * \param[out] output Device pointer to the output buffer (will be zeroed first).
 * \param[in] inWidth  Input width.
 * \param[in] inHeight Input height.
 * \param[in] outWidth  Output width.
 * \param[in] outHeight Output height.
 * \param[in] ksize Kernel size (must be odd).
 * \param[in] stride Stride (output = input * stride).
 * \param[in] padding Padding (default 0).
 * \param[in] block Block dimensions (default 16×16 threads).
 */
extern "C" void launch_conv_transposed_atomic(const float* input,
                                              const float* kernel,
                                              float* output,
                                              int inWidth, int inHeight,
                                              int outWidth, int outHeight,
                                              int ksize, int stride, int padding = 0,
                                              dim3 block = dim3(16,16,1)) {
    cudaMemset(output, 0, sizeof(float) * outWidth * outHeight);
    dim3 grid((inWidth + block.x - 1) / block.x,
              (inHeight + block.y - 1) / block.y);
    conv_transposed_atomic<<<grid, block>>>(input, kernel, output, inWidth, inHeight, outWidth, outHeight, ksize, stride, padding);
    cudaDeviceSynchronize();
}