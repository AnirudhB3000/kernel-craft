/**
 * \file conv_naive.cu
 * \brief Naïve 2‑D convolution implementation.
 *
 * This file provides a straightforward CUDA kernel that performs a
 * convolution by having each thread compute a single output pixel.
 * It includes a host‑side launcher for convenient invocation.
 */

#include <cuda_runtime.h>

/**
 * \brief Naïve convolution kernel.
 *
 * Each thread computes one output pixel using direct global‑memory reads.
 * Zero‑padding is applied for out‑of‑bounds coordinates.
 *
 * @param input   Pointer to the input image (row‑major, size = width×height).
 * @param kernel  Pointer to the convolution kernel (row‑major, size = ksize×ksize).
 * @param output  Pointer to the output image buffer (same size as input).
 * @param width   Width of the input and output images.
 * @param height  Height of the input and output images.
 * @param ksize   Width and height of the square kernel (must be odd).
 */
extern "C" __global__ void conv_naive(const float* __restrict__ input,
                                     const float* __restrict__ kernel,
                                     float* __restrict__ output,
                                     int width, int height,
                                     int ksize) {
    // Compute output coordinates
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int kHalf = ksize / 2;

    if (ox >= width || oy >= height) return;

    // Use double precision accumulator to minimise rounding differences
    double sum = 0.0;
    // Apply kernel centered at (ox, oy)
    for (int ky = 0; ky < ksize; ++ky) {
        int iy = oy + ky - kHalf;
        if (iy < 0 || iy >= height) continue; // zero padding
        for (int kx = 0; kx < ksize; ++kx) {
            int ix = ox + kx - kHalf;
            if (ix < 0 || ix >= width) continue; // zero padding
            float val = input[iy * width + ix];
            float kval = kernel[ky * ksize + kx];
            sum += static_cast<double>(val) * static_cast<double>(kval);
        }
    }
    output[oy * width + ox] = static_cast<float>(sum);
}

/**
 * \brief Host‑side convenience launcher for the naïve convolution kernel.
 *
 * This function configures a 2‑D grid based on the image dimensions and
 * launches the \c conv_naive kernel. It synchronises the device before
 * returning.
 *
 * @param input   Device pointer to the input image.
 * @param kernel  Device pointer to the convolution kernel.
 * @param output  Device pointer to the output buffer.
 * @param width   Image width.
 * @param height  Image height.
 * @param ksize   Kernel size (must be odd).
 * @param block   Block dimensions (default 16×16 threads).
 */
extern "C" void launch_conv_naive(const float* input,
                                 const float* kernel,
                                 float* output,
                                 int width, int height, int ksize,
                                 dim3 block = dim3(16,16,1)) {
    // ----- Hand off to GPU -----
    // Compute grid dimensions based on image size
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    // Launch the device kernel
    conv_naive<<<grid, block>>>(input, kernel, output, width, height, ksize);
    // Wait for GPU to finish and return control to CPU
    cudaDeviceSynchronize();
    // ----- Returned to CPU -----
}
