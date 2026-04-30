/**
 * \file conv_tiled.cu
 * \brief Shared‑memory tiled 2‑D convolution kernel.
 *
 * This implementation reduces global memory traffic by loading a tile of the
 * input image (including halo elements required for the convolution) into
 * shared memory. Each thread then computes one output pixel using the values
 * stored in the shared tile.
 *
 * \par Tile Configuration
 * The tile dimensions are configurable via the compile‑time parameters
 * \p TILE_W and \p TILE_H. The default is 8×8, which provides
 * the best overall performance for 3×3 kernels.
 *
 * \par Numerical Accuracy
 * The kernel accumulates results in double‑precision (\p double) to preserve
 * numerical accuracy, then writes a single‑precision (\p float) result back
 * to global memory.
 */

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Configurable tile size (default 16×16). Change via compile‑time flags or edit
// the values directly. The kernel will automatically compute the required
// shared‑memory size based on these parameters and the kernel size.
// ---------------------------------------------------------------------------
#ifndef TILE_W
#define TILE_W 16
#endif
#ifndef TILE_H
#define TILE_H 16
#endif

/**
 * \brief Tiled convolution kernel.
 *
 * \param[in] input  Input image (row‑major, float).
 * \param[in] kernel Convolution kernel (row‑major, float, odd dimension).
 * \param[out] output Output image buffer (row‑major, float).
 * \param[in] width  Image width.
 * \param[in] height Image height.
 * \param[in] ksize Kernel size (must be odd).
 */
extern "C" __global__ void conv_tiled(const float* __restrict__ input,
                                      const float* __restrict__ kernel,
                                      float* __restrict__ output,
                                      int width, int height, int ksize) {
    // ----- Hand off to GPU -----
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int ox = blockIdx.x * blockDim.x + tx; // output x coordinate
    const int oy = blockIdx.y * blockDim.y + ty; // output y coordinate

    const int kHalf = ksize / 2;

    // Shared‑memory tile dimensions include halo region required for the kernel.
    const int sharedW = blockDim.x + ksize - 1; // width of shared tile
    const int sharedH = blockDim.y + ksize - 1; // height of shared tile
    extern __shared__ float tile[]; // dynamically sized shared memory

    // Load input elements (including halo) into shared memory.
    // Each thread may load multiple elements to cover the full tile.
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
    // ----- Returned to CPU -----

    // Only compute if output pixel lies inside the image.
    if (ox < width && oy < height) {
        double sum = 0.0;
        // Convolution using values from shared memory.
        for (int ky = 0; ky < ksize; ++ky) {
            for (int kx = 0; kx < ksize; ++kx) {
                float imgVal = tile[(ty + ky) * sharedW + (tx + kx)];
                float kVal   = kernel[ky * ksize + kx];
                sum += static_cast<double>(imgVal) * static_cast<double>(kVal);
            }
        }
        output[oy * width + ox] = static_cast<float>(sum);
    }
    // ----- Returned to CPU -----
}

/**
 * \brief Host launcher for the tiled convolution kernel.
 *
 * Mirrors the interface of \p launch_conv_naive. The default block size is
 * \p dim3(TILE_W, TILE_H, 1), which matches the compile‑time tile dimensions.
 * Users may override the block size on the call site if desired.
 *
 * \param[in] d_input  Device pointer to input image.
 * \param[in] d_kernel Device pointer to convolution kernel.
 * \param[out] d_output Device pointer to output buffer.
 * \param[in] width   Image width.
 * \param[in] height  Image height.
 * \param[in] ksize  Kernel size (must be odd).
 * \param[in] block  Block dimensions (default matches TILE_W×TILE_H).
 */
extern "C" void launch_conv_tiled(const float* d_input,
                                  const float* d_kernel,
                                  float* d_output,
                                  int width, int height, int ksize,
                                  dim3 block = dim3(TILE_W, TILE_H, 1)) {
    // ----- Hand off to GPU -----
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    // Compute shared‑memory size needed for a single block.
    size_t sharedMemBytes = (block.x + ksize - 1) * (block.y + ksize - 1) * sizeof(float);
    conv_tiled<<<grid, block, sharedMemBytes>>>(d_input, d_kernel, d_output,
                                                width, height, ksize);
    cudaDeviceSynchronize();
    // ----- Returned to CPU -----
}

extern "C" void conv_tiled_nosync(const float* d_input,
                            const float* d_kernel,
                            float* d_output,
                            int width, int height, int ksize,
                            dim3 block) {
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    size_t sharedMemBytes = (block.x + ksize - 1) * (block.y + ksize - 1) * sizeof(float);
    conv_tiled<<<grid, block, sharedMemBytes>>>(d_input, d_kernel, d_output,
                                                width, height, ksize);
}




