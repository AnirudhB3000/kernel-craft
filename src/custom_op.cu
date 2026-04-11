/**
 * \file custom_op.cu
 * \brief Custom sparse convolution kernel for ML workloads.
 *
 * This implementation handles sparse kernels where most weights are zero,
 * avoiding unnecessary multiplications. Uses a simple coordinate list format
 * storing only non-zero kernel values and their positions.
 *
 * \par Sparse Format
 * - kernel_nonzero: array of non-zero kernel values (size = nnz)
 * - kernel_rows:   row offsets for each non-zero (size = nnz)
 * - kernel_cols:   column offsets for each non-zero (size = nnz)
 * - nnz:          number of non-zero elements
 *
 * \par Example (3x3 kernel with only center and corners non-zero)
 * \code
 *   kernel_nonzero = [1.0, 1.0, 1.0, 1.0, 1.0]  // 5 non-zeros
 *   kernel_rows   = [0, 0, 1, 2, 2]
 *   kernel_cols  = [0, 2, 1, 0, 2]
 * \endcode
 *
 * \par Performance
 * Sparse convolution avoids multiplications with zero kernel values,
 * providing speedup proportional to kernel sparsity.
 */

#include <cuda_runtime.h>

#define MAX_NONZERO 16

/**
 * \brief Sparse convolution kernel.
 *
 * Each thread computes one output pixel by iterating only over non-zero
 * kernel elements. Skips multiplications where kernel value is zero.
 *
 * \param[in] input         Input image (row-major, float).
 * \param[in] kernel_nonzero  Array of non-zero kernel values (size = nnz).
 * \param[in] kernel_rows    Row offsets for each non-zero.
 * \param[in] kernel_cols   Column offsets for each non-zero.
 * \param[in] nnz          Number of non-zero kernel elements.
 * \param[out] output       Output image buffer (row-major, float).
 * \param[in] width        Image width.
 * \param[in] height       Image height.
 * \param[in] ksize        Kernel size (must be odd).
 */
extern "C" __global__ void conv_sparse(const float* __restrict__ input,
                                    const float* __restrict__ kernel_nonzero,
                                    const int* __restrict__ kernel_rows,
                                    const int* __restrict__ kernel_cols,
                                    const int nnz,
                                    float* __restrict__ output,
                                    int width, int height, int ksize) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int kHalf = ksize / 2;

    if (ox >= width || oy >= height) return;

    double sum = 0.0;
    for (int i = 0; i < nnz; ++i) {
        int ky = kernel_rows[i];
        int kx = kernel_cols[i];
        int iy = oy + ky - kHalf;
        int ix = ox + kx - kHalf;
        if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
            float val = input[iy * width + ix];
            float kval = kernel_nonzero[i];
            sum += static_cast<double>(val) * static_cast<double>(kval);
        }
    }
    output[oy * width + ox] = static_cast<float>(sum);
}

/**
 * \brief Host launcher for sparse convolution kernel.
 *
 * Configures grid dimensions and launches the \p conv_sparse kernel.
 * Uses default 16x16 thread block if not specified.
 *
 * \param[in] d_input        Device pointer to input image.
 * \param[in] d_kernel_nonzero Device pointer to non-zero kernel values.
 * \param[in] d_kernel_rows   Device pointer to row offsets.
 * \param[in] d_kernel_cols  Device pointer to column offsets.
 * \param[in] nnz          Number of non-zero elements.
 * \param[out] d_output     Device pointer to output buffer.
 * \param[in] width        Image width.
 * \param[in] height       Image height.
 * \param[in] ksize       Kernel size.
 * \param[in] block       Block dimensions (default 16x16).
 */
extern "C" void launch_conv_sparse(const float* d_input,
                                    const float* d_kernel_nonzero,
                                    const int* d_kernel_rows,
                                    const int* d_kernel_cols,
                                    int nnz,
                                    float* d_output,
                                    int width, int height, int ksize,
                                    dim3 block = dim3(16, 16, 1)) {
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    conv_sparse<<<grid, block>>>(d_input, d_kernel_nonzero, d_kernel_rows,
                                 d_kernel_cols, nnz, d_output, width, height, ksize);
    cudaDeviceSynchronize();
}

/**
 * \brief Dense convolution kernel (fallback for comparison).
 *
 * Standard convolution that iterates over all kernel elements.
 * Used as baseline to compare against sparse implementation.
 *
 * \param[in] input   Input image (row-major, float).
 * \param[in] kernel  Dense kernel (row-major, float).
 * \param[out] output Output image buffer (row-major, float).
 * \param[in] width   Image width.
 * \param[in] height  Image height.
 * \param[in] ksize  Kernel size (must be odd).
 */
extern "C" __global__ void conv_dense_fallback(const float* __restrict__ input,
                                            const float* __restrict__ kernel,
                                            float* __restrict__ output,
                                            int width, int height, int ksize) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int kHalf = ksize / 2;

    if (ox >= width || oy >= height) return;

    double sum = 0.0;
    for (int ky = 0; ky < ksize; ++ky) {
        int iy = oy + ky - kHalf;
        if (iy < 0 || iy >= height) continue;
        for (int kx = 0; kx < ksize; ++kx) {
            int ix = ox + kx - kHalf;
            if (ix < 0 || ix >= width) continue;
            float val = input[iy * width + ix];
            float kval = kernel[ky * ksize + kx];
            sum += static_cast<double>(val) * static_cast<double>(kval);
        }
    }
    output[oy * width + ox] = static_cast<float>(sum);
}

/**
 * \brief Host launcher for dense convolution fallback.
 *
 * Configures grid dimensions and launches the \p conv_dense_fallback kernel.
 * Used as baseline for sparse vs dense comparison.
 *
 * \param[in] d_input   Device pointer to input image.
 * \param[in] d_kernel  Device pointer to dense kernel.
 * \param[out] d_output Device pointer to output buffer.
 * \param[in] width    Image width.
 * \param[in] height   Image height.
 * \param[in] ksize    Kernel size.
 * \param[in] block    Block dimensions (default 16x16).
 */
extern "C" void launch_conv_dense_fallback(const float* d_input,
                                            const float* d_kernel,
                                            float* d_output,
                                            int width, int height, int ksize,
                                            dim3 block = dim3(16, 16, 1)) {
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    conv_dense_fallback<<<grid, block>>>(d_input, d_kernel, d_output,
                                        width, height, ksize);
    cudaDeviceSynchronize();
}