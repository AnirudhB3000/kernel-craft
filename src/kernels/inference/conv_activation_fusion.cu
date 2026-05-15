/**
 * \file conv_activation_fusion.cu
 * \brief Fused convolution + activation kernels for CNN inference.
 *
 * Fuses convolution with activation functions (ReLU, Leaky ReLU, Sigmoid)
 * into a single kernel to reduce memory traffic and improve performance.
 *
 * \par Benefits
 * - Eliminates intermediate memory writes/reads
 * - Reduces kernel launch overhead
 * - Better cache locality
 *
 * \par Supported Activations
 * - ReLU: max(0, x)
 * - Leaky ReLU: x > 0 ? x : alpha * x
 * - Sigmoid: 1 / (1 + exp(-x))
 */

#include <cuda_runtime.h>
#include <cmath>

// ---------------------------------------------------------------------------
// Activation functions (device)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float relu_activation(float x) {
    return fmaxf(0.0f, x);
}

__device__ __forceinline__ float leaky_relu_activation(float x, float alpha = 0.01f) {
    return x > 0.0f ? x : alpha * x;
}

__device__ __forceinline__ float sigmoid_activation(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
 * \brief Fused convolution + ReLU activation (naive implementation).
 *
 * \param[in] input  Input image (row‑major, float).
 * \param[in] kernel Convolution kernel (row‑major, float).
 * \param[out] output Output image buffer (row‑major, float).
 * \param[in] width  Image width.
 * \param[in] height Image height.
 * \param[in] ksize Kernel size (must be odd).
 */
extern "C" __global__ void conv_relu_naive(const float* __restrict__ input,
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
            sum += static_cast<double>(input[iy * width + ix]) * 
                   static_cast<double>(kernel[ky * ksize + kx]);
        }
    }
    
    float conv_result = static_cast<float>(sum);
    output[oy * width + ox] = relu_activation(conv_result);
}

/**
 * \brief Fused convolution + Leaky ReLU activation (naive implementation).
 *
 * \param[in] input  Input image (row‑major, float).
 * \param[in] kernel Convolution kernel (row‑major, float).
 * \param[out] output Output image buffer (row‑major, float).
 * \param[in] width  Image width.
 * \param[in] height Image height.
 * \param[in] ksize Kernel size (must be odd).
 * \param[in] alpha Negative slope coefficient.
 */
extern "C" __global__ void conv_leaky_relu_naive(const float* __restrict__ input,
                                                  const float* __restrict__ kernel,
                                                  float* __restrict__ output,
                                                  int width, int height, int ksize,
                                                  float alpha) {
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
            sum += static_cast<double>(input[iy * width + ix]) * 
                   static_cast<double>(kernel[ky * ksize + kx]);
        }
    }
    
    float conv_result = static_cast<float>(sum);
    output[oy * width + ox] = leaky_relu_activation(conv_result, alpha);
}

/**
 * \brief Fused convolution + ReLU activation (tiled implementation).
 *
 * Uses shared memory tiling for better performance.
 */
extern "C" __global__ void conv_relu_tiled(const float* __restrict__ input,
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

    // Load input into shared memory
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
                float kVal = kernel[ky * ksize + kx];
                sum += static_cast<double>(imgVal) * static_cast<double>(kVal);
            }
        }
        
        float conv_result = static_cast<float>(sum);
        output[oy * width + ox] = relu_activation(conv_result);
    }
}

/**
 * \brief Fused convolution + Sigmoid activation (naive implementation).
 */
extern "C" __global__ void conv_sigmoid_naive(const float* __restrict__ input,
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
            sum += static_cast<double>(input[iy * width + ix]) * 
                   static_cast<double>(kernel[ky * ksize + kx]);
        }
    }
    
    float conv_result = static_cast<float>(sum);
    output[oy * width + ox] = sigmoid_activation(conv_result);
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

/**
 * \brief Launch naive fused conv+ReLU kernel.
 *
 * \param[in]  d_input  Device pointer to input image.
 * \param[in]  d_kernel Device pointer to convolution kernel weights.
 * \param[out] d_output Device pointer to output image.
 * \param[in]  width    Image width in pixels.
 * \param[in]  height   Image height in pixels.
 * \param[in]  ksize    Kernel side length.
 * \param[in]  block    CUDA block dimensions.
 */
extern "C" void launch_conv_relu_naive(const float* d_input,
                                       const float* d_kernel,
                                       float* d_output,
                                       int width, int height, int ksize,
                                       dim3 block = dim3(16, 16, 1)) {
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    conv_relu_naive<<<grid, block>>>(d_input, d_kernel, d_output, width, height, ksize);
    cudaDeviceSynchronize();
}

/**
 * \brief Launch naive fused conv+LeakyReLU kernel.
 *
 * \param[in]  d_input  Device pointer to input image.
 * \param[in]  d_kernel Device pointer to convolution kernel weights.
 * \param[out] d_output Device pointer to output image.
 * \param[in]  width    Image width in pixels.
 * \param[in]  height   Image height in pixels.
 * \param[in]  ksize    Kernel side length.
 * \param[in]  alpha    Negative slope for LeakyReLU.
 * \param[in]  block    CUDA block dimensions.
 */
extern "C" void launch_conv_leaky_relu_naive(const float* d_input,
                                              const float* d_kernel,
                                              float* d_output,
                                              int width, int height, int ksize,
                                              float alpha = 0.01f,
                                              dim3 block = dim3(16, 16, 1)) {
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    conv_leaky_relu_naive<<<grid, block>>>(d_input, d_kernel, d_output, width, height, ksize, alpha);
    cudaDeviceSynchronize();
}

/**
 * \brief Launch tiled fused conv+ReLU kernel with shared memory.
 *
 * \param[in]  d_input  Device pointer to input image.
 * \param[in]  d_kernel Device pointer to convolution kernel weights.
 * \param[out] d_output Device pointer to output image.
 * \param[in]  width    Image width in pixels.
 * \param[in]  height   Image height in pixels.
 * \param[in]  ksize    Kernel side length.
 * \param[in]  block    CUDA block dimensions controlling tile size.
 */
extern "C" void launch_conv_relu_tiled(const float* d_input,
                                       const float* d_kernel,
                                       float* d_output,
                                       int width, int height, int ksize,
                                       dim3 block = dim3(16, 16, 1)) {
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    size_t sharedMemBytes = (block.x + ksize - 1) * (block.y + ksize - 1) * sizeof(float);
    conv_relu_tiled<<<grid, block, sharedMemBytes>>>(d_input, d_kernel, d_output, width, height, ksize);
    cudaDeviceSynchronize();
}

/**
 * \brief Launch naive fused conv+Sigmoid kernel.
 *
 * \param[in]  d_input  Device pointer to input image.
 * \param[in]  d_kernel Device pointer to convolution kernel weights.
 * \param[out] d_output Device pointer to output image.
 * \param[in]  width    Image width in pixels.
 * \param[in]  height   Image height in pixels.
 * \param[in]  ksize    Kernel side length.
 * \param[in]  block    CUDA block dimensions.
 */
extern "C" void launch_conv_sigmoid_naive(const float* d_input,
                                          const float* d_kernel,
                                          float* d_output,
                                          int width, int height, int ksize,
                                          dim3 block = dim3(16, 16, 1)) {
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    conv_sigmoid_naive<<<grid, block>>>(d_input, d_kernel, d_output, width, height, ksize);
    cudaDeviceSynchronize();
}
