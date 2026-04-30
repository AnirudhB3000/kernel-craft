/**
 * \file conv_int8.cu
 * \brief INT8 quantized 2‑D convolution for CNN inference.
 *
 * This implementation demonstrates quantization-aware convolution where
 * weights and activations are quantized to INT8 for faster inference
 * on hardware with INT8 Tensor Core support (Turing+).
 *
 * \par Quantization Flow
 * 1. Quantize float input/weights to INT8 using scale factors
 * 2. Perform convolution in INT8 (with INT32 accumulation)
 * 3. Dequantize output back to float
 *
 * \par Key Benefits
 * - 4x memory bandwidth reduction vs FP32
 * - 2-4x compute throughput on Tensor Cores
 * - Minimal accuracy loss (~0.1-1% for CNNs)
 */

#include <cuda_runtime.h>
#include <cmath>

// ---------------------------------------------------------------------------
// Helper: Quantize float to INT8 with scaling
// ---------------------------------------------------------------------------
__device__ __forceinline__ int8_t quantize_float_to_int8(float val, float scale, float zero_point = 0.0f) {
    float q = val / scale + zero_point;
    q = fmaxf(-128.0f, fminf(127.0f, q));
    return static_cast<int8_t>(static_cast<int>(roundf(q)));
}

// ---------------------------------------------------------------------------
// Helper: Dequantize INT8 to float
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dequantize_int8_to_float(int8_t val, float scale, float zero_point = 0.0f) {
    return (static_cast<float>(val) - zero_point) * scale;
}

// ---------------------------------------------------------------------------
// Helper: Compute maximum absolute value for symmetric quantization
// ---------------------------------------------------------------------------
__device__ float block_max_abs(const float* data, int size) {
    extern __shared__ float shared_max[];
    float local_max = 0.0f;
    
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < size; i += blockDim.x * blockDim.y) {
        local_max = fmaxf(local_max, fabsf(data[i]));
    }
    
    shared_max[threadIdx.y * blockDim.x + threadIdx.x] = local_max;
    __syncthreads();
    
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        float max_val = 0.0f;
        for (int i = 0; i < blockDim.x * blockDim.y; ++i) {
            max_val = fmaxf(max_val, shared_max[i]);
        }
        shared_max[0] = max_val;
    }
    __syncthreads();
    
    return shared_max[0];
}

/**
 * \brief INT8 quantized convolution kernel (naive, per-tensor quantization).
 *
 * \param[in] input  Input image (row‑major, float).
 * \param[in] kernel Convolution kernel (row‑major, float).
 * \param[out] output Output image buffer (row‑major, float).
 * \param[in] width  Image width.
 * \param[in] height Image height.
 * \param[in] ksize Kernel size (must be odd).
 * \param[in] input_scale  Scale factor for input quantization.
 * \param[in] kernel_scale Scale factor for kernel quantization.
 * \param[in] output_scale Scale factor for output dequantization.
 */
extern "C" __global__ void conv_int8_naive(const float* __restrict__ input,
                                           const float* __restrict__ kernel,
                                           float* __restrict__ output,
                                           int width, int height, int ksize,
                                           float input_scale,
                                           float kernel_scale,
                                           float output_scale) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int kHalf = ksize / 2;

    if (ox >= width || oy >= height) return;

    // INT32 accumulator (standard for INT8 conv)
    int32_t sum = 0;

    // Perform convolution in INT8
    for (int ky = 0; ky < ksize; ++ky) {
        int iy = oy + ky - kHalf;
        if (iy < 0 || iy >= height) continue;
        for (int kx = 0; kx < ksize; ++kx) {
            int ix = ox + kx - kHalf;
            if (ix < 0 || ix >= width) continue;

            // Quantize to INT8 on-the-fly (in practice, pre-quantized)
            float in_val = input[iy * width + ix];
            float k_val = kernel[ky * ksize + kx];
            
            int8_t in_q = quantize_float_to_int8(in_val, input_scale);
            int8_t k_q = quantize_float_to_int8(k_val, kernel_scale);

            sum += static_cast<int32_t>(in_q) * static_cast<int32_t>(k_q);
        }
    }

    // Dequantize: output = sum * input_scale * kernel_scale / output_scale
    float dequant = static_cast<float>(sum) * input_scale * kernel_scale / output_scale;
    output[oy * width + ox] = dequant;
}

/**
 * \brief INT8 tiled convolution kernel with shared memory.
 *
 * Similar to conv_tiled but operates in INT8 with quantized values
 * stored in shared memory.
 */
extern "C" __global__ void conv_int8_tiled(const float* __restrict__ input,
                                           const float* __restrict__ kernel,
                                           float* __restrict__ output,
                                           int width, int height, int ksize,
                                           float input_scale,
                                           float kernel_scale,
                                           float output_scale) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int ox = blockIdx.x * blockDim.x + tx;
    const int oy = blockIdx.y * blockDim.y + ty;
    const int kHalf = ksize / 2;

    // Shared memory for quantized tiles (INT8)
    const int sharedW = blockDim.x + ksize - 1;
    const int sharedH = blockDim.y + ksize - 1;
    extern __shared__ int8_t tile[];

    // Load and quantize input to shared memory
    for (int dy = ty; dy < sharedH; dy += blockDim.y) {
        for (int dx = tx; dx < sharedW; dx += blockDim.x) {
            int img_x = blockIdx.x * blockDim.x + dx - kHalf;
            int img_y = blockIdx.y * blockDim.y + dy - kHalf;
            
            float val = 0.0f;
            if (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height) {
                val = input[img_y * width + img_x];
            }
            tile[dy * sharedW + dx] = quantize_float_to_int8(val, input_scale);
        }
    }
    __syncthreads();

    if (ox < width && oy < height) {
        int32_t sum = 0;
        
        // Pre-quantize kernel values (in practice, done once on CPU)
        for (int ky = 0; ky < ksize; ++ky) {
            for (int kx = 0; kx < ksize; ++kx) {
                int8_t img_q = tile[(ty + ky) * sharedW + (tx + kx)];
                int8_t k_q = quantize_float_to_int8(kernel[ky * ksize + kx], kernel_scale);
                sum += static_cast<int32_t>(img_q) * static_cast<int32_t>(k_q);
            }
        }
        
        float dequant = static_cast<float>(sum) * input_scale * kernel_scale / output_scale;
        output[oy * width + ox] = dequant;
    }
}

/**
 * \brief Host launcher for INT8 naive convolution.
 *
 * \param[in] d_input  Device pointer to input image.
 * \param[in] d_kernel Device pointer to convolution kernel.
 * \param[out] d_output Device pointer to output buffer.
 * \param[in] width   Image width.
 * \param[in] height  Image height.
 * \param[in] ksize  Kernel size (must be odd).
 * \param[in] input_scale  Scale factor for input quantization.
 * \param[in] kernel_scale Scale factor for kernel quantization.
 * \param[in] output_scale Scale factor for output dequantization.
 * \param[in] block  Block dimensions (default 16×16).
 */
extern "C" void launch_conv_int8_naive(const float* d_input,
                                       const float* d_kernel,
                                       float* d_output,
                                       int width, int height, int ksize,
                                       float input_scale,
                                       float kernel_scale,
                                       float output_scale,
                                       dim3 block = dim3(16, 16, 1)) {
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    conv_int8_naive<<<grid, block>>>(d_input, d_kernel, d_output,
                                     width, height, ksize,
                                     input_scale, kernel_scale, output_scale);
    cudaDeviceSynchronize();
}

/**
 * \brief Host launcher for INT8 tiled convolution.
 *
 * \param[in] d_input  Device pointer to input image.
 * \param[in] d_kernel Device pointer to convolution kernel.
 * \param[out] d_output Device pointer to output buffer.
 * \param[in] width   Image width.
 * \param[in] height  Image height.
 * \param[in] ksize  Kernel size (must be odd).
 * \param[in] input_scale  Scale factor for input quantization.
 * \param[in] kernel_scale Scale factor for kernel quantization.
 * \param[in] output_scale Scale factor for output dequantization.
 * \param[in] block  Block dimensions (default 16×16).
 */
extern "C" void launch_conv_int8_tiled(const float* d_input,
                                       const float* d_kernel,
                                       float* d_output,
                                       int width, int height, int ksize,
                                       float input_scale,
                                       float kernel_scale,
                                       float output_scale,
                                       dim3 block = dim3(16, 16, 1)) {
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    size_t sharedMemBytes = (block.x + ksize - 1) * (block.y + ksize - 1) * sizeof(int8_t);
    conv_int8_tiled<<<grid, block, sharedMemBytes>>>(d_input, d_kernel, d_output,
                                                     width, height, ksize,
                                                     input_scale, kernel_scale, output_scale);
    cudaDeviceSynchronize();
}

/**
 * \brief Compute symmetric quantization scale for a tensor.
 *
 * For symmetric quantization: scale = max_abs / 127.0
 *
 * \param[in] h_data Host pointer to float data.
 * \param[in] size Number of elements.
 * \return Scale factor for INT8 quantization.
 */
extern "C" float compute_quantization_scale(const float* h_data, int size) {
    float max_abs = 0.0f;
    for (int i = 0; i < size; ++i) {
        max_abs = fmaxf(max_abs, fabsf(h_data[i]));
    }
    return max_abs / 127.0f;
}
