/**
 * \file fp8_quant.cu
 * \brief FP8 (E4M3) activation quantization kernels for LLM inference.
 *
 * Implements FP8 E4M3 (4-bit exponent, 3-bit mantissa) quantization using
 * CUDA's native `__nv_fp8_e4m3` type (requires CUDA 11.8+, Ada/Hopper GPUs).
 *
 * \par Format: E4M3
 * - 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits
 * - Range: ~±448.0, precision ~1/16 relative
 * - Special: 0x7F and 0xFF are NaN (no Inf representation)
 *
 * \par SmoothQuant compatibility
 * Per-channel activation smoothing migrates quantization difficulty from
 * activations to weights: x_smooth = x / smooth_scale, w_smooth = w * smooth_scale.
 * After smoothing both tensors are easier to quantize to INT8/FP8.
 *
 * \par Scaling strategies
 * - Per-tensor: single scale for entire activation tensor
 * - Per-token:  one scale per row (token) — better accuracy for activations
 * - Per-channel: one scale per column — better for weight quantization
 */

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <float.h>
#include <math.h>

// Maximum representable E4M3 value (448.0)
#define FP8_E4M3_MAX 448.0f

// ---------------------------------------------------------------------------
// Reduction helper (warp-level max)
// ---------------------------------------------------------------------------

/**
 * \brief Warp-level reduction to find maximum absolute value.
 * \param[in] val Per-thread value.
 * \return Maximum across all active warp lanes.
 */
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset >= 1; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

// ---------------------------------------------------------------------------
// Quantization kernels
// ---------------------------------------------------------------------------

/**
 * \brief Per-tensor FP32→FP8 quantization kernel.
 *
 * Applies a single scale to the whole tensor: q = clamp(x / scale, -E4M3_MAX, E4M3_MAX).
 *
 * \param[in]  input    FP32 input tensor [count].
 * \param[out] output   FP8 output stored as int8_t [count].
 * \param[in]  scale    Quantization scale (max_abs / E4M3_MAX).
 * \param[in]  count    Number of elements.
 */
__global__ void quantize_fp8_per_tensor_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    float scale,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    float scaled = input[idx] / scale;
    scaled = fmaxf(-FP8_E4M3_MAX, fminf(FP8_E4M3_MAX, scaled));
    __nv_fp8_e4m3 fp8_val = __nv_fp8_e4m3(scaled);
    output[idx] = *reinterpret_cast<const int8_t*>(&fp8_val);
}

/**
 * \brief Per-token FP32→FP8 quantization kernel.
 *
 * Each row (token) gets its own scale derived from its absolute maximum.
 * Grid: (rows,). Block: (cols, 1) — one block per token row.
 *
 * \param[in]  input    FP32 input [rows, cols].
 * \param[out] output   FP8 output as int8_t [rows, cols].
 * \param[out] scales   Per-token scales [rows].
 * \param[in]  rows     Number of tokens.
 * \param[in]  cols     Embedding dimension (must be ≤ 1024 for one block).
 */
__global__ void quantize_fp8_per_token_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int rows, int cols)
{
    int row  = blockIdx.x;
    int lane = threadIdx.x;
    if (row >= rows) return;

    const float* row_in  = input  + row * cols;
    int8_t*      row_out = output + row * cols;

    // Compute per-token max abs via warp reduction
    float local_max = 0.0f;
    for (int c = lane; c < cols; c += blockDim.x)
        local_max = fmaxf(local_max, fabsf(row_in[c]));

    float warp_max = warp_reduce_max(local_max);

    __shared__ float s_max;
    if (lane % 32 == 0) {
        // Each warp leader writes its max; use atomicMax via reinterpret
        atomicMax((int*)&s_max, __float_as_int(warp_max));
    }
    if (lane == 0) s_max = 0.0f;
    __syncthreads();
    if (lane % 32 == 0)
        atomicMax((int*)&s_max, __float_as_int(warp_max));
    __syncthreads();

    float max_abs = __int_as_float(*(int*)&s_max);
    float scale = (max_abs > 0.0f) ? (max_abs / FP8_E4M3_MAX) : 1.0f;
    if (lane == 0) scales[row] = scale;

    // Quantize
    for (int c = lane; c < cols; c += blockDim.x) {
        float scaled = row_in[c] / scale;
        scaled = fmaxf(-FP8_E4M3_MAX, fminf(FP8_E4M3_MAX, scaled));
        __nv_fp8_e4m3 fp8_val = __nv_fp8_e4m3(scaled);
        row_out[c] = *reinterpret_cast<const int8_t*>(&fp8_val);
    }
}

/**
 * \brief FP8→FP32 dequantization kernel.
 *
 * Reconstructs FP32 values: x = fp8_val * scale[token].
 *
 * \param[in]  input    FP8 values as int8_t [rows, cols].
 * \param[in]  scales   Per-token (or per-tensor) scales [rows or 1].
 * \param[out] output   FP32 output [rows, cols].
 * \param[in]  rows     Number of tokens.
 * \param[in]  cols     Embedding dimension.
 * \param[in]  per_token True if scales has one entry per row; false for per-tensor.
 */
__global__ void dequantize_fp8_kernel(
    const int8_t* __restrict__ input,
    const float*  __restrict__ scales,
    float* __restrict__ output,
    int rows, int cols, bool per_token)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;

    int row = idx / cols;
    float scale = per_token ? scales[row] : scales[0];

    int8_t raw = input[idx];
    __nv_fp8_e4m3 fp8_val = *reinterpret_cast<const __nv_fp8_e4m3*>(&raw);
    output[idx] = (float)fp8_val * scale;
}

/**
 * \brief SmoothQuant channel-wise scaling kernel.
 *
 * Applies per-channel smooth scales to activations before quantization:
 * output[row, col] = input[row, col] / smooth_scale[col].
 *
 * \param[in]  input         FP32 activations [rows, cols].
 * \param[in]  smooth_scales Per-channel scales [cols].
 * \param[out] output        Smoothed activations [rows, cols].
 * \param[in]  rows          Number of tokens.
 * \param[in]  cols          Embedding dimension.
 */
__global__ void smoothquant_scale_kernel(
    const float* __restrict__ input,
    const float* __restrict__ smooth_scales,
    float* __restrict__ output,
    int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int col = idx % cols;
    output[idx] = input[idx] / smooth_scales[col];
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

/**
 * \brief Quantize FP32 activations to FP8 (per-tensor or per-token).
 *
 * \param[in]  d_input    Device FP32 input [rows, cols].
 * \param[out] d_output   Device FP8 output as int8_t [rows, cols].
 * \param[out] d_scales   Device scales: [1] for per-tensor, [rows] for per-token.
 * \param[in]  rows       Number of tokens.
 * \param[in]  cols       Embedding dimension.
 * \param[in]  per_token  Per-token scaling when true; per-tensor otherwise.
 * \param[in]  stream     CUDA stream (0 = default).
 */
extern "C" void launch_quantize_fp8(
    const float* d_input,
    int8_t*      d_output,
    float*       d_scales,
    int rows, int cols,
    bool per_token,
    cudaStream_t stream)
{
    int total = rows * cols;
    if (per_token) {
        int block = (cols < 1024) ? cols : 1024;
        quantize_fp8_per_token_kernel<<<rows, block, 0, stream>>>(
            d_input, d_output, d_scales, rows, cols);
    } else {
        // Compute per-tensor scale from max abs on CPU side would require
        // device→host copy; for simplicity, caller provides scale in d_scales[0]
        float h_scale;
        cudaMemcpyAsync(&h_scale, d_scales, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        dim3 block(256);
        dim3 grid((total + 255) / 256);
        quantize_fp8_per_tensor_kernel<<<grid, block, 0, stream>>>(
            d_input, d_output, h_scale, total);
    }
    cudaStreamSynchronize(stream);
}

/**
 * \brief Dequantize FP8 activations back to FP32.
 *
 * \param[in]  d_input    Device FP8 input as int8_t [rows, cols].
 * \param[in]  d_scales   Device scales: [1] or [rows].
 * \param[out] d_output   Device FP32 output [rows, cols].
 * \param[in]  rows       Number of tokens.
 * \param[in]  cols       Embedding dimension.
 * \param[in]  per_token  Match per_token flag used during quantization.
 * \param[in]  stream     CUDA stream (0 = default).
 */
extern "C" void launch_dequantize_fp8(
    const int8_t* d_input,
    const float*  d_scales,
    float*        d_output,
    int rows, int cols,
    bool per_token,
    cudaStream_t stream)
{
    int total = rows * cols;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    dequantize_fp8_kernel<<<grid, block, 0, stream>>>(
        d_input, d_scales, d_output, rows, cols, per_token);
    cudaStreamSynchronize(stream);
}

/**
 * \brief Apply SmoothQuant per-channel scaling to activations.
 *
 * \param[in]  d_input         Device FP32 activations [rows, cols].
 * \param[in]  d_smooth_scales Device per-channel scales [cols].
 * \param[out] d_output        Device smoothed output [rows, cols].
 * \param[in]  rows            Number of tokens.
 * \param[in]  cols            Embedding dimension.
 * \param[in]  stream          CUDA stream (0 = default).
 */
extern "C" void launch_smoothquant_scale(
    const float* d_input,
    const float* d_smooth_scales,
    float*       d_output,
    int rows, int cols,
    cudaStream_t stream)
{
    int total = rows * cols;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    smoothquant_scale_kernel<<<grid, block, 0, stream>>>(
        d_input, d_smooth_scales, d_output, rows, cols);
    cudaStreamSynchronize(stream);
}
