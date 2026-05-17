/**
 * \file quant_int4.cu
 * \brief INT4 weight quantization kernels for LLM inference.
 *
 * Implements GPTQ/AWQ-style INT4 weight quantization:
 * - Two INT4 values are packed per byte (low nibble = first value)
 * - Scales are stored per group of `group_size` elements (typically 128)
 * - Dequantization: w_fp32 = (w_int4 - zero_point) * scale
 *
 * \par Memory layout
 * packed_weights: [rows, cols/2] — two INT4 per byte, row-major
 * scales:         [rows, cols/group_size] — one FP32 scale per group
 * zeros:          [rows, cols/group_size] — one INT4 zero-point per group
 *
 * \par Key insight
 * INT4 reduces weight memory by 8× vs FP32 and 2× vs INT8. Combined with
 * Tensor Core GEMM on dequantized weights, this enables high-throughput
 * LLM inference at minimal accuracy cost (<1% degradation with group_size=128).
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

/**
 * \brief Unpack low nibble (bits 3:0) from a packed byte.
 * \param[in] packed Packed byte containing two INT4 values.
 * \return Signed INT4 in range [-8, 7].
 */
__device__ __forceinline__ int8_t unpack_low_nibble(uint8_t packed) {
    int8_t v = static_cast<int8_t>(packed & 0x0F);
    // Sign-extend from 4 bits to 8 bits
    return (v & 0x08) ? (v | 0xF0) : v;
}

/**
 * \brief Unpack high nibble (bits 7:4) from a packed byte.
 * \param[in] packed Packed byte containing two INT4 values.
 * \return Signed INT4 in range [-8, 7].
 */
__device__ __forceinline__ int8_t unpack_high_nibble(uint8_t packed) {
    int8_t v = static_cast<int8_t>((packed >> 4) & 0x0F);
    return (v & 0x08) ? (v | 0xF0) : v;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/**
 * \brief Dequantize INT4 packed weights to FP32.
 *
 * Each thread handles one pair of INT4 values (one packed byte).
 * Grid: (cols/2 / blockDim.x, rows). Block: (256, 1).
 *
 * \param[in]  packed    Packed INT4 weights [rows, cols/2].
 * \param[in]  scales    FP32 scales [rows, cols/group_size].
 * \param[in]  zeros     INT4 zero points [rows, cols/group_size], packed same as weights.
 * \param[out] output    Dequantized FP32 weights [rows, cols].
 * \param[in]  rows      Number of output rows.
 * \param[in]  cols      Number of output columns (even).
 * \param[in]  group_size Elements sharing one scale (typically 128).
 */
__global__ void dequantize_int4_kernel(
    const uint8_t* __restrict__ packed,
    const float*   __restrict__ scales,
    const uint8_t* __restrict__ zeros,
    float* __restrict__ output,
    int rows, int cols, int group_size)
{
    int byte_col = blockIdx.x * blockDim.x + threadIdx.x;  // index into cols/2
    int row      = blockIdx.y;
    if (row >= rows || byte_col * 2 >= cols) return;

    int col0 = byte_col * 2;
    int col1 = col0 + 1;

    // Load packed byte
    uint8_t p = packed[row * (cols / 2) + byte_col];

    // Determine group and load scale/zero
    int grp0 = col0 / group_size;
    int grp1 = col1 / group_size;

    float scale0 = scales[row * (cols / group_size) + grp0];
    float scale1 = scales[row * (cols / group_size) + grp1];

    // Unpack zero points (stored as unsigned 0-15, symmetric around 8)
    uint8_t z_byte0 = zeros[row * (cols / group_size / 2) + grp0 / 2];
    uint8_t z_byte1 = zeros[row * (cols / group_size / 2) + grp1 / 2];
    int8_t  zp0 = (grp0 % 2 == 0) ? (z_byte0 & 0x0F) : (z_byte0 >> 4);
    int8_t  zp1 = (grp1 % 2 == 0) ? (z_byte1 & 0x0F) : (z_byte1 >> 4);

    // Dequantize: w = (w_int4 - zero_point) * scale
    int8_t w0 = unpack_low_nibble(p);
    int8_t w1 = unpack_high_nibble(p);

    output[row * cols + col0] = (float)(w0 - zp0) * scale0;
    if (col1 < cols)
        output[row * cols + col1] = (float)(w1 - zp1) * scale1;
}

// GEMV block size: fixed 256 threads (each loops over multiple bytes for wide cols)
#define GEMV_BLOCK 256

/**
 * \brief INT4-quantized matrix-vector multiplication: y = W_int4 * x.
 *
 * Dequantizes on-the-fly during the dot product, avoiding storing FP32 weights.
 * Each thread block computes one output element y[row].
 * Grid: (rows,). Block: (GEMV_BLOCK, 1). Each thread strides over cols/2 bytes.
 *
 * \param[in]  packed    Packed INT4 weights [rows, cols/2].
 * \param[in]  scales    FP32 scales [rows, cols/group_size].
 * \param[in]  zeros     INT4 zero points [rows, cols/group_size], packed.
 * \param[in]  x         Input vector [cols] (FP32).
 * \param[out] y         Output vector [rows] (FP32).
 * \param[in]  rows      Weight matrix rows.
 * \param[in]  cols      Weight matrix columns (must be even).
 * \param[in]  group_size Elements sharing one scale.
 */
__global__ void gemv_int4_kernel(
    const uint8_t* __restrict__ packed,
    const float*   __restrict__ scales,
    const uint8_t* __restrict__ zeros,
    const float*   __restrict__ x,
    float* __restrict__ y,
    int rows, int cols, int group_size)
{
    int row  = blockIdx.x;
    int lane = threadIdx.x;
    if (row >= rows) return;

    __shared__ float s_partial[GEMV_BLOCK];

    float partial = 0.0f;
    int num_bytes = cols / 2;
    // Each thread strides over its share of packed bytes
    for (int byte_idx = lane; byte_idx < num_bytes; byte_idx += GEMV_BLOCK) {
        int col0 = byte_idx * 2;
        uint8_t p = packed[row * num_bytes + byte_idx];
        int grp0 = col0 / group_size;
        int grp1 = (col0 + 1) / group_size;

        float scale0 = scales[row * (cols / group_size) + grp0];
        float scale1 = scales[row * (cols / group_size) + grp1];

        int zc = cols / group_size / 2;
        uint8_t z_byte0 = zeros[row * zc + grp0 / 2];
        uint8_t z_byte1 = zeros[row * zc + grp1 / 2];
        int8_t  zp0 = (grp0 % 2 == 0) ? (z_byte0 & 0x0F) : (z_byte0 >> 4);
        int8_t  zp1 = (grp1 % 2 == 0) ? (z_byte1 & 0x0F) : (z_byte1 >> 4);

        float w0 = (float)(unpack_low_nibble(p)  - zp0) * scale0;
        float w1 = (float)(unpack_high_nibble(p) - zp1) * scale1;

        partial += w0 * x[col0];
        if (col0 + 1 < cols)
            partial += w1 * x[col0 + 1];
    }
    s_partial[lane] = partial;
    __syncthreads();

    // Tree reduction within GEMV_BLOCK threads
    for (int stride = GEMV_BLOCK / 2; stride >= 1; stride >>= 1) {
        if (lane < stride)
            s_partial[lane] += s_partial[lane + stride];
        __syncthreads();
    }
    if (lane == 0) y[row] = s_partial[0];
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

/**
 * \brief Dequantize INT4 packed weights to FP32 on the device.
 *
 * \param[in]  d_packed     Device packed INT4 weights [rows, cols/2].
 * \param[in]  d_scales     Device FP32 scales [rows, cols/group_size].
 * \param[in]  d_zeros      Device packed zero points [rows, cols/group_size/2].
 * \param[out] d_output     Device output FP32 weights [rows, cols].
 * \param[in]  rows         Weight matrix rows.
 * \param[in]  cols         Weight matrix columns (even).
 * \param[in]  group_size   Elements per quantization group.
 * \param[in]  stream       CUDA stream (0 = default).
 */
extern "C" void launch_dequantize_int4(
    const uint8_t* d_packed,
    const float*   d_scales,
    const uint8_t* d_zeros,
    float*         d_output,
    int rows, int cols, int group_size,
    cudaStream_t stream)
{
    dim3 block(256, 1);
    dim3 grid((cols / 2 + 255) / 256, rows);
    dequantize_int4_kernel<<<grid, block, 0, stream>>>(
        d_packed, d_scales, d_zeros, d_output, rows, cols, group_size);
    cudaStreamSynchronize(stream);
}

/**
 * \brief INT4 weight GEMV: y = W_int4 * x (dequantize on the fly).
 *
 * \param[in]  d_packed     Device packed INT4 weights [rows, cols/2].
 * \param[in]  d_scales     Device FP32 scales [rows, cols/group_size].
 * \param[in]  d_zeros      Device packed zero points [rows, cols/group_size/2].
 * \param[in]  d_x          Device input vector [cols].
 * \param[out] d_y          Device output vector [rows].
 * \param[in]  rows         Weight rows.
 * \param[in]  cols         Weight cols (even; cols/2 threads per block).
 * \param[in]  group_size   Elements per quantization group.
 * \param[in]  stream       CUDA stream (0 = default).
 */
extern "C" void launch_gemv_int4(
    const uint8_t* d_packed,
    const float*   d_scales,
    const uint8_t* d_zeros,
    const float*   d_x,
    float*         d_y,
    int rows, int cols, int group_size,
    cudaStream_t stream)
{
    size_t shared = GEMV_BLOCK * sizeof(float);
    gemv_int4_kernel<<<rows, GEMV_BLOCK, shared, stream>>>(
        d_packed, d_scales, d_zeros, d_x, d_y, rows, cols, group_size);
    cudaStreamSynchronize(stream);
}
