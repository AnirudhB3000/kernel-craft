/**
 * \file depthwise_conv1d.cu
 * \brief Causal depthwise 1-D convolution for Mamba preprocessing.
 *
 * Applies a per-channel 1-D filter with causal (left-only) padding.
 * At each output position t the kernel reads inputs at positions
 * t-(d_conv-1), ..., t (positions below 0 are treated as zero).
 *
 * \par Tensor layout
 * Channels-first: input and output are [B, D, L].
 * Weight is [D, d_conv]; bias (optional) is [D].
 *
 * \par Performance
 * One thread computes one output element.  For d_conv=4, the inner loop
 * is 4 iterations with sequential global reads; for larger sequences the
 * memory access pattern is coalesced across threads in the L dimension.
 *
 * \par Usage in Mamba
 * The Mamba block applies this before the selective scan:
 * \code
 *   x_conv = depthwise_conv1d(x_proj, conv_weight, conv_bias)
 *   x_ssm  = silu(x_conv)
 * \endcode
 */

#include <cuda_runtime.h>

#define DW_BLOCK 256  ///< Threads per block

/**
 * \brief Causal depthwise conv1d kernel.
 *
 * Grid: ceil(B*D*L / DW_BLOCK) blocks.
 *
 * \param[in]  x      Input [B, D, L].
 * \param[in]  w      Weight [D, d_conv].
 * \param[in]  bias   Bias [D] or NULL.
 * \param[out] y      Output [B, D, L].
 * \param[in]  B_     Batch size.
 * \param[in]  D      Channel dimension.
 * \param[in]  L      Sequence length.
 * \param[in]  d_conv Convolution kernel width (typically 4).
 */
__global__ void depthwise_conv1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int B_, int D, int L, int d_conv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B_ * D * L) return;

    int t = idx % L;
    int d = (idx / L) % D;
    int b = idx / (D * L);

    const float* x_bd = x + (b * D + d) * L;
    const float* w_d  = w + d * d_conv;

    float val = 0.f;
    for (int k = 0; k < d_conv; ++k) {
        int src = t - (d_conv - 1 - k);  /* causal: look left only */
        if (src >= 0)
            val += x_bd[src] * w_d[k];
    }
    if (bias) val += bias[d];
    y[(b * D + d) * L + t] = val;
}

/**
 * \brief Launch causal depthwise conv1d.
 *
 * \param[in]  d_x      Device input [B, D, L].
 * \param[in]  d_w      Device weight [D, d_conv].
 * \param[in]  d_bias   Device bias [D] (may be NULL).
 * \param[out] d_y      Device output [B, D, L].
 * \param[in]  B_       Batch size.
 * \param[in]  D        Channel dimension.
 * \param[in]  L        Sequence length.
 * \param[in]  d_conv   Kernel width.
 * \param[in]  stream   CUDA stream.
 */
extern "C" void launch_depthwise_conv1d(
    const float* d_x, const float* d_w, const float* d_bias,
    float* d_y, int B_, int D, int L, int d_conv,
    cudaStream_t stream)
{
    int total  = B_ * D * L;
    int blocks = (total + DW_BLOCK - 1) / DW_BLOCK;
    depthwise_conv1d_kernel<<<blocks, DW_BLOCK, 0, stream>>>(
        d_x, d_w, d_bias, d_y, B_, D, L, d_conv);
}
