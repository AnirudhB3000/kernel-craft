/**
 * \file selective_scan.cu
 * \brief Mamba-1 selective state-space model (SSM) scan kernel.
 *
 * Implements the zero-order hold (ZOH) discretized SSM recurrence:
 *
 * \code
 *   A_bar[b,t,d,n] = exp(delta[b,t,d] * A_log[d,n])
 *   B_bar[b,t,d,n] = delta[b,t,d] * B[b,t,n]
 *   h[b,t,d,n]     = A_bar[b,t,d,n] * h[b,t-1,d,n] + B_bar[b,t,d,n] * u[b,t,d]
 *   y[b,t,d]       = sum_n C[b,t,n] * h[b,t,d,n]
 * \endcode
 *
 * \par Parallelism
 * Each (batch, channel) pair (b, d) is processed by one thread. The B*D pairs
 * run in parallel, giving O(B*D) GPU concurrency. Within each pair the L
 * timesteps run sequentially; the N_state hidden values are kept in registers.
 *
 * \par Constraints
 * N_state must be ≤ SS_MAX_N (32). For Mamba-1 the default N_state=16 fits
 * comfortably.
 *
 * \par Tensor layouts (all row-major)
 * - u:     [B, L, D]
 * - A_log: [D, N_state]  (shared across batch and time)
 * - B:     [B, L, N_state]
 * - C:     [B, L, N_state]
 * - delta: [B, L, D]
 * - y:     [B, L, D]
 */

#include <cuda_runtime.h>
#include <math.h>

#define SS_BLOCK  256   ///< Threads per block (one thread = one (b, d) pair)
#define SS_MAX_N   32   ///< Maximum supported state dimension

/**
 * \brief Selective scan kernel.
 *
 * \param[in]  u       Input [B, L, D].
 * \param[in]  A_log   Log-scale state decay [D, N_state].
 * \param[in]  B       Input projection [B, L, N_state].
 * \param[in]  C       Output projection [B, L, N_state].
 * \param[in]  delta   Timestep [B, L, D].
 * \param[out] y       Output [B, L, D].
 * \param[in]  B_      Batch size.
 * \param[in]  L       Sequence length.
 * \param[in]  D       Channel dimension.
 * \param[in]  N_state Hidden state dimension (≤ SS_MAX_N).
 */
__global__ void selective_scan_kernel(
    const float* __restrict__ u,
    const float* __restrict__ A_log,
    const float* __restrict__ B,
    const float* __restrict__ C,
    const float* __restrict__ delta,
    float* __restrict__ y,
    int B_, int L, int D, int N_state)
{
    int bd = blockIdx.x * blockDim.x + threadIdx.x;
    if (bd >= B_ * D) return;
    int b = bd / D;
    int d = bd % D;

    /* Hidden state kept in registers for N_state ≤ SS_MAX_N */
    float h[SS_MAX_N];
    for (int n = 0; n < N_state; ++n) h[n] = 0.f;

    for (int t = 0; t < L; ++t) {
        float dt     = delta[(b * L + t) * D + d];
        float u_td   = u    [(b * L + t) * D + d];
        float y_td   = 0.f;

        const float* Bt = B + (b * L + t) * N_state;
        const float* Ct = C + (b * L + t) * N_state;
        const float* Ad = A_log + d * N_state;

        for (int n = 0; n < N_state; ++n) {
            float a_bar = expf(dt * Ad[n]);
            float b_bar = dt * Bt[n];
            h[n]  = a_bar * h[n] + b_bar * u_td;
            y_td += Ct[n] * h[n];
        }
        y[(b * L + t) * D + d] = y_td;
    }
}

/**
 * \brief Launch selective scan.
 *
 * \param[in]  d_u       Device input [B, L, D].
 * \param[in]  d_A_log   Device log-scale decay [D, N_state].
 * \param[in]  d_B       Device input projection [B, L, N_state].
 * \param[in]  d_C       Device output projection [B, L, N_state].
 * \param[in]  d_delta   Device timestep [B, L, D].
 * \param[out] d_y       Device output [B, L, D].
 * \param[in]  B_        Batch size.
 * \param[in]  L         Sequence length.
 * \param[in]  D         Channel dimension.
 * \param[in]  N_state   Hidden state dimension (≤ 32).
 * \param[in]  stream    CUDA stream.
 */
extern "C" void launch_selective_scan(
    const float* d_u,     const float* d_A_log,
    const float* d_B,     const float* d_C,
    const float* d_delta, float* d_y,
    int B_, int L, int D, int N_state,
    cudaStream_t stream)
{
    int total  = B_ * D;
    int blocks = (total + SS_BLOCK - 1) / SS_BLOCK;
    selective_scan_kernel<<<blocks, SS_BLOCK, 0, stream>>>(
        d_u, d_A_log, d_B, d_C, d_delta, d_y, B_, L, D, N_state);
}
