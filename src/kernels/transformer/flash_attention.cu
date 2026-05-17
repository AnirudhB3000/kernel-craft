/**
 * \file flash_attention.cu
 * \brief FlashAttention: memory-efficient scaled dot-product attention.
 *
 * Implements the tiled online-softmax algorithm (Dao et al., 2022) that avoids
 * materializing the full N×N attention matrix. Streams K/V tiles through shared
 * memory while maintaining running softmax statistics (m, l) per Q row.
 *
 * \par Supported variants
 * - Multi-Head Attention (MHA): num_kv_heads == num_q_heads
 * - Grouped-Query Attention (GQA): num_q_heads divisible by num_kv_heads
 * - Multi-Query Attention (MQA): num_kv_heads == 1
 *
 * \par Tensor layout
 * All tensors row-major: [B, H, N, d] (batch, heads, seq_len, head_dim).
 *
 * \par Constraints
 * - head_dim d must be 32, 64, or 128 (compile-time template dispatch)
 * - FLASH_BR Q-rows per block, FLASH_BC K/V-rows per tile iteration
 */

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define FLASH_BR 16  ///< Q rows processed per thread block
#define FLASH_BC 16  ///< K/V rows loaded per tile iteration

/**
 * \brief Core FlashAttention kernel templated on head dimension D.
 *
 * Grid: (ceil(N/BR), H, B). Block: (BR, 1, 1) — one thread per Q row in tile.
 * Each thread independently tracks its online softmax state in registers.
 *
 * \param[in]  Q      Query tensor [B, H, N, D].
 * \param[in]  K      Key tensor [B, H_kv, N, D].
 * \param[in]  V      Value tensor [B, H_kv, N, D].
 * \param[out] O      Output tensor [B, H, N, D] (pre-zeroed by caller).
 * \param[in]  B      Batch size.
 * \param[in]  H      Number of query heads.
 * \param[in]  H_kv   Number of KV heads (H % H_kv == 0).
 * \param[in]  N      Sequence length.
 * \param[in]  scale  Attention scale factor (1/sqrt(D)).
 * \param[in]  causal Apply causal lower-triangular mask when true.
 */
template <int D>
__global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int H, int H_kv, int N,
    float scale, bool causal)
{
    int b     = blockIdx.z;
    int h     = blockIdx.y;
    int q_blk = blockIdx.x;
    int lane  = threadIdx.x;  // [0, FLASH_BR)

    int q_row = q_blk * FLASH_BR + lane;
    // GQA/MQA head mapping: multiple Q heads share one KV head
    int h_kv  = h * H_kv / H;

    const float* Qb = Q + ((long long)(b * H      + h   ) * N * D);
    const float* Kb = K + ((long long)(b * H_kv   + h_kv) * N * D);
    const float* Vb = V + ((long long)(b * H_kv   + h_kv) * N * D);
    float*       Ob = O + ((long long)(b * H      + h   ) * N * D);

    // Online softmax state in registers
    float m = -FLT_MAX;
    float l = 0.0f;
    float acc[D];
    #pragma unroll
    for (int i = 0; i < D; ++i) acc[i] = 0.0f;

    // Load scaled Q row into registers
    float q_reg[D];
    bool valid_q = (q_row < N);
    #pragma unroll
    for (int i = 0; i < D; ++i)
        q_reg[i] = valid_q ? Qb[(long long)q_row * D + i] * scale : 0.0f;

    // Shared memory tiles for K and V
    __shared__ float K_tile[FLASH_BC][D];
    __shared__ float V_tile[FLASH_BC][D];

    int num_kv_tiles = (N + FLASH_BC - 1) / FLASH_BC;

    for (int kv_blk = 0; kv_blk < num_kv_tiles; ++kv_blk) {
        // Cooperative load: BR threads cover BC rows (each loads BC/BR rows)
        for (int r = lane; r < FLASH_BC; r += FLASH_BR) {
            int kv_row = kv_blk * FLASH_BC + r;
            if (kv_row < N) {
                #pragma unroll
                for (int i = 0; i < D; ++i) {
                    K_tile[r][i] = Kb[(long long)kv_row * D + i];
                    V_tile[r][i] = Vb[(long long)kv_row * D + i];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < D; ++i) {
                    K_tile[r][i] = 0.0f;
                    V_tile[r][i] = 0.0f;
                }
            }
        }
        __syncthreads();

        if (!valid_q) { __syncthreads(); continue; }

        // Compute S[j] = q_reg · K_tile[j] and track tile max
        float s[FLASH_BC];
        float tile_max = -FLT_MAX;

        #pragma unroll
        for (int j = 0; j < FLASH_BC; ++j) {
            int kv_row = kv_blk * FLASH_BC + j;
            if (kv_row >= N || (causal && kv_row > q_row)) {
                s[j] = -FLT_MAX;
                continue;
            }
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < D; ++i)
                dot += q_reg[i] * K_tile[j][i];
            s[j] = dot;
            if (dot > tile_max) tile_max = dot;
        }

        // Online softmax: merge tile statistics into running (m, l, acc)
        if (tile_max > -FLT_MAX) {
            float m_new   = fmaxf(m, tile_max);
            float rescale = expf(m - m_new);
            l *= rescale;
            #pragma unroll
            for (int i = 0; i < D; ++i) acc[i] *= rescale;

            #pragma unroll
            for (int j = 0; j < FLASH_BC; ++j) {
                if (s[j] == -FLT_MAX) continue;
                float p = expf(s[j] - m_new);
                l += p;
                #pragma unroll
                for (int i = 0; i < D; ++i)
                    acc[i] += p * V_tile[j][i];
            }
            m = m_new;
        }

        __syncthreads();
    }

    // Normalize accumulator and write output
    if (valid_q) {
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        #pragma unroll
        for (int i = 0; i < D; ++i)
            Ob[(long long)q_row * D + i] = acc[i] * inv_l;
    }
}

/**
 * \brief Launch FlashAttention, dispatching on head dimension d.
 *
 * \param[in]  Q      Device query tensor [B, H, N, d].
 * \param[in]  K      Device key tensor [B, H_kv, N, d].
 * \param[in]  V      Device value tensor [B, H_kv, N, d].
 * \param[out] O      Device output tensor [B, H, N, d].
 * \param[in]  B      Batch size.
 * \param[in]  H      Number of query heads.
 * \param[in]  H_kv   Number of KV heads (H_kv ≤ H; must divide H evenly).
 * \param[in]  N      Sequence length.
 * \param[in]  d      Head dimension (32, 64, or 128).
 * \param[in]  causal True for causal (decoder) attention.
 * \param[in]  stream CUDA stream (0 = default stream).
 */
extern "C" void launch_flash_attention(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int H_kv, int N, int d,
    bool causal, cudaStream_t stream)
{
    float scale = 1.0f / sqrtf((float)d);
    int num_q_tiles = (N + FLASH_BR - 1) / FLASH_BR;
    dim3 grid(num_q_tiles, H, B);
    dim3 block(FLASH_BR, 1, 1);

    switch (d) {
        case 32:
            flash_attention_kernel<32><<<grid, block, 0, stream>>>(
                Q, K, V, O, B, H, H_kv, N, scale, causal);
            break;
        case 64:
            flash_attention_kernel<64><<<grid, block, 0, stream>>>(
                Q, K, V, O, B, H, H_kv, N, scale, causal);
            break;
        case 128:
            flash_attention_kernel<128><<<grid, block, 0, stream>>>(
                Q, K, V, O, B, H, H_kv, N, scale, causal);
            break;
        default:
            break;
    }
    cudaStreamSynchronize(stream);
}
