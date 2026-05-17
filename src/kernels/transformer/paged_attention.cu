/**
 * \file paged_attention.cu
 * \brief PagedAttention: attention over non-contiguous paged KV caches.
 *
 * Implements the vLLM PagedAttention algorithm (Kwon et al., 2023).
 * The KV cache is partitioned into fixed-size physical blocks (pages) that
 * are mapped to logical positions via a block table. This enables
 * memory-efficient serving with variable-length sequences.
 *
 * \par Tensor layouts
 * - Q:          [B, H, d]   — single decode-step query (one token per sequence)
 * - K_pool:     [num_phys_blocks, block_size, H_kv, d] — physical KV pool
 * - V_pool:     [num_phys_blocks, block_size, H_kv, d]
 * - block_table:[B, max_blocks] — logical→physical block index mapping
 * - seq_lens:   [B]          — actual sequence lengths per batch item
 * - O:          [B, H, d]   — output (one vector per sequence head)
 *
 * \par Algorithm
 * For each (batch, head): stream through all logical KV blocks via the block
 * table, computing dot-product attention with online softmax (no O(N²) memory).
 *
 * \par Supported head dims
 * d = 32, 64, or 128 (compile-time template dispatch).
 */

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

/**
 * \brief PagedAttention decode kernel templated on head dimension D.
 *
 * Grid: (H, B). Block: (D, 1, 1) — one thread per head-dim element.
 * Shared memory: reduction buffer for Q·K dot products.
 *
 * \param[in]  Q           Device query [B, H, D].
 * \param[in]  block_table Device block table [B, max_blocks].
 * \param[in]  K_pool      Device KV pool keys [num_phys, block_size, H_kv, D].
 * \param[in]  V_pool      Device KV pool values [num_phys, block_size, H_kv, D].
 * \param[out] O           Device output [B, H, D].
 * \param[in]  seq_lens    Device sequence lengths [B].
 * \param[in]  H           Number of query heads.
 * \param[in]  H_kv        Number of KV heads.
 * \param[in]  block_size  Tokens per physical page.
 * \param[in]  max_blocks  Max logical blocks per sequence (block_table cols).
 * \param[in]  scale       Attention scale (1/sqrt(D)).
 */
template <int D>
__global__ void paged_attention_kernel(
    const float* __restrict__ Q,
    const int*   __restrict__ block_table,
    const float* __restrict__ K_pool,
    const float* __restrict__ V_pool,
    float* __restrict__ O,
    const int* __restrict__ seq_lens,
    int H, int H_kv,
    int block_size, int max_blocks,
    float scale)
{
    int b    = blockIdx.y;
    int h    = blockIdx.x;
    int lane = threadIdx.x;  // [0, D)

    int h_kv   = h * H_kv / H;
    int seq_len = seq_lens[b];

    // Each thread loads its element of the query (scaled)
    float q_val = Q[(b * H + h) * D + lane] * scale;

    // Shared reduction buffer for the Q·K dot product
    __shared__ float s_partial[D];

    // Online softmax state
    float m   = -FLT_MAX;
    float l   = 0.0f;
    float acc = 0.0f;  // per-thread accumulator for dimension `lane`

    int num_log_blocks = (seq_len + block_size - 1) / block_size;

    for (int blk_idx = 0; blk_idx < num_log_blocks; ++blk_idx) {
        int phys_block   = block_table[b * max_blocks + blk_idx];
        int tokens_in_blk = (blk_idx == num_log_blocks - 1)
                            ? (seq_len - blk_idx * block_size)
                            : block_size;

        for (int tok = 0; tok < tokens_in_blk; ++tok) {
            // Physical KV offset: [phys_block, tok, h_kv, :]
            long long kv_base = ((long long)phys_block * block_size + tok) * H_kv + h_kv;
            const float* k = K_pool + kv_base * D;
            const float* v = V_pool + kv_base * D;

            // Parallel dot product: each thread contributes q_val[lane]*k[lane]
            s_partial[lane] = q_val * k[lane];
            __syncthreads();

            // Parallel tree reduction to get full dot product
            for (int stride = D / 2; stride >= 1; stride >>= 1) {
                if (lane < stride)
                    s_partial[lane] += s_partial[lane + stride];
                __syncthreads();
            }
            float qk = s_partial[0];  // broadcast after reduction

            // Online softmax update
            float m_new   = fmaxf(m, qk);
            float p       = expf(qk - m_new);
            float rescale = expf(m - m_new);
            acc = acc * rescale + p * v[lane];
            l   = l   * rescale + p;
            m   = m_new;
        }
    }

    // Normalize and store
    O[(b * H + h) * D + lane] = (l > 0.0f) ? acc / l : 0.0f;
}

/**
 * \brief Launch PagedAttention decode, dispatching on head dimension d.
 *
 * \param[in]  Q           Device query [B, H, d].
 * \param[in]  block_table Device block table [B, max_blocks].
 * \param[in]  K_pool      Device KV pool [num_phys_blocks, block_size, H_kv, d].
 * \param[in]  V_pool      Device KV pool [num_phys_blocks, block_size, H_kv, d].
 * \param[out] O           Device output [B, H, d].
 * \param[in]  seq_lens    Device sequence lengths [B].
 * \param[in]  B           Batch size.
 * \param[in]  H           Number of query heads.
 * \param[in]  H_kv        Number of KV heads.
 * \param[in]  d           Head dimension (32, 64, or 128).
 * \param[in]  block_size  Tokens per physical KV block/page.
 * \param[in]  max_blocks  Columns in block_table.
 * \param[in]  stream      CUDA stream (0 = default).
 */
extern "C" void launch_paged_attention(
    const float* Q,
    const int*   block_table,
    const float* K_pool,
    const float* V_pool,
    float*       O,
    const int*   seq_lens,
    int B, int H, int H_kv, int d,
    int block_size, int max_blocks,
    cudaStream_t stream)
{
    float scale = 1.0f / sqrtf((float)d);
    dim3 grid(H, B);

    switch (d) {
        case 32:
            paged_attention_kernel<32><<<grid, dim3(32), 0, stream>>>(
                Q, block_table, K_pool, V_pool, O, seq_lens,
                H, H_kv, block_size, max_blocks, scale);
            break;
        case 64:
            paged_attention_kernel<64><<<grid, dim3(64), 0, stream>>>(
                Q, block_table, K_pool, V_pool, O, seq_lens,
                H, H_kv, block_size, max_blocks, scale);
            break;
        case 128:
            paged_attention_kernel<128><<<grid, dim3(128), 0, stream>>>(
                Q, block_table, K_pool, V_pool, O, seq_lens,
                H, H_kv, block_size, max_blocks, scale);
            break;
        default:
            break;
    }
    cudaStreamSynchronize(stream);
}
