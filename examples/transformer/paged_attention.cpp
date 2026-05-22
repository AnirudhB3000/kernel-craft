/**
 * \file transformer/paged_attention.cpp
 * \brief PagedAttention: non-contiguous KV-cache access for LLM decode.
 *
 * \par Purpose
 * Demonstrates the PagedAttention kernel (src/kernels/transformer/paged_attention.cu)
 * and explains how paged KV-cache management is the enabling technology behind
 * vLLM's high memory efficiency and throughput.
 *
 * \par The KV-cache memory problem
 * During autoregressive decoding each generated token's key and value vectors
 * are appended to the KV cache and must remain accessible for future tokens.
 * A naïve contiguous-allocation scheme reserves max_seq_len × H × d × 2 bytes
 * per sequence regardless of actual length, wasting up to 60–80% of GPU VRAM
 * on short sequences (the original vLLM paper's finding).
 *
 * \par vLLM's solution — physical pages + block table
 * vLLM divides the KV cache into fixed-size *physical pages* (blocks), each
 * holding `block_size` tokens.  A per-sequence *block table* maps logical
 * page indices to physical page indices, exactly like a virtual-memory page
 * table.  Key benefits:
 * - Pages are allocated on demand (no reserved slack).
 * - Pages from different sequences can be interleaved in physical memory.
 * - Prefix-sharing: sequences with the same system prompt share their initial
 *   KV pages (copy-on-write semantics).
 *
 * \par Physical KV cache layout
 * \code
 *   K_pool[num_phys_pages][block_size][H_kv][d]   // float32
 *   V_pool[num_phys_pages][block_size][H_kv][d]   // float32
 *   block_table[B][max_blocks_per_seq]             // int32
 * \endcode
 *
 * \par Decode step
 * Given a new query q[B, H, d] the kernel:
 * 1. Iterates over the block table for each sequence.
 * 2. For each logical page index, looks up the physical page in K_pool/V_pool.
 * 3. Accumulates scaled dot-product attention across all past tokens.
 * The kernel never requires the KV cache to be contiguous.
 *
 * \par Integration in vLLM
 * \code
 *   // vLLM calls this in KernelCraftAttentionBackend.forward() for decode
 *   launch_paged_attention(d_Q,          // new token query [B, H, d]
 *                          d_block_table, // [B, max_blocks]
 *                          d_K_pool,     // [num_phys, block_size, H_kv, d]
 *                          d_V_pool,     // [num_phys, block_size, H_kv, d]
 *                          d_O,          // output [B, H, d]
 *                          d_seq_lens,   // [B] actual seq len per request
 *                          B, H, H_kv, d, block_size, max_blocks, stream);
 * \endcode
 * See src/python/kernel_craft_vllm_backend.py for the full vLLM backend.
 *
 * \par Expected latency on RTX 4070 (B=1, H=8, H_kv=8, d=64)
 * | Seq len | Page size | Time  |
 * |---------|-----------|-------|
 * | 256     | 16 / 32   | ~0.13 ms |
 * | 1024    | 16 / 32   | ~0.48 ms |
 * Cost scales linearly with seq_len; page size has minimal impact on latency.
 *
 * \par Compile and run
 * \code
 * cmake --build build --target example_paged_attention
 * ./build/examples/example_paged_attention
 * \endcode
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

/**
 * \brief PagedAttention decode kernel (src/kernels/transformer/paged_attention.cu).
 *
 * Performs single-token attention decode using a paged KV cache.
 * Each block (CTA) processes one batch element and one attention head.
 *
 * \param[in]  d_Q          float32 device [B, H, d] — new token query.
 * \param[in]  d_block_table int32  device [B, max_blocks] — logical→physical map.
 * \param[in]  d_K_pool     float32 device [num_phys, block_size, H_kv, d].
 * \param[in]  d_V_pool     float32 device [num_phys, block_size, H_kv, d].
 * \param[out] d_O          float32 device [B, H, d] — output.
 * \param[in]  d_seq_lens   int32   device [B] — actual sequence length per request.
 * \param[in]  B            Batch size.
 * \param[in]  H            Number of query heads.
 * \param[in]  H_kv         Number of KV heads (GQA: H_kv ≤ H).
 * \param[in]  d            Head dimension.
 * \param[in]  block_size   Tokens per physical page (e.g. 16 or 32).
 * \param[in]  max_blocks   Maximum number of logical pages per sequence.
 * \param[in]  stream       CUDA stream (0 = default stream).
 */
extern "C" void launch_paged_attention(const float* d_Q,
                                       const int*   d_block_table,
                                       const float* d_K_pool,
                                       const float* d_V_pool,
                                       float*       d_O,
                                       const int*   d_seq_lens,
                                       int B, int H, int H_kv, int d,
                                       int block_size, int max_blocks,
                                       cudaStream_t stream);

int main() {
    /* ---- Configuration ---- */
    const int B          = 2;    /* Two concurrent sequences (typical decode batch) */
    const int H          = 8;    /* Query heads */
    const int H_kv       = 4;    /* KV heads (GQA 2:1 ratio) */
    const int head_d     = 64;   /* Head dimension */
    const int block_size = 16;   /* Tokens per physical KV page */
    const int seq_len_0  = 128;  /* Sequence 0: 128 tokens in cache */
    const int seq_len_1  = 64;   /* Sequence 1: 64 tokens in cache */

    /* ---- Derived sizes ---- */
    const int max_blocks = (std::max(seq_len_0, seq_len_1) + block_size - 1) / block_size;
    const int n_phys     = (seq_len_0 + seq_len_1 + block_size - 1) / block_size + 4;

    const size_t q_elems   = (size_t)B * H    * head_d;
    const size_t kv_elems  = (size_t)n_phys * block_size * H_kv * head_d;
    const size_t bt_elems  = (size_t)B * max_blocks;

    /* ---- Host data ---- */
    std::vector<float> h_Q(q_elems, 0.1f);
    std::vector<float> h_K(kv_elems, 0.05f);
    std::vector<float> h_V(kv_elems, 0.05f);
    std::vector<int>   h_bt(bt_elems, -1);  /* -1 = unused slot */
    std::vector<int>   h_sl = {seq_len_0, seq_len_1};

    /* Assign physical pages to each sequence's block table.
     * Seq 0 gets pages 0..n0-1, seq 1 gets pages n0..n0+n1-1.
     * In vLLM the PageScheduler fills this table dynamically. */
    int n0 = (seq_len_0 + block_size - 1) / block_size;
    int n1 = (seq_len_1 + block_size - 1) / block_size;
    for (int i = 0; i < n0; ++i) h_bt[0 * max_blocks + i] = i;
    for (int i = 0; i < n1; ++i) h_bt[1 * max_blocks + i] = n0 + i;

    /* ---- Device allocation ---- */
    float *d_Q, *d_K, *d_V, *d_O;
    int   *d_bt, *d_sl;

    cudaMalloc(&d_Q,  q_elems   * sizeof(float));
    cudaMalloc(&d_K,  kv_elems  * sizeof(float));
    cudaMalloc(&d_V,  kv_elems  * sizeof(float));
    cudaMalloc(&d_O,  q_elems   * sizeof(float));
    cudaMalloc(&d_bt, bt_elems  * sizeof(int));
    cudaMalloc(&d_sl, B         * sizeof(int));

    cudaMemcpy(d_Q,  h_Q.data(),  q_elems   * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K,  h_K.data(),  kv_elems  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V,  h_V.data(),  kv_elems  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bt, h_bt.data(), bt_elems  * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_sl, h_sl.data(), B         * sizeof(int),   cudaMemcpyHostToDevice);

    /* ---- Timing ---- */
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    /* Warm-up */
    launch_paged_attention(d_Q, d_bt, d_K, d_V, d_O, d_sl,
                           B, H, H_kv, head_d, block_size, max_blocks, 0);
    cudaDeviceSynchronize();

    const int ITERS = 50;
    cudaEventRecord(t0);
    for (int i = 0; i < ITERS; ++i)
        launch_paged_attention(d_Q, d_bt, d_K, d_V, d_O, d_sl,
                               B, H, H_kv, head_d, block_size, max_blocks, 0);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    /* Read back to check output is finite. */
    std::vector<float> h_O(q_elems);
    cudaMemcpy(h_O.data(), d_O, q_elems * sizeof(float), cudaMemcpyDeviceToHost);
    bool finite = true;
    for (float v : h_O) if (!std::isfinite(v)) { finite = false; break; }

    printf("PagedAttention Decode Example\n");
    printf("  B=%d  H=%d  H_kv=%d  d=%d  block_size=%d\n",
           B, H, H_kv, head_d, block_size);
    printf("  Seq lens : [%d, %d]  (max_blocks=%d)\n", seq_len_0, seq_len_1, max_blocks);
    printf("  Time     : %.3f ms/iter (%d iters)\n", ms / ITERS, ITERS);
    printf("  Output   : %s\n", finite ? "finite (PASS)" : "NaN/Inf (FAIL)");
    printf("\n  vLLM integration:\n");
    printf("  - block_table is filled by vLLM's BlockManager per request\n");
    printf("  - K/V pools are registered with vLLM's KVCacheManager\n");
    printf("  - Use GQA (H_kv < H) to cut KV cache memory by H/H_kv\n");
    printf("\nSee src/kernels/transformer/paged_attention.cu for the implementation.\n");
    printf("See src/python/kernel_craft_vllm_backend.py for the full vLLM backend.\n");

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    cudaFree(d_bt); cudaFree(d_sl);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return finite ? EXIT_SUCCESS : EXIT_FAILURE;
}
