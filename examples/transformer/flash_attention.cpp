/**
 * \file transformer/flash_attention.cpp
 * \brief FlashAttention: tiled online-softmax attention for LLM prefill.
 *
 * \par Purpose
 * Demonstrates the FlashAttention kernel (src/kernels/transformer/flash_attention.cu)
 * and explains how it fits into the prefill stage of an LLM inference engine
 * such as vLLM, TensorRT-LLM, or a custom serving framework.
 *
 * \par Background — the memory wall of standard attention
 * Standard attention materialises the full N×N score matrix in global memory:
 * \code
 *   S = Q @ K^T     // [B, H, N, N] — N² elements
 *   P = softmax(S)  // [B, H, N, N]
 *   O = P @ V       // [B, H, N, d]
 * \endcode
 * For N=4096, H=32 heads, B=1 this is 4096² × 32 × 4 bytes ≈ 2 GB, which
 * exceeds the L2 cache on every current GPU and forces all intermediate
 * reads/writes to global memory (bandwidth-bound, not compute-bound).
 *
 * \par FlashAttention's solution — online softmax in shared memory
 * The kernel tiles Q, K, V into blocks that fit in shared memory and computes
 * a numerically stable running softmax without ever writing S to global memory:
 * \code
 *   for each K/V tile:
 *     S_tile = Q_block @ K_tile^T                  // in registers
 *     m_new  = max(m_old, rowmax(S_tile))          // running max
 *     l_new  = e^(m_old-m_new)*l_old + rowsum(e^(S_tile-m_new))  // running sum
 *     O      = diag(l_old/l_new)*O + e^(S_tile-m_new) @ V_tile   // accumulate
 * \endcode
 * Memory complexity: O(N·d·H) instead of O(N²·H) — fits in L2 for long contexts.
 *
 * \par LLM inference phases and which kernel to use
 * | Phase   | Seq len N       | Kernel                    |
 * |---------|-----------------|---------------------------|
 * | Prefill | Full prompt (N) | launch_flash_attention    |
 * | Decode  | 1 new token     | launch_paged_attention    |
 *
 * During prefill, all N tokens are known simultaneously → FlashAttention.
 * During decode, the new query attends over the KV cache (non-contiguous
 * pages) → PagedAttention (see example_paged_attention.cpp).
 *
 * \par Multi-head vs Grouped-Query Attention (GQA)
 * Set H_kv < H to enable GQA (used in Llama-3, Mistral, Falcon):
 * - H_kv = H  → standard MHA (one KV head per Q head)
 * - H_kv = 1  → MQA (all Q heads share one KV head)
 * - H_kv = 2  → GQA-2 (H/2 Q heads share each KV head)
 * GQA reduces KV cache size by H/H_kv at the cost of slightly lower model
 * quality.  The kernel dispatches the correct KV head index via h_kv = h % H_kv.
 *
 * \par Expected throughput on RTX 4070 (B=1, H=8, H_kv=8, d=64)
 * | N    | Causal | Time (ms) | Gflops |
 * |------|--------|-----------|--------|
 * | 512  | No     | ~0.80     | ~675   |
 * | 1024 | No     | ~3.08     | ~698   |
 * | 1024 | Yes    | ~2.18     | ~985   |
 * Causal attention is faster because ~half the KV tiles are skipped (the
 * tile's max stays -FLT_MAX so the entire tile contributes zero to output).
 *
 * \par Compile and run
 * \code
 * cmake --build build --target example_flash_attention
 * ./build/examples/example_flash_attention
 * \endcode
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

/**
 * \brief FlashAttention forward pass launcher (src/kernels/transformer/flash_attention.cu).
 *
 * Implements tiled online-softmax attention with O(N·d) shared-memory usage.
 * Supports MHA, GQA, and MQA via the H_kv parameter.
 *
 * \param[in]  d_Q     float32 device [B, H,    N, d] — query tensor.
 * \param[in]  d_K     float32 device [B, H_kv, N, d] — key tensor.
 * \param[in]  d_V     float32 device [B, H_kv, N, d] — value tensor.
 * \param[out] d_O     float32 device [B, H,    N, d] — output tensor.
 * \param[in]  B       Batch size.
 * \param[in]  H       Number of query heads.
 * \param[in]  H_kv    Number of KV heads (H_kv ≤ H; H_kv=H for standard MHA).
 * \param[in]  N       Sequence length (number of tokens).
 * \param[in]  d       Head dimension (typically 64 or 128).
 * \param[in]  causal  If true, apply causal (lower-triangular) mask.
 * \param[in]  stream  CUDA stream (0 = default stream).
 */
extern "C" void launch_flash_attention(const float* d_Q, const float* d_K,
                                       const float* d_V, float* d_O,
                                       int B, int H, int H_kv,
                                       int N, int d, bool causal,
                                       cudaStream_t stream);

/** \brief Simple reference: scaled dot-product attention on the CPU. */
static void attention_cpu(const float* Q, const float* K, const float* V,
                           float* O, int H, int N, int d) {
    const float scale = 1.f / std::sqrt((float)d);
    std::vector<float> S(N * N), P(N * N);
    for (int h = 0; h < H; ++h) {
        /* S = Q[h] @ K[h]^T * scale */
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                float s = 0.f;
                for (int k = 0; k < d; ++k)
                    s += Q[h*N*d + i*d + k] * K[h*N*d + j*d + k];
                S[i*N + j] = s * scale;
            }
        /* softmax per row */
        for (int i = 0; i < N; ++i) {
            float m = -1e30f;
            for (int j = 0; j < N; ++j) m = std::max(m, S[i*N+j]);
            float sum = 0.f;
            for (int j = 0; j < N; ++j) { P[i*N+j] = std::exp(S[i*N+j]-m); sum += P[i*N+j]; }
            for (int j = 0; j < N; ++j) P[i*N+j] /= sum;
        }
        /* O[h] = P @ V[h] */
        for (int i = 0; i < N; ++i)
            for (int k = 0; k < d; ++k) {
                float o = 0.f;
                for (int j = 0; j < N; ++j) o += P[i*N+j] * V[h*N*d + j*d + k];
                O[h*N*d + i*d + k] = o;
            }
    }
}

int main() {
    /* ---- Configuration (MHA, single batch, causal decoder) ---- */
    const int B = 1, H = 4, H_kv = 4, N = 64, d = 64;
    const bool causal = true;

    const size_t qkv_elems = (size_t)B * H * N * d;
    const size_t kv_elems  = (size_t)B * H_kv * N * d;

    /* Host buffers */
    std::vector<float> h_Q(qkv_elems), h_K(kv_elems), h_V(kv_elems);
    std::vector<float> h_O_gpu(qkv_elems), h_O_cpu(qkv_elems);

    /* Initialise with small random values — avoids softmax overflow. */
    srand(42);
    auto randf = []{ return (rand() % 200 - 100) / 100.f * 0.1f; };
    for (auto& x : h_Q) x = randf();
    for (auto& x : h_K) x = randf();
    for (auto& x : h_V) x = randf();

    /* ---- Device allocation ---- */
    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, qkv_elems * sizeof(float));
    cudaMalloc(&d_K, kv_elems  * sizeof(float));
    cudaMalloc(&d_V, kv_elems  * sizeof(float));
    cudaMalloc(&d_O, qkv_elems * sizeof(float));

    cudaMemcpy(d_Q, h_Q.data(), qkv_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), kv_elems  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), kv_elems  * sizeof(float), cudaMemcpyHostToDevice);

    /* ---- Launch FlashAttention ---- */
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    /* Warm-up */
    launch_flash_attention(d_Q, d_K, d_V, d_O, B, H, H_kv, N, d, causal, 0);
    cudaDeviceSynchronize();

    const int ITERS = 10;
    cudaEventRecord(t0);
    for (int i = 0; i < ITERS; ++i)
        launch_flash_attention(d_Q, d_K, d_V, d_O, B, H, H_kv, N, d, causal, 0);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    cudaMemcpy(h_O_gpu.data(), d_O, qkv_elems * sizeof(float), cudaMemcpyDeviceToHost);

    /* ---- CPU reference (only runs on small N for speed) ---- */
    attention_cpu(h_Q.data(), h_K.data(), h_V.data(),
                  h_O_cpu.data(), H, N, d);

    float max_err = 0.f;
    for (size_t i = 0; i < qkv_elems; ++i)
        max_err = std::max(max_err, std::fabs(h_O_gpu[i] - h_O_cpu[i]));

    /* Flops: 2 × B × H × N² × d (QK^T) + 2 × B × H × N² × d (PV) */
    double flops = 4.0 * B * H * (double)N * N * d;
    double gflops = flops / (ms / ITERS * 1e-3) / 1e9;

    printf("FlashAttention Example\n");
    printf("  Config  : B=%d H=%d H_kv=%d N=%d d=%d causal=%s\n",
           B, H, H_kv, N, d, causal ? "yes" : "no");
    printf("  Time    : %.3f ms/iter (%d iters)\n", ms / ITERS, ITERS);
    printf("  Throughput : %.1f Gflops\n", gflops);
    printf("  Max error vs CPU : %.2e %s\n", max_err,
           max_err < 1e-3f ? "(PASS)" : "(FAIL — check kernel)");
    printf("\n  Integration note:\n");
    printf("  - Prefill: call with the full prompt length N\n");
    printf("  - Decode : use launch_paged_attention (example_paged_attention.cpp)\n");
    printf("  - GQA    : set H_kv < H (e.g. H=32, H_kv=8 for Llama-3-8B)\n");
    printf("\nSee src/kernels/transformer/flash_attention.cu for the implementation.\n");

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return max_err < 1e-3f ? EXIT_SUCCESS : EXIT_FAILURE;
}
