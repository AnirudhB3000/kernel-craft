/**
 * \file test_flash_attention.cpp
 * \brief Unit tests for FlashAttention and GQA/MQA variants.
 *
 * Verifies correctness against a naive O(N²) CPU reference implementation.
 * Tests: MHA, GQA, MQA, causal masking, multiple head dimensions.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

extern "C" void launch_flash_attention(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int H_kv, int N, int d,
    bool causal, cudaStream_t stream);

// ---------------------------------------------------------------------------
// CPU reference: naive scaled dot-product attention O(N²)
// ---------------------------------------------------------------------------

/**
 * \brief Reference attention computation on CPU.
 *
 * \param[in]  Q_h   Host query [B, H, N, d].
 * \param[in]  K_h   Host key [B, H_kv, N, d].
 * \param[in]  V_h   Host value [B, H_kv, N, d].
 * \param[out] O_h   Host output [B, H, N, d].
 * \param[in]  B     Batch size.
 * \param[in]  H     Query heads.
 * \param[in]  H_kv  KV heads.
 * \param[in]  N     Sequence length.
 * \param[in]  d     Head dimension.
 * \param[in]  causal Causal mask flag.
 */
static void attention_cpu_ref(
    const float* Q_h, const float* K_h, const float* V_h, float* O_h,
    int B, int H, int H_kv, int N, int d, bool causal)
{
    float scale = 1.0f / sqrtf((float)d);
    std::vector<float> scores(N);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            int h_kv = h * H_kv / H;
            const float* Qb = Q_h + (b * H     + h   ) * N * d;
            const float* Kb = K_h + (b * H_kv  + h_kv) * N * d;
            const float* Vb = V_h + (b * H_kv  + h_kv) * N * d;
            float*       Ob = O_h + (b * H     + h   ) * N * d;

            for (int i = 0; i < N; ++i) {
                // Compute scores[j] = Q[i] · K[j] * scale
                float max_s = -1e38f;
                for (int j = 0; j < N; ++j) {
                    if (causal && j > i) { scores[j] = -1e38f; continue; }
                    float dot = 0.0f;
                    for (int k = 0; k < d; ++k)
                        dot += Qb[i * d + k] * Kb[j * d + k];
                    scores[j] = dot * scale;
                    if (scores[j] > max_s) max_s = scores[j];
                }

                // Softmax
                float sum = 0.0f;
                for (int j = 0; j < N; ++j) {
                    scores[j] = expf(scores[j] - max_s);
                    if (causal && j > i) scores[j] = 0.0f;
                    sum += scores[j];
                }
                for (int j = 0; j < N; ++j) scores[j] /= (sum + 1e-9f);

                // Weighted sum of V
                for (int k = 0; k < d; ++k) {
                    float acc = 0.0f;
                    for (int j = 0; j < N; ++j)
                        acc += scores[j] * Vb[j * d + k];
                    Ob[i * d + k] = acc;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const char* name) {
    if (ok) { printf("[PASS] %s\n", name); ++g_pass; }
    else    { printf("[FAIL] %s\n", name); ++g_fail; }
}

static float max_abs_err(const float* a, const float* b, int n) {
    float err = 0.0f;
    for (int i = 0; i < n; ++i) err = fmaxf(err, fabsf(a[i] - b[i]));
    return err;
}

static void fill_rand(float* buf, int n, float lo = -1.f, float hi = 1.f) {
    for (int i = 0; i < n; ++i)
        buf[i] = lo + (hi - lo) * ((float)rand() / RAND_MAX);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_mha_correctness() {
    int B = 2, H = 4, H_kv = 4, N = 64, d = 64;
    int Q_sz  = B * H    * N * d;
    int KV_sz = B * H_kv * N * d;

    std::vector<float> Q(Q_sz), K(KV_sz), V(KV_sz);
    std::vector<float> O_ref(Q_sz, 0.f), O_gpu(Q_sz, 0.f);

    srand(42);
    fill_rand(Q.data(), Q_sz);
    fill_rand(K.data(), KV_sz);
    fill_rand(V.data(), KV_sz);

    attention_cpu_ref(Q.data(), K.data(), V.data(), O_ref.data(),
                      B, H, H_kv, N, d, false);

    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, Q_sz  * sizeof(float));
    cudaMalloc(&dK, KV_sz * sizeof(float));
    cudaMalloc(&dV, KV_sz * sizeof(float));
    cudaMalloc(&dO, Q_sz  * sizeof(float));

    cudaMemcpy(dQ, Q.data(), Q_sz  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K.data(), KV_sz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), KV_sz * sizeof(float), cudaMemcpyHostToDevice);

    launch_flash_attention(dQ, dK, dV, dO, B, H, H_kv, N, d, false, 0);
    cudaMemcpy(O_gpu.data(), dO, Q_sz * sizeof(float), cudaMemcpyDeviceToHost);

    float err = max_abs_err(O_ref.data(), O_gpu.data(), Q_sz);
    check(err < 1e-3f, "MHA correctness (d=64, B=2, H=4, N=64)");
    if (err >= 1e-3f) printf("  max abs error = %e\n", err);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
}

static void test_gqa_correctness() {
    // GQA: 4 Q heads, 2 KV heads
    int B = 1, H = 4, H_kv = 2, N = 32, d = 64;
    int Q_sz  = B * H    * N * d;
    int KV_sz = B * H_kv * N * d;

    std::vector<float> Q(Q_sz), K(KV_sz), V(KV_sz);
    std::vector<float> O_ref(Q_sz, 0.f), O_gpu(Q_sz, 0.f);

    srand(7);
    fill_rand(Q.data(), Q_sz);
    fill_rand(K.data(), KV_sz);
    fill_rand(V.data(), KV_sz);

    attention_cpu_ref(Q.data(), K.data(), V.data(), O_ref.data(),
                      B, H, H_kv, N, d, false);

    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, Q_sz  * sizeof(float));
    cudaMalloc(&dK, KV_sz * sizeof(float));
    cudaMalloc(&dV, KV_sz * sizeof(float));
    cudaMalloc(&dO, Q_sz  * sizeof(float));

    cudaMemcpy(dQ, Q.data(), Q_sz  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K.data(), KV_sz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), KV_sz * sizeof(float), cudaMemcpyHostToDevice);

    launch_flash_attention(dQ, dK, dV, dO, B, H, H_kv, N, d, false, 0);
    cudaMemcpy(O_gpu.data(), dO, Q_sz * sizeof(float), cudaMemcpyDeviceToHost);

    float err = max_abs_err(O_ref.data(), O_gpu.data(), Q_sz);
    check(err < 1e-3f, "GQA correctness (H=4, H_kv=2, d=64)");
    if (err >= 1e-3f) printf("  max abs error = %e\n", err);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
}

static void test_mqa_correctness() {
    // MQA: 4 Q heads, 1 KV head
    int B = 1, H = 4, H_kv = 1, N = 32, d = 32;
    int Q_sz  = B * H    * N * d;
    int KV_sz = B * H_kv * N * d;

    std::vector<float> Q(Q_sz), K(KV_sz), V(KV_sz);
    std::vector<float> O_ref(Q_sz, 0.f), O_gpu(Q_sz, 0.f);

    srand(13);
    fill_rand(Q.data(), Q_sz);
    fill_rand(K.data(), KV_sz);
    fill_rand(V.data(), KV_sz);

    attention_cpu_ref(Q.data(), K.data(), V.data(), O_ref.data(),
                      B, H, H_kv, N, d, false);

    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, Q_sz  * sizeof(float));
    cudaMalloc(&dK, KV_sz * sizeof(float));
    cudaMalloc(&dV, KV_sz * sizeof(float));
    cudaMalloc(&dO, Q_sz  * sizeof(float));

    cudaMemcpy(dQ, Q.data(), Q_sz  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K.data(), KV_sz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), KV_sz * sizeof(float), cudaMemcpyHostToDevice);

    launch_flash_attention(dQ, dK, dV, dO, B, H, H_kv, N, d, false, 0);
    cudaMemcpy(O_gpu.data(), dO, Q_sz * sizeof(float), cudaMemcpyDeviceToHost);

    float err = max_abs_err(O_ref.data(), O_gpu.data(), Q_sz);
    check(err < 1e-3f, "MQA correctness (H=4, H_kv=1, d=32)");
    if (err >= 1e-3f) printf("  max abs error = %e\n", err);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
}

static void test_causal_masking() {
    int B = 1, H = 2, H_kv = 2, N = 48, d = 64;
    int Q_sz  = B * H * N * d;

    std::vector<float> Q(Q_sz), K(Q_sz), V(Q_sz);
    std::vector<float> O_ref(Q_sz, 0.f), O_gpu(Q_sz, 0.f);

    srand(99);
    fill_rand(Q.data(), Q_sz);
    fill_rand(K.data(), Q_sz);
    fill_rand(V.data(), Q_sz);

    attention_cpu_ref(Q.data(), K.data(), V.data(), O_ref.data(),
                      B, H, H_kv, N, d, true);

    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, Q_sz * sizeof(float));
    cudaMalloc(&dK, Q_sz * sizeof(float));
    cudaMalloc(&dV, Q_sz * sizeof(float));
    cudaMalloc(&dO, Q_sz * sizeof(float));

    cudaMemcpy(dQ, Q.data(), Q_sz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K.data(), Q_sz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), Q_sz * sizeof(float), cudaMemcpyHostToDevice);

    launch_flash_attention(dQ, dK, dV, dO, B, H, H_kv, N, d, true, 0);
    cudaMemcpy(O_gpu.data(), dO, Q_sz * sizeof(float), cudaMemcpyDeviceToHost);

    float err = max_abs_err(O_ref.data(), O_gpu.data(), Q_sz);
    check(err < 1e-3f, "Causal masking correctness (N=48, d=64)");
    if (err >= 1e-3f) printf("  max abs error = %e\n", err);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
}

static void test_d128_correctness() {
    int B = 1, H = 2, H_kv = 2, N = 32, d = 128;
    int Q_sz  = B * H * N * d;

    std::vector<float> Q(Q_sz), K(Q_sz), V(Q_sz);
    std::vector<float> O_ref(Q_sz, 0.f), O_gpu(Q_sz, 0.f);

    srand(55);
    fill_rand(Q.data(), Q_sz, -0.5f, 0.5f);
    fill_rand(K.data(), Q_sz, -0.5f, 0.5f);
    fill_rand(V.data(), Q_sz, -0.5f, 0.5f);

    attention_cpu_ref(Q.data(), K.data(), V.data(), O_ref.data(),
                      B, H, H_kv, N, d, false);

    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, Q_sz * sizeof(float));
    cudaMalloc(&dK, Q_sz * sizeof(float));
    cudaMalloc(&dV, Q_sz * sizeof(float));
    cudaMalloc(&dO, Q_sz * sizeof(float));

    cudaMemcpy(dQ, Q.data(), Q_sz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K.data(), Q_sz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), Q_sz * sizeof(float), cudaMemcpyHostToDevice);

    launch_flash_attention(dQ, dK, dV, dO, B, H, H_kv, N, d, false, 0);
    cudaMemcpy(O_gpu.data(), dO, Q_sz * sizeof(float), cudaMemcpyDeviceToHost);

    float err = max_abs_err(O_ref.data(), O_gpu.data(), Q_sz);
    check(err < 2e-3f, "FlashAttention d=128 correctness");
    if (err >= 2e-3f) printf("  max abs error = %e\n", err);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
}

// ---------------------------------------------------------------------------

int main() {
    printf("=== FlashAttention Tests ===\n");
    test_mha_correctness();
    test_gqa_correctness();
    test_mqa_correctness();
    test_causal_masking();
    test_d128_correctness();
    printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
