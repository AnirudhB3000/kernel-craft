/**
 * \file test_paged_attention.cpp
 * \brief Unit tests for PagedAttention decode kernel.
 *
 * Verifies correctness by comparing paged KV access against a contiguous
 * (standard) FlashAttention reference on identical KV data.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

extern "C" void launch_paged_attention(
    const float* Q, const int* block_table,
    const float* K_pool, const float* V_pool,
    float* O, const int* seq_lens,
    int B, int H, int H_kv, int d,
    int block_size, int max_blocks,
    cudaStream_t stream);

// CPU reference: standard decode attention (contiguous KV)
static void decode_attention_cpu(
    const float* q,   // [B, H, d]
    const float* K,   // [B, H_kv, seq_len, d] — contiguous
    const float* V,   // [B, H_kv, seq_len, d]
    float* O,         // [B, H, d]
    int B, int H, int H_kv, int d, int seq_len)
{
    float scale = 1.0f / sqrtf((float)d);
    std::vector<float> scores(seq_len);
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            int h_kv = h * H_kv / H;
            const float* qbh = q + (b * H + h) * d;
            const float* Kbh = K + (b * H_kv + h_kv) * seq_len * d;
            const float* Vbh = V + (b * H_kv + h_kv) * seq_len * d;
            float* Obh = O + (b * H + h) * d;

            float max_s = -1e38f;
            for (int j = 0; j < seq_len; ++j) {
                float dot = 0.0f;
                for (int k = 0; k < d; ++k) dot += qbh[k] * Kbh[j * d + k];
                scores[j] = dot * scale;
                if (scores[j] > max_s) max_s = scores[j];
            }
            float sum = 0.0f;
            for (int j = 0; j < seq_len; ++j) { scores[j] = expf(scores[j] - max_s); sum += scores[j]; }
            for (int j = 0; j < seq_len; ++j) scores[j] /= sum;

            for (int k = 0; k < d; ++k) {
                float acc = 0.0f;
                for (int j = 0; j < seq_len; ++j) acc += scores[j] * Vbh[j * d + k];
                Obh[k] = acc;
            }
        }
    }
}

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

// Build paged KV pool from contiguous KV and a block table
static void build_paged_kv(
    const float* K_contig,  // [B, H_kv, seq_len, d]
    const float* V_contig,
    float* K_pool,          // [num_phys_blocks, block_size, H_kv, d]
    float* V_pool,
    const int* block_table, // [B, max_blocks]
    int B, int H_kv, int seq_len, int d,
    int block_size, int max_blocks)
{
    for (int b = 0; b < B; ++b) {
        int num_blocks = (seq_len + block_size - 1) / block_size;
        for (int blk = 0; blk < num_blocks; ++blk) {
            int phys = block_table[b * max_blocks + blk];
            for (int tok = 0; tok < block_size; ++tok) {
                int seq_pos = blk * block_size + tok;
                if (seq_pos >= seq_len) break;
                for (int h_kv = 0; h_kv < H_kv; ++h_kv) {
                    long long pool_off = ((long long)phys * block_size + tok) * H_kv + h_kv;
                    long long contig_off = (long long)(b * H_kv + h_kv) * seq_len + seq_pos;
                    for (int k = 0; k < d; ++k) {
                        K_pool[pool_off * d + k] = K_contig[contig_off * d + k];
                        V_pool[pool_off * d + k] = V_contig[contig_off * d + k];
                    }
                }
            }
        }
    }
}

static void test_single_batch() {
    int B = 1, H = 4, H_kv = 4, d = 64;
    int seq_len = 32, block_size = 16;
    int max_blocks = (seq_len + block_size - 1) / block_size;  // 2

    std::vector<float> Q(B * H * d), K(B * H_kv * seq_len * d), V(B * H_kv * seq_len * d);
    srand(42);
    for (auto& v : Q) v = ((float)rand() / RAND_MAX) * 2.f - 1.f;
    for (auto& v : K) v = ((float)rand() / RAND_MAX) * 2.f - 1.f;
    for (auto& v : V) v = ((float)rand() / RAND_MAX) * 2.f - 1.f;

    std::vector<float> O_ref(B * H * d, 0.f), O_gpu(B * H * d, 0.f);
    decode_attention_cpu(Q.data(), K.data(), V.data(), O_ref.data(),
                         B, H, H_kv, d, seq_len);

    // Block table: sequential physical blocks
    std::vector<int> block_table(B * max_blocks);
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < max_blocks; ++i)
            block_table[b * max_blocks + i] = b * max_blocks + i;

    int num_phys = B * max_blocks;
    std::vector<float> K_pool(num_phys * block_size * H_kv * d, 0.f);
    std::vector<float> V_pool(num_phys * block_size * H_kv * d, 0.f);
    build_paged_kv(K.data(), V.data(), K_pool.data(), V_pool.data(),
                   block_table.data(), B, H_kv, seq_len, d, block_size, max_blocks);

    std::vector<int> seq_lens = {seq_len};

    float *dQ, *dBT, *dKp, *dVp, *dO, *dSL;
    cudaMalloc(&dQ,  B * H * d           * sizeof(float));
    cudaMalloc(&dBT, B * max_blocks       * sizeof(int));
    cudaMalloc(&dKp, (long long)num_phys * block_size * H_kv * d * sizeof(float));
    cudaMalloc(&dVp, (long long)num_phys * block_size * H_kv * d * sizeof(float));
    cudaMalloc(&dO,  B * H * d           * sizeof(float));
    cudaMalloc(&dSL, B                   * sizeof(int));

    cudaMemcpy(dQ,  Q.data(),          B * H * d           * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dBT, block_table.data(),B * max_blocks       * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(dKp, K_pool.data(),     (long long)num_phys * block_size * H_kv * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dVp, V_pool.data(),     (long long)num_phys * block_size * H_kv * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dSL, seq_lens.data(),   B * sizeof(int),      cudaMemcpyHostToDevice);

    launch_paged_attention(dQ, (int*)dBT, dKp, dVp, dO, (int*)dSL,
                           B, H, H_kv, d, block_size, max_blocks, 0);
    cudaMemcpy(O_gpu.data(), dO, B * H * d * sizeof(float), cudaMemcpyDeviceToHost);

    float err = max_abs_err(O_ref.data(), O_gpu.data(), B * H * d);
    check(err < 1e-3f, "PagedAttention single batch (seq=32, bs=16)");
    if (err >= 1e-3f) printf("  max abs error = %e\n", err);

    cudaFree(dQ); cudaFree(dBT); cudaFree(dKp); cudaFree(dVp); cudaFree(dO); cudaFree(dSL);
}

static void test_multi_batch_variable_seq() {
    // B=2, different sequence lengths
    int B = 2, H = 2, H_kv = 1, d = 64;
    int seq_lens_h[2] = {16, 24};
    int max_seq = 24;
    int block_size = 8;
    int max_blocks = (max_seq + block_size - 1) / block_size;  // 3

    // Build contiguous KV (pad to max_seq)
    std::vector<float> Q(B * H * d), K(B * H_kv * max_seq * d), V(B * H_kv * max_seq * d);
    srand(7);
    for (auto& v : Q) v = ((float)rand() / RAND_MAX) * 2.f - 1.f;
    for (auto& v : K) v = ((float)rand() / RAND_MAX) * 2.f - 1.f;
    for (auto& v : V) v = ((float)rand() / RAND_MAX) * 2.f - 1.f;

    // CPU ref per batch item with actual seq_lens
    std::vector<float> O_ref(B * H * d, 0.f);
    for (int b = 0; b < B; ++b) {
        int sl = seq_lens_h[b];
        // Extract per-batch K/V slice
        std::vector<float> Qb(H * d), Kb(H_kv * sl * d), Vb(H_kv * sl * d), Ob(H * d);
        for (int h = 0; h < H; ++h)
            for (int k = 0; k < d; ++k) Qb[h*d+k] = Q[(b*H+h)*d+k];
        for (int h_kv = 0; h_kv < H_kv; ++h_kv)
            for (int j = 0; j < sl; ++j)
                for (int k = 0; k < d; ++k)
                    Kb[(h_kv*sl+j)*d+k] = K[(b*H_kv+h_kv)*max_seq*d+j*d+k],
                    Vb[(h_kv*sl+j)*d+k] = V[(b*H_kv+h_kv)*max_seq*d+j*d+k];
        decode_attention_cpu(Qb.data(), Kb.data(), Vb.data(), Ob.data(), 1, H, H_kv, d, sl);
        for (int h = 0; h < H; ++h)
            for (int k = 0; k < d; ++k)
                O_ref[(b*H+h)*d+k] = Ob[h*d+k];
    }

    // Build block tables (sequential physical blocks)
    std::vector<int> block_table(B * max_blocks, 0);
    int phys_counter = 0;
    for (int b = 0; b < B; ++b) {
        int nb = (seq_lens_h[b] + block_size - 1) / block_size;
        for (int i = 0; i < nb; ++i)
            block_table[b * max_blocks + i] = phys_counter++;
    }
    int num_phys = phys_counter;

    std::vector<float> K_pool(num_phys * block_size * H_kv * d, 0.f);
    std::vector<float> V_pool(num_phys * block_size * H_kv * d, 0.f);

    // Manually fill K_pool, V_pool from contiguous K, V per batch
    for (int b = 0; b < B; ++b) {
        int nb = (seq_lens_h[b] + block_size - 1) / block_size;
        for (int blk = 0; blk < nb; ++blk) {
            int phys = block_table[b * max_blocks + blk];
            for (int tok = 0; tok < block_size; ++tok) {
                int seq_pos = blk * block_size + tok;
                if (seq_pos >= seq_lens_h[b]) break;
                for (int h_kv = 0; h_kv < H_kv; ++h_kv) {
                    long long pool_off = ((long long)phys * block_size + tok) * H_kv + h_kv;
                    for (int k = 0; k < d; ++k) {
                        K_pool[pool_off * d + k] = K[(b * H_kv + h_kv) * max_seq * d + seq_pos * d + k];
                        V_pool[pool_off * d + k] = V[(b * H_kv + h_kv) * max_seq * d + seq_pos * d + k];
                    }
                }
            }
        }
    }

    std::vector<float> O_gpu(B * H * d, 0.f);
    float *dQ, *dBT, *dKp, *dVp, *dO, *dSL;
    cudaMalloc(&dQ,  B * H * d           * sizeof(float));
    cudaMalloc(&dBT, B * max_blocks       * sizeof(int));
    cudaMalloc(&dKp, (long long)num_phys * block_size * H_kv * d * sizeof(float));
    cudaMalloc(&dVp, (long long)num_phys * block_size * H_kv * d * sizeof(float));
    cudaMalloc(&dO,  B * H * d           * sizeof(float));
    cudaMalloc(&dSL, B                   * sizeof(int));

    cudaMemcpy(dQ,  Q.data(),          B * H * d     * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dBT, block_table.data(),B * max_blocks * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(dKp, K_pool.data(), (long long)num_phys * block_size * H_kv * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dVp, V_pool.data(), (long long)num_phys * block_size * H_kv * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dSL, seq_lens_h,    B * sizeof(int), cudaMemcpyHostToDevice);

    launch_paged_attention(dQ, (int*)dBT, dKp, dVp, dO, (int*)dSL,
                           B, H, H_kv, d, block_size, max_blocks, 0);
    cudaMemcpy(O_gpu.data(), dO, B * H * d * sizeof(float), cudaMemcpyDeviceToHost);

    float err = max_abs_err(O_ref.data(), O_gpu.data(), B * H * d);
    check(err < 1e-3f, "PagedAttention multi-batch variable seq (B=2, H_kv=1)");
    if (err >= 1e-3f) printf("  max abs error = %e\n", err);

    cudaFree(dQ); cudaFree(dBT); cudaFree(dKp); cudaFree(dVp); cudaFree(dO); cudaFree(dSL);
}

static void test_lifecycle() {
    // Minimal smoke test: allocate, run with seq_len=1, free
    int B = 1, H = 2, H_kv = 2, d = 64, seq_len = 1, block_size = 16, max_blocks = 1;
    std::vector<float> Q(B*H*d, 1.f), K_pool(block_size*H_kv*d, 1.f), V_pool(block_size*H_kv*d, 0.5f);
    std::vector<int> bt = {0}, sl = {seq_len};
    std::vector<float> O(B*H*d, 0.f);

    float *dQ, *dKp, *dVp, *dO; int *dBT, *dSL;
    cudaMalloc(&dQ,  B*H*d*sizeof(float));
    cudaMalloc(&dKp, block_size*H_kv*d*sizeof(float));
    cudaMalloc(&dVp, block_size*H_kv*d*sizeof(float));
    cudaMalloc(&dO,  B*H*d*sizeof(float));
    cudaMalloc(&dBT, B*max_blocks*sizeof(int));
    cudaMalloc(&dSL, B*sizeof(int));
    cudaMemcpy(dQ,  Q.data(),      B*H*d*sizeof(float),           cudaMemcpyHostToDevice);
    cudaMemcpy(dKp, K_pool.data(), block_size*H_kv*d*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dVp, V_pool.data(), block_size*H_kv*d*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dBT, bt.data(),     B*max_blocks*sizeof(int),       cudaMemcpyHostToDevice);
    cudaMemcpy(dSL, sl.data(),     B*sizeof(int),                  cudaMemcpyHostToDevice);

    launch_paged_attention(dQ, dBT, dKp, dVp, dO, dSL, B, H, H_kv, d, block_size, max_blocks, 0);
    cudaMemcpy(O.data(), dO, B*H*d*sizeof(float), cudaMemcpyDeviceToHost);

    // With Q=1, K=1, V=0.5 and seq_len=1: output should be ~0.5
    bool ok = true;
    for (int i = 0; i < B*H*d; ++i) ok = ok && (fabsf(O[i] - 0.5f) < 1e-4f);
    check(ok, "PagedAttention lifecycle (seq_len=1, output ~= V)");

    cudaFree(dQ); cudaFree(dKp); cudaFree(dVp); cudaFree(dO); cudaFree(dBT); cudaFree(dSL);
}

int main() {
    printf("=== PagedAttention Tests ===\n");
    test_lifecycle();
    test_single_batch();
    test_multi_batch_variable_seq();
    printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
