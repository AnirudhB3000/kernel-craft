/**
 * \file test_speculative_decoding.cpp
 * \brief Unit tests for speculative decoding draft-token verification.
 *
 * Tests: all-accept (target ≥ draft), all-reject (target = 0), prefix length,
 * corrected token comes from residual distribution.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

extern "C" void launch_verify_draft_tokens(
    const float* d_draft_probs, const float* d_target_probs,
    const int* d_draft_tokens, const float* d_rand_vals, const float* d_rand_vals2,
    int* d_accepted, int* d_corrected,
    int num_tokens, int vocab_size, cudaStream_t stream);

extern "C" void launch_compute_prefix_length(
    const int* d_accepted, int* d_prefix_len, int num_tokens, cudaStream_t stream);

static int g_pass = 0, g_fail = 0;
static void check(bool ok, const char* name) {
    if (ok) { printf("[PASS] %s\n", name); ++g_pass; }
    else    { printf("[FAIL] %s\n", name); ++g_fail; }
}

static void test_all_accept() {
    // When target_prob[token] >= draft_prob[token] for all tokens,
    // accept_prob = min(1, p/q) = 1.0, all tokens should be accepted.
    int num_tokens = 4, vocab_size = 8;

    // All tokens are token_id=0
    std::vector<int>   draft_tokens(num_tokens, 0);
    std::vector<float> draft_probs(num_tokens * vocab_size, 0.0f);
    std::vector<float> target_probs(num_tokens * vocab_size, 0.0f);

    // draft_prob[0] = 0.5, target_prob[0] = 1.0 → accept_prob = 1.0
    for (int i = 0; i < num_tokens; ++i) {
        draft_probs[i * vocab_size + 0]  = 0.5f;
        target_probs[i * vocab_size + 0] = 1.0f;
    }

    // rand_vals < 1.0 always → accepted
    std::vector<float> rand_vals(num_tokens, 0.5f);
    std::vector<float> rand_vals2(num_tokens, 0.5f);
    std::vector<int> accepted(num_tokens, -1);
    std::vector<int> corrected(num_tokens, -1);

    float *dd, *dt, *dr, *dr2; int *ddt, *da, *dc;
    cudaMalloc(&dd,  num_tokens * vocab_size * sizeof(float));
    cudaMalloc(&dt,  num_tokens * vocab_size * sizeof(float));
    cudaMalloc(&dr,  num_tokens * sizeof(float));
    cudaMalloc(&dr2, num_tokens * sizeof(float));
    cudaMalloc(&ddt, num_tokens * sizeof(int));
    cudaMalloc(&da,  num_tokens * sizeof(int));
    cudaMalloc(&dc,  num_tokens * sizeof(int));

    cudaMemcpy(dd,  draft_probs.data(),  num_tokens * vocab_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dt,  target_probs.data(), num_tokens * vocab_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dr,  rand_vals.data(),    num_tokens * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dr2, rand_vals2.data(),   num_tokens * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ddt, draft_tokens.data(), num_tokens * sizeof(int),   cudaMemcpyHostToDevice);

    launch_verify_draft_tokens(dd, dt, ddt, dr, dr2, da, dc, num_tokens, vocab_size, 0);
    cudaMemcpy(accepted.data(),  da, num_tokens * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(corrected.data(), dc, num_tokens * sizeof(int), cudaMemcpyDeviceToHost);

    bool all_ok = true;
    for (int i = 0; i < num_tokens; ++i)
        if (accepted[i] != 1) { all_ok = false; break; }
    check(all_ok, "All tokens accepted when target >= draft");

    cudaFree(dd); cudaFree(dt); cudaFree(dr); cudaFree(dr2);
    cudaFree(ddt); cudaFree(da); cudaFree(dc);
}

static void test_all_reject() {
    // When target_prob[token] = 0, accept_prob = min(1, 0/q) = 0,
    // all tokens should be rejected.
    int num_tokens = 4, vocab_size = 8;

    std::vector<int>   draft_tokens(num_tokens, 0);
    std::vector<float> draft_probs(num_tokens * vocab_size, 0.0f);
    std::vector<float> target_probs(num_tokens * vocab_size, 0.0f);

    for (int i = 0; i < num_tokens; ++i) {
        draft_probs[i * vocab_size + 0]  = 0.8f;
        draft_probs[i * vocab_size + 1]  = 0.2f;
        // target assigns mass to token 2, not token 0
        target_probs[i * vocab_size + 0] = 0.0f;
        target_probs[i * vocab_size + 2] = 1.0f;
    }

    // rand_vals = 0.5 > 0 = accept_prob → rejected
    std::vector<float> rand_vals(num_tokens, 0.5f);
    std::vector<float> rand_vals2(num_tokens, 0.1f);  // sample from residual
    std::vector<int> accepted(num_tokens, -1);
    std::vector<int> corrected(num_tokens, -1);

    float *dd, *dt, *dr, *dr2; int *ddt, *da, *dc;
    cudaMalloc(&dd,  num_tokens * vocab_size * sizeof(float));
    cudaMalloc(&dt,  num_tokens * vocab_size * sizeof(float));
    cudaMalloc(&dr,  num_tokens * sizeof(float));
    cudaMalloc(&dr2, num_tokens * sizeof(float));
    cudaMalloc(&ddt, num_tokens * sizeof(int));
    cudaMalloc(&da,  num_tokens * sizeof(int));
    cudaMalloc(&dc,  num_tokens * sizeof(int));

    cudaMemcpy(dd,  draft_probs.data(),  num_tokens * vocab_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dt,  target_probs.data(), num_tokens * vocab_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dr,  rand_vals.data(),    num_tokens * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dr2, rand_vals2.data(),   num_tokens * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ddt, draft_tokens.data(), num_tokens * sizeof(int),   cudaMemcpyHostToDevice);

    launch_verify_draft_tokens(dd, dt, ddt, dr, dr2, da, dc, num_tokens, vocab_size, 0);
    cudaMemcpy(accepted.data(),  da, num_tokens * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(corrected.data(), dc, num_tokens * sizeof(int), cudaMemcpyDeviceToHost);

    bool all_rejected = true;
    bool corrected_ok = true;
    for (int i = 0; i < num_tokens; ++i) {
        if (accepted[i] != 0) { all_rejected = false; }
        // Residual: max(0, target - draft)[v] = {0, 0, 1, 0, ...} → corrected = 2
        if (corrected[i] != 2) { corrected_ok = false; }
    }
    check(all_rejected, "All tokens rejected when target_prob[draft_token]=0");
    check(corrected_ok, "Corrected token sampled from residual distribution");

    cudaFree(dd); cudaFree(dt); cudaFree(dr); cudaFree(dr2);
    cudaFree(ddt); cudaFree(da); cudaFree(dc);
}

static void test_prefix_length() {
    // accepted = [1, 1, 0, 1] → prefix_len = 2 (stops at first 0)
    int num_tokens = 4;
    std::vector<int> accepted_h = {1, 1, 0, 1};

    int *da, *dpl;
    cudaMalloc(&da,  num_tokens * sizeof(int));
    cudaMalloc(&dpl, sizeof(int));
    cudaMemcpy(da, accepted_h.data(), num_tokens * sizeof(int), cudaMemcpyHostToDevice);

    launch_compute_prefix_length(da, dpl, num_tokens, 0);

    int prefix_len = -1;
    cudaMemcpy(&prefix_len, dpl, sizeof(int), cudaMemcpyDeviceToHost);
    check(prefix_len == 2, "Prefix length = 2 for [1,1,0,1]");

    cudaFree(da); cudaFree(dpl);
}

static void test_prefix_all_accepted() {
    int num_tokens = 5;
    std::vector<int> accepted_h = {1, 1, 1, 1, 1};

    int *da, *dpl;
    cudaMalloc(&da,  num_tokens * sizeof(int));
    cudaMalloc(&dpl, sizeof(int));
    cudaMemcpy(da, accepted_h.data(), num_tokens * sizeof(int), cudaMemcpyHostToDevice);

    launch_compute_prefix_length(da, dpl, num_tokens, 0);

    int prefix_len = -1;
    cudaMemcpy(&prefix_len, dpl, sizeof(int), cudaMemcpyDeviceToHost);
    check(prefix_len == 5, "Prefix length = 5 when all accepted");

    cudaFree(da); cudaFree(dpl);
}

int main() {
    printf("=== Speculative Decoding Tests ===\n");
    test_all_accept();
    test_all_reject();
    test_prefix_length();
    test_prefix_all_accepted();
    printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
