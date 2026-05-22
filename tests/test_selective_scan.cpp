/**
 * \file test_selective_scan.cpp
 * \brief Correctness tests for the Mamba-1 selective scan kernel.
 *
 * Each test runs a GPU scan and compares against a sequential CPU reference.
 * Pass criterion: max absolute error < 1e-4 for single-precision arithmetic.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

extern "C" void launch_selective_scan(
    const float* d_u,     const float* d_A_log,
    const float* d_B,     const float* d_C,
    const float* d_delta, float* d_y,
    int B_, int L, int D, int N_state,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

/**
 * \brief Sequential CPU selective scan (ZOH formulation).
 *
 * \param[in]  u       [B, L, D]
 * \param[in]  A_log   [D, N]
 * \param[in]  B       [B, L, N]
 * \param[in]  C       [B, L, N]
 * \param[in]  delta   [B, L, D]
 * \param[out] y       [B, L, D]
 */
static void selective_scan_cpu(
    const float* u, const float* A_log,
    const float* B, const float* C,
    const float* delta, float* y,
    int B_, int L, int D, int N)
{
    std::vector<float> h(N);
    for (int b = 0; b < B_; ++b) {
        for (int d = 0; d < D; ++d) {
            std::fill(h.begin(), h.end(), 0.f);
            for (int t = 0; t < L; ++t) {
                float dt   = delta[(b * L + t) * D + d];
                float u_td = u    [(b * L + t) * D + d];
                float y_td = 0.f;
                for (int n = 0; n < N; ++n) {
                    float a_bar = std::exp(dt * A_log[d * N + n]);
                    float b_bar = dt * B[(b * L + t) * N + n];
                    h[n]  = a_bar * h[n] + b_bar * u_td;
                    y_td += C[(b * L + t) * N + n] * h[n];
                }
                y[(b * L + t) * D + d] = y_td;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void fill_rand(std::vector<float>& v, float lo, float hi) {
    for (auto& x : v)
        x = lo + (hi - lo) * (float)rand() / RAND_MAX;
}

static float max_abs_err(const float* a, const float* b, int n) {
    float e = 0.f;
    for (int i = 0; i < n; ++i) e = std::max(e, std::abs(a[i] - b[i]));
    return e;
}

struct GPU {
    float* ptr = nullptr;
    explicit GPU(size_t bytes) { cudaMalloc(&ptr, bytes); }
    ~GPU() { cudaFree(ptr); }
    void upload(const float* h, size_t bytes) { cudaMemcpy(ptr, h, bytes, cudaMemcpyHostToDevice); }
    void download(float* h, size_t bytes) const { cudaMemcpy(h, ptr, bytes, cudaMemcpyDeviceToHost); }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/**
 * \brief Minimal smoke test: B=1, L=4, D=2, N=2 — hand-verifiable.
 */
static bool test_small_scan()
{
    int B_ = 1, L = 4, D = 2, N = 2;
    int u_sz    = B_ * L * D;
    int alog_sz = D * N;
    int bn_sz   = B_ * L * N;
    int y_sz    = B_ * L * D;

    std::vector<float> u(u_sz), A_log(alog_sz), B(bn_sz), C(bn_sz), delta(u_sz);
    srand(42);
    fill_rand(u,     0.f, 1.f);
    fill_rand(A_log, -1.f, 0.f);   /* negative log → |A| < 1 → stable */
    fill_rand(B,      0.f, 0.5f);
    fill_rand(C,      0.f, 0.5f);
    fill_rand(delta,  0.1f, 0.5f);

    std::vector<float> y_cpu(y_sz, 0.f), y_gpu(y_sz, 0.f);
    selective_scan_cpu(u.data(), A_log.data(), B.data(), C.data(),
                       delta.data(), y_cpu.data(), B_, L, D, N);

    GPU du(u_sz*4), dAlog(alog_sz*4), dB(bn_sz*4),
        dC(bn_sz*4), dd(u_sz*4), dy(y_sz*4);
    du.upload(u.data(), u_sz*4);
    dAlog.upload(A_log.data(), alog_sz*4);
    dB.upload(B.data(), bn_sz*4);
    dC.upload(C.data(), bn_sz*4);
    dd.upload(delta.data(), u_sz*4);
    cudaMemset(dy.ptr, 0, y_sz*4);

    launch_selective_scan(du.ptr, dAlog.ptr, dB.ptr, dC.ptr, dd.ptr, dy.ptr,
                          B_, L, D, N, 0);
    cudaDeviceSynchronize();
    dy.download(y_gpu.data(), y_sz*4);

    float err = max_abs_err(y_cpu.data(), y_gpu.data(), y_sz);
    printf("[test_small_scan] max_abs_err=%.2e %s\n", err, err < 1e-4f ? "PASS" : "FAIL");
    return err < 1e-4f;
}

/**
 * \brief Typical Mamba config: B=2, L=64, D=64, N=16.
 */
static bool test_mamba_typical()
{
    int B_ = 2, L = 64, D = 64, N = 16;
    int u_sz    = B_ * L * D;
    int alog_sz = D * N;
    int bn_sz   = B_ * L * N;

    std::vector<float> u(u_sz), A_log(alog_sz), B(bn_sz), C(bn_sz), delta(u_sz);
    srand(7);
    fill_rand(u,     -0.5f, 0.5f);
    fill_rand(A_log, -2.f, -0.01f);
    fill_rand(B,      0.f,  0.3f);
    fill_rand(C,      0.f,  0.3f);
    fill_rand(delta,  0.05f, 0.5f);

    std::vector<float> y_cpu(u_sz, 0.f), y_gpu(u_sz, 0.f);
    selective_scan_cpu(u.data(), A_log.data(), B.data(), C.data(),
                       delta.data(), y_cpu.data(), B_, L, D, N);

    GPU du(u_sz*4), dAlog(alog_sz*4), dB(bn_sz*4),
        dC(bn_sz*4), dd(u_sz*4), dy(u_sz*4);
    du.upload(u.data(), u_sz*4);
    dAlog.upload(A_log.data(), alog_sz*4);
    dB.upload(B.data(), bn_sz*4);
    dC.upload(C.data(), bn_sz*4);
    dd.upload(delta.data(), u_sz*4);
    cudaMemset(dy.ptr, 0, u_sz*4);

    launch_selective_scan(du.ptr, dAlog.ptr, dB.ptr, dC.ptr, dd.ptr, dy.ptr,
                          B_, L, D, N, 0);
    cudaDeviceSynchronize();
    dy.download(y_gpu.data(), u_sz*4);

    float err = max_abs_err(y_cpu.data(), y_gpu.data(), u_sz);
    printf("[test_mamba_typical] max_abs_err=%.2e %s\n", err, err < 1e-4f ? "PASS" : "FAIL");
    return err < 1e-4f;
}

/**
 * \brief Long sequence: B=1, L=1024, D=128, N=16.
 */
static bool test_long_sequence()
{
    int B_ = 1, L = 1024, D = 128, N = 16;
    int u_sz    = B_ * L * D;
    int alog_sz = D * N;
    int bn_sz   = B_ * L * N;

    std::vector<float> u(u_sz), A_log(alog_sz), B(bn_sz), C(bn_sz), delta(u_sz);
    srand(99);
    fill_rand(u,     -0.3f, 0.3f);
    fill_rand(A_log, -1.5f, -0.05f);
    fill_rand(B,      0.f,  0.2f);
    fill_rand(C,      0.f,  0.2f);
    fill_rand(delta,  0.1f, 0.4f);

    std::vector<float> y_cpu(u_sz, 0.f), y_gpu(u_sz, 0.f);
    selective_scan_cpu(u.data(), A_log.data(), B.data(), C.data(),
                       delta.data(), y_cpu.data(), B_, L, D, N);

    GPU du(u_sz*4), dAlog(alog_sz*4), dB(bn_sz*4),
        dC(bn_sz*4), dd(u_sz*4), dy(u_sz*4);
    du.upload(u.data(), u_sz*4);
    dAlog.upload(A_log.data(), alog_sz*4);
    dB.upload(B.data(), bn_sz*4);
    dC.upload(C.data(), bn_sz*4);
    dd.upload(delta.data(), u_sz*4);
    cudaMemset(dy.ptr, 0, u_sz*4);

    launch_selective_scan(du.ptr, dAlog.ptr, dB.ptr, dC.ptr, dd.ptr, dy.ptr,
                          B_, L, D, N, 0);
    cudaDeviceSynchronize();
    dy.download(y_gpu.data(), u_sz*4);

    float err = max_abs_err(y_cpu.data(), y_gpu.data(), u_sz);
    printf("[test_long_sequence] max_abs_err=%.2e %s\n", err, err < 1e-4f ? "PASS" : "FAIL");
    return err < 1e-4f;
}

/**
 * \brief Zero input → zero output.
 */
static bool test_zero_input()
{
    int B_ = 2, L = 32, D = 16, N = 8;
    int u_sz    = B_ * L * D;
    int alog_sz = D * N;
    int bn_sz   = B_ * L * N;

    std::vector<float> u(u_sz, 0.f), A_log(alog_sz), B(bn_sz), C(bn_sz), delta(u_sz);
    srand(11);
    fill_rand(A_log, -1.f, 0.f);
    fill_rand(B,      0.f, 0.5f);
    fill_rand(C,      0.f, 0.5f);
    fill_rand(delta,  0.1f, 0.5f);

    GPU du(u_sz*4), dAlog(alog_sz*4), dB(bn_sz*4),
        dC(bn_sz*4), dd(u_sz*4), dy(u_sz*4);
    du.upload(u.data(), u_sz*4);
    dAlog.upload(A_log.data(), alog_sz*4);
    dB.upload(B.data(), bn_sz*4);
    dC.upload(C.data(), bn_sz*4);
    dd.upload(delta.data(), u_sz*4);
    cudaMemset(dy.ptr, 0, u_sz*4);

    launch_selective_scan(du.ptr, dAlog.ptr, dB.ptr, dC.ptr, dd.ptr, dy.ptr,
                          B_, L, D, N, 0);
    cudaDeviceSynchronize();

    std::vector<float> y_gpu(u_sz);
    dy.download(y_gpu.data(), u_sz*4);

    float err = *std::max_element(y_gpu.begin(), y_gpu.end());
    err = std::max(err, -*std::min_element(y_gpu.begin(), y_gpu.end()));
    printf("[test_zero_input] max_abs_y=%.2e %s\n", err, err < 1e-6f ? "PASS" : "FAIL");
    return err < 1e-6f;
}

/**
 * \brief N_state=32 (maximum supported).
 */
static bool test_max_state()
{
    int B_ = 1, L = 64, D = 32, N = 32;
    int u_sz    = B_ * L * D;
    int alog_sz = D * N;
    int bn_sz   = B_ * L * N;

    std::vector<float> u(u_sz), A_log(alog_sz), B(bn_sz), C(bn_sz), delta(u_sz);
    srand(55);
    fill_rand(u,     -0.5f, 0.5f);
    fill_rand(A_log, -1.f, -0.01f);
    fill_rand(B,      0.f,  0.3f);
    fill_rand(C,      0.f,  0.3f);
    fill_rand(delta,  0.1f, 0.5f);

    std::vector<float> y_cpu(u_sz, 0.f), y_gpu(u_sz, 0.f);
    selective_scan_cpu(u.data(), A_log.data(), B.data(), C.data(),
                       delta.data(), y_cpu.data(), B_, L, D, N);

    GPU du(u_sz*4), dAlog(alog_sz*4), dB(bn_sz*4),
        dC(bn_sz*4), dd(u_sz*4), dy(u_sz*4);
    du.upload(u.data(), u_sz*4);
    dAlog.upload(A_log.data(), alog_sz*4);
    dB.upload(B.data(), bn_sz*4);
    dC.upload(C.data(), bn_sz*4);
    dd.upload(delta.data(), u_sz*4);
    cudaMemset(dy.ptr, 0, u_sz*4);

    launch_selective_scan(du.ptr, dAlog.ptr, dB.ptr, dC.ptr, dd.ptr, dy.ptr,
                          B_, L, D, N, 0);
    cudaDeviceSynchronize();
    dy.download(y_gpu.data(), u_sz*4);

    float err = max_abs_err(y_cpu.data(), y_gpu.data(), u_sz);
    printf("[test_max_state] max_abs_err=%.2e %s\n", err, err < 1e-4f ? "PASS" : "FAIL");
    return err < 1e-4f;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main()
{
    printf("=== Selective Scan Tests ===\n");
    int pass = 0, total = 0;
#define RUN(fn) do { ++total; if (fn()) ++pass; } while(0)
    RUN(test_small_scan);
    RUN(test_mamba_typical);
    RUN(test_long_sequence);
    RUN(test_zero_input);
    RUN(test_max_state);
#undef RUN
    printf("\n%d / %d tests passed\n", pass, total);
    return pass == total ? 0 : 1;
}
