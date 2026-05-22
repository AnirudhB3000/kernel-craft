/**
 * \file test_mamba_ops.cpp
 * \brief Correctness tests for depthwise conv1d and RMSNorm kernels.
 *
 * Tests:
 *  - depthwise_conv1d: causal output matches CPU loop; bias-free and bias variants.
 *  - rmsnorm: output matches CPU formula; various batch+dim combinations.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

extern "C" void launch_depthwise_conv1d(
    const float* d_x, const float* d_w, const float* d_bias,
    float* d_y, int B_, int D, int L, int d_conv,
    cudaStream_t stream);

extern "C" void launch_rmsnorm(
    const float* d_x, const float* d_g, float* d_y,
    int rows, int D, float eps, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void fill_rand(std::vector<float>& v, float lo, float hi) {
    for (auto& x : v) x = lo + (hi - lo) * (float)rand() / RAND_MAX;
}

static float max_abs_err(const float* a, const float* b, int n) {
    float e = 0.f;
    for (int i = 0; i < n; ++i) e = std::max(e, std::fabs(a[i] - b[i]));
    return e;
}

struct GPU {
    float* ptr = nullptr;
    explicit GPU(size_t bytes) { cudaMalloc(&ptr, bytes); }
    ~GPU() { cudaFree(ptr); }
    void upload(const void* h, size_t bytes) { cudaMemcpy(ptr, h, bytes, cudaMemcpyHostToDevice); }
    void download(void* h, size_t bytes) const { cudaMemcpy(h, ptr, bytes, cudaMemcpyDeviceToHost); }
};

// ---------------------------------------------------------------------------
// CPU references
// ---------------------------------------------------------------------------

/**
 * \brief Causal depthwise conv1d reference.
 *
 * \param[in]  x      [B, D, L]
 * \param[in]  w      [D, d_conv]
 * \param[in]  bias   [D] (may be nullptr)
 * \param[out] y      [B, D, L]
 */
static void depthwise_conv1d_cpu(
    const float* x, const float* w, const float* bias,
    float* y, int B_, int D, int L, int d_conv)
{
    for (int b = 0; b < B_; ++b)
    for (int d = 0; d < D;  ++d) {
        const float* x_bd = x + (b * D + d) * L;
        const float* w_d  = w + d * d_conv;
        float* y_bd = y + (b * D + d) * L;
        for (int t = 0; t < L; ++t) {
            float val = 0.f;
            for (int k = 0; k < d_conv; ++k) {
                int src = t - (d_conv - 1 - k);
                if (src >= 0) val += x_bd[src] * w_d[k];
            }
            if (bias) val += bias[d];
            y_bd[t] = val;
        }
    }
}

/**
 * \brief RMSNorm reference.
 *
 * \param[in]  x    [rows, D]
 * \param[in]  g    [D]
 * \param[out] y    [rows, D]
 */
static void rmsnorm_cpu(
    const float* x, const float* g, float* y,
    int rows, int D, float eps)
{
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + r * D;
        float* yr = y + r * D;
        float ss = 0.f;
        for (int i = 0; i < D; ++i) ss += xr[i] * xr[i];
        float rms_inv = 1.f / std::sqrt(ss / D + eps);
        for (int i = 0; i < D; ++i) yr[i] = g[i] * xr[i] * rms_inv;
    }
}

// ---------------------------------------------------------------------------
// depthwise_conv1d tests
// ---------------------------------------------------------------------------

/**
 * \brief Basic 1-channel, d_conv=4, L=8 — hand-checkable.
 */
static bool test_dwconv_basic()
{
    int B_ = 1, D = 1, L = 8, d_conv = 4;
    std::vector<float> x(B_*D*L), w(D*d_conv), y_cpu(B_*D*L, 0.f), y_gpu(B_*D*L, 0.f);
    srand(1);
    fill_rand(x, 0.f, 1.f);
    fill_rand(w, -1.f, 1.f);

    depthwise_conv1d_cpu(x.data(), w.data(), nullptr, y_cpu.data(), B_, D, L, d_conv);

    GPU gx(B_*D*L*4), gw(D*d_conv*4), gy(B_*D*L*4);
    gx.upload(x.data(), B_*D*L*4);
    gw.upload(w.data(), D*d_conv*4);
    launch_depthwise_conv1d(gx.ptr, gw.ptr, nullptr, gy.ptr, B_, D, L, d_conv, 0);
    cudaDeviceSynchronize();
    gy.download(y_gpu.data(), B_*D*L*4);

    float err = max_abs_err(y_cpu.data(), y_gpu.data(), B_*D*L);
    printf("[test_dwconv_basic] max_abs_err=%.2e %s\n", err, err < 1e-5f ? "PASS" : "FAIL");
    return err < 1e-5f;
}

/**
 * \brief Multi-channel, d_conv=4, with bias.
 */
static bool test_dwconv_multichannel()
{
    int B_ = 2, D = 32, L = 64, d_conv = 4;
    std::vector<float> x(B_*D*L), w(D*d_conv), bias(D);
    std::vector<float> y_cpu(B_*D*L, 0.f), y_gpu(B_*D*L, 0.f);
    srand(42);
    fill_rand(x, -1.f, 1.f);
    fill_rand(w, -0.5f, 0.5f);
    fill_rand(bias, -0.1f, 0.1f);

    depthwise_conv1d_cpu(x.data(), w.data(), bias.data(), y_cpu.data(), B_, D, L, d_conv);

    GPU gx(B_*D*L*4), gw(D*d_conv*4), gbias(D*4), gy(B_*D*L*4);
    gx.upload(x.data(), B_*D*L*4);
    gw.upload(w.data(), D*d_conv*4);
    gbias.upload(bias.data(), D*4);
    launch_depthwise_conv1d(gx.ptr, gw.ptr, gbias.ptr, gy.ptr, B_, D, L, d_conv, 0);
    cudaDeviceSynchronize();
    gy.download(y_gpu.data(), B_*D*L*4);

    float err = max_abs_err(y_cpu.data(), y_gpu.data(), B_*D*L);
    printf("[test_dwconv_multichannel] max_abs_err=%.2e %s\n", err, err < 1e-4f ? "PASS" : "FAIL");
    return err < 1e-4f;
}

/**
 * \brief d_conv=1 (no context): y[b,d,t] = x[b,d,t] * w[d,0].
 */
static bool test_dwconv_kernel1()
{
    int B_ = 1, D = 8, L = 16, d_conv = 1;
    std::vector<float> x(B_*D*L), w(D*1);
    std::vector<float> y_cpu(B_*D*L, 0.f), y_gpu(B_*D*L, 0.f);
    srand(7);
    fill_rand(x, -1.f, 1.f);
    fill_rand(w, -1.f, 1.f);

    depthwise_conv1d_cpu(x.data(), w.data(), nullptr, y_cpu.data(), B_, D, L, d_conv);

    GPU gx(B_*D*L*4), gw(D*4), gy(B_*D*L*4);
    gx.upload(x.data(), B_*D*L*4);
    gw.upload(w.data(), D*4);
    launch_depthwise_conv1d(gx.ptr, gw.ptr, nullptr, gy.ptr, B_, D, L, d_conv, 0);
    cudaDeviceSynchronize();
    gy.download(y_gpu.data(), B_*D*L*4);

    float err = max_abs_err(y_cpu.data(), y_gpu.data(), B_*D*L);
    printf("[test_dwconv_kernel1] max_abs_err=%.2e %s\n", err, err < 1e-5f ? "PASS" : "FAIL");
    return err < 1e-5f;
}

/**
 * \brief Causality check: output at t=0 only depends on x[t=0] (d_conv=4 → 3 zero-padded).
 */
static bool test_dwconv_causal()
{
    int B_ = 1, D = 1, L = 8, d_conv = 4;
    std::vector<float> x(B_*D*L, 0.f), w(D*d_conv, 1.f);
    x[0] = 1.f;  /* only position 0 is non-zero */

    std::vector<float> y_cpu(B_*D*L, 0.f), y_gpu(B_*D*L, 0.f);
    depthwise_conv1d_cpu(x.data(), w.data(), nullptr, y_cpu.data(), B_, D, L, d_conv);

    GPU gx(B_*D*L*4), gw(D*d_conv*4), gy(B_*D*L*4);
    gx.upload(x.data(), B_*D*L*4);
    gw.upload(w.data(), D*d_conv*4);
    launch_depthwise_conv1d(gx.ptr, gw.ptr, nullptr, gy.ptr, B_, D, L, d_conv, 0);
    cudaDeviceSynchronize();
    gy.download(y_gpu.data(), B_*D*L*4);

    /* x[0]=1 should appear in y[0..d_conv-1] (with causal padding zeros before) */
    bool causal_ok = (y_gpu[0] > 0.f);                     /* y[0] sees x[0] */
    bool no_future = (std::fabs(y_gpu[0] - y_cpu[0]) < 1e-5f);

    float err = max_abs_err(y_cpu.data(), y_gpu.data(), B_*D*L);
    bool pass = causal_ok && no_future && err < 1e-5f;
    printf("[test_dwconv_causal] causal_ok=%d max_abs_err=%.2e %s\n",
           (int)causal_ok, err, pass ? "PASS" : "FAIL");
    return pass;
}

// ---------------------------------------------------------------------------
// rmsnorm tests
// ---------------------------------------------------------------------------

/**
 * \brief Basic: B=2, T=4, D=8.
 */
static bool test_rmsnorm_basic()
{
    int rows = 8, D = 8;
    float eps = 1e-6f;
    std::vector<float> x(rows*D), g(D);
    std::vector<float> y_cpu(rows*D, 0.f), y_gpu(rows*D, 0.f);
    srand(3);
    fill_rand(x, -1.f, 1.f);
    fill_rand(g,  0.5f, 1.5f);

    rmsnorm_cpu(x.data(), g.data(), y_cpu.data(), rows, D, eps);

    GPU gx(rows*D*4), gg(D*4), gy(rows*D*4);
    gx.upload(x.data(), rows*D*4);
    gg.upload(g.data(), D*4);
    launch_rmsnorm(gx.ptr, gg.ptr, gy.ptr, rows, D, eps, 0);
    cudaDeviceSynchronize();
    gy.download(y_gpu.data(), rows*D*4);

    float err = max_abs_err(y_cpu.data(), y_gpu.data(), rows*D);
    printf("[test_rmsnorm_basic] max_abs_err=%.2e %s\n", err, err < 1e-5f ? "PASS" : "FAIL");
    return err < 1e-5f;
}

/**
 * \brief Large hidden dim: rows=128, D=768.
 */
static bool test_rmsnorm_large_dim()
{
    int rows = 128, D = 768;
    float eps = 1e-6f;
    std::vector<float> x(rows*D), g(D);
    std::vector<float> y_cpu(rows*D, 0.f), y_gpu(rows*D, 0.f);
    srand(77);
    fill_rand(x, -2.f, 2.f);
    fill_rand(g,  0.8f, 1.2f);

    rmsnorm_cpu(x.data(), g.data(), y_cpu.data(), rows, D, eps);

    GPU gx(rows*D*4), gg(D*4), gy(rows*D*4);
    gx.upload(x.data(), rows*D*4);
    gg.upload(g.data(), D*4);
    launch_rmsnorm(gx.ptr, gg.ptr, gy.ptr, rows, D, eps, 0);
    cudaDeviceSynchronize();
    gy.download(y_gpu.data(), rows*D*4);

    float err = max_abs_err(y_cpu.data(), y_gpu.data(), rows*D);
    printf("[test_rmsnorm_large_dim] max_abs_err=%.2e %s\n", err, err < 1e-4f ? "PASS" : "FAIL");
    return err < 1e-4f;
}

/**
 * \brief D=4096 (Mamba-2.8B hidden dim).
 */
static bool test_rmsnorm_d4096()
{
    int rows = 16, D = 4096;
    float eps = 1e-6f;
    std::vector<float> x(rows*D), g(D);
    std::vector<float> y_cpu(rows*D, 0.f), y_gpu(rows*D, 0.f);
    srand(13);
    fill_rand(x, -1.f, 1.f);
    fill_rand(g,  0.9f, 1.1f);

    rmsnorm_cpu(x.data(), g.data(), y_cpu.data(), rows, D, eps);

    GPU gx(rows*D*4), gg(D*4), gy(rows*D*4);
    gx.upload(x.data(), rows*D*4);
    gg.upload(g.data(), D*4);
    launch_rmsnorm(gx.ptr, gg.ptr, gy.ptr, rows, D, eps, 0);
    cudaDeviceSynchronize();
    gy.download(y_gpu.data(), rows*D*4);

    float err = max_abs_err(y_cpu.data(), y_gpu.data(), rows*D);
    printf("[test_rmsnorm_d4096] max_abs_err=%.2e %s\n", err, err < 1e-4f ? "PASS" : "FAIL");
    return err < 1e-4f;
}

/**
 * \brief Constant input x=c, g=1: output should be 1 everywhere.
 */
static bool test_rmsnorm_constant()
{
    int rows = 4, D = 64;
    float eps = 1e-6f;
    std::vector<float> x(rows*D, 2.f), g(D, 1.f);
    std::vector<float> y_gpu(rows*D, 0.f);

    GPU gx(rows*D*4), gg(D*4), gy(rows*D*4);
    gx.upload(x.data(), rows*D*4);
    gg.upload(g.data(), D*4);
    launch_rmsnorm(gx.ptr, gg.ptr, gy.ptr, rows, D, eps, 0);
    cudaDeviceSynchronize();
    gy.download(y_gpu.data(), rows*D*4);

    /* x=c, g=1 → rms=c → y=1 */
    float err = max_abs_err(y_gpu.data(), std::vector<float>(rows*D, 1.f).data(), rows*D);
    printf("[test_rmsnorm_constant] max_abs_err=%.2e %s\n", err, err < 1e-5f ? "PASS" : "FAIL");
    return err < 1e-5f;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main()
{
    printf("=== Mamba Ops Tests (depthwise_conv1d + rmsnorm) ===\n");
    int pass = 0, total = 0;
#define RUN(fn) do { ++total; if (fn()) ++pass; } while(0)
    RUN(test_dwconv_basic);
    RUN(test_dwconv_multichannel);
    RUN(test_dwconv_kernel1);
    RUN(test_dwconv_causal);
    RUN(test_rmsnorm_basic);
    RUN(test_rmsnorm_large_dim);
    RUN(test_rmsnorm_d4096);
    RUN(test_rmsnorm_constant);
#undef RUN
    printf("\n%d / %d tests passed\n", pass, total);
    return pass == total ? 0 : 1;
}
