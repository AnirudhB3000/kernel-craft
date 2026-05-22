/**
 * \file benchmark_selective_scan.cpp
 * \brief Throughput benchmarks for Mamba selective scan and supporting ops.
 *
 * Measures:
 *  - Selective scan: throughput (Gflops, tokens/sec) vs sequence length L
 *  - Depthwise conv1d: throughput (GB/s) vs L
 *  - RMSNorm: throughput (GB/s) vs rows and hidden dim D
 *
 * Results are written to stdout and to reports/benchmark_selective_scan.txt.
 *
 * \par Selective scan flop count
 * Per (b, d, t): N_state * (1 expf + 3 muls + 2 adds) ≈ 6*N_state flops.
 * Total: B * L * D * 6 * N_state flops.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

extern "C" void launch_selective_scan(
    const float* d_u,     const float* d_A_log,
    const float* d_B,     const float* d_C,
    const float* d_delta, float* d_y,
    int B_, int L, int D, int N_state,
    cudaStream_t stream);

extern "C" void launch_depthwise_conv1d(
    const float* d_x, const float* d_w, const float* d_bias,
    float* d_y, int B_, int D, int L, int d_conv,
    cudaStream_t stream);

extern "C" void launch_rmsnorm(
    const float* d_x, const float* d_g, float* d_y,
    int rows, int D, float eps, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

static double bench_kernel(int reps, void (*fn)(void*), void* ctx)
{
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    /* Warm-up */
    fn(ctx);
    cudaDeviceSynchronize();

    cudaEventRecord(t0);
    for (int i = 0; i < reps; ++i) fn(ctx);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return (double)ms / reps;
}

// ---------------------------------------------------------------------------
// Selective scan benchmark
// ---------------------------------------------------------------------------

struct SsCtx {
    float *u, *A_log, *B, *C, *delta, *y;
    int B_, L, D, N;
};

static void run_ss(void* p) {
    auto* c = (SsCtx*)p;
    launch_selective_scan(c->u, c->A_log, c->B, c->C, c->delta, c->y,
                          c->B_, c->L, c->D, c->N, 0);
}

/**
 * \brief Benchmark selective scan for increasing sequence lengths.
 *
 * Fixed: B=1, D=512, N_state=16 (typical Mamba-370M config).
 */
static void bench_selective_scan(FILE* out)
{
    const int B_ = 1, D = 512, N = 16;
    const int Ls[] = {64, 256, 1024, 4096, 16384};
    const int REPS = 50;

    fprintf(out, "\n--- Selective Scan (B=%d D=%d N=%d) ---\n", B_, D, N);
    fprintf(out, "%-8s  %-10s  %-12s  %-12s\n",
            "L", "ms/iter", "Gflops", "Mtokens/s");

    for (int L : Ls) {
        size_t u_bytes    = (size_t)B_ * L * D  * sizeof(float);
        size_t alog_bytes = (size_t)D * N        * sizeof(float);
        size_t bn_bytes   = (size_t)B_ * L * N   * sizeof(float);

        float *du, *dAlog, *dB, *dC, *dd, *dy;
        cudaMalloc(&du,    u_bytes);
        cudaMalloc(&dAlog, alog_bytes);
        cudaMalloc(&dB,    bn_bytes);
        cudaMalloc(&dC,    bn_bytes);
        cudaMalloc(&dd,    u_bytes);
        cudaMalloc(&dy,    u_bytes);

        /* Initialize with random-ish values */
        cudaMemset(du, 0, u_bytes);
        cudaMemset(dAlog, 0, alog_bytes);
        cudaMemset(dB, 0, bn_bytes);
        cudaMemset(dC, 0, bn_bytes);
        cudaMemset(dd, 0, u_bytes);
        cudaMemset(dy, 0, u_bytes);

        SsCtx ctx = {du, dAlog, dB, dC, dd, dy, B_, L, D, N};
        double ms = bench_kernel(REPS, run_ss, &ctx);

        /* Flops: B * L * D * N_state * 6 (exp + 3 mul + 2 add per state) */
        double flops   = (double)B_ * L * D * N * 6.0;
        double gflops  = flops / (ms * 1e6);
        double mtok_s  = (double)B_ * L / (ms * 1e-3) / 1e6;

        fprintf(out, "%-8d  %-10.3f  %-12.2f  %-12.2f\n", L, ms, gflops, mtok_s);

        cudaFree(du); cudaFree(dAlog); cudaFree(dB);
        cudaFree(dC); cudaFree(dd);    cudaFree(dy);
    }
}

// ---------------------------------------------------------------------------
// Depthwise conv1d benchmark
// ---------------------------------------------------------------------------

struct DwCtx {
    float *x, *w, *y;
    int B_, D, L, d_conv;
};

static void run_dw(void* p) {
    auto* c = (DwCtx*)p;
    launch_depthwise_conv1d(c->x, c->w, nullptr, c->y, c->B_, c->D, c->L, c->d_conv, 0);
}

/**
 * \brief Benchmark depthwise conv1d: varying L, d_conv=4, B=1, D=2048.
 */
static void bench_depthwise_conv1d(FILE* out)
{
    const int B_ = 1, D = 2048, d_conv = 4;
    const int Ls[] = {64, 256, 1024, 4096, 16384};
    const int REPS = 100;

    fprintf(out, "\n--- Depthwise Conv1d (B=%d D=%d d_conv=%d) ---\n", B_, D, d_conv);
    fprintf(out, "%-8s  %-10s  %-10s\n", "L", "ms/iter", "GB/s");

    for (int L : Ls) {
        size_t x_bytes = (size_t)B_ * D * L * sizeof(float);
        size_t w_bytes = (size_t)D * d_conv    * sizeof(float);

        float *gx, *gw, *gy;
        cudaMalloc(&gx, x_bytes);
        cudaMalloc(&gw, w_bytes);
        cudaMalloc(&gy, x_bytes);
        cudaMemset(gx, 0, x_bytes);
        cudaMemset(gw, 0, w_bytes);

        DwCtx ctx = {gx, gw, gy, B_, D, L, d_conv};
        double ms = bench_kernel(REPS, run_dw, &ctx);

        /* Bytes: read x (d_conv passes per output) + write y */
        double bytes_io = (double)x_bytes * (1 + d_conv) + (double)x_bytes;
        double gbps = bytes_io / (ms * 1e-3) / 1e9;

        fprintf(out, "%-8d  %-10.3f  %-10.2f\n", L, ms, gbps);

        cudaFree(gx); cudaFree(gw); cudaFree(gy);
    }
}

// ---------------------------------------------------------------------------
// RMSNorm benchmark
// ---------------------------------------------------------------------------

struct RmsCtx {
    float *x, *g, *y;
    int rows, D;
    float eps;
};

static void run_rms(void* p) {
    auto* c = (RmsCtx*)p;
    launch_rmsnorm(c->x, c->g, c->y, c->rows, c->D, c->eps, 0);
}

/**
 * \brief Benchmark RMSNorm for typical hidden sizes.
 */
static void bench_rmsnorm(FILE* out)
{
    struct Cfg { int rows, D; };
    const Cfg cfgs[] = {
        {128, 512},
        {128, 1024},
        {128, 2048},
        {128, 4096},
        {512, 4096},
    };
    const int REPS = 200;

    fprintf(out, "\n--- RMSNorm ---\n");
    fprintf(out, "%-6s  %-6s  %-10s  %-10s\n", "rows", "D", "ms/iter", "GB/s");

    for (auto& cfg : cfgs) {
        size_t x_bytes = (size_t)cfg.rows * cfg.D * sizeof(float);
        size_t g_bytes = (size_t)cfg.D * sizeof(float);

        float *gx, *gg, *gy;
        cudaMalloc(&gx, x_bytes);
        cudaMalloc(&gg, g_bytes);
        cudaMalloc(&gy, x_bytes);
        cudaMemset(gx, 0, x_bytes);
        cudaMemset(gg, 0, g_bytes);

        RmsCtx ctx = {gx, gg, gy, cfg.rows, cfg.D, 1e-6f};
        double ms = bench_kernel(REPS, run_rms, &ctx);

        /* 2 reads (x, g) + 1 write (y) */
        double gbps = (2.0 * x_bytes + g_bytes) / (ms * 1e-3) / 1e9;

        fprintf(out, "%-6d  %-6d  %-10.3f  %-10.2f\n", cfg.rows, cfg.D, ms, gbps);

        cudaFree(gx); cudaFree(gg); cudaFree(gy);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main()
{
    printf("=== Mamba Kernel Benchmarks ===\n\n");

    const char* report_path = "reports/benchmark_selective_scan.txt";
    FILE* fout = fopen(report_path, "w");
    if (!fout) {
        fprintf(stderr, "Warning: cannot open %s for writing; printing to stdout only\n",
                report_path);
        fout = stdout;
    }

    bench_selective_scan(stdout);
    bench_depthwise_conv1d(stdout);
    bench_rmsnorm(stdout);

    if (fout != stdout) {
        bench_selective_scan(fout);
        bench_depthwise_conv1d(fout);
        bench_rmsnorm(fout);
        fclose(fout);
        printf("\nResults saved to %s\n", report_path);
    }

    return 0;
}
