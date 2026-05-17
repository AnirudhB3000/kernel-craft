/**
 * \file benchmark_flash_attention.cpp
 * \brief Benchmark FlashAttention vs naive attention for various configs.
 *
 * Measures throughput (tokens/sec) and memory bandwidth utilization.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <sys/stat.h>

extern "C" void launch_flash_attention(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int H_kv, int N, int d,
    bool causal, cudaStream_t stream);

static float benchmark_flash_attention(
    int B, int H, int H_kv, int N, int d, bool causal, int iters = 20)
{
    long long Q_sz  = (long long)B * H    * N * d;
    long long KV_sz = (long long)B * H_kv * N * d;

    std::vector<float> Q(Q_sz, 1.f), K(KV_sz, 1.f), V(KV_sz, 1.f);

    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, Q_sz  * sizeof(float));
    cudaMalloc(&dK, KV_sz * sizeof(float));
    cudaMalloc(&dV, KV_sz * sizeof(float));
    cudaMalloc(&dO, Q_sz  * sizeof(float));
    cudaMemcpy(dQ, Q.data(), Q_sz  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K.data(), KV_sz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), KV_sz * sizeof(float), cudaMemcpyHostToDevice);

    // Warmup
    launch_flash_attention(dQ, dK, dV, dO, B, H, H_kv, N, d, causal, 0);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i)
        launch_flash_attention(dQ, dK, dV, dO, B, H, H_kv, N, d, causal, 0);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms / iters;
}

int main() {
    printf("=== FlashAttention Benchmark ===\n");
    printf("%-20s %-6s %-6s %-6s %-6s %-10s %-12s %-12s\n",
           "Config", "B", "H", "N", "d", "Causal", "Time(ms)", "Tput(Gflops)");

    struct Config { int B, H, H_kv, N, d; bool causal; const char* name; };
    Config configs[] = {
        {1, 8, 8,  512, 64,  false, "MHA-512"},
        {1, 8, 8, 1024, 64,  false, "MHA-1024"},
        {1, 8, 8, 2048, 64,  false, "MHA-2048"},
        {1, 8, 8,  512, 64,  true,  "Causal-512"},
        {1, 8, 8, 1024, 64,  true,  "Causal-1024"},
        {1, 8, 2,  512, 64,  false, "GQA-4:1-512"},
        {1, 8, 1,  512, 64,  false, "MQA-512"},
        {2, 4, 4,  512, 128, false, "d128-B2"},
    };

    mkdir("/home/aniru/kernel-craft/reports", 0755);
    FILE* rpt = fopen("/home/aniru/kernel-craft/reports/benchmark_flash_attention.txt", "w");
    if (rpt) fprintf(rpt, "FlashAttention Benchmark\n\n");

    for (auto& cfg : configs) {
        float ms = benchmark_flash_attention(
            cfg.B, cfg.H, cfg.H_kv, cfg.N, cfg.d, cfg.causal);
        // FLOPS: 2*B*H*N*N*d (QK) + 2*B*H*N*N*d (AV) ≈ 4*B*H*N²*d
        double flops = 4.0 * cfg.B * cfg.H * (double)cfg.N * cfg.N * cfg.d;
        double gflops = flops / (ms * 1e6);

        printf("%-20s %-6d %-6d %-6d %-6d %-10s %-12.3f %-12.2f\n",
               cfg.name, cfg.B, cfg.H, cfg.N, cfg.d,
               cfg.causal ? "yes" : "no", ms, gflops);

        if (rpt)
            fprintf(rpt, "%-20s B=%d H=%d N=%d d=%d causal=%s  %.3f ms  %.2f Gflops\n",
                    cfg.name, cfg.B, cfg.H, cfg.N, cfg.d,
                    cfg.causal ? "yes" : "no", ms, gflops);
    }

    if (rpt) fclose(rpt);
    printf("\nReport written to reports/benchmark_flash_attention.txt\n");
    return 0;
}
