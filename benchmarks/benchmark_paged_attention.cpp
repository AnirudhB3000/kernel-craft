/**
 * \file benchmark_paged_attention.cpp
 * \brief Benchmark PagedAttention decode across page sizes and sequence lengths.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <sys/stat.h>

extern "C" void launch_paged_attention(
    const float* Q, const int* block_table,
    const float* K_pool, const float* V_pool,
    float* O, const int* seq_lens,
    int B, int H, int H_kv, int d,
    int block_size, int max_blocks, cudaStream_t stream);

static float benchmark_paged(
    int B, int H, int H_kv, int d, int seq_len, int block_size, int iters = 20)
{
    int max_blocks = (seq_len + block_size - 1) / block_size;
    int num_phys   = B * max_blocks;

    std::vector<float> Q(B * H * d, 1.f);
    std::vector<float> K_pool((long long)num_phys * block_size * H_kv * d, 0.1f);
    std::vector<float> V_pool((long long)num_phys * block_size * H_kv * d, 0.1f);
    std::vector<int>   bt(B * max_blocks);
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < max_blocks; ++i)
            bt[b * max_blocks + i] = b * max_blocks + i;
    std::vector<int> sl(B, seq_len);

    float *dQ, *dKp, *dVp, *dO; int *dBT, *dSL;
    cudaMalloc(&dQ,  B * H * d                        * sizeof(float));
    cudaMalloc(&dKp, (long long)num_phys * block_size * H_kv * d * sizeof(float));
    cudaMalloc(&dVp, (long long)num_phys * block_size * H_kv * d * sizeof(float));
    cudaMalloc(&dO,  B * H * d                        * sizeof(float));
    cudaMalloc(&dBT, B * max_blocks                   * sizeof(int));
    cudaMalloc(&dSL, B                                * sizeof(int));

    cudaMemcpy(dQ,  Q.data(),      B * H * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dKp, K_pool.data(), (long long)num_phys * block_size * H_kv * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dVp, V_pool.data(), (long long)num_phys * block_size * H_kv * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dBT, bt.data(),     B * max_blocks * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(dSL, sl.data(),     B             * sizeof(int),   cudaMemcpyHostToDevice);

    // Warmup
    launch_paged_attention(dQ, dBT, dKp, dVp, dO, dSL, B, H, H_kv, d, block_size, max_blocks, 0);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i)
        launch_paged_attention(dQ, dBT, dKp, dVp, dO, dSL, B, H, H_kv, d, block_size, max_blocks, 0);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);

    cudaFree(dQ); cudaFree(dKp); cudaFree(dVp); cudaFree(dO); cudaFree(dBT); cudaFree(dSL);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms / iters;
}

int main() {
    printf("=== PagedAttention Benchmark ===\n");
    printf("%-10s %-8s %-8s %-8s %-10s\n",
           "Seq", "Block", "B", "H", "Time(ms)");

    mkdir("/home/aniru/kernel-craft/reports", 0755);
    FILE* rpt = fopen("/home/aniru/kernel-craft/reports/benchmark_paged_attention.txt", "w");
    if (rpt) fprintf(rpt, "PagedAttention Benchmark\n\n");

    struct Cfg { int B, H, H_kv, d, seq_len, block_size; };
    Cfg configs[] = {
        {1, 8, 8, 64,  256, 16},
        {1, 8, 8, 64,  256, 32},
        {1, 8, 8, 64,  256, 64},
        {1, 8, 8, 64, 1024, 16},
        {1, 8, 8, 64, 1024, 32},
        {1, 8, 8, 64, 1024, 64},
        {4, 8, 8, 64,  512, 32},
        {1, 8, 2, 64, 1024, 32},
    };

    for (auto& c : configs) {
        float ms = benchmark_paged(c.B, c.H, c.H_kv, c.d, c.seq_len, c.block_size);
        printf("%-10d %-8d %-8d %-8d %-10.3f\n", c.seq_len, c.block_size, c.B, c.H, ms);
        if (rpt)
            fprintf(rpt, "seq=%d block=%d B=%d H=%d H_kv=%d d=%d  %.3f ms\n",
                    c.seq_len, c.block_size, c.B, c.H, c.H_kv, c.d, ms);
    }

    if (rpt) fclose(rpt);
    printf("\nReport written to reports/benchmark_paged_attention.txt\n");
    return 0;
}
