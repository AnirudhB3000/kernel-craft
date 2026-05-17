/**
 * \file benchmark_quant.cpp
 * \brief Benchmark INT4 dequantization, GEMV, and FP8 quantization throughput.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <sys/stat.h>

extern "C" void launch_dequantize_int4(
    const uint8_t* d_packed, const float* d_scales, const uint8_t* d_zeros,
    float* d_output, int rows, int cols, int group_size, cudaStream_t stream);

extern "C" void launch_gemv_int4(
    const uint8_t* d_packed, const float* d_scales, const uint8_t* d_zeros,
    const float* d_x, float* d_y, int rows, int cols, int group_size,
    cudaStream_t stream);

extern "C" void launch_quantize_fp8(
    const float* d_input, int8_t* d_output, float* d_scales,
    int rows, int cols, bool per_token, cudaStream_t stream);

extern "C" void launch_dequantize_fp8(
    const int8_t* d_input, const float* d_scales, float* d_output,
    int rows, int cols, bool per_token, cudaStream_t stream);

template <typename Fn>
static float time_kernel(int iters, Fn fn) {
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    fn();  // warmup
    cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i) fn();
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms / iters;
}

int main() {
    printf("=== Quantization Benchmark ===\n");
    mkdir("/home/aniru/kernel-craft/reports", 0755);
    FILE* rpt = fopen("/home/aniru/kernel-craft/reports/benchmark_quant.txt", "w");
    if (rpt) fprintf(rpt, "Quantization Benchmark\n\n");

    // --- INT4 dequantization throughput ---
    printf("\nINT4 Dequantization (rows x cols, group_size=128)\n");
    printf("%-20s %-12s %-14s\n", "Config", "Time(ms)", "GB/s");
    if (rpt) fprintf(rpt, "INT4 Dequantization\n");

    for (auto [rows, cols] : std::vector<std::pair<int,int>>{{4096, 4096}, {4096, 8192}, {8192, 4096}}) {
        int gs = 128;
        int pc = cols / 2, sc = cols / gs, zc = sc / 2 + sc % 2;

        std::vector<uint8_t> packed(rows * pc, 0x77);
        std::vector<float>   scales(rows * sc, 0.1f);
        std::vector<uint8_t> zeros(rows * zc, 0x00);
        uint8_t *dp, *dz; float *ds, *dout;
        cudaMalloc(&dp,   rows * pc * sizeof(uint8_t));
        cudaMalloc(&dz,   rows * zc * sizeof(uint8_t));
        cudaMalloc(&ds,   rows * sc * sizeof(float));
        cudaMalloc(&dout, rows * cols * sizeof(float));
        cudaMemcpy(dp, packed.data(), rows * pc * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dz, zeros.data(),  rows * zc * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(ds, scales.data(), rows * sc * sizeof(float),   cudaMemcpyHostToDevice);

        float ms = time_kernel(50, [&]{
            launch_dequantize_int4(dp, ds, dz, dout, rows, cols, gs, 0);
        });
        // Bandwidth: read packed (rows*pc bytes) + write output (rows*cols*4 bytes)
        double bytes = (double)rows * pc + (double)rows * cols * 4;
        double gbs = bytes / (ms * 1e6);
        printf("  %4d x %4d          %-12.3f %-14.2f\n", rows, cols, ms, gbs);
        if (rpt) fprintf(rpt, "  INT4-deq %dx%d: %.3f ms, %.2f GB/s\n", rows, cols, ms, gbs);
        cudaFree(dp); cudaFree(dz); cudaFree(ds); cudaFree(dout);
    }

    // --- INT4 GEMV throughput ---
    printf("\nINT4 GEMV (y = W_int4 * x, group_size=128)\n");
    for (auto [rows, cols] : std::vector<std::pair<int,int>>{{4096, 4096}, {4096, 8192}}) {
        int gs = 128;
        int pc = cols / 2, sc = cols / gs, zc = sc / 2 + sc % 2;

        std::vector<uint8_t> packed(rows * pc, 0x77);
        std::vector<float>   scales(rows * sc, 0.1f);
        std::vector<uint8_t> zeros(rows * zc, 0x00);
        std::vector<float>   x(cols, 1.f);
        uint8_t *dp, *dz; float *ds, *dx, *dy;
        cudaMalloc(&dp, rows * pc * sizeof(uint8_t));
        cudaMalloc(&dz, rows * zc * sizeof(uint8_t));
        cudaMalloc(&ds, rows * sc * sizeof(float));
        cudaMalloc(&dx, cols      * sizeof(float));
        cudaMalloc(&dy, rows      * sizeof(float));
        cudaMemcpy(dp, packed.data(), rows * pc * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dz, zeros.data(),  rows * zc * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(ds, scales.data(), rows * sc * sizeof(float),   cudaMemcpyHostToDevice);
        cudaMemcpy(dx, x.data(),      cols      * sizeof(float),   cudaMemcpyHostToDevice);

        float ms = time_kernel(50, [&]{
            launch_gemv_int4(dp, ds, dz, dx, dy, rows, cols, gs, 0);
        });
        double gflops = 2.0 * rows * cols / (ms * 1e6);
        printf("  %4d x %4d          %-12.3f %-14.2f Gflops\n", rows, cols, ms, gflops);
        if (rpt) fprintf(rpt, "  INT4-gemv %dx%d: %.3f ms, %.2f Gflops\n", rows, cols, ms, gflops);
        cudaFree(dp); cudaFree(dz); cudaFree(ds); cudaFree(dx); cudaFree(dy);
    }

    // --- FP8 quantization throughput ---
    printf("\nFP8 Per-Token Quantization (rows x cols)\n");
    for (auto [rows, cols] : std::vector<std::pair<int,int>>{{512, 4096}, {1024, 4096}, {512, 8192}}) {
        std::vector<float> inp(rows * cols, 1.f);
        float *di, *ds_fp8, *dout_f; int8_t *dq;
        cudaMalloc(&di,      rows * cols * sizeof(float));
        cudaMalloc(&ds_fp8,  rows        * sizeof(float));
        cudaMalloc(&dout_f,  rows * cols * sizeof(float));
        cudaMalloc(&dq,      rows * cols * sizeof(int8_t));
        cudaMemcpy(di, inp.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);

        float ms_q = time_kernel(50, [&]{
            launch_quantize_fp8(di, dq, ds_fp8, rows, cols, true, 0);
        });
        float ms_dq = time_kernel(50, [&]{
            launch_dequantize_fp8(dq, ds_fp8, dout_f, rows, cols, true, 0);
        });
        double bytes = (double)rows * cols * sizeof(float);
        double gbs_q  = bytes / (ms_q  * 1e6);
        double gbs_dq = bytes / (ms_dq * 1e6);
        printf("  %4d x %4d  quant: %-8.3f ms (%.1f GB/s)  dequant: %-8.3f ms (%.1f GB/s)\n",
               rows, cols, ms_q, gbs_q, ms_dq, gbs_dq);
        if (rpt)
            fprintf(rpt, "  FP8 %dx%d: quant=%.3fms(%.1fGB/s) dequant=%.3fms(%.1fGB/s)\n",
                    rows, cols, ms_q, gbs_q, ms_dq, gbs_dq);
        cudaFree(di); cudaFree(ds_fp8); cudaFree(dout_f); cudaFree(dq);
    }

    if (rpt) fclose(rpt);
    printf("\nReport written to reports/benchmark_quant.txt\n");
    return 0;
}
