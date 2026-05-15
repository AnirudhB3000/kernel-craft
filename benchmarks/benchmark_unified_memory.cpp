/**
 * \file benchmark_unified_memory.cpp
 * \brief Benchmark unified memory (with/without prefetch) vs explicit transfers.
 *
 * \par Description
 * Compares three memory management strategies for convolution:
 * - Explicit cudaMemcpy (baseline)
 * - Unified memory without prefetch (page-fault driven)
 * - Unified memory with cudaMemPrefetchAsync
 *
 * \par Usage
 * \code
 * ./benchmark_unified_memory [width] [height]
 * \endcode
 * Defaults: 512x512.
 */
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>

extern "C" {
    void unified_benchmark(int width, int height, int ksize,
                           float* explicit_ms, float* um_fault_ms, float* um_prefetch_ms);
}

void ensure_reports_dir() {
    mkdir("/home/aniru/kernel-craft/reports", 0755);
}

int main(int argc, char** argv) {
    const int width  = (argc > 1) ? atoi(argv[1]) : 512;
    const int height = (argc > 2) ? atoi(argv[2]) : 512;
    const int ksize  = 3;

    float explicit_ms = 0.0f, fault_ms = 0.0f, prefetch_ms = 0.0f;
    unified_benchmark(width, height, ksize, &explicit_ms, &fault_ms, &prefetch_ms);

    printf("=== Unified Memory Benchmark ===\n");
    printf("Image: %dx%d\n", width, height);
    printf("Explicit cudaMemcpy: %.3f ms\n", explicit_ms);
    printf("Unified (no prefetch): %.3f ms\n", fault_ms);
    printf("Unified (prefetch):    %.3f ms\n", prefetch_ms);
    printf("Prefetch speedup vs no-prefetch: %.2fx\n", fault_ms / prefetch_ms);

    ensure_reports_dir();
    FILE* f = fopen("/home/aniru/kernel-craft/reports/benchmark_unified_memory.txt", "w");
    if (f) {
        fprintf(f, "=== Unified Memory Benchmark ===\n");
        fprintf(f, "Image: %dx%d\n", width, height);
        fprintf(f, "Explicit cudaMemcpy: %.3f ms\n", explicit_ms);
        fprintf(f, "Unified (no prefetch): %.3f ms\n", fault_ms);
        fprintf(f, "Unified (prefetch):    %.3f ms\n", prefetch_ms);
        fprintf(f, "Prefetch speedup vs no-prefetch: %.2fx\n", fault_ms / prefetch_ms);
        fclose(f);
    }
    return EXIT_SUCCESS;
}
