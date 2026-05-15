/**
 * \file benchmark_async_streams.cpp
 * \brief Benchmark async double-buffered streams vs synchronous baseline.
 *
 * \par Description
 * Measures the throughput gain from overlapping H2D transfers with GPU compute
 * using CUDA streams and pinned (page-locked) host memory.
 *
 * \par Usage
 * \code
 * ./benchmark_async_streams [width] [height] [num_batches]
 * \endcode
 * Defaults: 512x512, 16 batches.
 */
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>

extern "C" {
    void async_benchmark(int width, int height, int ksize, int num_batches,
                         float* sync_ms, float* async_ms);
}

void ensure_reports_dir() {
    mkdir("/home/aniru/kernel-craft/reports", 0755);
}

int main(int argc, char** argv) {
    const int width      = (argc > 1) ? atoi(argv[1]) : 512;
    const int height     = (argc > 2) ? atoi(argv[2]) : 512;
    const int num_batches = (argc > 3) ? atoi(argv[3]) : 16;
    const int ksize      = 3;

    float sync_ms = 0.0f, async_ms = 0.0f;
    async_benchmark(width, height, ksize, num_batches, &sync_ms, &async_ms);

    double sync_throughput  = (double)(width * height * num_batches) / (sync_ms  * 1e-3) / 1e6;
    double async_throughput = (double)(width * height * num_batches) / (async_ms * 1e-3) / 1e6;
    double speedup = sync_ms / async_ms;

    printf("=== Async Streams Benchmark ===\n");
    printf("Image: %dx%d, batches: %d\n", width, height, num_batches);
    printf("Sync:  %.3f ms (%.3f MPixels/s)\n", sync_ms,  sync_throughput);
    printf("Async: %.3f ms (%.3f MPixels/s)\n", async_ms, async_throughput);
    printf("Speedup: %.2fx\n", speedup);

    ensure_reports_dir();
    FILE* f = fopen("/home/aniru/kernel-craft/reports/benchmark_async_streams.txt", "w");
    if (f) {
        fprintf(f, "=== Async Streams Benchmark ===\n");
        fprintf(f, "Image: %dx%d, batches: %d\n", width, height, num_batches);
        fprintf(f, "Sync:  %.3f ms (%.3f MPixels/s)\n", sync_ms,  sync_throughput);
        fprintf(f, "Async: %.3f ms (%.3f MPixels/s)\n", async_ms, async_throughput);
        fprintf(f, "Speedup: %.2fx\n", speedup);
        fclose(f);
    }
    return EXIT_SUCCESS;
}
