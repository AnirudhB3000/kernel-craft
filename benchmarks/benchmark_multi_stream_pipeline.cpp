/**
 * \file benchmark_multi_stream_pipeline.cpp
 * \brief Benchmark multi-stream (concurrent preprocess+inference) vs single-stream.
 *
 * \par Description
 * Measures the latency benefit of running GPU preprocessing (resize + normalize)
 * concurrently with convolution inference using separate CUDA streams.
 *
 * \par Usage
 * \code
 * ./benchmark_multi_stream_pipeline [src_w] [src_h] [dst_w] [dst_h] [iterations]
 * \endcode
 * Defaults: 1024x1024 → 512x512, 50 iterations.
 */
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>

extern "C" {
    void multi_stream_benchmark(int src_w, int src_h, int dst_w, int dst_h,
                                int ksize, int num_iterations,
                                float* single_ms, float* multi_ms);
}

void ensure_reports_dir() {
    mkdir("/home/aniru/kernel-craft/reports", 0755);
}

int main(int argc, char** argv) {
    const int src_w = (argc > 1) ? atoi(argv[1]) : 1024;
    const int src_h = (argc > 2) ? atoi(argv[2]) : 1024;
    const int dst_w = (argc > 3) ? atoi(argv[3]) : 512;
    const int dst_h = (argc > 4) ? atoi(argv[4]) : 512;
    const int iters = (argc > 5) ? atoi(argv[5]) : 50;
    const int ksize = 3;

    float single_ms = 0.0f, multi_ms = 0.0f;
    multi_stream_benchmark(src_w, src_h, dst_w, dst_h, ksize, iters,
                           &single_ms, &multi_ms);

    double speedup = single_ms / multi_ms;

    printf("=== Multi-Stream Pipeline Benchmark ===\n");
    printf("Src: %dx%d -> Dst: %dx%d, iterations: %d\n",
           src_w, src_h, dst_w, dst_h, iters);
    printf("Single-stream: %.3f ms\n", single_ms);
    printf("Multi-stream:  %.3f ms\n", multi_ms);
    printf("Speedup: %.2fx\n", speedup);

    ensure_reports_dir();
    FILE* f = fopen("/home/aniru/kernel-craft/reports/benchmark_multi_stream_pipeline.txt", "w");
    if (f) {
        fprintf(f, "=== Multi-Stream Pipeline Benchmark ===\n");
        fprintf(f, "Src: %dx%d -> Dst: %dx%d, iterations: %d\n",
                src_w, src_h, dst_w, dst_h, iters);
        fprintf(f, "Single-stream: %.3f ms\n", single_ms);
        fprintf(f, "Multi-stream:  %.3f ms\n", multi_ms);
        fprintf(f, "Speedup: %.2fx\n", speedup);
        fclose(f);
    }
    return EXIT_SUCCESS;
}
