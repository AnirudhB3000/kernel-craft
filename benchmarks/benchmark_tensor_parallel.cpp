/**
 * \file benchmark_tensor_parallel.cpp
 * \brief Bandwidth and throughput benchmarks for tensor-parallelism primitives.
 *
 * Measures:
 *  - Simulation ring all-reduce: effective bandwidth (GB/s) vs message size
 *  - Simulation all-gather: effective bandwidth (GB/s) vs message size
 *  - Col-parallel linear TFLOPS vs matrix dimensions
 *  - Row-parallel linear TFLOPS vs matrix dimensions
 *  - NCCL ring all-reduce bandwidth vs message size (requires HAVE_NCCL)
 *
 * Results are written to stdout and appended to reports/benchmark_tensor_parallel.txt.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

extern "C" void launch_ring_allreduce(
    float** d_bufs, int count, int num_ranks, cudaStream_t stream);

extern "C" void launch_allgather(
    float** d_chunks, float* d_output, int chunk_size, int num_ranks,
    cudaStream_t stream);

extern "C" void launch_col_parallel_linear(
    const float* d_x, const float* d_W, float* d_y,
    int M, int N_rank, int K, cudaStream_t stream);

extern "C" void launch_row_parallel_linear(
    const float* d_x_rank, const float* d_W, float* d_out,
    int M, int N, int K_rank, cudaStream_t stream);

#ifdef HAVE_NCCL
#include <nccl.h>

extern "C" void nccl_comm_init_all(void** comms, int num_comms, const int* devs);
extern "C" void nccl_comm_destroy(void* comm);
extern "C" void nccl_group_start(void);
extern "C" void nccl_group_end(void);
extern "C" void launch_ring_allreduce_nccl(
    void* comm, float* d_buf, int count, cudaStream_t stream);
#endif

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

static double bench_ms(cudaEvent_t start, cudaEvent_t stop, int reps,
                       void (*run)(void*), void* ctx)
{
    cudaEventRecord(start);
    for (int i = 0; i < reps; ++i) run(ctx);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    return (double)ms / reps;
}

// ---------------------------------------------------------------------------
// Simulation all-reduce bandwidth
// ---------------------------------------------------------------------------

struct AllreduceCtx {
    float** d_bufs;
    int count;
    int num_ranks;
};
static void run_allreduce(void* p) {
    auto* c = (AllreduceCtx*)p;
    launch_ring_allreduce(c->d_bufs, c->count, c->num_ranks, 0);
}

static void bench_sim_allreduce(FILE* fp, int num_ranks)
{
    // Message sizes: 1 KB → 64 MB (in floats)
    const int sizes[] = {256, 4096, 65536, 1<<20, 1<<24};
    const int nsizes  = (int)(sizeof(sizes) / sizeof(sizes[0]));
    const int reps    = 20;

    printf("\n[Sim ring all-reduce, %d ranks] count -> GB/s\n", num_ranks);
    fprintf(fp, "\n[Sim ring all-reduce, %d ranks]\n", num_ranks);
    fprintf(fp, "  count     bytes       ms      GB/s\n");

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    for (int si = 0; si < nsizes; ++si) {
        int count = sizes[si];
        size_t bytes = (size_t)count * sizeof(float);

        std::vector<float*> d_bufs(num_ranks);
        for (int r = 0; r < num_ranks; ++r) {
            cudaMalloc(&d_bufs[r], bytes);
            cudaMemset(d_bufs[r], 0, bytes);
        }

        AllreduceCtx ctx = { d_bufs.data(), count, num_ranks };

        // Warmup
        for (int w = 0; w < 3; ++w) run_allreduce(&ctx);
        cudaDeviceSynchronize();

        cudaEventRecord(ev_start);
        for (int i = 0; i < reps; ++i) run_allreduce(&ctx);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);

        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        double ms_avg = (double)ms / reps;

        // Effective bandwidth: 2*(R-1)/R * bytes in + bytes out (ring allreduce model)
        double eff_bytes = 2.0 * (num_ranks - 1.0) / num_ranks * bytes;
        double gbps = (eff_bytes / 1e9) / (ms_avg / 1e3);

        printf("  count=%7d  %6.1f KB  %7.3f ms  %6.2f GB/s\n",
               count, bytes / 1024.0, ms_avg, gbps);
        fprintf(fp, "  %8d  %8zu  %7.3f  %6.2f\n", count, bytes, ms_avg, gbps);

        for (int r = 0; r < num_ranks; ++r) cudaFree(d_bufs[r]);
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
}

// ---------------------------------------------------------------------------
// Simulation all-gather bandwidth
// ---------------------------------------------------------------------------

struct AllgatherCtx {
    float** d_chunks;
    float*  d_output;
    int chunk_size;
    int num_ranks;
};
static void run_allgather(void* p) {
    auto* c = (AllgatherCtx*)p;
    launch_allgather(c->d_chunks, c->d_output, c->chunk_size, c->num_ranks, 0);
}

static void bench_sim_allgather(FILE* fp, int num_ranks)
{
    const int sizes[] = {256, 4096, 65536, 1<<20, 1<<24};
    const int nsizes  = (int)(sizeof(sizes) / sizeof(sizes[0]));
    const int reps    = 20;

    printf("\n[Sim all-gather, %d ranks] chunk_size -> GB/s\n", num_ranks);
    fprintf(fp, "\n[Sim all-gather, %d ranks]\n", num_ranks);
    fprintf(fp, "  chunk      bytes       ms      GB/s\n");

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    for (int si = 0; si < nsizes; ++si) {
        int chunk = sizes[si];
        size_t chunk_bytes = (size_t)chunk * sizeof(float);

        std::vector<float*> d_chunks(num_ranks);
        for (int r = 0; r < num_ranks; ++r) {
            cudaMalloc(&d_chunks[r], chunk_bytes);
            cudaMemset(d_chunks[r], 0, chunk_bytes);
        }
        float* d_out;
        cudaMalloc(&d_out, (size_t)num_ranks * chunk_bytes);

        AllgatherCtx ctx = { d_chunks.data(), d_out, chunk, num_ranks };

        for (int w = 0; w < 3; ++w) run_allgather(&ctx);
        cudaDeviceSynchronize();

        cudaEventRecord(ev_start);
        for (int i = 0; i < reps; ++i) run_allgather(&ctx);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);

        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        double ms_avg = (double)ms / reps;
        double total_bytes = (double)num_ranks * chunk_bytes;
        double gbps = (total_bytes / 1e9) / (ms_avg / 1e3);

        printf("  chunk=%7d  %6.1f KB  %7.3f ms  %6.2f GB/s\n",
               chunk, chunk_bytes / 1024.0, ms_avg, gbps);
        fprintf(fp, "  %8d  %8zu  %7.3f  %6.2f\n",
                chunk, chunk_bytes, ms_avg, gbps);

        for (int r = 0; r < num_ranks; ++r) cudaFree(d_chunks[r]);
        cudaFree(d_out);
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
}

// ---------------------------------------------------------------------------
// Col/row parallel linear TFLOPS
// ---------------------------------------------------------------------------

static void bench_parallel_linear(FILE* fp)
{
    // configs: (M, N, K) representing different LLM hidden-dim scenarios
    struct Config { int M, N, K; const char* label; };
    const Config cfgs[] = {
        {  1,  4096, 4096, "decode  M=1    N=4096 K=4096" },
        {  8,  4096, 4096, "decode  M=8    N=4096 K=4096" },
        { 32,  4096, 4096, "prefill M=32   N=4096 K=4096" },
        {128,  4096, 4096, "prefill M=128  N=4096 K=4096" },
        { 32, 11008, 4096, "FFN-up  M=32  N=11008 K=4096" },
    };
    const int ncfg = (int)(sizeof(cfgs) / sizeof(cfgs[0]));
    const int reps = 50;

    printf("\n[Col-parallel linear (single rank = full GEMM)] M N K -> TFLOPS\n");
    fprintf(fp, "\n[Col-parallel linear TFLOPS]\n");
    fprintf(fp, "  M      N      K       ms     TFLOPS  label\n");

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    for (int ci = 0; ci < ncfg; ++ci) {
        int M = cfgs[ci].M, N = cfgs[ci].N, K = cfgs[ci].K;

        float *d_x, *d_W, *d_y;
        cudaMalloc(&d_x, (size_t)M * K * sizeof(float));
        cudaMalloc(&d_W, (size_t)N * K * sizeof(float));
        cudaMalloc(&d_y, (size_t)M * N * sizeof(float));
        cudaMemset(d_x, 0, (size_t)M * K * sizeof(float));
        cudaMemset(d_W, 0, (size_t)N * K * sizeof(float));

        for (int w = 0; w < 3; ++w)
            launch_col_parallel_linear(d_x, d_W, d_y, M, N, K, 0);
        cudaDeviceSynchronize();

        cudaEventRecord(ev_start);
        for (int i = 0; i < reps; ++i)
            launch_col_parallel_linear(d_x, d_W, d_y, M, N, K, 0);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);

        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        double ms_avg = (double)ms / reps;
        // FLOPs: 2 * M * N * K (multiply-accumulate)
        double tflops = 2.0 * M * N * K / (ms_avg / 1e3) / 1e12;

        printf("  M=%4d N=%5d K=%5d  %7.3f ms  %6.3f TFLOPS  %s\n",
               M, N, K, ms_avg, tflops, cfgs[ci].label);
        fprintf(fp, "  %4d  %5d  %5d  %7.3f  %6.3f  %s\n",
                M, N, K, ms_avg, tflops, cfgs[ci].label);

        cudaFree(d_x); cudaFree(d_W); cudaFree(d_y);
    }

    printf("\n[Row-parallel linear (single rank = full GEMM)] M N K -> TFLOPS\n");
    fprintf(fp, "\n[Row-parallel linear TFLOPS]\n");
    fprintf(fp, "  M      N      K       ms     TFLOPS  label\n");

    for (int ci = 0; ci < ncfg; ++ci) {
        int M = cfgs[ci].M, N = cfgs[ci].N, K = cfgs[ci].K;

        float *d_xr, *d_W, *d_out;
        cudaMalloc(&d_xr, (size_t)M * K * sizeof(float));
        cudaMalloc(&d_W,  (size_t)N * K * sizeof(float));
        cudaMalloc(&d_out, (size_t)M * N * sizeof(float));
        cudaMemset(d_xr, 0, (size_t)M * K * sizeof(float));
        cudaMemset(d_W,  0, (size_t)N * K * sizeof(float));

        for (int w = 0; w < 3; ++w)
            launch_row_parallel_linear(d_xr, d_W, d_out, M, N, K, 0);
        cudaDeviceSynchronize();

        cudaEventRecord(ev_start);
        for (int i = 0; i < reps; ++i)
            launch_row_parallel_linear(d_xr, d_W, d_out, M, N, K, 0);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);

        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        double ms_avg = (double)ms / reps;
        double tflops = 2.0 * M * N * K / (ms_avg / 1e3) / 1e12;

        printf("  M=%4d N=%5d K=%5d  %7.3f ms  %6.3f TFLOPS  %s\n",
               M, N, K, ms_avg, tflops, cfgs[ci].label);
        fprintf(fp, "  %4d  %5d  %5d  %7.3f  %6.3f  %s\n",
                M, N, K, ms_avg, tflops, cfgs[ci].label);

        cudaFree(d_xr); cudaFree(d_W); cudaFree(d_out);
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
}

// ---------------------------------------------------------------------------
// NCCL bandwidth (compiled only when HAVE_NCCL)
// ---------------------------------------------------------------------------

#ifdef HAVE_NCCL
static void bench_nccl_allreduce(FILE* fp, int num_ranks)
{
    const int sizes[] = {256, 4096, 65536, 1<<20, 1<<24};
    const int nsizes  = (int)(sizeof(sizes) / sizeof(sizes[0]));
    const int reps    = 20;

    const int devs[2] = {0, 0};  // same-device SHM transport
    void* comms[2] = {};
    nccl_comm_init_all(comms, num_ranks, devs);
    if (!comms[0]) {
        printf("[NCCL benchmark] ncclCommInitAll failed — skipping\n");
        fprintf(fp, "[NCCL benchmark] ncclCommInitAll failed\n");
        return;
    }

    printf("\n[NCCL ring all-reduce, %d ranks, SHM] count -> GB/s\n", num_ranks);
    fprintf(fp, "\n[NCCL ring all-reduce, %d ranks, SHM]\n", num_ranks);
    fprintf(fp, "  count      bytes       ms      GB/s\n");

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    // Two streams, one per rank
    cudaStream_t s0, s1;
    cudaStreamCreate(&s0);
    cudaStreamCreate(&s1);

    for (int si = 0; si < nsizes; ++si) {
        int count = sizes[si];
        size_t bytes = (size_t)count * sizeof(float);

        float *d_buf0, *d_buf1;
        cudaMalloc(&d_buf0, bytes); cudaMemset(d_buf0, 0, bytes);
        cudaMalloc(&d_buf1, bytes); cudaMemset(d_buf1, 0, bytes);

        // Warmup
        for (int w = 0; w < 3; ++w) {
            nccl_group_start();
            launch_ring_allreduce_nccl(comms[0], d_buf0, count, s0);
            launch_ring_allreduce_nccl(comms[1], d_buf1, count, s1);
            nccl_group_end();
            cudaStreamSynchronize(s0);
            cudaStreamSynchronize(s1);
        }

        cudaEventRecord(ev_start, s0);
        for (int i = 0; i < reps; ++i) {
            nccl_group_start();
            launch_ring_allreduce_nccl(comms[0], d_buf0, count, s0);
            launch_ring_allreduce_nccl(comms[1], d_buf1, count, s1);
            nccl_group_end();
        }
        cudaEventRecord(ev_stop, s0);
        cudaStreamSynchronize(s0);
        cudaStreamSynchronize(s1);

        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        double ms_avg = (double)ms / reps;
        double eff_bytes = 2.0 * (num_ranks - 1.0) / num_ranks * bytes;
        double gbps = (eff_bytes / 1e9) / (ms_avg / 1e3);

        printf("  count=%7d  %6.1f KB  %7.3f ms  %6.2f GB/s\n",
               count, bytes / 1024.0, ms_avg, gbps);
        fprintf(fp, "  %8d  %8zu  %7.3f  %6.2f\n", count, bytes, ms_avg, gbps);

        cudaFree(d_buf0); cudaFree(d_buf1);
    }

    cudaStreamDestroy(s0); cudaStreamDestroy(s1);
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
    for (int r = 0; r < num_ranks; ++r) nccl_comm_destroy(comms[r]);
}
#endif  // HAVE_NCCL

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main()
{
    cudaSetDevice(0);

    // Open report file
    const char* report_path = "reports/benchmark_tensor_parallel.txt";
    FILE* fp = fopen(report_path, "a");
    if (!fp) {
        // Try creating reports/ directory via a simple fallback
        fp = fopen("benchmark_tensor_parallel.txt", "a");
        if (!fp) fp = stdout;
    }

    printf("=== Tensor Parallel Benchmark (Phase 13) ===\n");
    fprintf(fp, "=== Tensor Parallel Benchmark ===\n");

    // Simulation benchmarks (always run)
    bench_sim_allreduce(fp, 2);
    bench_sim_allreduce(fp, 4);
    bench_sim_allgather(fp, 2);
    bench_sim_allgather(fp, 4);

    // Parallel linear TFLOPS
    bench_parallel_linear(fp);

#ifdef HAVE_NCCL
    // NCCL benchmarks (single-GPU SHM transport)
    printf("\n[NCCL available — running NCCL benchmarks with SHM transport]\n");
    fprintf(fp, "\n[NCCL benchmarks — SHM transport]\n");
    bench_nccl_allreduce(fp, 2);
#else
    printf("\n[NCCL not compiled in — skipping NCCL benchmarks]\n");
    fprintf(fp, "\n[NCCL not compiled in]\n");
#endif

    fprintf(fp, "\n");
    if (fp != stdout) fclose(fp);

    printf("\nBenchmark complete. Results appended to %s\n", report_path);
    return 0;
}
