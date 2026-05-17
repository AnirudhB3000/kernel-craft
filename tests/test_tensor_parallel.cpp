/**
 * \file test_tensor_parallel.cpp
 * \brief Unit tests for ring all-reduce and all-gather primitives.
 *
 * Tests single-GPU simulation with multiple virtual ranks (separate device buffers).
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

extern "C" void launch_ring_allreduce(
    float** d_bufs, int count, int num_ranks, cudaStream_t stream);

extern "C" void launch_allgather(
    float** d_chunks, float* d_output, int chunk_size, int num_ranks,
    cudaStream_t stream);

static int g_pass = 0, g_fail = 0;
static void check(bool ok, const char* name) {
    if (ok) { printf("[PASS] %s\n", name); ++g_pass; }
    else    { printf("[FAIL] %s\n", name); ++g_fail; }
}

static void test_allreduce_2ranks() {
    // 2 ranks, each holds [1, 2, 3, 4]; after allreduce → [2, 4, 6, 8]
    int count = 4, num_ranks = 2;
    std::vector<float> data0 = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> data1 = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> expected = {2.f, 4.f, 6.f, 8.f};

    float *d0, *d1;
    cudaMalloc(&d0, count * sizeof(float));
    cudaMalloc(&d1, count * sizeof(float));
    cudaMemcpy(d0, data0.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d1, data1.data(), count * sizeof(float), cudaMemcpyHostToDevice);

    float* d_bufs[2] = {d0, d1};
    launch_ring_allreduce(d_bufs, count, num_ranks, 0);

    std::vector<float> result0(count), result1(count);
    cudaMemcpy(result0.data(), d0, count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(result1.data(), d1, count * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok0 = true, ok1 = true;
    for (int i = 0; i < count; ++i) {
        if (fabsf(result0[i] - expected[i]) > 1e-4f) ok0 = false;
        if (fabsf(result1[i] - expected[i]) > 1e-4f) ok1 = false;
    }
    check(ok0 && ok1, "Ring allreduce 2 ranks: result = 2x input");

    cudaFree(d0); cudaFree(d1);
}

static void test_allreduce_4ranks() {
    // 4 ranks, each holds [1, 2, ..., 8]; after allreduce → [4, 8, 12, ...]
    int count = 8, num_ranks = 4;
    std::vector<std::vector<float>> data(num_ranks, std::vector<float>(count));
    for (int r = 0; r < num_ranks; ++r)
        for (int i = 0; i < count; ++i) data[r][i] = (float)(i + 1);

    std::vector<float*> d_bufs(num_ranks);
    for (int r = 0; r < num_ranks; ++r) {
        cudaMalloc(&d_bufs[r], count * sizeof(float));
        cudaMemcpy(d_bufs[r], data[r].data(), count * sizeof(float), cudaMemcpyHostToDevice);
    }

    launch_ring_allreduce(d_bufs.data(), count, num_ranks, 0);

    std::vector<float> expected(count);
    for (int i = 0; i < count; ++i) expected[i] = (float)num_ranks * (i + 1);

    bool all_ok = true;
    for (int r = 0; r < num_ranks; ++r) {
        std::vector<float> result(count);
        cudaMemcpy(result.data(), d_bufs[r], count * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < count; ++i)
            if (fabsf(result[i] - expected[i]) > 1e-3f) { all_ok = false; break; }
    }
    check(all_ok, "Ring allreduce 4 ranks: result = 4x input");

    for (int r = 0; r < num_ranks; ++r) cudaFree(d_bufs[r]);
}

static void test_allreduce_single_rank() {
    // With 1 rank, allreduce is a no-op (identity)
    int count = 16;
    std::vector<float> data(count);
    for (int i = 0; i < count; ++i) data[i] = (float)i;

    float* d0;
    cudaMalloc(&d0, count * sizeof(float));
    cudaMemcpy(d0, data.data(), count * sizeof(float), cudaMemcpyHostToDevice);

    float* d_bufs[1] = {d0};
    launch_ring_allreduce(d_bufs, count, 1, 0);

    std::vector<float> result(count);
    cudaMemcpy(result.data(), d0, count * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < count; ++i)
        if (fabsf(result[i] - data[i]) > 1e-6f) { ok = false; break; }
    check(ok, "Ring allreduce 1 rank: identity (no-op)");

    cudaFree(d0);
}

static void test_allgather() {
    // 4 ranks, each contributes chunk [r*10+0, r*10+1, r*10+2]
    int chunk_size = 3, num_ranks = 4;
    std::vector<float> expected;
    std::vector<float*> d_chunks(num_ranks);
    for (int r = 0; r < num_ranks; ++r) {
        std::vector<float> chunk(chunk_size);
        for (int i = 0; i < chunk_size; ++i) {
            chunk[i] = (float)(r * 10 + i);
            expected.push_back(chunk[i]);
        }
        cudaMalloc(&d_chunks[r], chunk_size * sizeof(float));
        cudaMemcpy(d_chunks[r], chunk.data(), chunk_size * sizeof(float), cudaMemcpyHostToDevice);
    }

    float* d_out;
    cudaMalloc(&d_out, num_ranks * chunk_size * sizeof(float));

    launch_allgather(d_chunks.data(), d_out, chunk_size, num_ranks, 0);

    std::vector<float> result(num_ranks * chunk_size);
    cudaMemcpy(result.data(), d_out, num_ranks * chunk_size * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < num_ranks * chunk_size; ++i)
        if (fabsf(result[i] - expected[i]) > 1e-6f) { ok = false; break; }
    check(ok, "All-gather 4 ranks: correct concatenation");

    for (int r = 0; r < num_ranks; ++r) cudaFree(d_chunks[r]);
    cudaFree(d_out);
}

int main() {
    printf("=== Tensor Parallel Tests ===\n");
    test_allreduce_single_rank();
    test_allreduce_2ranks();
    test_allreduce_4ranks();
    test_allgather();
    printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
