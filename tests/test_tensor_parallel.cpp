/**
 * \file test_tensor_parallel.cpp
 * \brief Unit tests for tensor-parallelism primitives.
 *
 * Tests cover:
 *  - Single-GPU simulation: ring all-reduce, all-gather (Phase 11)
 *  - Column-parallel and row-parallel linear forward shards (Phase 13)
 *  - NCCL-backed all-reduce and all-gather using same-device SHM transport
 *    (Phase 13; compiled in only when HAVE_NCCL is defined)
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Extern C declarations
// ---------------------------------------------------------------------------

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
extern "C" void launch_allgather_nccl(
    void* comm, const float* d_sendbuf, float* d_recvbuf,
    int sendcount, cudaStream_t stream);
extern "C" int nccl_comm_count(void* comm);
extern "C" int nccl_comm_user_rank(void* comm);
#endif

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

static int g_pass = 0, g_fail = 0, g_skip = 0;
static void check(bool ok, const char* name) {
    if (ok) { printf("[PASS] %s\n", name); ++g_pass; }
    else    { printf("[FAIL] %s\n", name); ++g_fail; }
}
static void skip_test(const char* name, const char* reason) {
    printf("[SKIP] %s (%s)\n", name, reason);
    ++g_skip;
}

// CPU reference: C[M,N] = A[M,K] * B[N,K]^T
static void cpu_gemm_nt(const std::vector<float>& A,
                        const std::vector<float>& B,
                        std::vector<float>& C,
                        int M, int N, int K)
{
    C.assign((size_t)M * N, 0.f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                C[(size_t)m * N + n] += A[(size_t)m * K + k] * B[(size_t)n * K + k];
}

// =========================================================================
// Phase 11 simulation tests (unchanged)
// =========================================================================

static void test_allreduce_2ranks()
{
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

    std::vector<float> r0(count), r1(count);
    cudaMemcpy(r0.data(), d0, count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(r1.data(), d1, count * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < count; ++i)
        if (fabsf(r0[i] - expected[i]) > 1e-4f || fabsf(r1[i] - expected[i]) > 1e-4f)
            ok = false;
    check(ok, "Sim ring allreduce 2 ranks: result = 2x input");

    cudaFree(d0); cudaFree(d1);
}

static void test_allreduce_4ranks()
{
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
    check(all_ok, "Sim ring allreduce 4 ranks: result = 4x input");

    for (int r = 0; r < num_ranks; ++r) cudaFree(d_bufs[r]);
}

static void test_allreduce_single_rank()
{
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
    check(ok, "Sim ring allreduce 1 rank: identity (no-op)");

    cudaFree(d0);
}

static void test_allgather()
{
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
    check(ok, "Sim all-gather 4 ranks: correct concatenation");

    for (int r = 0; r < num_ranks; ++r) cudaFree(d_chunks[r]);
    cudaFree(d_out);
}

// =========================================================================
// Phase 13: Column-parallel linear tests
// =========================================================================

/**
 * \brief Column-parallel linear, 2 ranks simulation.
 *
 * x [M=4, K=8], W [N=6, K=8].
 * Rank 0 holds W[0:3,:], rank 1 holds W[3:6,:].
 * After col-parallel GEMM + all-gather, result must equal x @ W^T.
 */
static void test_col_parallel_linear_2ranks()
{
    const int M = 4, N = 6, K = 8;
    const int N_rank = N / 2;  // 3 per rank

    std::vector<float> h_x(M * K), h_W(N * K);
    for (int i = 0; i < M * K; ++i) h_x[i] = (float)(i % 7) * 0.1f + 0.05f;
    for (int i = 0; i < N * K; ++i) h_W[i] = (float)((i * 3 + 1) % 11) * 0.1f - 0.5f;

    std::vector<float> ref;
    cpu_gemm_nt(h_x, h_W, ref, M, N, K);

    std::vector<float> h_W0(h_W.begin(),               h_W.begin() + N_rank * K);
    std::vector<float> h_W1(h_W.begin() + N_rank * K,  h_W.end());

    float *d_x, *d_W0, *d_W1, *d_y0, *d_y1;
    cudaMalloc(&d_x,  M * K      * sizeof(float));
    cudaMalloc(&d_W0, N_rank * K  * sizeof(float));
    cudaMalloc(&d_W1, N_rank * K  * sizeof(float));
    cudaMalloc(&d_y0, M * N_rank  * sizeof(float));
    cudaMalloc(&d_y1, M * N_rank  * sizeof(float));

    cudaMemcpy(d_x,  h_x.data(),  M * K     * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W0, h_W0.data(), N_rank * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, h_W1.data(), N_rank * K * sizeof(float), cudaMemcpyHostToDevice);

    launch_col_parallel_linear(d_x, d_W0, d_y0, M, N_rank, K, 0);
    launch_col_parallel_linear(d_x, d_W1, d_y1, M, N_rank, K, 0);
    cudaDeviceSynchronize();

    std::vector<float> h_y0(M * N_rank), h_y1(M * N_rank);
    cudaMemcpy(h_y0.data(), d_y0, M * N_rank * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y1.data(), d_y1, M * N_rank * sizeof(float), cudaMemcpyDeviceToHost);

    // Reconstruct full output row by row (all-gather simulation)
    std::vector<float> gathered(M * N);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N_rank; ++n)
            gathered[m * N + n]          = h_y0[m * N_rank + n];
        for (int n = 0; n < N_rank; ++n)
            gathered[m * N + N_rank + n] = h_y1[m * N_rank + n];
    }

    bool ok = true;
    for (int i = 0; i < M * N; ++i)
        if (fabsf(gathered[i] - ref[i]) > 1e-3f) { ok = false; break; }
    check(ok, "Col-parallel linear 2 ranks: gathered output matches x@W^T");

    cudaFree(d_x); cudaFree(d_W0); cudaFree(d_W1);
    cudaFree(d_y0); cudaFree(d_y1);
}

// =========================================================================
// Phase 13: Row-parallel linear tests
// =========================================================================

/**
 * \brief Row-parallel linear, 2 ranks simulation.
 *
 * x [M=4, K=8], W [N=6, K=8].
 * Rank 0: x0=x[:,0:4], W0=W[:,0:4]. Rank 1: x1=x[:,4:8], W1=W[:,4:8].
 * partial0 + partial1 must equal x @ W^T.
 */
static void test_row_parallel_linear_2ranks()
{
    const int M = 4, N = 6, K = 8;
    const int K_rank = K / 2;  // 4 per rank

    std::vector<float> h_x(M * K), h_W(N * K);
    for (int i = 0; i < M * K; ++i) h_x[i] = (float)(i % 5) * 0.2f + 0.1f;
    for (int i = 0; i < N * K; ++i) h_W[i] = (float)((i * 7 + 3) % 13) * 0.1f - 0.6f;

    std::vector<float> ref;
    cpu_gemm_nt(h_x, h_W, ref, M, N, K);

    // Split x along K
    std::vector<float> h_x0(M * K_rank), h_x1(M * K_rank);
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K_rank; ++k) {
            h_x0[m * K_rank + k] = h_x[m * K + k];
            h_x1[m * K_rank + k] = h_x[m * K + K_rank + k];
        }
    }
    // Split W along K (stored as [N, K])
    std::vector<float> h_W0(N * K_rank), h_W1(N * K_rank);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K_rank; ++k) {
            h_W0[n * K_rank + k] = h_W[n * K + k];
            h_W1[n * K_rank + k] = h_W[n * K + K_rank + k];
        }
    }

    float *d_x0, *d_x1, *d_W0, *d_W1, *d_p0, *d_p1;
    cudaMalloc(&d_x0, M * K_rank * sizeof(float));
    cudaMalloc(&d_x1, M * K_rank * sizeof(float));
    cudaMalloc(&d_W0, N * K_rank * sizeof(float));
    cudaMalloc(&d_W1, N * K_rank * sizeof(float));
    cudaMalloc(&d_p0, M * N      * sizeof(float));
    cudaMalloc(&d_p1, M * N      * sizeof(float));

    cudaMemcpy(d_x0, h_x0.data(), M * K_rank * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x1, h_x1.data(), M * K_rank * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W0, h_W0.data(), N * K_rank * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, h_W1.data(), N * K_rank * sizeof(float), cudaMemcpyHostToDevice);

    launch_row_parallel_linear(d_x0, d_W0, d_p0, M, N, K_rank, 0);
    launch_row_parallel_linear(d_x1, d_W1, d_p1, M, N, K_rank, 0);
    cudaDeviceSynchronize();

    std::vector<float> h_p0(M * N), h_p1(M * N);
    cudaMemcpy(h_p0.data(), d_p0, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_p1.data(), d_p1, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // all-reduce simulation: sum the two partials
    bool ok = true;
    for (int i = 0; i < M * N; ++i)
        if (fabsf(h_p0[i] + h_p1[i] - ref[i]) > 1e-3f) { ok = false; break; }
    check(ok, "Row-parallel linear 2 ranks: summed partials match x@W^T");

    cudaFree(d_x0); cudaFree(d_x1); cudaFree(d_W0); cudaFree(d_W1);
    cudaFree(d_p0); cudaFree(d_p1);
}

/**
 * \brief Verify col-parallel -> row-parallel chain (1 rank = identity).
 *
 * With a single rank, col_parallel computes the full first layer output,
 * and row_parallel computes the full second layer output. Result must match
 * a sequential CPU reference.
 */
static void test_col_then_row_parallel_identity()
{
    const int M = 3, K = 8, hidden = 6, N = 4;

    std::vector<float> h_x(M * K), h_W1(hidden * K), h_W2(N * hidden);
    for (int i = 0; i < M * K;      ++i) h_x[i]  = (float)(i % 7 + 1) * 0.1f;
    for (int i = 0; i < hidden * K; ++i) h_W1[i] = (float)(i % 5 - 2) * 0.1f;
    for (int i = 0; i < N * hidden; ++i) h_W2[i] = (float)(i % 3 - 1) * 0.15f;

    std::vector<float> ref_mid, ref_out;
    cpu_gemm_nt(h_x,    h_W1, ref_mid, M, hidden, K);
    cpu_gemm_nt(ref_mid, h_W2, ref_out, M, N,      hidden);

    float *d_x, *d_W1, *d_mid, *d_W2, *d_out;
    cudaMalloc(&d_x,   M * K       * sizeof(float));
    cudaMalloc(&d_W1,  hidden * K  * sizeof(float));
    cudaMalloc(&d_mid, M * hidden   * sizeof(float));
    cudaMalloc(&d_W2,  N * hidden   * sizeof(float));
    cudaMalloc(&d_out, M * N        * sizeof(float));

    cudaMemcpy(d_x,  h_x.data(),  M * K      * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, h_W1.data(), hidden * K  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2.data(), N * hidden  * sizeof(float), cudaMemcpyHostToDevice);

    launch_col_parallel_linear(d_x,  d_W1, d_mid, M, hidden, K,      0);
    launch_row_parallel_linear(d_mid, d_W2, d_out, M, N,     hidden,  0);
    cudaDeviceSynchronize();

    std::vector<float> h_out(M * N);
    cudaMemcpy(h_out.data(), d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < M * N; ++i)
        if (fabsf(h_out[i] - ref_out[i]) > 1e-3f) { ok = false; break; }
    check(ok, "Col-parallel -> row-parallel chain (1 rank): matches CPU reference");

    cudaFree(d_x); cudaFree(d_W1); cudaFree(d_mid); cudaFree(d_W2); cudaFree(d_out);
}

// =========================================================================
// Phase 13: NCCL tests (single-GPU SHM transport via devs={0,0})
// =========================================================================

#ifdef HAVE_NCCL

static void test_nccl_allreduce_same_device()
{
    const int R = 2;
    const int devs[R] = {0, 0};
    void* comms[R] = {};

    nccl_comm_init_all(comms, R, devs);
    if (!comms[0] || !comms[1]) {
        skip_test("NCCL allreduce same-device (SHM)", "ncclCommInitAll failed");
        skip_test("NCCL comm rank/count introspection", "ncclCommInitAll failed");
        return;
    }

    bool ranks_ok = (nccl_comm_user_rank(comms[0]) == 0) &&
                    (nccl_comm_user_rank(comms[1]) == 1) &&
                    (nccl_comm_count(comms[0]) == 2) &&
                    (nccl_comm_count(comms[1]) == 2);
    check(ranks_ok, "NCCL comm rank/count introspection");

    const int count = 1024;
    std::vector<float> h_in(count, 1.0f);

    float *d_buf0, *d_buf1;
    cudaMalloc(&d_buf0, count * sizeof(float));
    cudaMalloc(&d_buf1, count * sizeof(float));
    cudaMemcpy(d_buf0, h_in.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_buf1, h_in.data(), count * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t s0, s1;
    cudaStreamCreate(&s0);
    cudaStreamCreate(&s1);

    nccl_group_start();
    launch_ring_allreduce_nccl(comms[0], d_buf0, count, s0);
    launch_ring_allreduce_nccl(comms[1], d_buf1, count, s1);
    nccl_group_end();

    cudaStreamSynchronize(s0);
    cudaStreamSynchronize(s1);

    std::vector<float> r0(count), r1(count);
    cudaMemcpy(r0.data(), d_buf0, count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(r1.data(), d_buf1, count * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < count; ++i)
        if (fabsf(r0[i] - 2.f) > 1e-5f || fabsf(r1[i] - 2.f) > 1e-5f)
            { ok = false; break; }
    check(ok, "NCCL allreduce same-device (SHM): result = 2x input");

    cudaStreamDestroy(s0); cudaStreamDestroy(s1);
    cudaFree(d_buf0); cudaFree(d_buf1);
    nccl_comm_destroy(comms[0]);
    nccl_comm_destroy(comms[1]);
}

static void test_nccl_allgather_same_device()
{
    const int R = 2;
    const int devs[R] = {0, 0};
    void* comms[R] = {};

    nccl_comm_init_all(comms, R, devs);
    if (!comms[0] || !comms[1]) {
        skip_test("NCCL allgather same-device (SHM)", "ncclCommInitAll failed");
        return;
    }

    const int sendcount = 512;
    std::vector<float> h_send0(sendcount), h_send1(sendcount);
    for (int i = 0; i < sendcount; ++i) {
        h_send0[i] = (float)i;
        h_send1[i] = (float)(i + sendcount);
    }

    float *d_s0, *d_s1, *d_r0, *d_r1;
    cudaMalloc(&d_s0, sendcount * sizeof(float));
    cudaMalloc(&d_s1, sendcount * sizeof(float));
    cudaMalloc(&d_r0, R * sendcount * sizeof(float));
    cudaMalloc(&d_r1, R * sendcount * sizeof(float));

    cudaMemcpy(d_s0, h_send0.data(), sendcount * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s1, h_send1.data(), sendcount * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t s0, s1;
    cudaStreamCreate(&s0);
    cudaStreamCreate(&s1);

    nccl_group_start();
    launch_allgather_nccl(comms[0], d_s0, d_r0, sendcount, s0);
    launch_allgather_nccl(comms[1], d_s1, d_r1, sendcount, s1);
    nccl_group_end();

    cudaStreamSynchronize(s0);
    cudaStreamSynchronize(s1);

    std::vector<float> hr0(R * sendcount), hr1(R * sendcount);
    cudaMemcpy(hr0.data(), d_r0, R * sendcount * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hr1.data(), d_r1, R * sendcount * sizeof(float), cudaMemcpyDeviceToHost);

    // Both ranks should see [0..511, 512..1023]
    bool ok = true;
    for (int i = 0; i < R * sendcount; ++i)
        if (fabsf(hr0[i] - (float)i) > 1e-5f || fabsf(hr1[i] - (float)i) > 1e-5f)
            { ok = false; break; }
    check(ok, "NCCL allgather same-device (SHM): concatenated output matches");

    cudaStreamDestroy(s0); cudaStreamDestroy(s1);
    cudaFree(d_s0); cudaFree(d_s1); cudaFree(d_r0); cudaFree(d_r1);
    nccl_comm_destroy(comms[0]);
    nccl_comm_destroy(comms[1]);
}

#else  // HAVE_NCCL not defined — emit skip messages

static void test_nccl_allreduce_same_device()
{
    skip_test("NCCL comm rank/count introspection",
              "compiled without HAVE_NCCL — install libnccl and rebuild");
    skip_test("NCCL allreduce same-device (SHM)",
              "compiled without HAVE_NCCL");
}

static void test_nccl_allgather_same_device()
{
    skip_test("NCCL allgather same-device (SHM)",
              "compiled without HAVE_NCCL");
}

#endif  // HAVE_NCCL

// =========================================================================
// main
// =========================================================================

int main()
{
    printf("=== Tensor Parallel Tests (Phase 11 + Phase 13) ===\n\n");

    printf("-- Phase 11: Simulation (ring all-reduce, all-gather) --\n");
    test_allreduce_single_rank();
    test_allreduce_2ranks();
    test_allreduce_4ranks();
    test_allgather();

    printf("\n-- Phase 13: Column-parallel and row-parallel linear --\n");
    test_col_parallel_linear_2ranks();
    test_row_parallel_linear_2ranks();
    test_col_then_row_parallel_identity();

    printf("\n-- Phase 13: NCCL-backed collectives (SHM transport) --\n");
    test_nccl_allreduce_same_device();
    test_nccl_allgather_same_device();

    printf("\nResults: %d passed, %d failed, %d skipped\n",
           g_pass, g_fail, g_skip);
    return g_fail ? 1 : 0;
}
