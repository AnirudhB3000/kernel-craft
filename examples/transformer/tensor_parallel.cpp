/**
 * \file transformer/tensor_parallel.cpp
 * \brief Tensor parallelism: col/row parallel linear + collective communication.
 *
 * \par Purpose
 * Demonstrates the tensor-parallel linear kernels
 * (src/kernels/transformer/tensor_parallel.cu) and ring all-reduce / all-gather
 * collectives that enable sharding large transformer weight matrices across
 * multiple GPUs — the core mechanism in Megatron-LM style model parallelism.
 *
 * \par Why tensor parallelism is needed
 * Llama-3-70B has attention projection matrices of shape [8192, 8192].
 * At fp16 that is 128 MB per matrix × 4 matrices (Q,K,V,O) = 512 MB per layer
 * × 80 layers = 40 GB — more than any single consumer GPU.  Tensor parallelism
 * shards each weight matrix along one axis across R GPUs, reducing per-GPU
 * memory to 40 GB / R while keeping compute proportional.
 *
 * \par Column-parallel linear (Megatron-LM first linear in MLP/attention)
 * \code
 *   Total weight  W [N, K]  split column-wise across R ranks:
 *     Rank r holds W_r [N/R, K]
 *
 *   Each rank computes:   y_r = x @ W_r^T   → [M, N/R]
 *   Combine via all-gather: y  = cat(y_0, ..., y_{R-1}) → [M, N]
 * \endcode
 * Input x [M, K] is replicated on all ranks (or broadcast before the layer).
 *
 * \par Row-parallel linear (second linear in each MLP block)
 * \code
 *   Total weight  W [N, K]  split row-wise across R ranks:
 *     Rank r holds W_r [N, K/R]  and  x_r [M, K/R]  (pre-split input)
 *
 *   Each rank computes:   p_r = x_r @ W_r^T  → [M, N]   (partial sum)
 *   Combine via all-reduce: y  = sum(p_0, ..., p_{R-1}) → [M, N]
 * \endcode
 *
 * \par Collective communication kernels
 * | Collective   | Operation                          | Used after         |
 * |--------------|------------------------------------|--------------------|
 * | All-reduce   | sum partial results across ranks   | Row-parallel linear|
 * | All-gather   | concatenate rank shards            | Col-parallel linear|
 *
 * In a real multi-GPU setup these use NCCL (see tensor_parallel_nccl.cu).
 * This example uses single-GPU simulation with separate device buffers.
 *
 * \par Compile and run
 * \code
 * cmake --build build --target example_tensor_parallel
 * ./build/examples/example_tensor_parallel
 * \endcode
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>

/**
 * \brief Column-parallel linear: y_rank[M, N/R] = x[M, K] @ W_rank[N/R, K]^T
 *
 * Uses a tiled SGEMM with 16×16 shared-memory tiles and bank-conflict-free
 * padding (+1 column per tile).  Each rank independently processes its shard
 * without communication; the caller all-gathers results afterward.
 *
 * \param[in]  d_x      float32 device [M, K] — replicated input.
 * \param[in]  d_W      float32 device [N_rank, K] — this rank's weight shard.
 * \param[out] d_y      float32 device [M, N_rank] — partial output.
 * \param[in]  M        Batch / sequence length.
 * \param[in]  N_rank   Output features for this rank (= N_total / num_ranks).
 * \param[in]  K        Input features (shared across all ranks).
 * \param[in]  stream   CUDA stream.
 */
extern "C" void launch_col_parallel_linear(const float* d_x, const float* d_W,
                                           float* d_y,
                                           int M, int N_rank, int K,
                                           cudaStream_t stream);

/**
 * \brief Row-parallel linear: partial[M, N] = x_rank[M, K/R] @ W_rank[N, K/R]^T
 *
 * Same tiled SGEMM; the caller all-reduces partial results across ranks.
 *
 * \param[in]  d_x_rank float32 device [M, K_rank] — this rank's input shard.
 * \param[in]  d_W      float32 device [N, K_rank] — this rank's weight shard.
 * \param[out] d_out    float32 device [M, N] — partial sum (needs all-reduce).
 * \param[in]  M        Batch / sequence length.
 * \param[in]  N        Output features (same on all ranks).
 * \param[in]  K_rank   Input features for this rank (= K_total / num_ranks).
 * \param[in]  stream   CUDA stream.
 */
extern "C" void launch_row_parallel_linear(const float* d_x_rank, const float* d_W,
                                           float* d_out,
                                           int M, int N, int K_rank,
                                           cudaStream_t stream);

/**
 * \brief Simulated ring all-reduce over R device buffers (single-GPU).
 *
 * In production, replace with ncclAllReduce (see tensor_parallel_nccl.cu).
 * Each element of each buffer is accumulated (summed) in-place across all R
 * virtual ranks.  The single-GPU simulation uses sequential reduce-scatter +
 * all-gather phases to avoid aliasing between ranks' buffers.
 *
 * \param[in,out] d_bufs Array of R device pointers, each pointing to `count` floats.
 * \param[in]     count  Number of elements per rank buffer.
 * \param[in]     R      Number of simulated ranks.
 * \param[in]     stream CUDA stream.
 */
extern "C" void launch_ring_allreduce(float** d_bufs, int count, int R,
                                      cudaStream_t stream);

/**
 * \brief Simulated all-gather: concatenate R chunks into a single output.
 *
 * In production, replace with ncclAllGather (see tensor_parallel_nccl.cu).
 *
 * \param[in]  d_chunks Array of R device pointers, each pointing to `chunk_size` floats.
 * \param[out] d_out    Device pointer to chunk_size × R floats (concatenated result).
 * \param[in]  chunk_size  Elements per rank chunk.
 * \param[in]  R           Number of simulated ranks.
 * \param[in]  stream      CUDA stream.
 */
extern "C" void launch_allgather(float** d_chunks, float* d_out,
                                 int chunk_size, int R, cudaStream_t stream);

int main() {
    /* ---- Configuration: simulate 2-way tensor parallelism ---- */
    const int R      = 2;    /* simulated ranks (GPUs) */
    const int M      = 8;    /* batch × seq_len (decode batch) */
    const int N      = 256;  /* total output features */
    const int K      = 256;  /* input features */
    const int N_rank = N / R; /* output features per rank (col-parallel) */
    const int K_rank = K / R; /* input features per rank (row-parallel) */

    printf("Tensor Parallelism Example  (R=%d ranks, M=%d, N=%d, K=%d)\n\n",
           R, M, N, K);

    /* ---- Column-parallel linear demo ---- */
    printf("--- Column-parallel linear ---\n");
    printf("  W[%d,%d] split column-wise: each rank holds W_rank[%d,%d]\n",
           N, K, N_rank, K);

    /* Input x replicated on all ranks */
    std::vector<float> h_x(M * K, 1.f);
    /* Weight shards: rank r has W_r = 0.01 * r (for easy verification) */
    std::vector<std::vector<float>> h_W_col(R, std::vector<float>(N_rank * K));
    for (int r = 0; r < R; ++r)
        for (auto& w : h_W_col[r]) w = 0.01f * (r + 1);

    float *d_x;
    cudaMalloc(&d_x, M * K * sizeof(float));
    cudaMemcpy(d_x, h_x.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float*> d_W_col(R), d_y_col(R);
    float* d_gathered;
    cudaMalloc(&d_gathered, M * N * sizeof(float));
    for (int r = 0; r < R; ++r) {
        cudaMalloc(&d_W_col[r], N_rank * K * sizeof(float));
        cudaMalloc(&d_y_col[r], M * N_rank * sizeof(float));
        cudaMemcpy(d_W_col[r], h_W_col[r].data(), N_rank * K * sizeof(float),
                   cudaMemcpyHostToDevice);
        /* Each rank computes y_rank = x @ W_rank^T */
        launch_col_parallel_linear(d_x, d_W_col[r], d_y_col[r], M, N_rank, K, 0);
    }
    /* All-gather: y = [y_0 | y_1] along the N axis */
    launch_allgather(d_y_col.data(), d_gathered, M * N_rank, R, 0);
    cudaDeviceSynchronize();

    std::vector<float> h_gathered(M * N);
    cudaMemcpy(h_gathered.data(), d_gathered, M * N * sizeof(float),
               cudaMemcpyDeviceToHost);
    printf("  y[0,0]   = %.4f (rank-0 cols, expected ~%d × 0.01 × K = %.2f)\n",
           h_gathered[0], M, (float)K * 0.01f);
    printf("  y[0,%d] = %.4f (rank-1 cols, expected ~%d × 0.02 × K = %.2f)\n",
           N_rank, h_gathered[N_rank], M, (float)K * 0.02f);

    /* ---- Row-parallel linear demo ---- */
    printf("\n--- Row-parallel linear ---\n");
    printf("  W[%d,%d] split row-wise: each rank holds W_rank[%d,%d]\n",
           N, K, N, K_rank);

    std::vector<std::vector<float>> h_x_rank(R, std::vector<float>(M * K_rank, 1.f));
    std::vector<std::vector<float>> h_W_row(R, std::vector<float>(N * K_rank, 0.01f));

    std::vector<float*> d_xr(R), d_W_row(R), d_partial(R);
    for (int r = 0; r < R; ++r) {
        cudaMalloc(&d_xr[r],      M * K_rank * sizeof(float));
        cudaMalloc(&d_W_row[r],   N * K_rank * sizeof(float));
        cudaMalloc(&d_partial[r], M * N      * sizeof(float));
        cudaMemcpy(d_xr[r],    h_x_rank[r].data(), M * K_rank * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_W_row[r], h_W_row[r].data(),  N * K_rank * sizeof(float),
                   cudaMemcpyHostToDevice);
        /* Each rank computes partial = x_rank @ W_rank^T */
        launch_row_parallel_linear(d_xr[r], d_W_row[r], d_partial[r],
                                   M, N, K_rank, 0);
    }
    /* All-reduce: y = sum(partial_0, partial_1) */
    launch_ring_allreduce(d_partial.data(), M * N, R, 0);
    cudaDeviceSynchronize();

    std::vector<float> h_partial(M * N);
    cudaMemcpy(h_partial.data(), d_partial[0], M * N * sizeof(float),
               cudaMemcpyDeviceToHost);
    float expected_row = K * 0.01f; /* K_rank × 0.01 × 2 ranks = K × 0.01 */
    printf("  y[0,0] = %.4f  (expected K×0.01 = %.4f)\n",
           h_partial[0], expected_row);

    /* ---- Cleanup ---- */
    cudaFree(d_x); cudaFree(d_gathered);
    for (int r = 0; r < R; ++r) {
        cudaFree(d_W_col[r]); cudaFree(d_y_col[r]);
        cudaFree(d_xr[r]); cudaFree(d_W_row[r]); cudaFree(d_partial[r]);
    }

    printf("\nKey insight:\n");
    printf("  Col-parallel: all-gather after  → N_rank per rank, cheap comm\n");
    printf("  Row-parallel: all-reduce after  → full N on each rank, heavier comm\n");
    printf("  In Megatron-LM: FFN = col-parallel → GELU → row-parallel\n");
    printf("  (only one all-reduce per FFN block, not two)\n");
    printf("\nFor multi-GPU: replace launch_ring_allreduce/launch_allgather\n");
    printf("with nccl_allreduce/nccl_allgather in tensor_parallel_nccl.cu.\n");
    return EXIT_SUCCESS;
}
