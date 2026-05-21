/**
 * \file tensor_parallel.cu
 * \brief Tensor parallelism primitives: ring all-reduce and all-gather.
 *
 * Implements the communication patterns needed for tensor-parallel LLM
 * inference across multiple GPU ranks.
 *
 * \par Ring All-Reduce
 * Partitions the buffer into `num_ranks` chunks. In `num_ranks - 1` reduce-scatter
 * steps each rank receives partial sums from its neighbor; in `num_ranks - 1`
 * all-gather steps the fully summed chunks are broadcast. Final result: every
 * rank holds the global sum.
 *
 * \par Multi-GPU vs single-GPU simulation
 * On multi-GPU systems, ranks exchange data via NCCL or peer cudaMemcpy.
 * This file provides:
 * 1. A single-GPU ring all-reduce simulation using multiple CUDA streams
 *    (each "rank" is a distinct region of device memory).
 * 2. All-gather primitive following the same simulation model.
 *
 * \par Real deployment
 * In production, replace the simulated peer copies with:
 *   ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, comm, stream)
 * or equivalent. The kernel logic here handles the per-GPU compute; only the
 * transport layer changes.
 *
 * \par Scaling
 * For a model with hidden_dim=4096 split across 4 GPUs:
 * - Each GPU holds 4096/4 = 1024 columns of a weight matrix.
 * - After local GEMM each rank holds a partial output.
 * - All-reduce sums the partials → each rank holds the full output.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

// ---------------------------------------------------------------------------
// In-place element-wise addition kernel (reduce-scatter step)
// ---------------------------------------------------------------------------

/**
 * \brief Element-wise add: dst[i] += src[i] in-place.
 *
 * \param[in,out] dst  Target buffer.
 * \param[in]     src  Source buffer.
 * \param[in]     n    Number of elements.
 */
__global__ void add_inplace_kernel(float* __restrict__ dst,
                                   const float* __restrict__ src,
                                   int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] += src[idx];
}

/**
 * \brief Element-wise copy: dst[i] = src[i].
 *
 * \param[out] dst Target buffer.
 * \param[in]  src Source buffer.
 * \param[in]  n   Number of elements.
 */
__global__ void copy_kernel(float* __restrict__ dst,
                            const float* __restrict__ src,
                            int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[idx];
}

// ---------------------------------------------------------------------------
// Host simulation of ring all-reduce
// ---------------------------------------------------------------------------

/**
 * \brief Simulate ring all-reduce over `num_ranks` device buffers on one GPU.
 *
 * Each "rank" is a contiguous buffer of `count` floats. After the call, all
 * rank buffers contain the element-wise sum of all input buffers.
 *
 * \par Algorithm (reduce-scatter then all-gather)
 * Reduce-scatter: for step s in [0, R-1): rank r accumulates its chunk s into
 * the chunk of rank (r+1) % R.
 * All-gather: for step s in [0, R-1): rank r copies its summed chunk to
 * the chunk slot of rank (r+1) % R.
 *
 * \param[in,out] d_bufs     Array of `num_ranks` device pointers, each [count].
 * \param[in]     count      Elements per rank buffer.
 * \param[in]     num_ranks  Number of ranks to simulate.
 * \param[in]     stream     CUDA stream (0 = default).
 */
extern "C" void launch_ring_allreduce(
    float** d_bufs,
    int count,
    int num_ranks,
    cudaStream_t stream)
{
    if (num_ranks <= 1) return;  // trivial: buffer already has the result

    int chunk = (count + num_ranks - 1) / num_ranks;
    dim3 block(256);

    // Phase 1: Reduce-scatter — chunk c is accumulated into rank c's buffer.
    // Each rank c sums contributions from all other ranks into its home chunk c,
    // doing so one source rank at a time (safe for single-GPU simulation where
    // all "rank" buffers share device memory without aliasing between chunks).
    for (int c = 0; c < num_ranks; ++c) {
        int chunk_start = c * chunk;
        int chunk_elems = ((chunk_start + chunk) <= count)
                          ? chunk : (count - chunk_start);
        if (chunk_elems <= 0) continue;

        dim3 grid((chunk_elems + 255) / 256);
        for (int r = 0; r < num_ranks; ++r) {
            if (r == c) continue;  // rank c already holds its own data
            add_inplace_kernel<<<grid, block, 0, stream>>>(
                d_bufs[c] + chunk_start,  // accumulate INTO rank c
                d_bufs[r] + chunk_start,  // FROM rank r
                chunk_elems);
        }
    }
    cudaStreamSynchronize(stream);

    // Phase 2: All-gather — rank c broadcasts its fully summed chunk to all others.
    for (int c = 0; c < num_ranks; ++c) {
        int chunk_start = c * chunk;
        int chunk_elems = ((chunk_start + chunk) <= count)
                          ? chunk : (count - chunk_start);
        if (chunk_elems <= 0) continue;

        dim3 grid((chunk_elems + 255) / 256);
        for (int r = 0; r < num_ranks; ++r) {
            if (r == c) continue;
            copy_kernel<<<grid, block, 0, stream>>>(
                d_bufs[r] + chunk_start,  // write to rank r's chunk c slot
                d_bufs[c] + chunk_start,  // read from rank c (home rank)
                chunk_elems);
        }
    }
    cudaStreamSynchronize(stream);
}

// ---------------------------------------------------------------------------
// Tiled SGEMM for parallel-linear shards  (C = A × B^T)
// A [M, K], B [N, K] (weight stored row-major), C [M, N]
// C[m,n] = sum_k A[m,k] * B[n,k]
// ---------------------------------------------------------------------------

#define TP_TILE 16

/**
 * \brief Tiled SGEMM: C[M,N] = A[M,K] × B[N,K]^T.
 *
 * Thread (tx,ty) computes C[blockIdx.y*T+ty, blockIdx.x*T+tx].
 *
 * \param[in]  A  Left  operand [M, K].
 * \param[in]  B  Right operand [N, K] (accessed transposed).
 * \param[out] C  Output [M, N].
 * \param[in]  M  Rows of A / rows of C.
 * \param[in]  N  Rows of B / columns of C.
 * \param[in]  K  Shared inner dimension.
 */
__global__ static void sgemm_nt_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K)
{
    __shared__ float sA[TP_TILE][TP_TILE + 1];
    __shared__ float sB[TP_TILE][TP_TILE + 1];

    int row = blockIdx.y * TP_TILE + threadIdx.y;
    int col = blockIdx.x * TP_TILE + threadIdx.x;
    float acc = 0.f;

    for (int t = 0; t < (K + TP_TILE - 1) / TP_TILE; ++t) {
        int k = t * TP_TILE + threadIdx.x;
        // Load A tile: sA[ty][tx] = A[row, t*T+tx]
        sA[threadIdx.y][threadIdx.x] = (row < M && k < K) ? A[row * K + k] : 0.f;
        // Load B tile (transposed): sB[ty][tx] = B[(blockIdx.x*T+ty), (t*T+tx)]
        int n_b = blockIdx.x * TP_TILE + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] = (n_b < N && k < K) ? B[n_b * K + k] : 0.f;
        __syncthreads();

        // Accumulate: acc += sA[ty][k] * sB[tx][k]  (sB[tx] = B-row for col)
        for (int ki = 0; ki < TP_TILE; ++ki)
            acc += sA[threadIdx.y][ki] * sB[threadIdx.x][ki];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

/**
 * \brief Column-parallel linear forward shard.
 *
 * Each rank holds an output-dimension shard of the weight matrix.
 * Computes y_rank[M, N_rank] = x[M, K] × W_rank[N_rank, K]^T.
 * After this call, all ranks must all-gather to form y[M, N_total].
 *
 * \param[in]  d_x     Input [M, K] — replicated across ranks.
 * \param[in]  d_W     Weight shard [N_rank, K].
 * \param[out] d_y     Output shard [M, N_rank].
 * \param[in]  M       Batch size (rows of x).
 * \param[in]  N_rank  Output features on this rank.
 * \param[in]  K       Input features.
 * \param[in]  stream  CUDA stream.
 */
extern "C" void launch_col_parallel_linear(
    const float* d_x, const float* d_W, float* d_y,
    int M, int N_rank, int K, cudaStream_t stream)
{
    dim3 block(TP_TILE, TP_TILE);
    dim3 grid((N_rank + TP_TILE - 1) / TP_TILE,
              (M      + TP_TILE - 1) / TP_TILE);
    sgemm_nt_kernel<<<grid, block, 0, stream>>>(d_x, d_W, d_y, M, N_rank, K);
}

/**
 * \brief Row-parallel linear forward shard.
 *
 * Each rank holds an input-dimension shard of the weight matrix.
 * Computes partial[M, N] = x_rank[M, K_rank] × W_rank[N, K_rank]^T.
 * After this call, all ranks must all-reduce (sum) partial results to
 * obtain y[M, N].
 *
 * \param[in]  d_x_rank  Input shard [M, K_rank] — pre-split.
 * \param[in]  d_W       Weight shard [N, K_rank].
 * \param[out] d_out     Partial output [M, N].
 * \param[in]  M         Batch size.
 * \param[in]  N         Output features (same on every rank).
 * \param[in]  K_rank    Input features on this rank.
 * \param[in]  stream    CUDA stream.
 */
extern "C" void launch_row_parallel_linear(
    const float* d_x_rank, const float* d_W, float* d_out,
    int M, int N, int K_rank, cudaStream_t stream)
{
    dim3 block(TP_TILE, TP_TILE);
    dim3 grid((N      + TP_TILE - 1) / TP_TILE,
              (M      + TP_TILE - 1) / TP_TILE);
    sgemm_nt_kernel<<<grid, block, 0, stream>>>(d_x_rank, d_W, d_out, M, N, K_rank);
}

/**
 * \brief Simulate all-gather over `num_ranks` device buffers on one GPU.
 *
 * Each rank contributes a chunk of `chunk_size` elements.
 * After the call, `d_output` contains all chunks concatenated:
 * [rank0_chunk | rank1_chunk | ... | rank{R-1}_chunk].
 *
 * \param[in]  d_chunks   Array of `num_ranks` device pointers, each [chunk_size].
 * \param[out] d_output   Device output buffer [num_ranks * chunk_size].
 * \param[in]  chunk_size Elements contributed by each rank.
 * \param[in]  num_ranks  Number of ranks.
 * \param[in]  stream     CUDA stream (0 = default).
 */
extern "C" void launch_allgather(
    float** d_chunks,
    float*  d_output,
    int chunk_size,
    int num_ranks,
    cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((chunk_size + 255) / 256);
    for (int r = 0; r < num_ranks; ++r) {
        copy_kernel<<<grid, block, 0, stream>>>(
            d_output + r * chunk_size,
            d_chunks[r],
            chunk_size);
    }
    cudaStreamSynchronize(stream);
}
