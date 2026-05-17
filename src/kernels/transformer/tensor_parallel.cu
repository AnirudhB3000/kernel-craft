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
