/**
 * \file pybind_transformer.cpp
 * \brief Python bindings for Phase 11 transformer/LLM inference kernels.
 *
 * Exposes CUDA transformer kernels to Python via pybind11.
 * Follows the same numpy-array pattern as pybind_cuda.cpp.
 *
 * \par Exposed functions
 * - flash_attention(Q, K, V, H_kv, causal) → O
 * - paged_attention(Q, block_table, K_pool, V_pool, seq_lens, block_size) → O
 * - quant_int4_dequant(packed, scales, zeros, rows, cols, group_size) → fp32
 * - gemv_int4(packed, scales, zeros, x, rows, cols, group_size) → y
 * - fp8_quantize(input, per_token) → (quantized, scales)
 * - fp8_dequantize(quantized, scales, rows, cols, per_token) → output
 * - smoothquant_scale(input, smooth_scales) → output
 * - speculative_decode(draft_probs, target_probs, draft_tokens, rand_vals, rand_vals2)
 *                       → (accepted, corrected)
 * - ring_allreduce(list_of_arrays) → list_of_arrays (in-place, returns same)
 * - allgather(list_of_chunks) → concatenated array
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Extern C declarations
// ---------------------------------------------------------------------------

extern "C" void launch_flash_attention(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int H_kv, int N, int d,
    bool causal, cudaStream_t stream);

extern "C" void launch_paged_attention(
    const float* Q, const int* block_table,
    const float* K_pool, const float* V_pool,
    float* O, const int* seq_lens,
    int B, int H, int H_kv, int d,
    int block_size, int max_blocks, cudaStream_t stream);

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

extern "C" void launch_smoothquant_scale(
    const float* d_input, const float* d_smooth_scales, float* d_output,
    int rows, int cols, cudaStream_t stream);

extern "C" void launch_verify_draft_tokens(
    const float* d_draft_probs, const float* d_target_probs,
    const int* d_draft_tokens, const float* d_rand_vals, const float* d_rand_vals2,
    int* d_accepted, int* d_corrected,
    int num_tokens, int vocab_size, cudaStream_t stream);

extern "C" void launch_compute_prefix_length(
    const int* d_accepted, int* d_prefix_len, int num_tokens, cudaStream_t stream);

extern "C" void launch_ring_allreduce(
    float** d_bufs, int count, int num_ranks, cudaStream_t stream);

extern "C" void launch_allgather(
    float** d_chunks, float* d_output, int chunk_size, int num_ranks,
    cudaStream_t stream);

// Phase 13: col/row parallel linear
extern "C" void launch_col_parallel_linear(
    const float* d_x, const float* d_W, float* d_y,
    int M, int N_rank, int K, cudaStream_t stream);

extern "C" void launch_row_parallel_linear(
    const float* d_x_rank, const float* d_W, float* d_out,
    int M, int N, int K_rank, cudaStream_t stream);

// Phase 14: Mamba SSM kernels
extern "C" void launch_selective_scan(
    const float* d_u,     const float* d_A_log,
    const float* d_B,     const float* d_C,
    const float* d_delta, float* d_y,
    int B_, int L, int D, int N_state,
    cudaStream_t stream);

extern "C" void launch_depthwise_conv1d(
    const float* d_x, const float* d_w, const float* d_bias,
    float* d_y, int B_, int D, int L, int d_conv,
    cudaStream_t stream);

extern "C" void launch_rmsnorm(
    const float* d_x, const float* d_g, float* d_y,
    int rows, int D, float eps, cudaStream_t stream);

// Phase 13: NCCL collectives
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void* device_alloc(size_t bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess)
        throw std::runtime_error("cudaMalloc failed");
    return p;
}

static void h2d(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}
static void d2h(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}

// ---------------------------------------------------------------------------
// Python-exposed functions
// ---------------------------------------------------------------------------

/**
 * \brief FlashAttention forward pass.
 *
 * \param[in] Q      numpy float32 [B, H, N, d]
 * \param[in] K      numpy float32 [B, H_kv, N, d]
 * \param[in] V      numpy float32 [B, H_kv, N, d]
 * \param[in] H_kv   number of KV heads
 * \param[in] causal apply causal mask
 * \return numpy float32 [B, H, N, d]
 */
py::array_t<float> py_flash_attention(
    py::array_t<float> Q, py::array_t<float> K, py::array_t<float> V,
    int H_kv, bool causal)
{
    auto Qbuf = Q.request(), Kbuf = K.request(), Vbuf = V.request();
    if (Qbuf.ndim != 4 || Kbuf.ndim != 4 || Vbuf.ndim != 4)
        throw std::runtime_error("Q, K, V must be 4-D [B, H, N, d]");

    int B = (int)Qbuf.shape[0];
    int H = (int)Qbuf.shape[1];
    int N = (int)Qbuf.shape[2];
    int d = (int)Qbuf.shape[3];

    size_t Q_bytes  = (size_t)B * H    * N * d * sizeof(float);
    size_t KV_bytes = (size_t)B * H_kv * N * d * sizeof(float);

    float *dQ = (float*)device_alloc(Q_bytes);
    float *dK = (float*)device_alloc(KV_bytes);
    float *dV = (float*)device_alloc(KV_bytes);
    float *dO = (float*)device_alloc(Q_bytes);

    h2d(dQ, Qbuf.ptr, Q_bytes);
    h2d(dK, Kbuf.ptr, KV_bytes);
    h2d(dV, Vbuf.ptr, KV_bytes);

    launch_flash_attention(dQ, dK, dV, dO, B, H, H_kv, N, d, causal, 0);

    std::vector<float> out((size_t)B * H * N * d);
    d2h(out.data(), dO, Q_bytes);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return py::array_t<float>(std::vector<ssize_t>{B, H, N, d}, out.data());
}

/**
 * \brief PagedAttention decode step.
 *
 * \param[in] Q           float32 [B, H, d]
 * \param[in] block_table int32   [B, max_blocks]
 * \param[in] K_pool      float32 [num_phys, block_size, H_kv, d]
 * \param[in] V_pool      float32 [num_phys, block_size, H_kv, d]
 * \param[in] seq_lens    int32   [B]
 * \param[in] H_kv        number of KV heads
 * \param[in] block_size  tokens per physical page
 * \return float32 [B, H, d]
 */
py::array_t<float> py_paged_attention(
    py::array_t<float> Q,
    py::array_t<int>   block_table,
    py::array_t<float> K_pool,
    py::array_t<float> V_pool,
    py::array_t<int>   seq_lens,
    int H_kv, int block_size)
{
    auto Qbuf  = Q.request();
    auto BTbuf = block_table.request();
    auto Kbuf  = K_pool.request();
    auto Vbuf  = V_pool.request();
    auto SLbuf = seq_lens.request();

    int B          = (int)Qbuf.shape[0];
    int H          = (int)Qbuf.shape[1];
    int d          = (int)Qbuf.shape[2];
    int max_blocks = (int)BTbuf.shape[1];

    size_t Q_bytes  = (size_t)B * H * d * sizeof(float);
    size_t BT_bytes = (size_t)BTbuf.size * sizeof(int);
    size_t K_bytes  = (size_t)Kbuf.size  * sizeof(float);
    size_t SL_bytes = (size_t)B          * sizeof(int);

    float *dQ, *dKp, *dVp, *dO; int *dBT, *dSL;
    dQ  = (float*)device_alloc(Q_bytes);
    dKp = (float*)device_alloc(K_bytes);
    dVp = (float*)device_alloc(K_bytes);
    dO  = (float*)device_alloc(Q_bytes);
    dBT = (int*)  device_alloc(BT_bytes);
    dSL = (int*)  device_alloc(SL_bytes);

    h2d(dQ,  Qbuf.ptr,  Q_bytes);
    h2d(dBT, BTbuf.ptr, BT_bytes);
    h2d(dKp, Kbuf.ptr,  K_bytes);
    h2d(dVp, Vbuf.ptr,  K_bytes);
    h2d(dSL, SLbuf.ptr, SL_bytes);

    launch_paged_attention(dQ, dBT, dKp, dVp, dO, dSL,
                           B, H, H_kv, d, block_size, max_blocks, 0);

    std::vector<float> out((size_t)B * H * d);
    d2h(out.data(), dO, Q_bytes);

    cudaFree(dQ); cudaFree(dKp); cudaFree(dVp); cudaFree(dO);
    cudaFree(dBT); cudaFree(dSL);
    return py::array_t<float>(std::vector<ssize_t>{B, H, d}, out.data());
}

/**
 * \brief Dequantize INT4 packed weights to FP32.
 *
 * \param[in] packed     uint8 numpy [rows, cols/2]
 * \param[in] scales     float32 numpy [rows, cols/group_size]
 * \param[in] zeros      uint8 numpy [rows, cols/group_size/2]
 * \param[in] rows       weight rows
 * \param[in] cols       weight cols
 * \param[in] group_size elements per quantization group
 * \return float32 [rows, cols]
 */
py::array_t<float> py_quant_int4_dequant(
    py::array_t<uint8_t> packed,
    py::array_t<float>   scales,
    py::array_t<uint8_t> zeros,
    int rows, int cols, int group_size)
{
    auto pb = packed.request(), sb = scales.request(), zb = zeros.request();

    uint8_t *dp, *dz; float *ds, *dout;
    dp   = (uint8_t*)device_alloc(pb.size * sizeof(uint8_t));
    dz   = (uint8_t*)device_alloc(zb.size * sizeof(uint8_t));
    ds   = (float*)  device_alloc(sb.size * sizeof(float));
    dout = (float*)  device_alloc((size_t)rows * cols * sizeof(float));

    h2d(dp, pb.ptr, pb.size * sizeof(uint8_t));
    h2d(dz, zb.ptr, zb.size * sizeof(uint8_t));
    h2d(ds, sb.ptr, sb.size * sizeof(float));

    launch_dequantize_int4(dp, ds, dz, dout, rows, cols, group_size, 0);

    std::vector<float> out((size_t)rows * cols);
    d2h(out.data(), dout, (size_t)rows * cols * sizeof(float));

    cudaFree(dp); cudaFree(dz); cudaFree(ds); cudaFree(dout);
    return py::array_t<float>(std::vector<ssize_t>{rows, cols}, out.data());
}

/**
 * \brief FP8 quantize activations, returns (quantized_int8, scales).
 *
 * \param[in] input     float32 numpy [rows, cols]
 * \param[in] per_token per-token scaling if true; per-tensor otherwise
 * \return tuple(int8 [rows, cols], float32 [rows] or [1])
 */
py::tuple py_fp8_quantize(py::array_t<float> input, bool per_token) {
    auto buf = input.request();
    int rows = (int)buf.shape[0];
    int cols = (int)buf.shape[1];
    int scale_count = per_token ? rows : 1;

    float  *di, *ds; int8_t *dq;
    di = (float*) device_alloc((size_t)rows * cols * sizeof(float));
    dq = (int8_t*)device_alloc((size_t)rows * cols * sizeof(int8_t));
    ds = (float*) device_alloc((size_t)scale_count  * sizeof(float));

    h2d(di, buf.ptr, (size_t)rows * cols * sizeof(float));
    if (!per_token) {
        // Compute per-tensor scale on CPU
        float max_abs = 0.f;
        for (int i = 0; i < rows * cols; ++i)
            max_abs = fmaxf(max_abs, fabsf(((float*)buf.ptr)[i]));
        float scale = (max_abs > 0.f) ? max_abs / 448.f : 1.f;
        h2d(ds, &scale, sizeof(float));
    }

    launch_quantize_fp8(di, dq, ds, rows, cols, per_token, 0);

    std::vector<int8_t> q_out((size_t)rows * cols);
    std::vector<float>  s_out(scale_count);
    d2h(q_out.data(), dq, (size_t)rows * cols * sizeof(int8_t));
    d2h(s_out.data(), ds, (size_t)scale_count  * sizeof(float));

    cudaFree(di); cudaFree(dq); cudaFree(ds);
    auto q_arr = py::array_t<int8_t>(std::vector<ssize_t>{rows, cols}, q_out.data());
    auto s_arr = py::array_t<float> (std::vector<ssize_t>{scale_count}, s_out.data());
    return py::make_tuple(q_arr, s_arr);
}

/**
 * \brief FP8 dequantize activations back to FP32.
 *
 * \param[in] quantized int8 numpy [rows, cols]
 * \param[in] scales    float32 numpy [rows] or [1]
 * \param[in] rows      token count
 * \param[in] cols      embedding dim
 * \param[in] per_token match flag from quantization
 * \return float32 [rows, cols]
 */
py::array_t<float> py_fp8_dequantize(
    py::array_t<int8_t> quantized, py::array_t<float> scales,
    int rows, int cols, bool per_token)
{
    auto qbuf = quantized.request(), sbuf = scales.request();

    int8_t *dq; float *ds, *dout;
    dq   = (int8_t*)device_alloc((size_t)rows * cols * sizeof(int8_t));
    ds   = (float*) device_alloc((size_t)sbuf.size   * sizeof(float));
    dout = (float*) device_alloc((size_t)rows * cols  * sizeof(float));

    h2d(dq, qbuf.ptr, (size_t)rows * cols * sizeof(int8_t));
    h2d(ds, sbuf.ptr, (size_t)sbuf.size   * sizeof(float));

    launch_dequantize_fp8(dq, ds, dout, rows, cols, per_token, 0);

    std::vector<float> out((size_t)rows * cols);
    d2h(out.data(), dout, (size_t)rows * cols * sizeof(float));

    cudaFree(dq); cudaFree(ds); cudaFree(dout);
    return py::array_t<float>(std::vector<ssize_t>{rows, cols}, out.data());
}

/**
 * \brief Verify draft tokens via rejection sampling.
 *
 * \param[in] draft_probs  float32 [num_tokens, vocab_size]
 * \param[in] target_probs float32 [num_tokens, vocab_size]
 * \param[in] draft_tokens int32   [num_tokens]
 * \param[in] rand_vals    float32 [num_tokens] acceptance test
 * \param[in] rand_vals2   float32 [num_tokens] residual sampling
 * \return tuple(accepted int32 [num_tokens], corrected int32 [num_tokens])
 */
py::tuple py_speculative_decode(
    py::array_t<float> draft_probs,
    py::array_t<float> target_probs,
    py::array_t<int>   draft_tokens,
    py::array_t<float> rand_vals,
    py::array_t<float> rand_vals2)
{
    auto dpbuf = draft_probs.request();
    int num_tokens  = (int)dpbuf.shape[0];
    int vocab_size  = (int)dpbuf.shape[1];

    float *dd, *dt, *dr, *dr2; int *ddt, *da, *dc;
    dd  = (float*)device_alloc((size_t)num_tokens * vocab_size * sizeof(float));
    dt  = (float*)device_alloc((size_t)num_tokens * vocab_size * sizeof(float));
    dr  = (float*)device_alloc((size_t)num_tokens * sizeof(float));
    dr2 = (float*)device_alloc((size_t)num_tokens * sizeof(float));
    ddt = (int*)  device_alloc((size_t)num_tokens * sizeof(int));
    da  = (int*)  device_alloc((size_t)num_tokens * sizeof(int));
    dc  = (int*)  device_alloc((size_t)num_tokens * sizeof(int));

    h2d(dd,  draft_probs.request().ptr,  (size_t)num_tokens * vocab_size * sizeof(float));
    h2d(dt,  target_probs.request().ptr, (size_t)num_tokens * vocab_size * sizeof(float));
    h2d(dr,  rand_vals.request().ptr,    (size_t)num_tokens * sizeof(float));
    h2d(dr2, rand_vals2.request().ptr,   (size_t)num_tokens * sizeof(float));
    h2d(ddt, draft_tokens.request().ptr, (size_t)num_tokens * sizeof(int));

    launch_verify_draft_tokens(dd, dt, ddt, dr, dr2, da, dc, num_tokens, vocab_size, 0);

    std::vector<int> accepted(num_tokens), corrected(num_tokens);
    d2h(accepted.data(),  da, (size_t)num_tokens * sizeof(int));
    d2h(corrected.data(), dc, (size_t)num_tokens * sizeof(int));

    cudaFree(dd); cudaFree(dt); cudaFree(dr); cudaFree(dr2);
    cudaFree(ddt); cudaFree(da); cudaFree(dc);

    auto a_arr = py::array_t<int>(std::vector<ssize_t>{num_tokens}, accepted.data());
    auto c_arr = py::array_t<int>(std::vector<ssize_t>{num_tokens}, corrected.data());
    return py::make_tuple(a_arr, c_arr);
}

// ---------------------------------------------------------------------------
// Phase 11: ring_allreduce / allgather (simulation, list-of-arrays interface)
// ---------------------------------------------------------------------------

/**
 * \brief Simulate ring all-reduce over a list of numpy arrays (in-place).
 *
 * :param arrays: list of float32 numpy arrays of equal length (one per rank).
 * :return: the same list, each array holding the global element-wise sum.
 */
py::list py_ring_allreduce(py::list arrays)
{
    int num_ranks = (int)arrays.size();
    if (num_ranks == 0) return arrays;

    auto buf0 = py::cast<py::array_t<float>>(arrays[0]).request();
    int count  = (int)buf0.size;

    std::vector<float*> d_bufs(num_ranks);
    for (int r = 0; r < num_ranks; ++r) {
        auto arr = py::cast<py::array_t<float>>(arrays[r]).request();
        d_bufs[r] = (float*)device_alloc((size_t)count * sizeof(float));
        h2d(d_bufs[r], arr.ptr, (size_t)count * sizeof(float));
    }

    launch_ring_allreduce(d_bufs.data(), count, num_ranks, 0);

    for (int r = 0; r < num_ranks; ++r) {
        auto arr = py::cast<py::array_t<float>>(arrays[r]).request();
        d2h(arr.ptr, d_bufs[r], (size_t)count * sizeof(float));
        cudaFree(d_bufs[r]);
    }
    return arrays;
}

/**
 * \brief Simulate all-gather: concatenate list of equal-length chunks.
 *
 * :param chunks: list of float32 numpy arrays [chunk_size] (one per rank).
 * :return: float32 numpy array [num_ranks * chunk_size].
 */
py::array_t<float> py_allgather(py::list chunks)
{
    int num_ranks  = (int)chunks.size();
    auto buf0      = py::cast<py::array_t<float>>(chunks[0]).request();
    int chunk_size = (int)buf0.size;

    std::vector<float*> d_chunks(num_ranks);
    for (int r = 0; r < num_ranks; ++r) {
        auto arr = py::cast<py::array_t<float>>(chunks[r]).request();
        d_chunks[r] = (float*)device_alloc((size_t)chunk_size * sizeof(float));
        h2d(d_chunks[r], arr.ptr, (size_t)chunk_size * sizeof(float));
    }

    float* d_out = (float*)device_alloc((size_t)num_ranks * chunk_size * sizeof(float));
    launch_allgather(d_chunks.data(), d_out, chunk_size, num_ranks, 0);

    std::vector<float> out((size_t)num_ranks * chunk_size);
    d2h(out.data(), d_out, (size_t)num_ranks * chunk_size * sizeof(float));

    for (int r = 0; r < num_ranks; ++r) cudaFree(d_chunks[r]);
    cudaFree(d_out);
    return py::array_t<float>(
        std::vector<ssize_t>{num_ranks * chunk_size}, out.data());
}

// ---------------------------------------------------------------------------
// Phase 13: column-parallel and row-parallel linear
// ---------------------------------------------------------------------------

/**
 * \brief Column-parallel linear forward shard.
 *
 * :param x:     float32 numpy [M, K] — input (replicated across ranks).
 * :param W:     float32 numpy [N_rank, K] — weight shard (output-dim split).
 * :return:      float32 numpy [M, N_rank] — output shard; all-gather to assemble.
 */
py::array_t<float> py_col_parallel_linear(
    py::array_t<float> x, py::array_t<float> W)
{
    auto xb = x.request(), wb = W.request();
    if (xb.ndim != 2 || wb.ndim != 2)
        throw std::runtime_error("x and W must be 2-D");

    int M      = (int)xb.shape[0];
    int K      = (int)xb.shape[1];
    int N_rank = (int)wb.shape[0];
    if ((int)wb.shape[1] != K)
        throw std::runtime_error("W.shape[1] must equal x.shape[1] (K)");

    float *dx, *dW, *dy;
    dx = (float*)device_alloc((size_t)M * K      * sizeof(float));
    dW = (float*)device_alloc((size_t)N_rank * K  * sizeof(float));
    dy = (float*)device_alloc((size_t)M * N_rank  * sizeof(float));

    h2d(dx, xb.ptr, (size_t)M * K     * sizeof(float));
    h2d(dW, wb.ptr, (size_t)N_rank * K * sizeof(float));

    launch_col_parallel_linear(dx, dW, dy, M, N_rank, K, 0);
    cudaDeviceSynchronize();

    std::vector<float> out((size_t)M * N_rank);
    d2h(out.data(), dy, (size_t)M * N_rank * sizeof(float));

    cudaFree(dx); cudaFree(dW); cudaFree(dy);
    return py::array_t<float>(std::vector<ssize_t>{M, N_rank}, out.data());
}

/**
 * \brief Row-parallel linear forward shard.
 *
 * :param x_rank: float32 numpy [M, K_rank] — input shard (pre-split along K).
 * :param W:      float32 numpy [N, K_rank] — weight shard (input-dim split).
 * :return:       float32 numpy [M, N] — partial output; all-reduce to sum.
 */
py::array_t<float> py_row_parallel_linear(
    py::array_t<float> x_rank, py::array_t<float> W)
{
    auto xb = x_rank.request(), wb = W.request();
    if (xb.ndim != 2 || wb.ndim != 2)
        throw std::runtime_error("x_rank and W must be 2-D");

    int M      = (int)xb.shape[0];
    int K_rank = (int)xb.shape[1];
    int N      = (int)wb.shape[0];
    if ((int)wb.shape[1] != K_rank)
        throw std::runtime_error("W.shape[1] must equal x_rank.shape[1] (K_rank)");

    float *dxr, *dW, *dout;
    dxr  = (float*)device_alloc((size_t)M * K_rank * sizeof(float));
    dW   = (float*)device_alloc((size_t)N * K_rank  * sizeof(float));
    dout = (float*)device_alloc((size_t)M * N        * sizeof(float));

    h2d(dxr, xb.ptr, (size_t)M * K_rank * sizeof(float));
    h2d(dW,  wb.ptr, (size_t)N * K_rank  * sizeof(float));

    launch_row_parallel_linear(dxr, dW, dout, M, N, K_rank, 0);
    cudaDeviceSynchronize();

    std::vector<float> out((size_t)M * N);
    d2h(out.data(), dout, (size_t)M * N * sizeof(float));

    cudaFree(dxr); cudaFree(dW); cudaFree(dout);
    return py::array_t<float>(std::vector<ssize_t>{M, N}, out.data());
}

// ---------------------------------------------------------------------------
// Phase 13: NCCL comm management (thin wrappers, opaque handle = uint64)
// ---------------------------------------------------------------------------

/**
 * \brief Initialise NCCL communicators for the given device list.
 *
 * :param devs: list of CUDA device indices (may repeat for SHM testing).
 * :return:     list of opaque uint64 comm handles (one per rank).
 *
 * When NCCL is not compiled in, all handles are 0 and a warning is printed.
 */
py::list py_nccl_comm_init(py::list devs)
{
    int n = (int)devs.size();
    std::vector<int>   dev_arr(n);
    std::vector<void*> comms(n, nullptr);
    for (int i = 0; i < n; ++i) dev_arr[i] = py::cast<int>(devs[i]);

    nccl_comm_init_all(comms.data(), n, dev_arr.data());

    py::list result;
    for (int i = 0; i < n; ++i)
        result.append(py::int_(reinterpret_cast<uint64_t>(comms[i])));
    return result;
}

/**
 * \brief Destroy one NCCL communicator by opaque handle.
 *
 * :param handle: uint64 returned by nccl_comm_init().
 */
void py_nccl_comm_destroy(uint64_t handle)
{
    nccl_comm_destroy(reinterpret_cast<void*>(handle));
}

/**
 * \brief NCCL in-place all-reduce (sum) for a single rank.
 *
 * Must be called inside a group (py_nccl_group_start / py_nccl_group_end)
 * when multiple ranks share one process.
 *
 * :param handle: uint64 comm handle for this rank.
 * :param arr:    float32 numpy array to reduce in-place.
 */
void py_nccl_allreduce(uint64_t handle, py::array_t<float> arr)
{
    auto buf = arr.request();
    int count = (int)buf.size;

    float* d_buf = (float*)device_alloc((size_t)count * sizeof(float));
    h2d(d_buf, buf.ptr, (size_t)count * sizeof(float));

    launch_ring_allreduce_nccl(reinterpret_cast<void*>(handle), d_buf, count, 0);
    cudaDeviceSynchronize();

    d2h(buf.ptr, d_buf, (size_t)count * sizeof(float));
    cudaFree(d_buf);
}

void py_nccl_group_start() { nccl_group_start(); }
void py_nccl_group_end()   { nccl_group_end();   }

// ---------------------------------------------------------------------------
// Phase 14: Mamba SSM wrappers
// ---------------------------------------------------------------------------

/**
 * \brief Mamba-1 selective scan.
 *
 * :param u:      float32 numpy [B, L, D] — input.
 * :param A_log:  float32 numpy [D, N]    — log-scale state decay.
 * :param B:      float32 numpy [B, L, N] — input projection.
 * :param C:      float32 numpy [B, L, N] — output projection.
 * :param delta:  float32 numpy [B, L, D] — timestep.
 * :return:       float32 numpy [B, L, D] — output.
 */
py::array_t<float> py_selective_scan(
    py::array_t<float> u,
    py::array_t<float> A_log,
    py::array_t<float> B,
    py::array_t<float> C,
    py::array_t<float> delta)
{
    auto ub = u.request(), Ab = A_log.request(),
         Bb = B.request(), Cb = C.request(), db = delta.request();

    if (ub.ndim != 3 || db.ndim != 3)
        throw std::runtime_error("u and delta must be 3-D [B, L, D]");
    if (Ab.ndim != 2)
        throw std::runtime_error("A_log must be 2-D [D, N]");
    if (Bb.ndim != 3 || Cb.ndim != 3)
        throw std::runtime_error("B and C must be 3-D [B, L, N]");

    int B_ = (int)ub.shape[0];
    int L  = (int)ub.shape[1];
    int D  = (int)ub.shape[2];
    int N  = (int)Ab.shape[1];

    if ((int)Ab.shape[0] != D)
        throw std::runtime_error("A_log.shape[0] must equal D");
    if ((int)Bb.shape[2] != N || (int)Cb.shape[2] != N)
        throw std::runtime_error("B and C must have N state dimensions");

    size_t uD_bytes  = (size_t)B_ * L * D * sizeof(float);
    size_t uN_bytes  = (size_t)B_ * L * N * sizeof(float);
    size_t AN_bytes  = (size_t)D * N        * sizeof(float);

    float *du  = (float*)device_alloc(uD_bytes);
    float *dAl = (float*)device_alloc(AN_bytes);
    float *dB  = (float*)device_alloc(uN_bytes);
    float *dC  = (float*)device_alloc(uN_bytes);
    float *dd  = (float*)device_alloc(uD_bytes);
    float *dy  = (float*)device_alloc(uD_bytes);

    h2d(du,  ub.ptr, uD_bytes);
    h2d(dAl, Ab.ptr, AN_bytes);
    h2d(dB,  Bb.ptr, uN_bytes);
    h2d(dC,  Cb.ptr, uN_bytes);
    h2d(dd,  db.ptr, uD_bytes);

    launch_selective_scan(du, dAl, dB, dC, dd, dy, B_, L, D, N, 0);
    cudaDeviceSynchronize();

    std::vector<float> out((size_t)B_ * L * D);
    d2h(out.data(), dy, uD_bytes);

    cudaFree(du); cudaFree(dAl); cudaFree(dB);
    cudaFree(dC); cudaFree(dd);  cudaFree(dy);

    return py::array_t<float>(
        std::vector<ssize_t>{B_, L, D}, out.data());
}

/**
 * \brief Causal depthwise conv1d (channels-first layout).
 *
 * :param x:      float32 numpy [B, D, L] — input.
 * :param w:      float32 numpy [D, d_conv] — weight.
 * :param bias:   float32 numpy [D] or None.
 * :return:       float32 numpy [B, D, L] — output.
 */
py::array_t<float> py_depthwise_conv1d(
    py::array_t<float> x,
    py::array_t<float> w,
    py::object bias_obj)
{
    auto xb = x.request(), wb = w.request();
    if (xb.ndim != 3)
        throw std::runtime_error("x must be 3-D [B, D, L]");
    if (wb.ndim != 2)
        throw std::runtime_error("w must be 2-D [D, d_conv]");

    int B_     = (int)xb.shape[0];
    int D      = (int)xb.shape[1];
    int L      = (int)xb.shape[2];
    int d_conv = (int)wb.shape[1];

    if ((int)wb.shape[0] != D)
        throw std::runtime_error("w.shape[0] must equal D");

    size_t x_bytes = (size_t)B_ * D * L * sizeof(float);
    size_t w_bytes = (size_t)D * d_conv  * sizeof(float);

    float *gx = (float*)device_alloc(x_bytes);
    float *gw = (float*)device_alloc(w_bytes);
    float *gy = (float*)device_alloc(x_bytes);
    float *gbias = nullptr;

    h2d(gx, xb.ptr, x_bytes);
    h2d(gw, wb.ptr, w_bytes);

    if (!bias_obj.is_none()) {
        auto ba = bias_obj.cast<py::array_t<float>>().request();
        gbias = (float*)device_alloc((size_t)D * sizeof(float));
        h2d(gbias, ba.ptr, (size_t)D * sizeof(float));
    }

    launch_depthwise_conv1d(gx, gw, gbias, gy, B_, D, L, d_conv, 0);
    cudaDeviceSynchronize();

    std::vector<float> out((size_t)B_ * D * L);
    d2h(out.data(), gy, x_bytes);

    cudaFree(gx); cudaFree(gw); cudaFree(gy);
    if (gbias) cudaFree(gbias);

    return py::array_t<float>(
        std::vector<ssize_t>{B_, D, L}, out.data());
}

/**
 * \brief RMSNorm: normalize each row by root-mean-square then scale.
 *
 * :param x:    float32 numpy [rows, D] — input.
 * :param g:    float32 numpy [D]       — scale weight.
 * :param eps:  float — stability term (default 1e-6).
 * :return:     float32 numpy [rows, D] — output.
 */
py::array_t<float> py_rmsnorm(
    py::array_t<float> x,
    py::array_t<float> g,
    float eps = 1e-6f)
{
    auto xb = x.request(), gb = g.request();
    if (xb.ndim != 2)
        throw std::runtime_error("x must be 2-D [rows, D]");
    if (gb.ndim != 1)
        throw std::runtime_error("g must be 1-D [D]");

    int rows = (int)xb.shape[0];
    int D    = (int)xb.shape[1];
    if ((int)gb.shape[0] != D)
        throw std::runtime_error("g.shape[0] must equal D");

    size_t x_bytes = (size_t)rows * D * sizeof(float);
    size_t g_bytes = (size_t)D         * sizeof(float);

    float *gx = (float*)device_alloc(x_bytes);
    float *gg = (float*)device_alloc(g_bytes);
    float *gy = (float*)device_alloc(x_bytes);

    h2d(gx, xb.ptr, x_bytes);
    h2d(gg, gb.ptr, g_bytes);

    launch_rmsnorm(gx, gg, gy, rows, D, eps, 0);
    cudaDeviceSynchronize();

    std::vector<float> out((size_t)rows * D);
    d2h(out.data(), gy, x_bytes);

    cudaFree(gx); cudaFree(gg); cudaFree(gy);

    return py::array_t<float>(
        std::vector<ssize_t>{rows, D}, out.data());
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(kernel_craft_transformer, m) {
    m.doc() = "kernel-craft Phase 11: Transformer/LLM inference kernels";

    m.def("flash_attention", &py_flash_attention,
        py::arg("Q"), py::arg("K"), py::arg("V"),
        py::arg("H_kv"), py::arg("causal") = false,
        "FlashAttention forward [B,H,N,d] → [B,H,N,d]");

    m.def("paged_attention", &py_paged_attention,
        py::arg("Q"), py::arg("block_table"), py::arg("K_pool"), py::arg("V_pool"),
        py::arg("seq_lens"), py::arg("H_kv"), py::arg("block_size"),
        "PagedAttention decode [B,H,d] → [B,H,d]");

    m.def("quant_int4_dequant", &py_quant_int4_dequant,
        py::arg("packed"), py::arg("scales"), py::arg("zeros"),
        py::arg("rows"), py::arg("cols"), py::arg("group_size") = 128,
        "Dequantize INT4 packed weights to FP32");

    m.def("fp8_quantize", &py_fp8_quantize,
        py::arg("input"), py::arg("per_token") = true,
        "Quantize FP32 activations to FP8 E4M3; returns (quantized, scales)");

    m.def("fp8_dequantize", &py_fp8_dequantize,
        py::arg("quantized"), py::arg("scales"),
        py::arg("rows"), py::arg("cols"), py::arg("per_token") = true,
        "Dequantize FP8 E4M3 back to FP32");

    m.def("speculative_decode", &py_speculative_decode,
        py::arg("draft_probs"), py::arg("target_probs"),
        py::arg("draft_tokens"), py::arg("rand_vals"), py::arg("rand_vals2"),
        "Verify draft tokens; returns (accepted_mask, corrected_tokens)");

    // Phase 11 (previously unregistered): simulation collectives
    m.def("ring_allreduce", &py_ring_allreduce,
        py::arg("arrays"),
        "In-place ring all-reduce over a list of numpy arrays (simulation)");

    m.def("allgather", &py_allgather,
        py::arg("chunks"),
        "All-gather a list of equal-length numpy chunks; returns concatenated array");

    // Phase 13: parallel linear shards
    m.def("col_parallel_linear", &py_col_parallel_linear,
        py::arg("x"), py::arg("W"),
        "Col-parallel linear shard: y[M,N_rank] = x[M,K] @ W[N_rank,K]^T");

    m.def("row_parallel_linear", &py_row_parallel_linear,
        py::arg("x_rank"), py::arg("W"),
        "Row-parallel linear shard: partial[M,N] = x_rank[M,K_rank] @ W[N,K_rank]^T");

    // Phase 13: NCCL comm management
    m.def("nccl_comm_init",    &py_nccl_comm_init,    py::arg("devs"),
        "Init NCCL comms for given device list; returns list of opaque uint64 handles");
    m.def("nccl_comm_destroy", &py_nccl_comm_destroy, py::arg("handle"),
        "Destroy one NCCL comm handle");
    m.def("nccl_allreduce",    &py_nccl_allreduce,
        py::arg("handle"), py::arg("arr"),
        "NCCL in-place all-reduce (sum) for this rank's comm handle");
    m.def("nccl_group_start",  &py_nccl_group_start,
        "Start NCCL group (bracket multi-rank calls in one process)");
    m.def("nccl_group_end",    &py_nccl_group_end,
        "End NCCL group and flush pending collectives");

    // Phase 14: Mamba SSM primitives
    m.def("selective_scan", &py_selective_scan,
        py::arg("u"), py::arg("A_log"), py::arg("B"), py::arg("C"), py::arg("delta"),
        "Mamba-1 selective scan: u[B,L,D] → y[B,L,D]");

    m.def("depthwise_conv1d", &py_depthwise_conv1d,
        py::arg("x"), py::arg("w"), py::arg("bias") = py::none(),
        "Causal depthwise conv1d (channels-first [B,D,L]); bias may be None");

    m.def("rmsnorm", &py_rmsnorm,
        py::arg("x"), py::arg("g"), py::arg("eps") = 1e-6f,
        "RMSNorm: normalize [rows,D] by root-mean-square then scale by g[D]");

#ifdef HAVE_NCCL
    m.attr("HAVE_NCCL") = true;
#else
    m.attr("HAVE_NCCL") = false;
#endif
}
