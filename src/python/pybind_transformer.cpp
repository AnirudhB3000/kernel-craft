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
// Module definition — merged into existing kernel_craft_python module
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
}
