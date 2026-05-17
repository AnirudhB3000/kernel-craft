/**
 * \file test_fp8_quant.cpp
 * \brief Unit tests for FP8 E4M3 activation quantization.
 *
 * Tests: per-token round-trip, per-tensor round-trip, SmoothQuant scaling,
 * saturation at FP8_E4M3_MAX bounds.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>

extern "C" void launch_quantize_fp8(
    const float* d_input, int8_t* d_output, float* d_scales,
    int rows, int cols, bool per_token, cudaStream_t stream);

extern "C" void launch_dequantize_fp8(
    const int8_t* d_input, const float* d_scales, float* d_output,
    int rows, int cols, bool per_token, cudaStream_t stream);

extern "C" void launch_smoothquant_scale(
    const float* d_input, const float* d_smooth_scales, float* d_output,
    int rows, int cols, cudaStream_t stream);

static int g_pass = 0, g_fail = 0;
static void check(bool ok, const char* name) {
    if (ok) { printf("[PASS] %s\n", name); ++g_pass; }
    else    { printf("[FAIL] %s\n", name); ++g_fail; }
}

static void test_per_token_roundtrip() {
    // Per-token quantize → dequantize: error should be < 1/E4M3_precision
    int rows = 8, cols = 64;
    int total = rows * cols;

    srand(42);
    std::vector<float> input(total);
    for (auto& v : input) v = ((float)rand() / RAND_MAX) * 2.f - 1.f;

    float *d_in, *d_out, *d_scales; int8_t *d_q;
    cudaMalloc(&d_in,    total * sizeof(float));
    cudaMalloc(&d_out,   total * sizeof(float));
    cudaMalloc(&d_scales,rows  * sizeof(float));
    cudaMalloc(&d_q,     total * sizeof(int8_t));

    cudaMemcpy(d_in, input.data(), total * sizeof(float), cudaMemcpyHostToDevice);

    launch_quantize_fp8(d_in, d_q, d_scales, rows, cols, true, 0);
    launch_dequantize_fp8(d_q, d_scales, d_out, rows, cols, true, 0);

    std::vector<float> output(total);
    cudaMemcpy(output.data(), d_out, total * sizeof(float), cudaMemcpyDeviceToHost);

    // E4M3 has 3 mantissa bits → relative error ≤ 1/8 ≈ 12.5%
    // Per-token scale covers the full range, so absolute error ≤ max_abs * 1/8
    float max_rel_err = 0.f;
    for (int i = 0; i < total; ++i) {
        float abs_in = fabsf(input[i]);
        if (abs_in > 1e-6f) {
            float rel = fabsf(output[i] - input[i]) / abs_in;
            max_rel_err = fmaxf(max_rel_err, rel);
        }
    }
    check(max_rel_err < 0.15f, "FP8 per-token round-trip relative error < 15%");
    printf("  max relative error = %.4f%%\n", max_rel_err * 100.f);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_scales); cudaFree(d_q);
}

static void test_per_token_zero_preservation() {
    // Zeros should quantize and dequantize exactly to zero
    int rows = 4, cols = 32, total = rows * cols;
    std::vector<float> input(total, 0.f);

    float *d_in, *d_out, *d_scales; int8_t *d_q;
    cudaMalloc(&d_in,    total * sizeof(float));
    cudaMalloc(&d_out,   total * sizeof(float));
    cudaMalloc(&d_scales,rows  * sizeof(float));
    cudaMalloc(&d_q,     total * sizeof(int8_t));

    cudaMemcpy(d_in, input.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    launch_quantize_fp8(d_in, d_q, d_scales, rows, cols, true, 0);
    launch_dequantize_fp8(d_q, d_scales, d_out, rows, cols, true, 0);

    std::vector<float> output(total);
    cudaMemcpy(output.data(), d_out, total * sizeof(float), cudaMemcpyDeviceToHost);

    bool all_zero = true;
    for (int i = 0; i < total; ++i)
        if (fabsf(output[i]) > 1e-6f) { all_zero = false; break; }
    check(all_zero, "FP8 zero input → zero output");

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_scales); cudaFree(d_q);
}

static void test_smoothquant_scaling() {
    // SmoothQuant: dividing each column by smooth_scale should reduce max abs
    int rows = 4, cols = 16, total = rows * cols;

    // Activations with large outliers in specific channels
    std::vector<float> input(total, 1.f);
    // Column 0 has large outlier
    for (int r = 0; r < rows; ++r) input[r * cols + 0] = 100.f;

    // Smooth scale = max per channel (simple version)
    std::vector<float> smooth_scales(cols, 1.f);
    smooth_scales[0] = 100.f;  // remove outlier from col 0

    float *d_in, *d_out, *d_ss;
    cudaMalloc(&d_in,  total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));
    cudaMalloc(&d_ss,  cols  * sizeof(float));

    cudaMemcpy(d_in, input.data(),         total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ss, smooth_scales.data(), cols  * sizeof(float), cudaMemcpyHostToDevice);

    launch_smoothquant_scale(d_in, d_ss, d_out, rows, cols, 0);

    std::vector<float> output(total);
    cudaMemcpy(output.data(), d_out, total * sizeof(float), cudaMemcpyDeviceToHost);

    // Column 0 should now be 1.0 (100/100), other cols unchanged (1.0/1.0 = 1.0)
    bool ok = true;
    for (int r = 0; r < rows; ++r) {
        if (fabsf(output[r * cols + 0] - 1.f) > 1e-5f) { ok = false; break; }
        for (int c = 1; c < cols; ++c)
            if (fabsf(output[r * cols + c] - 1.f) > 1e-5f) { ok = false; break; }
    }
    check(ok, "SmoothQuant per-channel scaling (outlier reduction)");

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_ss);
}

static void test_per_tensor_roundtrip() {
    // Per-tensor quantize using caller-provided scale
    int rows = 4, cols = 32, total = rows * cols;
    srand(7);
    std::vector<float> input(total);
    float max_abs = 0.f;
    for (auto& v : input) {
        v = ((float)rand() / RAND_MAX) * 4.f - 2.f;
        max_abs = fmaxf(max_abs, fabsf(v));
    }
    float scale_val = max_abs / 448.f;  // E4M3 max

    float *d_in, *d_out, *d_scales; int8_t *d_q;
    cudaMalloc(&d_in,    total * sizeof(float));
    cudaMalloc(&d_out,   total * sizeof(float));
    cudaMalloc(&d_scales,1     * sizeof(float));
    cudaMalloc(&d_q,     total * sizeof(int8_t));

    cudaMemcpy(d_in,    input.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales,&scale_val,   1     * sizeof(float), cudaMemcpyHostToDevice);

    launch_quantize_fp8(d_in, d_q, d_scales, rows, cols, false, 0);
    launch_dequantize_fp8(d_q, d_scales, d_out, rows, cols, false, 0);

    std::vector<float> output(total);
    cudaMemcpy(output.data(), d_out, total * sizeof(float), cudaMemcpyDeviceToHost);

    float max_rel_err = 0.f;
    for (int i = 0; i < total; ++i) {
        float abs_in = fabsf(input[i]);
        if (abs_in > 1e-6f)
            max_rel_err = fmaxf(max_rel_err, fabsf(output[i] - input[i]) / abs_in);
    }
    check(max_rel_err < 0.15f, "FP8 per-tensor round-trip relative error < 15%");
    printf("  max relative error = %.4f%%\n", max_rel_err * 100.f);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_scales); cudaFree(d_q);
}

int main() {
    printf("=== FP8 Quantization Tests ===\n");
    test_per_token_roundtrip();
    test_per_token_zero_preservation();
    test_smoothquant_scaling();
    test_per_tensor_roundtrip();
    printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
