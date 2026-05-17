/**
 * \file test_quant_int4.cpp
 * \brief Unit tests for INT4 weight dequantization and GEMV.
 *
 * Tests: dequantization round-trip error, GEMV correctness vs FP32 reference.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>

extern "C" void launch_dequantize_int4(
    const uint8_t* d_packed, const float* d_scales, const uint8_t* d_zeros,
    float* d_output, int rows, int cols, int group_size, cudaStream_t stream);

extern "C" void launch_gemv_int4(
    const uint8_t* d_packed, const float* d_scales, const uint8_t* d_zeros,
    const float* d_x, float* d_y, int rows, int cols, int group_size,
    cudaStream_t stream);

static int g_pass = 0, g_fail = 0;
static void check(bool ok, const char* name) {
    if (ok) { printf("[PASS] %s\n", name); ++g_pass; }
    else    { printf("[FAIL] %s\n", name); ++g_fail; }
}

// Pack two int4 values into one byte (low nibble = first, high nibble = second)
static uint8_t pack_nibbles(int8_t lo, int8_t hi) {
    return (uint8_t)((lo & 0x0F) | ((hi & 0x0F) << 4));
}

static void test_dequantize_correctness() {
    // Small matrix: 4 rows × 8 cols, group_size=8 (one group per row)
    int rows = 4, cols = 8, group_size = 8;
    int packed_cols = cols / 2;           // 4 bytes per row
    int scale_cols  = cols / group_size;  // 1 scale per row

    // INT4 values: -8 to 7
    std::vector<int8_t> weights_i4(rows * cols);
    for (int i = 0; i < rows * cols; ++i) weights_i4[i] = (int8_t)((i % 16) - 8);

    // Pack
    std::vector<uint8_t> packed(rows * packed_cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < packed_cols; ++c)
            packed[r * packed_cols + c] = pack_nibbles(
                weights_i4[r * cols + c * 2],
                weights_i4[r * cols + c * 2 + 1]);

    // Scales and zero points (zero_point = 0)
    std::vector<float>   scales(rows * scale_cols, 0.125f);  // 1/8
    // Zero points packed: 2 zeros per byte, zero_point=0 → all 0
    int zeros_cols = scale_cols / 2 + (scale_cols % 2);
    std::vector<uint8_t> zeros(rows * zeros_cols, 0x00);

    // Expected: w_f32 = w_i4 * scale (zero_point = 0)
    std::vector<float> expected(rows * cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            expected[r * cols + c] = (float)weights_i4[r * cols + c] * 0.125f;

    // Upload and dequantize on GPU
    uint8_t *dp, *dz; float *ds, *dout;
    cudaMalloc(&dp,  rows * packed_cols * sizeof(uint8_t));
    cudaMalloc(&dz,  rows * zeros_cols  * sizeof(uint8_t));
    cudaMalloc(&ds,  rows * scale_cols  * sizeof(float));
    cudaMalloc(&dout,rows * cols        * sizeof(float));

    cudaMemcpy(dp,  packed.data(), rows * packed_cols * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dz,  zeros.data(),  rows * zeros_cols  * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ds,  scales.data(), rows * scale_cols  * sizeof(float),   cudaMemcpyHostToDevice);

    launch_dequantize_int4(dp, ds, dz, dout, rows, cols, group_size, 0);

    std::vector<float> result(rows * cols);
    cudaMemcpy(result.data(), dout, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0.f;
    for (int i = 0; i < rows * cols; ++i)
        max_err = fmaxf(max_err, fabsf(result[i] - expected[i]));
    check(max_err < 1e-5f, "INT4 dequantization round-trip (4×8, gs=8)");
    if (max_err >= 1e-5f) printf("  max abs error = %e\n", max_err);

    cudaFree(dp); cudaFree(dz); cudaFree(ds); cudaFree(dout);
}

static void test_gemv_correctness() {
    // 16×16 matrix, group_size=8
    int rows = 16, cols = 16, group_size = 8;
    int packed_cols = cols / 2;
    int scale_cols  = cols / group_size;
    int zeros_cols  = scale_cols / 2 + (scale_cols % 2);

    srand(42);
    // Random INT4 weights in [-7, 7] (avoid -8 for simplicity with zero-point=0)
    std::vector<int8_t> weights_i4(rows * cols);
    for (auto& w : weights_i4) w = (int8_t)((rand() % 15) - 7);

    std::vector<uint8_t> packed(rows * packed_cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < packed_cols; ++c)
            packed[r * packed_cols + c] = pack_nibbles(
                weights_i4[r * cols + c * 2],
                weights_i4[r * cols + c * 2 + 1]);

    float scale_val = 0.1f;
    std::vector<float>   scales(rows * scale_cols, scale_val);
    std::vector<uint8_t> zeros(rows * zeros_cols, 0x00);

    // Random input vector
    std::vector<float> x(cols);
    for (auto& v : x) v = ((float)rand() / RAND_MAX) * 2.f - 1.f;

    // CPU reference: y = W * x
    std::vector<float> y_ref(rows, 0.f);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            y_ref[r] += (float)weights_i4[r * cols + c] * scale_val * x[c];

    uint8_t *dp, *dz; float *ds, *dx, *dy;
    cudaMalloc(&dp,  rows * packed_cols * sizeof(uint8_t));
    cudaMalloc(&dz,  rows * zeros_cols  * sizeof(uint8_t));
    cudaMalloc(&ds,  rows * scale_cols  * sizeof(float));
    cudaMalloc(&dx,  cols               * sizeof(float));
    cudaMalloc(&dy,  rows               * sizeof(float));

    cudaMemcpy(dp, packed.data(), rows * packed_cols * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dz, zeros.data(),  rows * zeros_cols  * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, scales.data(), rows * scale_cols  * sizeof(float),   cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(),      cols               * sizeof(float),   cudaMemcpyHostToDevice);

    launch_gemv_int4(dp, ds, dz, dx, dy, rows, cols, group_size, 0);

    std::vector<float> y_gpu(rows);
    cudaMemcpy(y_gpu.data(), dy, rows * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0.f;
    for (int i = 0; i < rows; ++i) max_err = fmaxf(max_err, fabsf(y_gpu[i] - y_ref[i]));
    check(max_err < 1e-4f, "INT4 GEMV correctness (16×16, gs=8)");
    if (max_err >= 1e-4f) printf("  max abs error = %e\n", max_err);

    cudaFree(dp); cudaFree(dz); cudaFree(ds); cudaFree(dx); cudaFree(dy);
}

static void test_multi_group() {
    // 4×128, group_size=64 (2 groups per row)
    int rows = 4, cols = 128, group_size = 64;
    int packed_cols = cols / 2;
    int scale_cols  = cols / group_size;  // 2
    int zeros_cols  = scale_cols / 2 + (scale_cols % 2);  // 1

    srand(13);
    std::vector<int8_t> weights_i4(rows * cols);
    for (auto& w : weights_i4) w = (int8_t)((rand() % 15) - 7);

    std::vector<uint8_t> packed(rows * packed_cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < packed_cols; ++c)
            packed[r * packed_cols + c] = pack_nibbles(
                weights_i4[r * cols + c * 2],
                weights_i4[r * cols + c * 2 + 1]);

    // Different scales per group
    std::vector<float> scales(rows * scale_cols);
    for (int i = 0; i < rows * scale_cols; ++i) scales[i] = 0.05f + 0.01f * i;
    std::vector<uint8_t> zeros(rows * zeros_cols, 0x00);

    std::vector<float> expected(rows * cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            expected[r * cols + c] = (float)weights_i4[r * cols + c]
                                   * scales[r * scale_cols + c / group_size];

    uint8_t *dp, *dz; float *ds, *dout;
    cudaMalloc(&dp,  rows * packed_cols * sizeof(uint8_t));
    cudaMalloc(&dz,  rows * zeros_cols  * sizeof(uint8_t));
    cudaMalloc(&ds,  rows * scale_cols  * sizeof(float));
    cudaMalloc(&dout,rows * cols        * sizeof(float));

    cudaMemcpy(dp,  packed.data(), rows * packed_cols * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dz,  zeros.data(),  rows * zeros_cols  * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ds,  scales.data(), rows * scale_cols  * sizeof(float),   cudaMemcpyHostToDevice);

    launch_dequantize_int4(dp, ds, dz, dout, rows, cols, group_size, 0);

    std::vector<float> result(rows * cols);
    cudaMemcpy(result.data(), dout, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0.f;
    for (int i = 0; i < rows * cols; ++i)
        max_err = fmaxf(max_err, fabsf(result[i] - expected[i]));
    check(max_err < 1e-5f, "INT4 dequant multi-group (4×128, gs=64)");
    if (max_err >= 1e-5f) printf("  max abs error = %e\n", max_err);

    cudaFree(dp); cudaFree(dz); cudaFree(ds); cudaFree(dout);
}

int main() {
    printf("=== INT4 Quantization Tests ===\n");
    test_dequantize_correctness();
    test_gemv_correctness();
    test_multi_group();
    printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
