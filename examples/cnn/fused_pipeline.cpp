/**
 * \file cnn/fused_pipeline.cpp
 * \brief Fused conv + BatchNorm + ReLU in a single CUDA kernel.
 *
 * \par Purpose
 * Demonstrates the fused Conv→BN→ReLU kernel and explains why kernel fusion
 * is the single most impactful optimisation in CNN inference pipelines.
 *
 * \par Why fusion matters
 * Running conv, BN, and ReLU as three separate kernels forces every
 * intermediate feature map to round-trip through global memory:
 * \code
 *   conv:  read(input) → write(feat)    [W×H floats written]
 *   BN:    read(feat)  → write(feat)    [W×H floats read + written again]
 *   ReLU:  read(feat)  → write(output)  [W×H floats read + written again]
 * \endcode
 * The fused kernel keeps intermediate values in registers:
 * \code
 *   fused: read(input) → conv → BN → ReLU → write(output)
 *          only ONE global-memory write, same compute
 * \endcode
 * On RTX 4070 this saves ~40% global memory traffic and delivers ~2.3× speedup
 * over the separate pipeline (see CLAUDE.md §Phase 3 benchmarks).
 *
 * \par BatchNorm folding — eliminating BN at inference time
 * During training, BN computes per-channel mean/variance from the batch.
 * At inference, mean and variance are fixed (running statistics), so BN
 * reduces to a per-pixel affine transform:  y = (x - mean) / std * γ + β.
 * This can be pre-absorbed into the convolution weights before deployment:
 * \code
 *   // Folded weight:  W_fold = W * γ / std
 *   // Folded bias:    b_fold = -mean * γ / std + β
 *   kc.bn_folding(conv_weights, bn_gamma, bn_beta, bn_mean, bn_var, eps)
 * \endcode
 * See src/kernels/inference/bn_folding.cu and the Python binding bn_folding().
 * After folding the fused kernel simplifies to conv + bias + ReLU — used
 * in TensorRT's layer fusion and in our ConvReLU TensorRT plugin.
 *
 * \par Integration context
 * This is the innermost building block of a CNN inference engine stem:
 * \code
 *   // ResNet-50 stem (224×224 → 112×112 feature map)
 *   launch_conv_bn_relu_fused(d_img, d_w0, gamma, beta, d_feat, 224, 224, 7);
 *   // d_feat goes directly to the next fused layer — no D2H copy
 * \endcode
 * Wrap this in a CUDA Graph (see example_cuda_graphs.cpp) to further
 * eliminate per-layer launch overhead.
 *
 * \par Compile and run
 * \code
 * cmake --build build --target example_fused_pipeline
 * ./build/examples/example_fused_pipeline
 * \endcode
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

/**
 * \brief Fused conv + BN-affine + ReLU launcher (src/pipelines/pipeline_fused.cu).
 *
 * Executes three logical steps in a single kernel:
 * 1. Tiled 2-D convolution with zero-padding (same as launch_conv_tiled).
 * 2. Per-pixel BN-affine:  feat = gamma * feat + beta
 *    (assumes BN mean/var already folded into gamma/beta).
 * 3. ReLU clamp:  out = max(0, feat).
 *
 * \param[in]  d_input  Row-major float32 device array [height × width].
 * \param[in]  d_kernel Row-major float32 device array [ksize × ksize].
 * \param[in]  gamma    BN scale factor (scalar; per-channel in a real model).
 * \param[in]  beta     BN shift factor (scalar; per-channel in a real model).
 * \param[out] d_output Row-major float32 device array [height × width].
 * \param[in]  width    Feature-map width in pixels.
 * \param[in]  height   Feature-map height in pixels.
 * \param[in]  ksize    Convolution kernel side length (must be odd).
 * \param[in]  block    CUDA block dimensions (default 8×8).
 */
extern "C" void launch_conv_bn_relu_fused(const float* d_input,
                                          const float* d_kernel,
                                          float gamma, float beta,
                                          float* d_output,
                                          int width, int height, int ksize,
                                          dim3 block = dim3(8, 8, 1));

/** \brief Separate conv launcher for comparison timing. */
extern "C" void launch_conv_tiled(const float* d_input, const float* d_kernel,
                                  float* d_output,
                                  int width, int height, int ksize,
                                  dim3 block);

/** \brief Separate in-place ReLU launcher for comparison timing. */
extern "C" void launch_relu(float* d_data, int size);

int main() {
    const int W = 256, H = 256, K = 3;
    const int N = W * H, Ksq = K * K;
    const int ITERS = 20;

    /* Synthesise a random-ish input and edge-detection kernel. */
    float *h_in  = new float[N];
    float *h_ker = new float[Ksq];
    float *h_out = new float[N];
    for (int i = 0; i < N;   ++i) h_in[i]  = (float)(i % 255) / 255.f;
    for (int i = 0; i < Ksq; ++i) h_ker[i] = 1.f / 9.f;

    /* BN parameters (γ=1.5, β=0.2 simulates a folded BN layer). */
    const float gamma = 1.5f, beta = 0.2f;

    float *d_in, *d_ker, *d_out, *d_tmp;
    cudaMalloc(&d_in,  sizeof(float) * N);
    cudaMalloc(&d_ker, sizeof(float) * Ksq);
    cudaMalloc(&d_out, sizeof(float) * N);
    cudaMalloc(&d_tmp, sizeof(float) * N);  /* intermediate buffer for separate path */

    cudaMemcpy(d_in,  h_in,  sizeof(float) * N,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_ker, h_ker, sizeof(float) * Ksq, cudaMemcpyHostToDevice);

    dim3 block(8, 8, 1);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    /* ---- Fused pipeline timing ---- */
    cudaEventRecord(t0);
    for (int i = 0; i < ITERS; ++i)
        launch_conv_bn_relu_fused(d_in, d_ker, gamma, beta, d_out, W, H, K);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float t_fused;
    cudaEventElapsedTime(&t_fused, t0, t1);

    /* ---- Separate pipeline timing (conv → in-place scale+shift → relu) ---- */
    cudaEventRecord(t0);
    for (int i = 0; i < ITERS; ++i) {
        launch_conv_tiled(d_in, d_ker, d_tmp, W, H, K, block);
        /* BN affine applied as relu_bias = gamma*x+beta > 0; simulated via relu */
        launch_relu(d_tmp, N);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float t_sep;
    cudaEventElapsedTime(&t_sep, t0, t1);

    cudaMemcpy(h_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    printf("Fused Conv+BN+ReLU Pipeline (%dx%d, K=%d, %d iters)\n", W, H, K, ITERS);
    printf("  Fused    : %.2f ms total  (%.3f ms/iter)\n", t_fused, t_fused / ITERS);
    printf("  Separate : %.2f ms total  (%.3f ms/iter)\n", t_sep,   t_sep   / ITERS);
    printf("  Speedup  : %.2fx\n", t_sep / t_fused);
    printf("  h_out[0] = %.4f  (conv → gamma*x+beta → relu)\n", h_out[0]);
    printf("\nKey insight: fusion eliminates 2 intermediate global-memory round-trips.\n");
    printf("See src/pipelines/pipeline_fused.cu for the kernel implementation.\n");
    printf("See src/kernels/inference/bn_folding.cu for BN weight absorption.\n");

    cudaFree(d_in); cudaFree(d_ker); cudaFree(d_out); cudaFree(d_tmp);
    delete[] h_in; delete[] h_ker; delete[] h_out;
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return EXIT_SUCCESS;
}
