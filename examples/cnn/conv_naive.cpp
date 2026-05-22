/**
 * \file cnn/conv_naive.cpp
 * \brief Baseline naive 2-D convolution on the GPU.
 *
 * \par Purpose
 * This example is the entry point for understanding how a convolution maps
 * onto CUDA's thread hierarchy.  Every output pixel is computed by one
 * thread; threads read directly from global memory on every access, making
 * this kernel memory-bandwidth-bound.  It serves as the unoptimised baseline
 * against which all faster variants (tiled, fused, INT8) are compared.
 *
 * \par Integration context — CNN inference engines
 * In a production CNN inference engine (TensorRT, ONNX Runtime) the first
 * convolution layer receives a pre-processed image tensor from the host and
 * produces an intermediate feature map that feeds into BatchNorm + ReLU.
 * The naive kernel is not used in production but its simplicity makes it
 * the correct starting point when porting a new layer type or verifying the
 * numerical correctness of a faster variant before enabling it in the engine.
 *
 * Typical engine integration flow:
 * \code
 * // 1. Host allocates persistent device buffers at model-load time.
 * cudaMalloc(&d_input,  batch * C * H * W * sizeof(float));
 * cudaMalloc(&d_output, batch * C_out * H * W * sizeof(float));
 *
 * // 2. Per-request: copy preprocessed frame and call the kernel.
 * cudaMemcpyAsync(d_input, host_frame, bytes, cudaMemcpyHostToDevice, stream);
 * launch_conv_naive(d_input, d_kernel_weights, d_output, W, H, ksize);
 *
 * // 3. Pass d_output directly to the next layer without touching host memory.
 * launch_bn_relu_fused(d_output, gamma, beta, d_output, W, H);
 * \endcode
 *
 * \par Performance notes
 * - Memory-bound: every thread re-reads ksize² values from global memory.
 * - Neighbouring threads re-fetch overlapping halos — no shared-memory reuse.
 * - See example_conv_tiled.cpp for the shared-memory version (~1.5–2× faster).
 * - See example_fused_pipeline.cpp to fuse this with BN + ReLU (saves ~40%
 *   global memory traffic).
 *
 * \par Compile and run
 * \code
 * cmake --build build --target example_conv_naive
 * ./build/examples/example_conv_naive
 * \endcode
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

/**
 * \brief GPU naive 2-D convolution launcher (src/kernels/core/conv_naive.cu).
 *
 * Grid: ⌈width/block.x⌉ × ⌈height/block.y⌉ blocks, each of size block.
 * Each thread writes exactly one output pixel — no shared memory is used.
 * Out-of-bounds reads are zero-padded (same-padding semantics).
 *
 * \param[in]  d_input   Row-major float32 device array [height × width].
 * \param[in]  d_kernel  Row-major float32 device array [ksize × ksize].
 * \param[out] d_output  Row-major float32 device array [height × width].
 * \param[in]  width     Image width in pixels.
 * \param[in]  height    Image height in pixels.
 * \param[in]  ksize     Kernel side length (must be odd, e.g. 3 or 5).
 * \param[in]  block     CUDA thread-block dimensions (default 16×16).
 */
extern "C" void launch_conv_naive(const float* d_input, const float* d_kernel,
                                  float* d_output,
                                  int width, int height, int ksize,
                                  dim3 block = dim3(16, 16, 1));

/** \brief CPU reference used for correctness verification. */
static void conv_cpu(const float* input, const float* kernel, float* output,
                     int width, int height, int ksize) {
    int kHalf = ksize / 2;
    for (int oy = 0; oy < height; ++oy)
        for (int ox = 0; ox < width; ++ox) {
            float sum = 0.f;
            for (int ky = 0; ky < ksize; ++ky) {
                int iy = oy + ky - kHalf;
                if (iy < 0 || iy >= height) continue;
                for (int kx = 0; kx < ksize; ++kx) {
                    int ix = ox + kx - kHalf;
                    if (ix < 0 || ix >= width) continue;
                    sum += input[iy * width + ix] * kernel[ky * ksize + kx];
                }
            }
            output[oy * width + ox] = sum;
        }
}

int main() {
    // -----------------------------------------------------------------------
    // Problem setup
    // 16×16 single-channel image with a 3×3 box-blur kernel.
    // In a real CNN (e.g. ResNet-50) the first layer takes a 224×224×3 input;
    // replace W/H with the actual feature-map spatial dimensions.
    // -----------------------------------------------------------------------
    const int W = 16, H = 16, K = 3;
    const int N = W * H, Ksq = K * K;

    float h_in[256], h_ker[9], h_out[256], h_ref[256];

    /* Synthetic ramp: easy to trace individual pixel values during debugging. */
    for (int i = 0; i < N;   ++i) h_in[i]  = static_cast<float>(i % 16);
    /* 3×3 box blur: every output pixel = average of its 3×3 neighbourhood. */
    for (int i = 0; i < Ksq; ++i) h_ker[i] = 1.f / 9.f;

    // -----------------------------------------------------------------------
    // Device allocation and H2D transfer
    // In a real inference server, buffers are allocated once at model-load
    // time and reused across requests.  See example_memory_pool.cpp.
    // -----------------------------------------------------------------------
    float *d_in, *d_ker, *d_out;
    cudaMalloc(&d_in,  sizeof(float) * N);
    cudaMalloc(&d_ker, sizeof(float) * Ksq);
    cudaMalloc(&d_out, sizeof(float) * N);

    cudaMemcpy(d_in,  h_in,  sizeof(float) * N,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_ker, h_ker, sizeof(float) * Ksq, cudaMemcpyHostToDevice);

    // -----------------------------------------------------------------------
    // Kernel launch
    // The 16×16 block maps each row to half a warp so consecutive threads
    // access consecutive addresses (coalesced reads from global memory).
    // Grid size is calculated inside the launcher to cover all output pixels.
    // -----------------------------------------------------------------------
    launch_conv_naive(d_in, d_ker, d_out, W, H, K);

    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // -----------------------------------------------------------------------
    // Correctness verification — always validate before measuring performance.
    // A max absolute error of 1e-4 is expected for float32 round-off at this
    // problem size; larger errors indicate a logic bug in the kernel.
    // -----------------------------------------------------------------------
    conv_cpu(h_in, h_ker, h_ref, W, H, K);
    bool ok = true;
    for (int i = 0; i < N; ++i)
        if (std::fabs(h_out[i] - h_ref[i]) > 1e-4f) { ok = false; break; }

    printf("Naive Convolution Example\n");
    printf("  Image %dx%d   Kernel %dx%d   Block 16x16\n", W, H, K, K);
    printf("  Correctness : %s\n", ok ? "PASS (GPU == CPU)" : "FAIL — MISMATCH");
    printf("  h_out[0]=%.4f  h_ref[0]=%.4f\n", h_out[0], h_ref[0]);

    cudaFree(d_in); cudaFree(d_ker); cudaFree(d_out);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
