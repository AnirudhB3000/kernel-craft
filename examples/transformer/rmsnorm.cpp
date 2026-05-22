/**
 * \file rmsnorm.cpp
 * \brief Fused RMSNorm (Root Mean Square Layer Norm) example.
 *
 * \par What this demonstrates
 * - Fused RMSNorm used in Mamba, LLaMA, Mistral, and most modern transformer
 *   variants as a lighter alternative to LayerNorm (no mean subtraction).
 * - GPU kernel (`launch_rmsnorm`) vs CPU reference.
 * - Memory-bandwidth analysis: the kernel reads x + g and writes y, all
 *   float — compare against GPU peak BW.
 * - Integration note: in Mamba, RMSNorm gates the final output
 *   (y = rmsnorm(residual) ⊙ silu(z)).  In LLaMA it precedes every
 *   attention and FFN block.  kernel-craft's launcher is a drop-in for
 *   torch's `F.rms_norm`.
 *
 * \par Kernel design
 * One thread block per row (B×T rows); parallel reduction over D using
 * shared memory.  Two passes:
 *   1. Parallel sum-of-squares → shared reduction → `rms_inv = rsqrtf(…)`
 *   2. Normalize and scale: `y[i] = x[i] * rms_inv * g[i]`
 * Block size RMS_BLOCK=256 handles D up to 65536 via grid-stride loops.
 *
 * \par Performance (RTX 4070 Laptop, 128 rows)
 * \code
 *   D=512  →  ~0.030 ms,  ~18 GB/s
 *   D=1024 →  ~0.023 ms,  ~46 GB/s
 *   D=2048 →  ~0.029 ms,  ~72 GB/s
 *   D=4096 →  ~0.025 ms, ~171 GB/s
 * \endcode
 * At D=4096 it approaches ~170 GB/s.  At small D, per-block launch overhead
 * is visible.
 *
 * \par Integration — LLaMA / Mamba layer replacement
 * \code
 *   // Replace torch.nn.RMSNorm forward:
 *   launch_rmsnorm(d_x, d_g, d_y, rows, D, 1e-6f, stream);
 * \endcode
 */

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" void launch_rmsnorm(
    const float* d_x, const float* d_g, float* d_y,
    int rows, int D, float eps, cudaStream_t stream);

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------
/**
 * \brief CPU reference for RMSNorm.
 *
 * \param[in]  x    Input [rows, D].
 * \param[in]  g    Scale [D].
 * \param[out] y    Output [rows, D].
 * \param[in]  eps  Stability epsilon.
 */
static void cpu_rmsnorm(const float* x, const float* g, float* y,
                        int rows, int D, float eps)
{
    for (int r = 0; r < rows; ++r) {
        float rms_sq = 0.f;
        for (int d = 0; d < D; ++d)
            rms_sq += x[r * D + d] * x[r * D + d];
        float inv = 1.f / sqrtf(rms_sq / D + eps);
        for (int d = 0; d < D; ++d)
            y[r * D + d] = x[r * D + d] * inv * g[d];
    }
}

static void fill_rand(std::vector<float>& v, float lo = -1.f, float hi = 1.f) {
    for (float& x : v)
        x = lo + (hi - lo) * (static_cast<float>(rand()) / RAND_MAX);
}

static float max_abs_error(const std::vector<float>& a,
                            const std::vector<float>& b) {
    float err = 0.f;
    for (size_t i = 0; i < a.size(); ++i)
        err = std::max(err, std::abs(a[i] - b[i]));
    return err;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    srand(42);

    const int rows = 128;
    const float eps = 1e-6f;
    const int dims[] = {512, 1024, 2048, 4096};

    printf("RMSNorm example\n");
    printf("rows=%d  eps=%.0e\n\n", rows, (double)eps);

    for (int D : dims) {
        std::vector<float> h_x(rows * D), h_g(D), h_y_gpu(rows * D), h_y_cpu(rows * D);
        fill_rand(h_x);
        fill_rand(h_g, 0.5f, 1.5f);

        cpu_rmsnorm(h_x.data(), h_g.data(), h_y_cpu.data(), rows, D, eps);

        float *d_x, *d_g, *d_y;
        cudaMalloc(&d_x, rows * D * sizeof(float));
        cudaMalloc(&d_g, D       * sizeof(float));
        cudaMalloc(&d_y, rows * D * sizeof(float));
        cudaMemcpy(d_x, h_x.data(), rows * D * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_g, h_g.data(), D       * sizeof(float), cudaMemcpyHostToDevice);

        // Warmup
        launch_rmsnorm(d_x, d_g, d_y, rows, D, eps, 0);
        cudaDeviceSynchronize();

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        const int reps = 200;
        cudaEventRecord(t0);
        for (int r = 0; r < reps; ++r)
            launch_rmsnorm(d_x, d_g, d_y, rows, D, eps, 0);
        cudaEventRecord(t1);
        cudaDeviceSynchronize();
        float ms = 0.f;
        cudaEventElapsedTime(&ms, t0, t1);
        ms /= reps;

        cudaMemcpy(h_y_gpu.data(), d_y, rows * D * sizeof(float),
                   cudaMemcpyDeviceToHost);

        // Bandwidth: read x + g, write y (approx)
        double bytes = (double)(rows * D * 2 + D) * sizeof(float);
        double gb_s  = bytes / (ms * 1e-3) * 1e-9;
        float  err   = max_abs_error(h_y_gpu, h_y_cpu);

        printf("  D=%5d  time=%6.3f ms  BW=%5.0f GB/s  max_err=%.2e\n",
               D, ms, gb_s, err);

        cudaFree(d_x); cudaFree(d_g); cudaFree(d_y);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

    printf("\nDone.\n");
    return 0;
}
