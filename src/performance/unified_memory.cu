/**
 * \file unified_memory.cu
 * \brief Unified (managed) memory variants of convolution kernels.
 *
 * CUDA Unified Memory (cudaMallocManaged) automatically migrates pages
 * between host and device on demand, removing explicit cudaMemcpy calls.
 * This simplifies memory management at the cost of potential page-fault
 * overhead on first access.
 *
 * \par Prefetching
 * Performance can be recovered by calling cudaMemPrefetchAsync before a
 * kernel, which pre-migrates pages asynchronously without page faults.
 *
 * \par Oversubscription
 * When the managed allocation exceeds GPU memory, the CUDA driver
 * automatically evicts cold pages to host memory and re-fetches on demand.
 * The test exercises this by allocating more than GPU VRAM.
 *
 * \par Usage
 * \code
 * float *um_input, *um_kernel, *um_output;
 * unified_alloc(&um_input, width * height);
 * unified_alloc(&um_kernel, ksize * ksize);
 * unified_alloc(&um_output, width * height);
 * unified_conv_tiled(um_input, um_kernel, um_output, width, height, ksize);
 * cudaDeviceSynchronize();  // ensure results are visible on host
 * unified_free(um_input); unified_free(um_kernel); unified_free(um_output);
 * \endcode
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifndef TILE_W
#define TILE_W 8
#endif
#ifndef TILE_H
#define TILE_H 8
#endif

/* -------------------------------------------------------------------------
 * Device kernels
 * ---------------------------------------------------------------------- */

/**
 * \brief Naive convolution kernel operating on unified memory.
 *
 * Identical computation to conv_naive but intended to be called with
 * managed pointers so the driver migrates pages transparently.
 *
 * \param[in]  input  Managed pointer to input image.
 * \param[in]  kernel Managed pointer to convolution weights.
 * \param[out] output Managed pointer to output image.
 * \param[in]  width  Image width in pixels.
 * \param[in]  height Image height in pixels.
 * \param[in]  ksize  Kernel side length.
 */
__global__ void unified_conv_naive_kernel(const float* __restrict__ input,
                                           const float* __restrict__ kernel,
                                           float* __restrict__ output,
                                           int width, int height, int ksize) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= width || oy >= height) return;

    int kHalf = ksize / 2;
    double sum = 0.0;
    for (int ky = 0; ky < ksize; ++ky) {
        int iy = oy + ky - kHalf;
        if (iy < 0 || iy >= height) continue;
        for (int kx = 0; kx < ksize; ++kx) {
            int ix = ox + kx - kHalf;
            if (ix < 0 || ix >= width) continue;
            sum += static_cast<double>(input[iy * width + ix]) *
                   static_cast<double>(kernel[ky * ksize + kx]);
        }
    }
    output[oy * width + ox] = static_cast<float>(sum);
}

/**
 * \brief Tiled convolution kernel operating on unified memory.
 *
 * Uses shared-memory tiling for better bandwidth efficiency. Managed
 * pointers are valid device pointers, so no special treatment needed.
 *
 * \param[in]  input  Managed pointer to input image.
 * \param[in]  kernel Managed pointer to convolution weights.
 * \param[out] output Managed pointer to output image.
 * \param[in]  width  Image width in pixels.
 * \param[in]  height Image height in pixels.
 * \param[in]  ksize  Kernel side length.
 */
__global__ void unified_conv_tiled_kernel(const float* __restrict__ input,
                                           const float* __restrict__ kernel,
                                           float* __restrict__ output,
                                           int width, int height, int ksize) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int ox = blockIdx.x * blockDim.x + tx;
    const int oy = blockIdx.y * blockDim.y + ty;
    const int kHalf  = ksize / 2;
    const int sharedW = blockDim.x + ksize - 1;
    const int sharedH = blockDim.y + ksize - 1;
    extern __shared__ float tile[];

    for (int dy = ty; dy < sharedH; dy += blockDim.y) {
        for (int dx = tx; dx < sharedW; dx += blockDim.x) {
            int img_x = blockIdx.x * blockDim.x + dx - kHalf;
            int img_y = blockIdx.y * blockDim.y + dy - kHalf;
            float val = 0.0f;
            if (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height)
                val = input[img_y * width + img_x];
            tile[dy * sharedW + dx] = val;
        }
    }
    __syncthreads();

    if (ox < width && oy < height) {
        double sum = 0.0;
        for (int ky = 0; ky < ksize; ++ky)
            for (int kx = 0; kx < ksize; ++kx)
                sum += static_cast<double>(tile[(ty + ky) * sharedW + (tx + kx)]) *
                       static_cast<double>(kernel[ky * ksize + kx]);
        output[oy * width + ox] = static_cast<float>(sum);
    }
}

/* -------------------------------------------------------------------------
 * Host API
 * ---------------------------------------------------------------------- */

/**
 * \brief Allocate a float array in CUDA unified memory.
 *
 * \param[out] ptr   Receives the managed pointer.
 * \param[in]  count Number of floats to allocate.
 * \return 0 on success, -1 on failure.
 */
extern "C" int unified_alloc(float** ptr, int count) {
    cudaError_t err = cudaMallocManaged(ptr, count * sizeof(float));
    return (err == cudaSuccess) ? 0 : -1;
}

/**
 * \brief Free a unified memory allocation.
 *
 * \param[in] ptr Managed pointer previously returned by unified_alloc.
 */
extern "C" void unified_free(float* ptr) {
    cudaFree(ptr);
}

/**
 * \brief Prefetch a unified memory region to the GPU asynchronously.
 *
 * Avoids page-fault overhead by migrating data before the kernel runs.
 *
 * \param[in] ptr    Managed pointer to prefetch.
 * \param[in] count  Number of floats.
 * \param[in] stream CUDA stream (0 for default).
 */
extern "C" void unified_prefetch_to_gpu(float* ptr, int count, cudaStream_t stream) {
    int device;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(ptr, count * sizeof(float), device, stream);
}

/**
 * \brief Prefetch a unified memory region back to the host asynchronously.
 *
 * Call this after kernel completion (before host reads) to avoid page faults
 * on the CPU side.
 *
 * \param[in] ptr    Managed pointer to prefetch.
 * \param[in] count  Number of floats.
 * \param[in] stream CUDA stream (0 for default).
 */
extern "C" void unified_prefetch_to_cpu(float* ptr, int count, cudaStream_t stream) {
    cudaMemPrefetchAsync(ptr, count * sizeof(float), cudaCpuDeviceId, stream);
}

/**
 * \brief Launch naive convolution using unified memory pointers.
 *
 * \param[in]  um_input  Managed pointer to input image.
 * \param[in]  um_kernel Managed pointer to convolution weights.
 * \param[out] um_output Managed pointer to output image.
 * \param[in]  width     Image width in pixels.
 * \param[in]  height    Image height in pixels.
 * \param[in]  ksize     Kernel side length.
 * \param[in]  block     CUDA block dimensions.
 */
extern "C" void unified_conv_naive(float* um_input, float* um_kernel, float* um_output,
                                    int width, int height, int ksize, dim3 block) {
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    unified_conv_naive_kernel<<<grid, block>>>(um_input, um_kernel, um_output,
                                               width, height, ksize);
    cudaDeviceSynchronize();
}

/**
 * \brief Launch tiled convolution using unified memory pointers with prefetching.
 *
 * Prefetches input and kernel to GPU before launching, then prefetches output
 * back to CPU after the kernel completes to avoid page-fault overhead.
 *
 * \param[in]  um_input  Managed pointer to input image.
 * \param[in]  um_kernel Managed pointer to convolution weights.
 * \param[out] um_output Managed pointer to output image.
 * \param[in]  width     Image width in pixels.
 * \param[in]  height    Image height in pixels.
 * \param[in]  ksize     Kernel side length.
 * \param[in]  block     CUDA block dimensions.
 */
extern "C" void unified_conv_tiled(float* um_input, float* um_kernel, float* um_output,
                                    int width, int height, int ksize, dim3 block) {
    unified_prefetch_to_gpu(um_input,  width * height,     0);
    unified_prefetch_to_gpu(um_kernel, ksize * ksize,      0);
    unified_prefetch_to_gpu(um_output, width * height,     0);

    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    size_t sharedBytes = (block.x + ksize - 1) * (block.y + ksize - 1) * sizeof(float);
    unified_conv_tiled_kernel<<<grid, block, sharedBytes>>>(um_input, um_kernel, um_output,
                                                             width, height, ksize);
    cudaDeviceSynchronize();

    unified_prefetch_to_cpu(um_output, width * height, 0);
}

/**
 * \brief Benchmark unified memory (with/without prefetch) vs explicit transfers.
 *
 * Runs all three approaches for the given image size and reports timing.
 *
 * \param[in]  width         Image width in pixels.
 * \param[in]  height        Image height in pixels.
 * \param[in]  ksize         Kernel side length.
 * \param[out] explicit_ms   Time for explicit cudaMemcpy approach (ms).
 * \param[out] um_fault_ms   Time for unified memory without prefetch (ms).
 * \param[out] um_prefetch_ms Time for unified memory with prefetch (ms).
 */
extern "C" void unified_benchmark(int width, int height, int ksize,
                                   float* explicit_ms,
                                   float* um_fault_ms,
                                   float* um_prefetch_ms) {
    int imgSize  = width * height;
    int kerSize  = ksize * ksize;
    int imgBytes = imgSize * sizeof(float);
    int kerBytes = kerSize * sizeof(float);
    dim3 block(TILE_W, TILE_H);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    /* --- Explicit transfer baseline --- */
    float *h_input  = (float*)malloc(imgBytes);
    float *h_kernel = (float*)malloc(kerBytes);
    float *h_output = (float*)malloc(imgBytes);
    for (int i = 0; i < imgSize; ++i) h_input[i]  = static_cast<float>(i + 1);
    for (int i = 0; i < kerSize; ++i) h_kernel[i] = 1.0f / kerSize;

    float *d_input, *d_kernel_dev, *d_output;
    cudaMalloc(&d_input,      imgBytes);
    cudaMalloc(&d_kernel_dev, kerBytes);
    cudaMalloc(&d_output,     imgBytes);

    cudaEventRecord(ev0);
    cudaMemcpy(d_input,      h_input,  imgBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_dev, h_kernel, kerBytes, cudaMemcpyHostToDevice);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    size_t sharedBytes = (block.x + ksize - 1) * (block.y + ksize - 1) * sizeof(float);
    unified_conv_tiled_kernel<<<grid, block, sharedBytes>>>(d_input, d_kernel_dev, d_output,
                                                             width, height, ksize);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, imgBytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(explicit_ms, ev0, ev1);

    cudaFree(d_input); cudaFree(d_kernel_dev); cudaFree(d_output);

    /* --- Unified memory without prefetch --- */
    float *um_input, *um_kernel, *um_output_nopf;
    unified_alloc(&um_input,       imgSize);
    unified_alloc(&um_kernel,      kerSize);
    unified_alloc(&um_output_nopf, imgSize);
    memcpy(um_input,  h_input,  imgBytes);
    memcpy(um_kernel, h_kernel, kerBytes);

    cudaEventRecord(ev0);
    unified_conv_naive(um_input, um_kernel, um_output_nopf, width, height, ksize, block);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(um_fault_ms, ev0, ev1);

    unified_free(um_input); unified_free(um_kernel); unified_free(um_output_nopf);

    /* --- Unified memory with prefetch --- */
    float *um_in2, *um_ker2, *um_out2;
    unified_alloc(&um_in2,  imgSize);
    unified_alloc(&um_ker2, kerSize);
    unified_alloc(&um_out2, imgSize);
    memcpy(um_in2,  h_input,  imgBytes);
    memcpy(um_ker2, h_kernel, kerBytes);

    cudaEventRecord(ev0);
    unified_conv_tiled(um_in2, um_ker2, um_out2, width, height, ksize, block);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(um_prefetch_ms, ev0, ev1);

    unified_free(um_in2); unified_free(um_ker2); unified_free(um_out2);

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    free(h_input); free(h_kernel); free(h_output);
}
