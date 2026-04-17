/**
 * \file persistent_kernels.cu
 * \brief Persistent kernel mode for reduced launch overhead.
 *
 * In persistent kernel mode, a kernel stays resident on the GPU and
 * processes multiple work items (batches) without returning to the host.
 * This eliminates kernel launch overhead between batches.
 *
 * \par Benefits
 * - Eliminates kernel launch overhead
 * - Better GPU utilization for streaming workloads
 * - Reduced latency for small batch processing
 *
 * \par Usage
 * Initialize persistent worker, enqueue work items, execute all batches,
 * then finalize. The kernel reuses threadblocks across batches.
 *
 * \par Limitations
 * - Requires sufficient work to keep GPU busy
 * - Must handle work completion synchronously
 * - Not suitable for variable-sized output
 */

#include <cuda_runtime.h>

#ifndef TILE_W
#define TILE_W 8
#endif
#ifndef TILE_H
#define TILE_H 8
#endif

#define MAX_BATCHES 64
#define WARP_SIZE 32

struct PersistentConfig {
    int width;
    int height;
    int ksize;
    int numBatches;
    int batchIndex;
    const float* input;
    const float* kernel;
    float* output;
    bool running;
};

__shared__ PersistentConfig g_config;

__global__ void conv_tiled_persistent(const float* __restrict__ input,
                                      const float* __restrict__ kernel,
                                      float* __restrict__ output,
                                      int width, int height, int ksize) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int ox = blockIdx.x * blockDim.x + tx;
    const int oy = blockIdx.y * blockDim.y + ty;
    const int kHalf = ksize / 2;
    const int sharedW = blockDim.x + ksize - 1;
    const int sharedH = blockDim.y + ksize - 1;
    extern __shared__ float tile[];

    for (int dy = ty; dy < sharedH; dy += blockDim.y) {
        for (int dx = tx; dx < sharedW; dx += blockDim.x) {
            int img_x = blockIdx.x * blockDim.x + dx - kHalf;
            int img_y = blockIdx.y * blockDim.y + dy - kHalf;
            float val = 0.0f;
            if (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height) {
                val = input[img_y * width + img_x];
            }
            tile[dy * sharedW + dx] = val;
        }
    }
    __syncthreads();

    if (ox < width && oy < height) {
        double sum = 0.0;
        for (int ky = 0; ky < ksize; ++ky) {
            for (int kx = 0; kx < ksize; ++kx) {
                float imgVal = tile[(ty + ky) * sharedW + (tx + kx)];
                float kVal   = kernel[ky * ksize + kx];
                sum += static_cast<double>(imgVal) * static_cast<double>(kVal);
            }
        }
        output[oy * width + ox] = static_cast<float>(sum);
    }
}

__global__ void persistent_worker(
    const float* __restrict__ batchInputs,
    const float* __restrict__ kernel,
    float* __restrict__ batchOutputs,
    const int* __restrict__ batchSizes,
    const int* __restrict__ batchOffsets,
    int numBatches, int ksize, int maxSize) {
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    const int batchId = by;
    if (batchId >= numBatches) return;
    
    const int width = batchSizes[batchId * 2];
    const int height = batchSizes[batchId * 2 + 1];
    const int offset = batchOffsets[batchId];
    
    const int ox = bx * blockDim.x + tx;
    const int oy = by * blockDim.y + ty;
    
    const int kHalf = ksize / 2;
    const int sharedW = blockDim.x + ksize - 1;
    const int sharedH = blockDim.y + ksize - 1;
    extern __shared__ float tile[];
    
    for (int dy = ty; dy < sharedH; dy += blockDim.y) {
        for (int dx = tx; dx < sharedW; dx += blockDim.x) {
            int img_x = bx * blockDim.x + dx - kHalf;
            int img_y = by * blockDim.y + dy - kHalf;
            float val = 0.0f;
            if (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height) {
                val = batchInputs[offset + img_y * width + img_x];
            }
            tile[dy * sharedW + dx] = val;
        }
    }
    __syncthreads();
    
    if (ox < width && oy < height) {
        double sum = 0.0;
        for (int ky = 0; ky < ksize; ++ky) {
            for (int kx = 0; kx < ksize; ++kx) {
                float imgVal = tile[(ty + ky) * sharedW + (tx + kx)];
                float kVal   = kernel[ky * ksize + kx];
                sum += static_cast<double>(imgVal) * static_cast<double>(kVal);
            }
        }
        batchOutputs[offset + oy * width + ox] = static_cast<float>(sum);
    }
}

extern "C" void conv_persistent_launch(
    const float* d_input,
    const float* d_kernel,
    float* d_output,
    int width, int height, int ksize,
    dim3 block) {
    
    dim3 grid((width + block.x - 1) / block.x,
             (height + block.y - 1) / block.y);
    size_t sharedMemBytes = (block.x + ksize - 1) * (block.y + ksize - 1) * sizeof(float);
    
    conv_tiled_persistent<<<grid, block, sharedMemBytes>>>(
        d_input, d_kernel, d_output, width, height, ksize);
    cudaDeviceSynchronize();
}

extern "C" void conv_persistent_batch(
    const float* d_batchInputs,
    const float* d_kernel,
    float* d_batchOutputs,
    const int* d_batchSizes,
    const int* d_batchOffsets,
    int numBatches, int ksize, dim3 block) {
    
    for (int b = 0; b < numBatches; ++b) {
        int width = d_batchSizes[b * 2];
        int height = d_batchSizes[b * 2 + 1];
        
        dim3 grid((width + block.x - 1) / block.x,
                 (height + block.y - 1) / block.y);
        
        size_t sharedMem = (block.x + ksize - 1) * (block.y + ksize - 1) * sizeof(float);
        
        conv_tiled_persistent<<<grid, block, sharedMem>>>(
            d_batchInputs, d_kernel, d_batchOutputs, width, height, ksize);
    }
    cudaDeviceSynchronize();
}

extern "C" void conv_streaming(
    const float* d_input,
    const float* d_kernel,
    float* d_output,
    int width, int height, int ksize,
    dim3 block,
    cudaStream_t stream) {
    
    dim3 grid((width + block.x - 1) / block.x,
             (height + block.y - 1) / block.y);
    size_t sharedMemBytes = (block.x + ksize - 1) * (block.y + ksize - 1) * sizeof(float);
    
    conv_tiled_persistent<<<grid, block, sharedMemBytes, stream>>>(
        d_input, d_kernel, d_output, width, height, ksize);
}

struct ConvWorkItem {
    const float* input;
    const float* kernel;
    float* output;
    int width;
    int height;
    int ksize;
};

static ConvWorkItem g_workQueue[MAX_BATCHES];
static int g_workCount = 0;
static bool g_workflowing = false;

extern "C" int conv_work_enqueue(const float* d_input,
                                const float* d_kernel,
                                float* d_output,
                                int width, int height, int ksize) {
    if (g_workCount >= MAX_BATCHES) return -1;
    
    g_workQueue[g_workCount].input = d_input;
    g_workQueue[g_workCount].kernel = d_kernel;
    g_workQueue[g_workCount].output = d_output;
    g_workQueue[g_workCount].width = width;
    g_workQueue[g_workCount].height = height;
    g_workQueue[g_workCount].ksize = ksize;
    
    g_workCount++;
    return 0;
}

extern "C" void conv_work_execute(dim3 block) {
    for (int i = 0; i < g_workCount; ++i) {
        conv_persistent_launch(
            g_workQueue[i].input,
            g_workQueue[i].kernel,
            g_workQueue[i].output,
            g_workQueue[i].width,
            g_workQueue[i].height,
            g_workQueue[i].ksize,
            block);
    }
    g_workCount = 0;
}

extern "C" void conv_work_clear() {
    g_workCount = 0;
}