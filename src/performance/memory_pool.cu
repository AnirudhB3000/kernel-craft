/**
 * \file memory_pool.cu
 * \brief Custom memory allocator with reusable buffers.
 *
 * This implementation provides a memory pool to reduce allocation overhead
 * for repeated GPU operations. It pre-allocates buffers and reuses them
 * across kernel calls, eliminating the overhead of cudaMalloc/cudaFree
 * for batch processing.
 *
 * \par Benefits
 * - Reduced allocation overhead (eliminates per-batch malloc/free)
 * - Better memory utilization
 * - Integrates with tiled convolution for zero-copy operations
 *
 * \par Usage
 * Create pool, allocate buffers, use for operations, destroy when done.
 */

#include <cuda_runtime.h>

#ifndef TILE_W
#define TILE_W 8
#endif
#ifndef TILE_H
#define TILE_H 8
#endif

extern "C" {
    void conv_tiled(const float* d_input, const float* d_kernel, float* d_output,
                    int width, int height, int ksize, dim3 block);
    void relu(float* d_data, int size);
}

#define MAX_POOLS 8
#define MAX_BUFFERS_PER_POOL 16

enum DeviceBufferState {
    BUFFER_FREE = 0,
    BUFFER_IN_USE = 1
};

struct DeviceBuffer {
    void* ptr;
    size_t size;
    int state;
    int id;
};

struct MemPool {
    DeviceBuffer buffers[MAX_BUFFERS_PER_POOL];
    int numBuffers;
    int width;
    int height;
    int ksize;
    size_t bufferSize;
    cudaStream_t stream;
    bool valid;
};

static MemPool g_pools[MAX_POOLS];
static int g_poolCount = 0;

extern "C" int mem_pool_reset() {
    g_poolCount = 0;
    for (int i = 0; i < MAX_POOLS; ++i) {
        g_pools[i].valid = false;
    }
    return 0;
}

extern "C" int mem_pool_create(MemPool** out_pool,
                             int width, int height, int ksize,
                             int numBuffers) {
    if (g_poolCount >= MAX_POOLS || numBuffers > MAX_BUFFERS_PER_POOL) {
        return -1;
    }

    MemPool* pool = &g_pools[g_poolCount++];
    pool->width = width;
    pool->height = height;
    pool->ksize = ksize;
    pool->numBuffers = numBuffers;
    pool->bufferSize = width * height * sizeof(float);
    pool->valid = true;
    cudaStreamCreate(&pool->stream);

    for (int i = 0; i < numBuffers; ++i) {
        cudaError_t err = cudaMalloc(&pool->buffers[i].ptr, pool->bufferSize);
        if (err != cudaSuccess) {
            for (int j = 0; j < i; ++j) {
                cudaFree(pool->buffers[j].ptr);
            }
            pool->valid = false;
            return -1;
        }
        pool->buffers[i].size = pool->bufferSize;
        pool->buffers[i].state = BUFFER_FREE;
        pool->buffers[i].id = i;
    }

    *out_pool = pool;
    return 0;
}

extern "C" int mem_pool_destroy(MemPool* pool) {
    if (!pool || !pool->valid) return -1;

    for (int i = 0; i < pool->numBuffers; ++i) {
        cudaFree(pool->buffers[i].ptr);
    }
    cudaStreamDestroy(pool->stream);
    pool->valid = false;
    return 0;
}

extern "C" void* mem_pool_alloc(MemPool* pool) {
    if (!pool || !pool->valid) return nullptr;

    for (int i = 0; i < pool->numBuffers; ++i) {
        if (pool->buffers[i].state == BUFFER_FREE) {
            pool->buffers[i].state = BUFFER_IN_USE;
            return pool->buffers[i].ptr;
        }
    }
    return nullptr;
}

extern "C" int mem_pool_free(MemPool* pool, void* ptr) {
    if (!pool || !pool->valid) return -1;

    for (int i = 0; i < pool->numBuffers; ++i) {
        if (pool->buffers[i].ptr == ptr) {
            pool->buffers[i].state = BUFFER_FREE;
            return 0;
        }
    }
    return -1;
}

typedef struct DeviceBuffer* DeviceBufferHandle;

extern "C" DeviceBufferHandle mem_pool_get_buffer(MemPool* pool, int idx) {
    if (!pool || !pool->valid || idx < 0 || idx >= pool->numBuffers) {
        return nullptr;
    }
    pool->buffers[idx].state = BUFFER_IN_USE;
    return &pool->buffers[idx];
}

extern "C" int mem_pool_release_buffer(MemPool* pool, DeviceBufferHandle handle) {
    if (!pool || !pool->valid || !handle) return -1;
    handle->state = BUFFER_FREE;
    return 0;
}

__global__ void conv_tiled_kernel(const float* __restrict__ input,
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

__global__ void relu_kernel(float* __restrict__ data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] > 0.0f ? data[idx] : 0.0f;
    }
}

extern "C" void conv_pipeline_with_pool(const float* d_input,
                                        const float* d_kernel,
                                        float* d_output,
                                        MemPool* pool,
                                        int width, int height, int ksize,
                                        dim3 block) {
    void* tempBuffer = mem_pool_alloc(pool);
    if (!tempBuffer) {
        return;
    }

    conv_tiled(d_input, d_kernel, (float*)tempBuffer, width, height, ksize, block);
    relu((float*)tempBuffer, width * height);
    
    cudaMemcpy(d_output, tempBuffer, width * height * sizeof(float),
              cudaMemcpyDeviceToDevice);
    
    mem_pool_free(pool, tempBuffer);
}

typedef struct {
    void* data;
    size_t size;
    int refcount;
} ManagedBuffer;

static ManagedBuffer g_managedBuffers[MAX_BUFFERS_PER_POOL];
static int g_managedCount = 0;

extern "C" void* managed_alloc(size_t size) {
    if (g_managedCount >= MAX_BUFFERS_PER_POOL) return nullptr;
    
    cudaMalloc(&g_managedBuffers[g_managedCount].data, size);
    g_managedBuffers[g_managedCount].size = size;
    g_managedBuffers[g_managedCount].refcount = 1;
    
    return g_managedBuffers[g_managedCount++].data;
}

extern "C" void managed_retain(void* ptr) {
    for (int i = 0; i < g_managedCount; ++i) {
        if (g_managedBuffers[i].data == ptr) {
            g_managedBuffers[i].refcount++;
            return;
        }
    }
}

extern "C" void managed_release(void* ptr) {
    for (int i = 0; i < g_managedCount; ++i) {
        if (g_managedBuffers[i].data == ptr) {
            g_managedBuffers[i].refcount--;
            if (g_managedBuffers[i].refcount == 0) {
                cudaFree(g_managedBuffers[i].data);
                g_managedBuffers[i].data = nullptr;
            }
            return;
        }
    }
}