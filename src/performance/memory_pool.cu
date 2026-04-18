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

/**
 * \brief Reset all memory pools.
 *
 * Clears the global pool count and marks all pools as invalid.
 * Should be called at program startup or when cleaning up.
 *
 * \return 0 on success.
 */
extern "C" int mem_pool_reset() {
    g_poolCount = 0;
    for (int i = 0; i < MAX_POOLS; ++i) {
        g_pools[i].valid = false;
    }
    return 0;
}

/**
 * \brief Create a memory pool with pre-allocated buffers.
 *
 * Allocates buffers of the specified size and number for reuse
 * across kernel calls. Reduces allocation overhead for batch processing.
 *
 * \param[out] out_pool Pointer to receive the created pool handle.
 * \param[in] width Image width for buffer size calculation.
 * \param[in] height Image height for buffer size calculation.
 * \param[in] ksize Kernel size.
 * \param[in] numBuffers Number of buffers to pre-allocate.
 * \return 0 on success, -1 on failure.
 */
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

/**
 * \brief Destroy a memory pool and free all buffers.
 *
 * Releases all pre-allocated device buffers and destroys the stream.
 *
 * \param pool Pool handle to destroy.
 * \return 0 on success, -1 if pool is invalid.
 */
extern "C" int mem_pool_destroy(MemPool* pool) {
    if (!pool || !pool->valid) return -1;

    for (int i = 0; i < pool->numBuffers; ++i) {
        cudaFree(pool->buffers[i].ptr);
    }
    cudaStreamDestroy(pool->stream);
    pool->valid = false;
    return 0;
}

/**
 * \brief Allocate a buffer from the pool.
 *
 * Returns a pointer to an available buffer or nullptr if none free.
 *
 * \param pool Pool to allocate from.
 * \return Pointer to buffer on success, nullptr on failure.
 */
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

/**
 * \brief Free a buffer back to the pool.
 *
 * Marks the buffer as available for reuse.
 *
 * \param pool Pool that owns the buffer.
 * \param ptr Pointer to buffer to free.
 * \return 0 on success, -1 if not found.
 */
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

/**
 * \brief Get a buffer by index.
 *
 * Returns a handle to the buffer at the specified index.
 *
 * \param pool Pool to query.
 * \param idx Buffer index.
 * \return Buffer handle on success, nullptr on failure.
 */
extern "C" DeviceBufferHandle mem_pool_get_buffer(MemPool* pool, int idx) {
    if (!pool || !pool->valid || idx < 0 || idx >= pool->numBuffers) {
        return nullptr;
    }
    pool->buffers[idx].state = BUFFER_IN_USE;
    return &pool->buffers[idx];
}

/**
 * \brief Release a buffer handle.
 *
 * Marks the buffer as available for reuse.
 *
 * \param pool Pool that owns the buffer.
 * \param handle Buffer handle to release.
 * \return 0 on success, -1 on failure.
 */
extern "C" int mem_pool_release_buffer(MemPool* pool, DeviceBufferHandle handle) {
    if (!pool || !pool->valid || !handle) return -1;
    handle->state = BUFFER_FREE;
    return 0;
}

/**
 * \brief Tiled convolution kernel using memory pool buffers.
 *
 * \param input  Input image.
 * \param kernel Convolution kernel.
 * \param output Output buffer.
 * \param width  Image width.
 * \param height Image height.
 * \param ksize  Kernel size.
 */
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

/**
 * \brief ReLU activation kernel.
 *
 * \param data Input/output activation data.
 * \param size Number of elements.
 */
__global__ void relu_kernel(float* __restrict__ data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] > 0.0f ? data[idx] : 0.0f;
    }
}

/**
 * \brief Execute convolution pipeline using memory pool buffers.
 *
 * Allocates temp buffer from pool, runs conv + relu, copies result,
 * returns buffer to pool.
 *
 * \param d_input  Input image device pointer.
 * \param d_kernel Kernel device pointer.
 * \param d_output Output device pointer.
 * \param pool     Memory pool to use.
 * \param width    Image width.
 * \param height   Image height.
 * \param ksize    Kernel size.
 * \param block    Block dimensions.
 */
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

/**
 * \brief Managed buffer with reference counting.
 *
 * Automatically frees memory when reference count drops to zero.
 */
typedef struct {
    void* data;
    size_t size;
    int refcount;
} ManagedBuffer;

static ManagedBuffer g_managedBuffers[MAX_BUFFERS_PER_POOL];
static int g_managedCount = 0;

/**
 * \brief Allocate a managed buffer.
 *
 * \param size Size in bytes to allocate.
 * \return Device pointer on success, nullptr on failure.
 */
extern "C" void* managed_alloc(size_t size) {
    if (g_managedCount >= MAX_BUFFERS_PER_POOL) return nullptr;
    
    cudaMalloc(&g_managedBuffers[g_managedCount].data, size);
    g_managedBuffers[g_managedCount].size = size;
    g_managedBuffers[g_managedCount].refcount = 1;
    
    return g_managedBuffers[g_managedCount++].data;
}

/**
 * \brief Retain a managed buffer (increment refcount).
 *
 * \param ptr Device pointer to retain.
 */
extern "C" void managed_retain(void* ptr) {
    for (int i = 0; i < g_managedCount; ++i) {
        if (g_managedBuffers[i].data == ptr) {
            g_managedBuffers[i].refcount++;
            return;
        }
    }
}

/**
 * \brief Release a managed buffer (decrement refcount).
 *
 * Frees memory when refcount reaches zero.
 *
 * \param ptr Device pointer to release.
 */
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