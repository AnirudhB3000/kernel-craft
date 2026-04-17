/**
 * \file example_memory_pool.cpp
 * \brief Example: Memory Pool for batch processing.
 *
 * This example demonstrates using a memory pool to reduce allocation
 * overhead when processing multiple batches through convolution.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
    void conv_tiled(const float* d_input, const float* d_kernel, float* d_output,
                    int width, int height, int ksize, dim3 block);
    void relu(float* d_data, int size);
}

struct MemPool;
extern "C" int mem_pool_create(MemPool** out_pool, int width, int height, int ksize, int numBuffers);
extern "C" int mem_pool_destroy(MemPool* pool);
extern "C" void* mem_pool_alloc(MemPool* pool);
extern "C" int mem_pool_free(MemPool* pool, void* ptr);

#define WIDTH 256
#define HEIGHT 256
#define KSIZE 3
#define NUM_BATCHES 16

int main() {
    printf("=== Memory Pool Example ===\n\n");

    float *h_input, *h_kernel, *h_output;
    float *d_input, *d_kernel, *d_output;

    h_input = (float*)malloc(WIDTH * HEIGHT * sizeof(float));
    h_kernel = (float*)malloc(KSIZE * KSIZE * sizeof(float));
    h_output = (float*)malloc(WIDTH * HEIGHT * sizeof(float));

    for (int i = 0; i < WIDTH * HEIGHT; ++i) h_input[i] = (float)(i % 256) / 255.0f;
    for (int i = 0; i < KSIZE * KSIZE; ++i) h_kernel[i] = 1.0f / 9.0f;

    cudaMalloc(&d_input, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc(&d_kernel, KSIZE * KSIZE * sizeof(float));
    cudaMalloc(&d_output, WIDTH * HEIGHT * sizeof(float));

    cudaMemcpy(d_input, h_input, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, KSIZE * KSIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(8, 8, 1);

    // Create memory pool
    MemPool* pool = nullptr;
    int ret = mem_pool_create(&pool, WIDTH, HEIGHT, KSIZE, NUM_BATCHES);
    if (ret != 0) {
        printf("Failed to create memory pool\n");
        return 1;
    }
    printf("Created memory pool with %d buffers\n", NUM_BATCHES);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Benchmark with memory pool
    printf("\nProcessing %d batches with memory pool...\n", NUM_BATCHES);
    cudaEventRecord(start);
    for (int i = 0; i < NUM_BATCHES; ++i) {
        void* temp = mem_pool_alloc(pool);
        if (temp) {
            conv_tiled(d_input, d_kernel, (float*)temp, WIDTH, HEIGHT, KSIZE, block);
            relu((float*)temp, WIDTH * HEIGHT);
            mem_pool_free(pool, temp);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float pool_time;
    cudaEventElapsedTime(&pool_time, start, stop);
    printf("  Total: %.2f ms (avg: %.2f ms per batch)\n", pool_time, pool_time / (float)NUM_BATCHES);

    // Cleanup
    mem_pool_destroy(pool);

    // Benchmark without pool (for comparison)
    printf("\nProcessing %d batches without pool...\n", NUM_BATCHES);
    cudaEventRecord(start);
    for (int i = 0; i < NUM_BATCHES; ++i) {
        float* temp;
        cudaMalloc(&temp, WIDTH * HEIGHT * sizeof(float));
        conv_tiled(d_input, d_kernel, temp, WIDTH, HEIGHT, KSIZE, block);
        relu(temp, WIDTH * HEIGHT);
        cudaFree(temp);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float nopool_time;
    cudaEventElapsedTime(&nopool_time, start, stop);
    printf("  Total: %.2f ms (avg: %.2f ms per batch)\n", nopool_time, nopool_time / (float)NUM_BATCHES);

    printf("\nMemory pool overhead reduction: %.1f%%\n", 
           (1.0f - pool_time / nopool_time) * 100.0f);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(h_input);
    free(h_kernel);
    free(h_output);

    printf("\n=== Example Complete ===\n");
    return 0;
}