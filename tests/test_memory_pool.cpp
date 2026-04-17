/**
 * \file test_memory_pool.cpp
 * \brief Unit test for memory pool functionality.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

struct MemPool;

extern "C" {
    int mem_pool_create(MemPool** out_pool,
                         int width, int height, int ksize,
                         int numBuffers);
    int mem_pool_destroy(MemPool* pool);
    void* mem_pool_alloc(MemPool* pool);
    int mem_pool_free(MemPool* pool, void* ptr);
    int mem_pool_reset();
}

void conv_naive_cpu(const float* input, const float* kernel, float* output,
                    int width, int height, int ksize) {
    int kHalf = ksize / 2;
    for (int oy = 0; oy < height; ++oy) {
        for (int ox = 0; ox < width; ++ox) {
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
    }
}

int test_pool_create_destroy() {
    MemPool* pool = nullptr;
    int result = mem_pool_create(&pool, 32, 32, 3, 4);
    if (result != 0) {
        fprintf(stderr, "FAIL: mem_pool_create returned %d\n", result);
        return 1;
    }
    if (pool == nullptr) {
        fprintf(stderr, "FAIL: pool is null after create\n");
        return 1;
    }
    result = mem_pool_destroy(pool);
    if (result != 0) {
        fprintf(stderr, "FAIL: mem_pool_destroy returned %d\n", result);
        return 1;
    }
    printf("test_pool_create_destroy passed.\n");
    return 0;
}

int test_pool_alloc_free() {
    MemPool* pool = nullptr;
    mem_pool_create(&pool, 32, 32, 3, 4);
    
    void* ptr1 = mem_pool_alloc(pool);
    if (ptr1 == nullptr) {
        fprintf(stderr, "FAIL: mem_pool_alloc returned null\n");
        mem_pool_destroy(pool);
        return 1;
    }
    
    int result = mem_pool_free(pool, ptr1);
    if (result != 0) {
        fprintf(stderr, "FAIL: mem_pool_free returned %d\n", result);
        mem_pool_destroy(pool);
        return 1;
    }
    
    void* ptr2 = mem_pool_alloc(pool);
    if (ptr2 == nullptr) {
        fprintf(stderr, "FAIL: second alloc returned null\n");
        mem_pool_destroy(pool);
        return 1;
    }
    
    mem_pool_free(pool, ptr2);
    mem_pool_destroy(pool);
    
    printf("test_pool_alloc_free passed.\n");
    return 0;
}

int test_pool_exhaustion() {
    MemPool* pool = nullptr;
    mem_pool_create(&pool, 32, 32, 3, 2);
    
    void* ptr1 = mem_pool_alloc(pool);
    void* ptr2 = mem_pool_alloc(pool);
    void* ptr3 = mem_pool_alloc(pool);
    
    if (ptr1 == nullptr || ptr2 == nullptr) {
        fprintf(stderr, "FAIL: should have allocated 2 buffers\n");
        mem_pool_destroy(pool);
        return 1;
    }
    if (ptr3 != nullptr) {
        fprintf(stderr, "FAIL: pool exhausted, should return null\n");
        mem_pool_destroy(pool);
        return 1;
    }
    
    mem_pool_free(pool, ptr1);
    void* ptr4 = mem_pool_alloc(pool);
    if (ptr4 == nullptr) {
        fprintf(stderr, "FAIL: should be able to realloc after free\n");
        mem_pool_destroy(pool);
        return 1;
    }
    
    mem_pool_free(pool, ptr2);
    mem_pool_free(pool, ptr4);
    mem_pool_destroy(pool);
    
    printf("test_pool_exhaustion passed.\n");
    return 0;
}

int test_pool_invalid_inputs() {
    int result;
    
    result = mem_pool_destroy(nullptr);
    if (result != -1) {
        fprintf(stderr, "FAIL: destroy null pool should return -1\n");
        return 1;
    }
    
    void* ptr = mem_pool_alloc(nullptr);
    if (ptr != nullptr) {
        fprintf(stderr, "FAIL: alloc from null pool should return null\n");
        return 1;
    }
    
    MemPool* pool = nullptr;
    mem_pool_create(&pool, 32, 32, 3, 1);
    result = mem_pool_free(pool, nullptr);
    if (result != -1) {
        fprintf(stderr, "FAIL: free null ptr should return -1\n");
        mem_pool_destroy(pool);
        return 1;
    }
    mem_pool_destroy(pool);
    
    printf("test_pool_invalid_inputs passed.\n");
    return 0;
}

int test_pool_too_many() {
    mem_pool_reset();
    MemPool* pool1 = nullptr;
    MemPool* pool2 = nullptr;
    MemPool* pool3 = nullptr;
    MemPool* pool4 = nullptr;
    MemPool* pool5 = nullptr;
    MemPool* pool6 = nullptr;
    MemPool* pool7 = nullptr;
    MemPool* pool8 = nullptr;
    MemPool* pool9 = nullptr;
    
    int r1 = mem_pool_create(&pool1, 32, 32, 3, 4);
    int r2 = mem_pool_create(&pool2, 32, 32, 3, 4);
    int r3 = mem_pool_create(&pool3, 32, 32, 3, 4);
    int r4 = mem_pool_create(&pool4, 32, 32, 3, 4);
    int r5 = mem_pool_create(&pool5, 32, 32, 3, 4);
    int r6 = mem_pool_create(&pool6, 32, 32, 3, 4);
    int r7 = mem_pool_create(&pool7, 32, 32, 3, 4);
    int r8 = mem_pool_create(&pool8, 32, 32, 3, 4);
    int r9 = mem_pool_create(&pool9, 32, 32, 3, 4);
    
    printf("DEBUG: r1=%d r2=%d r3=%d r4=%d r5=%d r6=%d r7=%d r8=%d r9=%d\n",
          r1, r2, r3, r4, r5, r6, r7, r8, r9);
    
    if (pool9 != nullptr) {
        fprintf(stderr, "FAIL: should not create more than MAX_POOLS (8)\n");
        if (pool1) mem_pool_destroy(pool1);
        if (pool2) mem_pool_destroy(pool2);
        if (pool3) mem_pool_destroy(pool3);
        if (pool4) mem_pool_destroy(pool4);
        if (pool5) mem_pool_destroy(pool5);
        if (pool6) mem_pool_destroy(pool6);
        if (pool7) mem_pool_destroy(pool7);
        if (pool8) mem_pool_destroy(pool8);
        return 1;
    }
    
    if (pool1) mem_pool_destroy(pool1);
    if (pool2) mem_pool_destroy(pool2);
    if (pool3) mem_pool_destroy(pool3);
    if (pool4) mem_pool_destroy(pool4);
    if (pool5) mem_pool_destroy(pool5);
    if (pool6) mem_pool_destroy(pool6);
    if (pool7) mem_pool_destroy(pool7);
    if (pool8) mem_pool_destroy(pool8);
    
    printf("test_pool_too_many passed.\n");
    return 0;
}

int main() {
    int failures = 0;
    
    failures += test_pool_create_destroy();
    failures += test_pool_alloc_free();
    failures += test_pool_exhaustion();
    failures += test_pool_invalid_inputs();
    failures += test_pool_too_many();
    
    if (failures == 0) {
        printf("\nAll memory pool tests passed.\n");
        return EXIT_SUCCESS;
    } else {
        printf("\n%d memory pool test(s) failed.\n", failures);
        return EXIT_FAILURE;
    }
}