/**
 * \file cnn/cuda_graphs.cpp
 * \brief CUDA Graphs for zero-overhead multi-kernel pipeline replay.
 *
 * \par Purpose
 * Demonstrates how to capture a multi-kernel pipeline (conv → ReLU → bias)
 * as a CUDA Graph and replay it with a single API call, eliminating the
 * per-kernel CPU launch overhead that dominates latency at small batch sizes.
 *
 * \par Why CUDA Graphs matter for inference engines
 * Each cudaLaunchKernel call on the CPU side takes ~5–15 µs.  A typical
 * ResNet-50 has ~50 conv layers; running three kernels per layer (conv, BN,
 * ReLU) means ~150 launches × 10 µs ≈ 1.5 ms of pure launch overhead per
 * image — comparable to the actual GPU compute time on modern hardware.
 *
 * CUDA Graphs solve this with a two-phase approach:
 * \par
 * - **Capture phase** (once at model-load time): run the pipeline inside
 *   cudaStreamBeginCapture / cudaStreamEndCapture.  CUDA records the kernel
 *   DAG (nodes, edges, parameters) into a cudaGraph_t object.
 * - **Replay phase** (per request): call cudaGraphLaunch — one CPU call
 *   dispatches all captured kernels with ~5 µs total overhead regardless of
 *   how many kernels are in the graph.
 *
 * \par Integration in vLLM / TensorRT-LLM
 * Both vLLM and TensorRT-LLM use CUDA Graphs to accelerate decode steps where
 * the same attention + FFN kernels execute for every generated token.  The
 * graph is captured at a fixed batch size during engine warm-up and replayed
 * until the sequence is complete.  Prefill (variable length) still uses eager
 * launches.
 *
 * \par Constraints
 * - Captured pointers must remain valid across replays (no cudaFree/cudaMalloc).
 * - Changing kernel arguments (e.g. batch size) requires re-capture or graph
 *   update via cudaGraphExecKernelNodeSetParams.
 * - Host callbacks and synchronisation primitives cannot be captured.
 *
 * \par Expected output on RTX 4070 (1024×1024, 10 iterations)
 * \code
 *   Separate launches : ~30 ms total  (3 ms/iter launch overhead visible)
 *   CUDA Graph replay : ~25 ms total  (~5 µs launch overhead per iter)
 * \endcode
 *
 * \par Compile and run
 * \code
 * cmake --build build --target example_cuda_graphs
 * ./build/examples/example_cuda_graphs
 * \endcode
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * \brief Tiled 2-D convolution (src/kernels/core/conv_tiled.cu).
 * \param[in]  d_input   float32 device array [height × width].
 * \param[in]  d_kernel  float32 device array [ksize × ksize].
 * \param[out] d_output  float32 device array [height × width].
 */
extern "C" void launch_conv_tiled(const float* d_input, const float* d_kernel,
                                  float* d_output,
                                  int width, int height, int ksize,
                                  dim3 block);

/**
 * \brief In-place ReLU activation (src/pipelines/pipeline_separate.cu).
 * \param[in,out] d_data  float32 device array [size].
 * \param[in]     size    Number of elements.
 */
extern "C" void relu(float* d_data, int size);

/**
 * \brief In-place scalar bias addition (src/pipelines/pipeline_separate.cu).
 * \param[in,out] d_data  float32 device array [size].
 * \param[in]     bias    Scalar bias to add to every element.
 * \param[in]     size    Number of elements.
 */
extern "C" void add_bias(float* d_data, float bias, int size);

#define W 1024
#define H 1024
#define K 3
#define ITERS 10

/** \brief Run conv → ReLU → bias as three separate kernel launches. */
static float benchmark_separate(const float* d_in, const float* d_ker,
                                 float* d_out, dim3 block) {
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    for (int i = 0; i < ITERS; ++i) {
        launch_conv_tiled(d_in, d_ker, d_out, W, H, K, block);
        relu(d_out, W * H);
        add_bias(d_out, 0.1f, W * H);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return ms;
}

/**
 * \brief Capture conv → ReLU → bias into a CUDA Graph, then replay it.
 *
 * \par Capture sequence
 * 1. cudaStreamBeginCapture — puts the stream into capture mode; subsequent
 *    kernel launches are recorded, not executed immediately.
 * 2. Launch kernels as normal (same calls as separate mode).
 * 3. cudaStreamEndCapture — finalises the graph.
 * 4. cudaGraphInstantiate — compiles graph into an executable cudaGraphExec_t.
 * 5. cudaGraphLaunch (ITERS times) — single CPU call per iteration.
 */
static float benchmark_graph(const float* d_in, const float* d_ker,
                              float* d_out, dim3 block) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* --- Capture phase (runs once at warm-up time in a real engine) --- */
    cudaGraph_t graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    launch_conv_tiled(d_in, d_ker, d_out, W, H, K, block);
    relu(d_out, W * H);
    add_bias(d_out, 0.1f, W * H);

    cudaStreamEndCapture(stream, &graph);

    cudaGraphExec_t exec;
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

    /* --- Replay phase (one call per inference request) --- */
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0, stream);
    for (int i = 0; i < ITERS; ++i)
        cudaGraphLaunch(exec, stream);
    cudaEventRecord(t1, stream);
    cudaStreamSynchronize(stream);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return ms;
}

int main() {
    printf("=== CUDA Graphs Example (%dx%d, %d iterations) ===\n\n", W, H, ITERS);

    const size_t img_bytes = (size_t)W * H * sizeof(float);
    const size_t ker_bytes = (size_t)K * K * sizeof(float);

    float *h_in = (float*)malloc(img_bytes);
    float *h_ker = (float*)malloc(ker_bytes);
    for (int i = 0; i < W * H; ++i) h_in[i]  = (float)(i % 256) / 255.f;
    for (int i = 0; i < K * K; ++i) h_ker[i] = 1.f / 9.f;

    float *d_in, *d_ker, *d_out;
    cudaMalloc(&d_in,  img_bytes);
    cudaMalloc(&d_ker, ker_bytes);
    cudaMalloc(&d_out, img_bytes);
    cudaMemcpy(d_in,  h_in,  img_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ker, h_ker, ker_bytes, cudaMemcpyHostToDevice);

    dim3 block(8, 8, 1);

    /* Warm up GPU clocks before timing. */
    launch_conv_tiled(d_in, d_ker, d_out, W, H, K, block);
    cudaDeviceSynchronize();

    float t_sep   = benchmark_separate(d_in, d_ker, d_out, block);
    float t_graph = benchmark_graph(d_in, d_ker, d_out, block);

    printf("Separate launches : %.2f ms total  (%.2f ms/iter)\n",
           t_sep, t_sep / ITERS);
    printf("CUDA Graph replay : %.2f ms total  (%.2f ms/iter)\n",
           t_graph, t_graph / ITERS);
    printf("Launch overhead saved : ~%.1f µs/iter (3 kernels × ~5 µs each)\n",
           (t_sep - t_graph) / ITERS * 1000.f);

    printf("\nKey insight: graph replay amortises CPU→GPU dispatch latency;\n");
    printf("compute time is identical — only scheduling overhead differs.\n");
    printf("See src/performance/cuda_graphs.cu for the full implementation.\n");

    cudaFree(d_in); cudaFree(d_ker); cudaFree(d_out);
    free(h_in); free(h_ker);
    return 0;
}
