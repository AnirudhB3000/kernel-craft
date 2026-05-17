/**
 * \file vllm_plugin_wrapper.cpp
 * \brief vLLM custom operator integration for kernel-craft transformer kernels.
 *
 * Exposes PagedAttention and FlashAttention as vLLM-compatible PyTorch custom ops.
 * vLLM uses `torch.ops.vllm.*` for custom CUDA kernels. This file follows that
 * pattern using the PyTorch custom-op registration API.
 *
 * \par Integration path
 * 1. Build this file as a shared library: `libvllm_kernel_craft.so`
 * 2. In Python: `torch.ops.load_library("libvllm_kernel_craft.so")`
 * 3. Call via: `torch.ops.kernel_craft.paged_attention(Q, block_table, ...)`
 *
 * \par Compilation note
 * Requires PyTorch C++ API (libtorch). Build with:
 *   nvcc -shared -fPIC vllm_plugin_wrapper.cpp \
 *     -I$(TORCH_INCLUDE) -L$(TORCH_LIB) -ltorch -ltorch_cuda \
 *     -o libvllm_kernel_craft.so
 *
 * \par Without PyTorch
 * When HAVE_TORCH is not defined, this file compiles to a plain C stub
 * that documents the expected interface for vLLM integration.
 */

#ifdef HAVE_TORCH
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// Forward declarations of kernel launchers
extern "C" void launch_flash_attention(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int H_kv, int N, int d,
    bool causal, cudaStream_t stream);

extern "C" void launch_paged_attention(
    const float* Q, const int* block_table,
    const float* K_pool, const float* V_pool,
    float* O, const int* seq_lens,
    int B, int H, int H_kv, int d,
    int block_size, int max_blocks, cudaStream_t stream);

/**
 * \brief vLLM-compatible FlashAttention wrapper.
 *
 * \param[in] Q      CUDA float tensor [B, H, N, d]
 * \param[in] K      CUDA float tensor [B, H_kv, N, d]
 * \param[in] V      CUDA float tensor [B, H_kv, N, d]
 * \param[in] H_kv   Number of KV heads
 * \param[in] causal Apply causal mask
 * \return CUDA float tensor [B, H, N, d]
 */
at::Tensor vllm_flash_attention(
    at::Tensor Q, at::Tensor K, at::Tensor V, int H_kv, bool causal)
{
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(),
                "Q, K, V must be CUDA tensors");
    TORCH_CHECK(Q.scalar_type() == at::kFloat, "Only float32 supported");

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int N = (int)Q.size(2);
    int d = (int)Q.size(3);

    auto O  = at::zeros({B, H, N, d}, Q.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    launch_flash_attention(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(),
        B, H, H_kv, N, d, causal, stream);
    return O;
}

/**
 * \brief vLLM-compatible PagedAttention wrapper.
 *
 * \param[in] Q           CUDA float [B, H, d]
 * \param[in] block_table CUDA int   [B, max_blocks]
 * \param[in] K_pool      CUDA float [num_phys, block_size, H_kv, d]
 * \param[in] V_pool      CUDA float [num_phys, block_size, H_kv, d]
 * \param[in] seq_lens    CUDA int   [B]
 * \param[in] H_kv        Number of KV heads
 * \param[in] block_size  Tokens per physical page
 * \return CUDA float [B, H, d]
 */
at::Tensor vllm_paged_attention(
    at::Tensor Q, at::Tensor block_table,
    at::Tensor K_pool, at::Tensor V_pool,
    at::Tensor seq_lens, int H_kv, int block_size)
{
    TORCH_CHECK(Q.is_cuda(), "All tensors must be CUDA tensors");

    int B          = (int)Q.size(0);
    int H          = (int)Q.size(1);
    int d          = (int)Q.size(2);
    int max_blocks = (int)block_table.size(1);

    auto O      = at::zeros({B, H, d}, Q.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    launch_paged_attention(
        Q.data_ptr<float>(),
        block_table.data_ptr<int>(),
        K_pool.data_ptr<float>(),
        V_pool.data_ptr<float>(),
        O.data_ptr<float>(),
        seq_lens.data_ptr<int>(),
        B, H, H_kv, d, block_size, max_blocks, stream);
    return O;
}

// Register as torch custom ops under namespace "kernel_craft"
TORCH_LIBRARY(kernel_craft, m) {
    m.def("flash_attention(Tensor Q, Tensor K, Tensor V, int H_kv, bool causal) -> Tensor");
    m.def("paged_attention(Tensor Q, Tensor block_table, Tensor K_pool, "
          "Tensor V_pool, Tensor seq_lens, int H_kv, int block_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(kernel_craft, CUDA, m) {
    m.impl("flash_attention", &vllm_flash_attention);
    m.impl("paged_attention", &vllm_paged_attention);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention", &vllm_flash_attention, "FlashAttention (CUDA)");
    m.def("paged_attention",  &vllm_paged_attention,  "PagedAttention decode (CUDA)");
}

#else // !HAVE_TORCH

// Stub documenting the expected vLLM custom op interface
#include <cstdio>

extern "C" void vllm_plugin_info() {
    printf("kernel-craft vLLM plugin: build with -DHAVE_TORCH to enable PyTorch integration.\n");
    printf("Expected usage:\n");
    printf("  import torch\n");
    printf("  torch.ops.load_library('libvllm_kernel_craft.so')\n");
    printf("  O = torch.ops.kernel_craft.paged_attention(Q, block_table, ...)\n");
}

#endif // HAVE_TORCH
