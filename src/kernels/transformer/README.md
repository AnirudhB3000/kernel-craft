# transformer/ — Transformer / LLM Inference Kernels

CUDA kernels targeting transformer-based language model inference. All kernels follow the same `extern "C"` launcher pattern as the CNN inference kernels.

## Files

| File | Purpose |
|------|---------|
| `flash_attention.cu` | Tiled multi-head attention (MHA/GQA/MQA) with online softmax and causal masking |
| `paged_attention.cu` | Attention over a non-contiguous paged KV-cache via block table lookup (vLLM-style) |
| `quant_int4.cu` | INT4 weight dequantization + GEMV (GPTQ/AWQ-style: 2×INT4 packed per byte, per-group FP16 scales) |
| `fp8_quant.cu` | FP8 E4M3 activation quantization — per-token and per-channel scaling, SmoothQuant-compatible |
| `speculative_decoding.cu` | Draft token verification via rejection sampling; outputs accepted token mask |
| `tensor_parallel.cu` | Ring all-reduce and all-gather primitives; NCCL wrapper with single-GPU simulation fallback |

## Hardware Requirements

| Feature | Minimum SM | Notes |
|---------|-----------|-------|
| FlashAttention | SM 6.0 | Shared memory tiling, online softmax |
| PagedAttention | SM 6.0 | Block table pointer arithmetic |
| INT4 GEMV | SM 8.0 | Tensor Core path for Ampere+; falls back to scalar on older hardware |
| FP8 E4M3 | SM 8.9 | `__nv_fp8_e4m3` type requires CUDA ≥ 11.8 and Ada/Hopper |
| Tensor Parallelism | SM 6.0 | Multi-GPU path requires peer-to-peer; single-GPU simulation always available |

## Launcher Signatures

```cpp
// flash_attention.cu
void launch_flash_attention(const float* Q, const float* K, const float* V,
                            float* O, int B, int H, int H_kv, int N, int d,
                            bool causal, cudaStream_t stream);

// paged_attention.cu
void launch_paged_attention(const float* Q, const int* block_table,
                            const float* K_pool, const float* V_pool,
                            float* O, const int* seq_lens,
                            int B, int H, int d, int block_size,
                            int max_blocks, cudaStream_t stream);

// quant_int4.cu
void launch_int4_dequant(const uint8_t* packed, const float* scales,
                         float* out, int rows, int cols, int group_size,
                         cudaStream_t stream);
void launch_int4_gemv(const uint8_t* packed, const float* scales,
                      const float* x, float* y,
                      int rows, int cols, int group_size, cudaStream_t stream);

// fp8_quant.cu
void launch_fp8_quantize(const float* in, __nv_fp8_e4m3* out, float* scales,
                         int rows, int cols, bool per_token, cudaStream_t stream);

// speculative_decoding.cu
void launch_speculative_decode(const float* draft_probs, const float* target_probs,
                               const int* draft_tokens, int* accepted_mask,
                               int B, int K, int vocab_size, cudaStream_t stream);
```

## Performance (RTX 4070 Laptop, SM 8.9, CUDA 12.0)

| Kernel | Config | Result |
|--------|--------|--------|
| FlashAttention MHA | N=1024, d=64, causal | ~2.18 ms, ~985 Gflops |
| FlashAttention GQA 4:1 | N=512, d=64 | ~0.67 ms, ~801 Gflops |
| PagedAttention decode | seq=1024, page=32 | ~0.48 ms |
| INT4 dequant | 4096×4096, group=128 | ~195 GB/s |
| FP8 quantize per-token | 1024×4096 | ~171 GB/s |

## Tests

```bash
./build/bin/test_flash_attention
./build/bin/test_paged_attention
./build/bin/test_quant_int4
./build/bin/test_fp8_quant
./build/bin/test_speculative_decoding
./build/bin/test_tensor_parallel
```
