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
| `tensor_parallel.cu` | Ring all-reduce, all-gather, column-parallel and row-parallel linear (tiled 16×16 SGEMM) |
| `tensor_parallel_nccl.cu` | NCCL-backed all-reduce and all-gather; compiles to stubs when `HAVE_NCCL` is not defined |
| `selective_scan.cu` | Mamba-1 ZOH selective scan over `[B, L, D]`; per-(b,d) thread, N_state ≤ 32 in registers |
| `depthwise_conv1d.cu` | Causal depthwise conv1d; channels-first `[B, D, L]` layout, configurable `d_conv` |
| `rmsnorm.cu` | Fused RMSNorm: two-pass parallel reduce → rms_inv → normalize + scale; D up to 65536 |

## Hardware Requirements

| Feature | Minimum SM | Notes |
|---------|-----------|-------|
| FlashAttention | SM 6.0 | Shared memory tiling, online softmax |
| PagedAttention | SM 6.0 | Block table pointer arithmetic |
| INT4 GEMV | SM 8.0 | Tensor Core path for Ampere+; falls back to scalar on older hardware |
| FP8 E4M3 | SM 8.9 | `__nv_fp8_e4m3` type requires CUDA ≥ 11.8 and Ada/Hopper |
| Tensor Parallelism (SGEMM) | SM 6.0 | Tiled SGEMM for col/row parallel linear; single-GPU simulation always available |
| Tensor Parallelism (NCCL) | SM 6.0 | Real multi-GPU all-reduce/all-gather; requires peer-to-peer and libnccl |
| Selective Scan | SM 6.0 | Sequential recurrence per (b,d) thread; N_state capped at 32 |
| Depthwise Conv1d | SM 6.0 | Causal index-gating; no explicit zero-pad buffer |
| RMSNorm | SM 6.0 | Standard parallel reduction; no special hardware requirement |

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

// tensor_parallel.cu
void launch_col_parallel_linear(const float* x, const float* W, float* y,
                                int M, int N_rank, int K, cudaStream_t stream);
void launch_row_parallel_linear(const float* x_rank, const float* W, float* partial,
                                int M, int N, int K_rank, cudaStream_t stream);

// tensor_parallel_nccl.cu  (requires HAVE_NCCL; otherwise no-ops)
void nccl_comm_init_all(void** comms, int n, const int* devs);
void nccl_comm_destroy(void* comm);
void launch_ring_allreduce_nccl(void* comm, float* d_buf, int count, cudaStream_t stream);
void launch_allgather_nccl(void* comm, const float* sendbuf, float* recvbuf,
                           int count, cudaStream_t stream);

// selective_scan.cu
void launch_selective_scan(const float* d_u, const float* d_A_log,
                           const float* d_B, const float* d_C,
                           const float* d_delta, float* d_y,
                           int B, int L, int D, int N_state,
                           cudaStream_t stream);

// depthwise_conv1d.cu  (input: channels-first [B, D, L])
void launch_depthwise_conv1d(const float* d_x, const float* d_w, const float* d_bias,
                             float* d_y, int B, int D, int L, int d_conv,
                             cudaStream_t stream);

// rmsnorm.cu
void launch_rmsnorm(const float* d_x, const float* d_g, float* d_y,
                    int rows, int D, float eps, cudaStream_t stream);
```

## Performance (RTX 4070 Laptop, SM 8.9, CUDA 12.0)

| Kernel | Config | Result |
|--------|--------|--------|
| FlashAttention MHA | N=1024, d=64, causal | ~2.18 ms, ~985 Gflops |
| FlashAttention GQA 4:1 | N=512, d=64 | ~0.67 ms, ~801 Gflops |
| PagedAttention decode | seq=1024, page=32 | ~0.48 ms |
| INT4 dequant | 4096×4096, group=128 | ~195 GB/s |
| FP8 quantize per-token | 1024×4096 | ~171 GB/s |
| Col-parallel linear | M=32, N=4096, K=4096 | ~0.94 TFLOPS |
| Row-parallel linear | M=32, N=11008, K=4096 | ~0.83 TFLOPS |
| Sim all-gather | 2 ranks, 4 MB | ~117 GB/s |
| Selective scan | B=1, D=512, L=4096, N=16 | ~5.57 ms, ~36 Gflops |
| Depthwise conv1d | B=1, D=2048, L=1024, d_conv=4 | ~0.068 ms, ~741 GB/s |
| RMSNorm | 128 rows, D=4096 | ~0.025 ms, ~171 GB/s |

## Tests

```bash
./build/bin/test_flash_attention
./build/bin/test_paged_attention
./build/bin/test_quant_int4
./build/bin/test_fp8_quant
./build/bin/test_speculative_decoding
./build/bin/test_tensor_parallel   # includes col/row parallel linear; NCCL tests skip without libnccl
./build/bin/test_selective_scan    # 5 tests: small/typical/long/zero/max-state
./build/bin/test_mamba_ops         # 8 tests: 4 conv1d variants + 4 rmsnorm variants

# Multi-process NCCL test (requires 2+ GPUs or same-device workaround):
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    tests/test_tensor_parallel_multiprocess.py
```
