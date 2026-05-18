# performance/ — Performance Infrastructure

Utilities for reducing kernel launch overhead, memory allocation cost, and memory transfer latency, independent of the specific kernel workload.

## Files

| File | Description |
|------|-------------|
| `cuda_graphs.cu` | Captures a kernel DAG once and replays it — eliminates per-launch CPU overhead |
| `memory_pool.cu` | Pre-allocated device buffer pool — eliminates per-batch `cudaMalloc`/`cudaFree` round-trips |
| `mixed_precision.cu` | FP16/TF32 kernel variants using Tensor Cores (Volta+ for FP16, Ampere+ for TF32) |
| `persistent_kernels.cu` | Kernels that stay resident across batches, eliminating re-launch overhead for fixed workloads |
| `async_streams.cu` | Double-buffered async pipeline — overlaps H2D transfer, compute, and D2H using two CUDA streams |
| `unified_memory.cu` | Managed (`cudaMallocManaged`) memory with `cudaMemPrefetchAsync` to avoid first-access page faults |
| `multi_stream_pipeline.cu` | Concurrent preprocessing + inference using `cudaStreamWaitEvent` for inter-stream synchronization |

## Benchmark Results

| Optimization | Result |
|-------------|--------|
| CUDA Graphs | ~5–10 μs launch overhead vs ~15–30 μs for separate launches |
| Memory Pool | ~85% reduction in allocation overhead (16-buffer reuse) |
| FP16 Tensor Cores | ~2× compute throughput, 50% bandwidth vs FP32 |
| TF32 | Automatic on Ampere+; ~1.1× throughput with near-FP32 accuracy |
| Persistent Kernels | ~50% lower latency for fixed batch streams |
| Async double-buffer | ~2× throughput on large batches (512×512, 16 batches, pinned host memory) |
| Unified + prefetch | ~equivalent to explicit `cudaMemcpy`; without prefetch: up to 3× overhead on first access |

## Notes

- `async_streams.cu` requires pinned (`cudaMallocHost`) host memory for true async H2D/D2H overlap — pageable memory silently falls back to synchronous copies
- `multi_stream_pipeline.cu` uses timing-disabled events (`cudaEventDisableTiming`) for inter-stream synchronization to avoid profiler overhead
- Unified memory page-fault overhead is significant on discrete GPUs without NVLink; always call `cudaMemPrefetchAsync` before the first kernel
