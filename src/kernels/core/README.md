# core/ — Core 2D Convolution Kernels

Foundational convolution implementations demonstrating progressive optimization from naive global memory access to shared memory tiling.

## Files

| File | Description |
|------|-------------|
| `conv_naive.cu` | One thread per output pixel; direct global memory reads. Baseline for benchmarking. |
| `conv_tiled.cu` | Shared memory tiling with halo regions; `__syncthreads()` between load and compute phases. |

## Key Concepts

- **Naive**: Simple mapping but redundant global memory reads — each input pixel is re-read by multiple threads
- **Tiled**: Loads an input tile + halo into shared memory once per block, then computes all outputs from fast on-chip memory
- **Tile size dispatch**: 8×8, 16×16, and 32×32 tiles are all supported; 8×8 performs best for 3×3 kernels across all tested image sizes

## Benchmark Results (3×3 kernel)

| Kernel | Image Size | Time (ms) |
|--------|------------|-----------|
| Naive | 256×256 | ~0.45 |
| Tiled 8×8 | 256×256 | ~0.36 |
| Tiled 8×8 | 1024×1024 | ~0.60 |
| Tiled 8×8 | 2048×2048 | ~1.34 |

See `CLAUDE.md` for the full tile-size comparison table.

## Tests

```bash
./build/bin/test_conv_naive
./build/bin/test_conv_tiled
```
