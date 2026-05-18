# custom/ — Custom Operations

Non-standard convolution variants and domain-specific CUDA operations not efficiently served by standard dense kernels.

## Files

| File | Description |
|------|-------------|
| `custom_op.cu` | Sparse convolution in coordinate-list (COO) format; only computes at non-zero input locations |

## Purpose

Demonstrates custom kernel design patterns for cases where standard convolution is wasteful:
- **Sparse inputs**: point clouds, event camera frames, sparse feature maps where most activations are zero
- **Domain-specific transforms**: custom filter shapes, non-rectangular kernels

## Design Notes

- Uses coordinate-list (COO) format: `(row, col, value)` triples for non-zero locations
- Each CUDA thread handles one non-zero input element and scatters contributions to output
- Correctness validated against dense reference; use when sparsity > ~70% for a practical speedup
