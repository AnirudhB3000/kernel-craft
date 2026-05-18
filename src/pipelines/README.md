# pipelines/ — ML Pipeline Implementations

End-to-end pipeline kernels demonstrating fused vs separate execution and GPU-side preprocessing.

## Files

| File | Description |
|------|-------------|
| `pipeline_fused.cu` | Single kernel: conv + batch normalization + ReLU fused into one pass |
| `pipeline_separate.cu` | Three separate kernel launches: conv, then BN, then ReLU |
| `preprocess_gpu.cu` | GPU preprocessing: resize, normalize, and horizontal flip for training data pipelines |

## Key Insight

Memory movement dominates cost over arithmetic. The fused pipeline avoids two intermediate global memory round-trips (conv→BN and BN→ReLU), achieving:

| Pipeline | Kernel Launches | Time (ms) @ 1024×1024 | Speedup |
|----------|----------------|-----------------------|---------|
| Separate | 3 | ~0.96 | baseline |
| Fused | 1 | ~0.42 | ~2.3× |

## GPU Preprocessing (`preprocess_gpu.cu`)

Moves data augmentation and normalization onto the GPU, eliminating the CPU preprocessing bottleneck in training pipelines. Operations:
- **Resize**: bilinear interpolation
- **Normalize**: per-channel mean subtraction and std division
- **Flip**: horizontal flip augmentation

Pairs with `src/performance/async_streams.cu` for double-buffered overlap of preprocessing and forward pass on concurrent CUDA streams.
