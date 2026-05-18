# variants/ — Convolution Variants

Extended convolution kernels for non-standard receptive fields, upsampling, and grouped channel processing.

## Files

| File | Description |
|------|-------------|
| `conv3d.cu` | 3D convolution for volumetric data (video, medical imaging) |
| `conv_dilated.cu` | Dilated convolution — expands receptive field without increasing parameter count |
| `conv_transposed.cu` | Transposed (deconvolution) — upsampling via strided expansion; uses atomicAdd for correct accumulation |
| `conv_grouped.cu` | Grouped convolution — each group processes independent channel subsets (ResNeXt / depthwise-style) |

## Key Insights

| Variant | Trade-off |
|---------|-----------|
| 3D | Cost scales cubically with spatial dims; shared memory pressure limits tile gains |
| Dilated | Receptive field grows without extra parameters, but strided access hurts cache efficiency |
| Transposed | Multiple input pixels contribute to each output → atomic adds required |
| Grouped | Parameters scale as `1/groups²`; enables depthwise separable convolutions at `groups = C_in` |

## Benchmark Results

| Kernel | Config | Naive GPU (ms) | Tiled (ms) |
|--------|--------|----------------|------------|
| 3D conv | 64³, 3×3×3 | ~8.5 | ~7.2 |
| 3D conv | 128³, 3×3×3 | ~65 | ~52 |
| Dilated (rate=2) | 1024×1024 | — | ~+15% vs rate=1 |
| Dilated (rate=8) | 1024×1024 | — | ~+120% vs rate=1 |
