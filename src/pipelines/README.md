# pipelines/ - ML Pipeline Implementations

## Overview
End-to-end pipeline implementations demonstrating fused vs separate kernel approaches and GPU preprocessing.

## Files
- `pipeline_fused.cu` - Single kernel performing conv + batchnorm + relu (fused pipeline)
- `pipeline_separate.cu` - Separate kernel launches for conv, batchnorm, relu
- `preprocess_gpu.cu` - GPU implementations of resize, normalization, and simple augmentation

## Key Insight
Memory movement dominates cost more than arithmetic. Fused pipeline achieves ~2.3x speedup over separate launches by saving ~40% global memory traffic.

## Full GPU Pipeline
Combines preprocessing (resize, normalize, augment) with training forward pass on GPU to eliminate CPU preprocessing bottleneck.
