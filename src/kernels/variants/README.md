# variants/ - Convolution Variants

## Overview
Extended convolution implementations covering specialized types used in modern CNN architectures.

## Files
- `conv3d.cu` - 3D convolution for volumetric data (video, medical imaging)
- `conv_dilated.cu` - Dilated convolution for expanded receptive fields without parameter increase
- `conv_transposed.cu` - Transposed convolution for upsampling (used in segmentation, GANs)
- `conv_grouped.cu` - Grouped convolution for ResNeXt-style architectures

## Key Insights
- **3D**: Scales cubically; tiling provides modest gains due to shared memory constraints
- **Dilated**: Expands receptive field but less efficient memory access patterns
- **Transposed**: Uses atomic additions for correct accumulation from multiple input pixels
- **Grouped**: Reduces parameters by factor of groups²; each group processes independent channel subsets
