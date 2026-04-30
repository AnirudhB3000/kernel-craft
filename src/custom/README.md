# custom/ - Custom Operations

## Overview
Non-standard convolution variants and domain-specific CUDA operations that extend beyond standard dense convolution.

## Files
- `custom_op.cu` - Custom non-standard convolution implementation (sparse convolution, domain-specific transforms, or specialized filter kernels)

## Purpose
Demonstrates how to implement convolution variants not efficiently supported by standard libraries. Used to compare against dense fallbacks and understand tradeoffs in custom kernel design.
