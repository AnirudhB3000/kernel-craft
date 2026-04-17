#!/usr/bin/env python3
"""
Example: Mixed precision (FP16) convolution.

This example demonstrates using half-precision floating point
for faster computation on GPUs with Tensor Cores (Volta+).
"""

import sys
sys.path.insert(0, '../../src/python/build')

import numpy as np
import kernel_craft_python as kc  # type: ignore


def main():
    print("=" * 60)
    print("kernel_craft Python Example: Mixed Precision (FP16)")
    print("=" * 60)

    np.random.seed(42)

    # Create input data
    width, height = 256, 256
    ksize = 3

    input_fp32 = np.random.rand(height, width).astype(np.float32)
    kernel = np.random.rand(ksize, ksize).astype(np.float32)

    print(f"\nInput shape: {input_fp32.shape}")
    print(f"Kernel shape: {kernel.shape}")

    # Run FP32 convolution (baseline)
    print("\nRunning FP32 convolution...")
    result_fp32 = kc.conv_tiled(input_fp32, kernel, 8, 8)
    print(f"FP32 result range: [{result_fp32.min():.4f}, {result_fp32.max():.4f}]")

    # Simulate FP16 by converting to float16 and back (for demonstration)
    input_fp16 = input_fp32.astype(np.float16).astype(np.float32)
    kernel_fp16 = kernel.astype(np.float16).astype(np.float32)

    print("\nRunning FP16 (simulated with float16 input)...")
    result_fp16 = kc.conv_tiled(input_fp16, kernel_fp16, 8, 8)
    print(f"FP16 result range: [{result_fp16.min():.4f}, {result_fp16.max():.4f}]")

    # Compare results
    diff = np.abs(result_fp32 - result_fp16)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"\nPrecision comparison:")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Relative error: {mean_diff / np.mean(np.abs(result_fp32)) * 100:.4f}%")

    print("\n" + "=" * 60)
    print("SUCCESS: Mixed precision example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()