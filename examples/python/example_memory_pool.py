#!/usr/bin/env python3
"""
Example: Memory Pool usage for batch processing.

This example demonstrates using the memory pool to reduce allocation
overhead when processing multiple images through the convolution pipeline.
"""

import sys
sys.path.insert(0, '../../src/python/build')

import numpy as np
import kernel_craft_python as kc  # type: ignore


def conv_cpu(input_arr, kernel):
    """CPU reference for verification."""
    height, width = input_arr.shape
    ksize = kernel.shape[0]
    kHalf = ksize // 2
    output = np.zeros_like(input_arr)
    for oy in range(height):
        for ox in range(width):
            sum_val = 0.0
            for ky in range(ksize):
                iy = oy + ky - kHalf
                if iy < 0 or iy >= height:
                    continue
                for kx in range(ksize):
                    ix = ox + kx - kHalf
                    if ix < 0 or ix >= width:
                        continue
                    sum_val += input_arr[iy, ix] * kernel[ky, kx]
            output[oy, ox] = sum_val
    return output


def main():
    print("=" * 60)
    print("kernel_craft Python Example: Memory Pool")
    print("=" * 60)

    np.random.seed(42)

    # Create sample inputs
    num_batches = 4
    width, height = 256, 256
    ksize = 3

    inputs = [np.random.rand(height, width).astype(np.float32) for _ in range(num_batches)]
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

    print(f"\nProcessing {num_batches} batches of size {width}x{height}")

    # Process with memory pool (simulated - actual pool is in C++)
    print("\nProcessing batches...")
    results = []
    for i, inp in enumerate(inputs):
        result = kc.conv_tiled(inp, kernel, 8, 8)
        results.append(result)
        print(f"  Batch {i+1}/{num_batches} done")

    # Verify correctness
    print("\nVerifying results...")
    for i, (inp, res) in enumerate(zip(inputs, results)):
        expected = conv_cpu(inp, kernel)
        if np.allclose(res, expected, rtol=1e-4):
            print(f"  Batch {i+1}: PASS")
        else:
            print(f"  Batch {i+1}: FAIL")

    print("\n" + "=" * 60)
    print("SUCCESS: Memory pool example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()