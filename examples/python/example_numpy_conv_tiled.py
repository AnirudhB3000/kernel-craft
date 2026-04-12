#!/usr/bin/env python3
"""
Example: Tiled convolution with numpy arrays using real image.

This example demonstrates using kernel_craft's conv_tiled function
with configurable tile sizes. Tiled convolution uses shared memory
for better memory bandwidth utilization.
"""

import sys
sys.path.insert(0, '../../src/python/build')

import numpy as np
from PIL import Image
import kernel_craft_python as kc


def load_image(path):
    """Load image and convert to float32 grayscale numpy array."""
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.float32) / 255.0


def main():
    print("=" * 60)
    print("kernel_craft Python Example: Tiled Convolution (numpy)")
    print("=" * 60)

    # Load the lena image
    input_arr = load_image('../../data/sample_images/lena_512x512.png')
    print(f"\nInput shape: {input_arr.shape}")

    # Create a 3x3 box blur kernel
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0

    print(f"Kernel shape: {kernel.shape}")
    print(f"Kernel (box blur):\n{kernel}")

    # Test different tile sizes and measure time
    import time

    tile_sizes = [(8, 8), (16, 16), (32, 32)]

    for tile_w, tile_h in tile_sizes:
        # Warmup
        _ = kc.conv_tiled(input_arr, kernel, tile_w, tile_h)

        # Timed run
        start = time.time()
        result = kc.conv_tiled(input_arr, kernel, tile_w, tile_h)
        elapsed = time.time() - start

        print(f"\n--- Tile {tile_w}x{tile_h} ---")
        print(f"Time: {elapsed*1000:.2f} ms")
        print(f"Output range: [{result.min():.4f}, {result.max():.4f}]")

    # Compare naive vs tiled
    print("\n--- Comparing naive vs tiled ---")

    # Warmup
    _ = kc.conv_naive(input_arr, kernel)
    _ = kc.conv_tiled(input_arr, kernel, 8, 8)

    start = time.time()
    result_naive = kc.conv_naive(input_arr, kernel)
    time_naive = time.time() - start

    start = time.time()
    result_tiled = kc.conv_tiled(input_arr, kernel, 8, 8)
    time_tiled = time.time() - start

    max_diff = np.max(np.abs(result_naive - result_tiled))
    print(f"Naive time: {time_naive*1000:.2f} ms")
    print(f"Tiled (8x8) time: {time_tiled*1000:.2f} ms")
    print(f"Max difference: {max_diff:.8f}")

    # Save output image
    output_img = (np.clip(result_tiled, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(output_img).save('../../data/outputs/output_tiled.png')
    print("\nSaved output to ../../data/outputs/output_tiled.png")

    print("\n" + "=" * 60)
    print("SUCCESS: Tiled convolution with lena image completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()