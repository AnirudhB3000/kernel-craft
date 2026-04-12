#!/usr/bin/env python3
"""
Example: Naive convolution with numpy arrays using real image.

This example demonstrates using kernel_craft's conv_naive function
with a real image (lena.png) and applies an edge detection kernel.
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
    print("kernel_craft Python Example: Naive Convolution (numpy)")
    print("=" * 60)

    # Load the lena image
    input_arr = load_image('../../data/sample_images/lena_512x512.png')
    print(f"\nInput shape: {input_arr.shape}")

    # Create a 3x3 edge detection kernel (Laplacian)
    kernel = np.array([
        [0,  1,  0],
        [1, -4,  1],
        [0,  1,  0],
    ], dtype=np.float32)

    print(f"Kernel shape: {kernel.shape}")
    print(f"Kernel:\n{kernel}")

    # Run convolution
    print("\nRunning conv_naive...")
    result = kc.conv_naive(input_arr, kernel)

    print(f"Output shape: {result.shape}")
    print(f"Output range: [{result.min():.4f}, {result.max():.4f}]")

    # Save output image
    output_img = (np.clip(result, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(output_img).save('../../data/outputs/output_naive.png')
    print("\nSaved output to ../../data/outputs/output_naive.png")

    print("\n" + "=" * 60)
    print("SUCCESS: Naive convolution with lena image completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()