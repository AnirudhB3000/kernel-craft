#!/usr/bin/env python3
"""
Example: Tiled convolution with PyTorch tensors.

This example demonstrates using kernel_craft's conv_tiled function
with PyTorch tensors on CUDA device with different tile sizes.
"""

import sys
sys.path.insert(0, '../../src/python/build')

import torch
import kernel_craft_python as kc
import time


def main():
    print("=" * 60)
    print("kernel_craft Python Example: Tiled Convolution (PyTorch)")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    print(f"\nCUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # Create input tensor on CUDA (512x512)
    input_tensor = torch.rand(512, 512, dtype=torch.float32, device='cuda')
    print(f"\nInput shape: {input_tensor.shape}")

    # Create box blur kernel (3x3)
    kernel = torch.ones(3, 3, dtype=torch.float32, device='cuda') / 9.0
    print(f"Kernel shape: {kernel.shape}")

    # Test different tile sizes
    tile_sizes = [(8, 8), (16, 16), (32, 32)]

    for tile_w, tile_h in tile_sizes:
        # Warmup
        _ = kc.conv_tiled(input_tensor, kernel, tile_w, tile_h)
        torch.cuda.synchronize()

        # Timed run
        start = time.time()
        result = kc.conv_tiled(input_tensor, kernel, tile_w, tile_h)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        print(f"\n--- Tile {tile_w}x{tile_h} ---")
        print(f"Time: {elapsed*1000:.2f} ms")
        print(f"Output range: [{result.min():.4f}, {result.max():.4f}]")

    # Compare naive vs tiled
    print("\n--- Comparing naive vs tiled ---")

    # Warmup
    _ = kc.conv_naive(input_tensor, kernel)
    _ = kc.conv_tiled(input_tensor, kernel, 8, 8)
    torch.cuda.synchronize()

    start = time.time()
    result_naive = kc.conv_naive(input_tensor, kernel)
    torch.cuda.synchronize()
    time_naive = time.time() - start

    start = time.time()
    result_tiled = kc.conv_tiled(input_tensor, kernel, 8, 8)
    torch.cuda.synchronize()
    time_tiled = time.time() - start

    max_diff = (result_naive - result_tiled).abs().max().item()
    print(f"Naive time: {time_naive*1000:.2f} ms")
    print(f"Tiled (8x8) time: {time_tiled*1000:.2f} ms")
    print(f"Max difference: {max_diff:.8f}")

    print("\n" + "=" * 60)
    print("SUCCESS: Tiled convolution with PyTorch completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()