#!/usr/bin/env python3
"""
Example: Naive convolution with PyTorch tensors.

This example demonstrates using kernel_craft's conv_naive function
with PyTorch tensors on CUDA device. No memory copying - operates
directly on GPU tensors.
"""

import sys
sys.path.insert(0, '../../src/python/build')

import torch
import kernel_craft_python as kc


def main():
    print("=" * 60)
    print("kernel_craft Python Example: Naive Convolution (PyTorch)")
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
    print(f"Input device: {input_tensor.device}")

    # Create kernel tensor on CUDA (3x3 edge detection)
    kernel = torch.tensor([
        [0,  1,  0],
        [1, -4,  1],
        [0,  1, 0],
    ], dtype=torch.float32, device='cuda')

    print(f"Kernel shape: {kernel.shape}")
    print(f"Kernel device: {kernel.device}")

    # Run convolution
    print("\nRunning conv_naive...")
    result = kc.conv_naive(input_tensor, kernel)

    print(f"Output shape: {result.shape}")
    print(f"Output device: {result.device}")
    print(f"Output range: [{result.min():.4f}, {result.max():.4f}]")

    # Verify output is on CUDA
    assert result.device.type == 'cuda', "Output should be on CUDA"

    print("\n" + "=" * 60)
    print("SUCCESS: Naive convolution with PyTorch completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()