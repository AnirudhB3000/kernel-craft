#!/usr/bin/env python3
"""
example_torch_conv_naive.py — kernel-craft naive convolution via PyTorch tensors.

API overview
------------
``kernel_craft_python.conv_naive(input_tensor, kernel_tensor)``

    When called with CUDA torch.Tensor arguments the module reads the raw
    device pointer via tensor.data_ptr() — no H2D copy, no temporary NumPy
    array.  The output is also a CUDA tensor allocated on the same device.

    Parameters
    ----------
    input_tensor  : torch.Tensor, float32, shape (H, W), device='cuda'
    kernel_tensor : torch.Tensor, float32, shape (k, k), device='cuda'

    Returns
    -------
    torch.Tensor, float32, shape (H, W), device='cuda'

Why zero-copy matters
---------------------
NumPy-backed calls (example_numpy_conv_naive.py) perform:
    H2D copy (host → GPU)  →  kernel  →  D2H copy (GPU → host)

PyTorch tensor calls perform only:
    kernel

When the input is already on the GPU (e.g. produced by a previous layer or
loaded by DALI directly into device memory) this eliminates ~2× the memory
traffic and the CPU synchronisation barrier implicit in cudaMemcpy.

Integration — custom training loop
-----------------------------------
You can use conv_naive as a differentiable-friendly operator inside a
PyTorch forward pass by wrapping it in torch.autograd.Function:

    class GpuConvNaive(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, k):
            ctx.save_for_backward(x, k)
            return kc.conv_naive(x, k)

        @staticmethod
        def backward(ctx, grad_output):
            x, k = ctx.saved_tensors
            # Gradient w.r.t. x: cross-correlate grad_output with flipped k
            grad_x = kc.conv_naive(grad_output, k.flip(0).flip(1))
            # Gradient w.r.t. k: cross-correlate x with grad_output
            grad_k = None  # implement if kernel is a learnable parameter
            return grad_x, grad_k

Integration — inference with torch.ops
----------------------------------------
For cleaner op composition use the torch.ops.kernel_craft namespace
(see example_torch_ops.py and src/python/kernel_craft_torch_ops.py):

    import kernel_craft_torch_ops  # registers kernel_craft namespace
    output = torch.ops.kernel_craft.flash_attention(Q, K, V, H_kv=4, causal=True)

Usage
-----
    cd examples/python
    python example_torch_conv_naive.py
"""

import sys
import os

_REPO = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_REPO, "src", "python", "build"))
sys.path.insert(0, os.path.join(_REPO, "src", "python"))

import torch
import numpy as np
import kernel_craft_python as kc  # type: ignore


def main() -> None:
    print("=" * 60)
    print("kernel-craft: naive convolution (PyTorch tensor interface)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available — cannot run this example.")
        return

    device = torch.device("cuda")
    print(f"\nDevice : {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # Input: 512×512 single-channel feature map already on the GPU.
    # In a real model this tensor arrives from the previous layer —
    # no H2D copy is needed.
    # ------------------------------------------------------------------
    H, W = 512, 512
    input_t = torch.rand(H, W, dtype=torch.float32, device=device)
    print(f"Input  : shape={tuple(input_t.shape)}  device={input_t.device}")

    # ------------------------------------------------------------------
    # 3×3 Laplacian edge-detection kernel on the GPU.
    # ------------------------------------------------------------------
    kernel_t = torch.tensor([
        [ 0.,  1.,  0.],
        [ 1., -4.,  1.],
        [ 0.,  1.,  0.],
    ], dtype=torch.float32, device=device)
    print(f"Kernel : shape={tuple(kernel_t.shape)}  device={kernel_t.device}")

    # ------------------------------------------------------------------
    # GPU conv — no H2D/D2H, operates directly on device pointers.
    # ------------------------------------------------------------------
    # Warm-up (triggers any lazy CUDA initialisation)
    _ = kc.conv_naive(input_t, kernel_t)
    torch.cuda.synchronize()

    # Timed run using CUDA events for accurate GPU elapsed time
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    result = kc.conv_naive(input_t, kernel_t)
    t1.record()
    torch.cuda.synchronize()
    elapsed_ms = t0.elapsed_time(t1)

    print(f"\nOutput : shape={tuple(result.shape)}  device={result.device}")
    print(f"Range  : [{result.min():.4f}, {result.max():.4f}]")
    print(f"Time   : {elapsed_ms:.3f} ms (GPU events, {H}×{W})")

    # ------------------------------------------------------------------
    # Correctness: compare to torch.nn.functional.conv2d (CPU reference)
    # ------------------------------------------------------------------
    ref = torch.nn.functional.conv2d(
        input_t.unsqueeze(0).unsqueeze(0),       # [1,1,H,W]
        kernel_t.unsqueeze(0).unsqueeze(0),       # [1,1,3,3]
        padding=1,
    ).squeeze()

    max_err = (result - ref).abs().max().item()
    print(f"\nMax error vs torch.nn.functional.conv2d: {max_err:.2e}  "
          f"{'PASS' if max_err < 1e-3 else 'FAIL'}")

    assert result.device.type == "cuda", "Output must remain on CUDA"

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
