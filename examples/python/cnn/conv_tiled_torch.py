#!/usr/bin/env python3
"""
example_torch_conv_tiled.py — kernel-craft tiled convolution via PyTorch tensors.

API overview
------------
``kernel_craft_python.conv_tiled(input_tensor, kernel_tensor, tile_w, tile_h)``

    Shared-memory tiled 2-D convolution operating directly on CUDA tensors.
    Parameters and return value identical to conv_naive but with explicit tile
    dimensions that control the CUDA thread-block size:

    tile_w, tile_h = 8, 8   — safest default; 8×8 tile + 3×3 halo = 10×10 shmem
    tile_w, tile_h = 16, 16 — faster on 1024+ images; 18×18 shmem tile
    tile_w, tile_h = 32, 32 — only beneficial on very large feature maps

Tile size vs occupancy trade-off
---------------------------------
Larger tiles mean more threads per block → higher potential occupancy, but
also more shared memory per block → fewer concurrent blocks on each SM:

    8×8   = 64  threads/block, 10×10×4 = 400 B shmem  → high occupancy
    16×16 = 256 threads/block, 18×18×4 = 1296 B shmem → medium occupancy
    32×32 = 1024 threads/block (max), 34×34×4 = 4624 B  → low occupancy

For 3×3 kernels 8×8 wins consistently across image sizes (see CLAUDE.md).
For 5×5+ kernels the halo grows so a 16×16 tile amortises shmem loading better.

Integration — PyTorch custom extension pattern
-----------------------------------------------
This module can be used as a drop-in replacement for F.conv2d in models that
need reproducible GPU-level control without going through cuDNN:

    import kernel_craft_python as kc

    class CustomConv2d(torch.nn.Module):
        def __init__(self, weight: torch.Tensor, tile: int = 8):
            super().__init__()
            self.weight = torch.nn.Parameter(weight)  # [k, k]
            self.tile   = tile

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [H, W] on CUDA (single-channel for demonstration)
            return kc.conv_tiled(x, self.weight, self.tile, self.tile)

Integration — CUDA Graphs
--------------------------
When the same image tensor and weights are used repeatedly (e.g. streaming
inference), wrap the tiled call in a CUDA Graph for zero launch overhead:

    # Capture once
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        result_static = kc.conv_tiled(input_static, kernel_static, 8, 8)

    # Replay per frame (no Python overhead, one GPU dispatch)
    input_static.copy_(new_frame)
    g.replay()
    frame_result = result_static.clone()

See example_cuda_graphs.cpp for the C++ version of this pattern.

Usage
-----
    cd examples/python
    python example_torch_conv_tiled.py
"""

import sys
import os
import time

_REPO = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_REPO, "src", "python", "build"))
sys.path.insert(0, os.path.join(_REPO, "src", "python"))

import torch
import kernel_craft_python as kc  # type: ignore


def _cuda_time_ms(fn, warmup: int = 2, reps: int = 10) -> float:
    """Time a callable with CUDA events, averaging over `reps` runs."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(reps):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / reps


def main() -> None:
    print("=" * 60)
    print("kernel-craft: tiled convolution (PyTorch tensor interface)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available — cannot run this example.")
        return

    device = torch.device("cuda")
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # 512×512 feature map on the GPU (simulates output of a previous layer)
    # ------------------------------------------------------------------
    H, W = 512, 512
    input_t = torch.rand(H, W, dtype=torch.float32, device=device)
    kernel_t = torch.ones(3, 3, dtype=torch.float32, device=device) / 9.0
    print(f"Input : {tuple(input_t.shape)}   Kernel: {tuple(kernel_t.shape)}")

    # ------------------------------------------------------------------
    # Tile-size sweep with CUDA event timing
    # ------------------------------------------------------------------
    print("\n--- Tile size sweep (CUDA event timing) ---")
    results = {}
    for tw, th in [(8, 8), (16, 16), (32, 32)]:
        ms = _cuda_time_ms(lambda tw=tw, th=th: kc.conv_tiled(input_t, kernel_t, tw, th))
        results[(tw, th)] = ms
        print(f"  tile {tw:2d}×{th:2d}  →  {ms:.3f} ms/call")

    best_tile = min(results, key=results.get)
    print(f"\nBest tile: {best_tile[0]}×{best_tile[1]}  ({results[best_tile]:.3f} ms)")

    # ------------------------------------------------------------------
    # Comparison vs conv_naive (GPU events only, no H2D overhead)
    # ------------------------------------------------------------------
    print("\n--- Tiled vs Naive (same input, CUDA events) ---")
    ms_naive = _cuda_time_ms(lambda: kc.conv_naive(input_t, kernel_t))
    ms_tiled = _cuda_time_ms(lambda: kc.conv_tiled(input_t, kernel_t, 8, 8))
    print(f"  Naive        : {ms_naive:.3f} ms")
    print(f"  Tiled (8×8)  : {ms_tiled:.3f} ms")
    print(f"  Speedup      : {ms_naive/ms_tiled:.2f}×")

    # ------------------------------------------------------------------
    # Correctness across all tile sizes
    # ------------------------------------------------------------------
    print("\n--- Correctness vs conv_naive ---")
    ref = kc.conv_naive(input_t, kernel_t)
    for tw, th in [(8, 8), (16, 16), (32, 32)]:
        out = kc.conv_tiled(input_t, kernel_t, tw, th)
        err = (out - ref).abs().max().item()
        print(f"  tile {tw}×{th}  max_err={err:.2e}  {'PASS' if err < 1e-4 else 'FAIL'}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
