#!/usr/bin/env python3
"""
example_numpy_conv_tiled.py — kernel-craft tiled convolution via NumPy arrays.

API overview
------------
``kernel_craft_python.conv_tiled(input, kernel, tile_w, tile_h)``

    Runs the shared-memory tiled 2-D GPU convolution on a NumPy float32 array.

    Parameters
    ----------
    input   : np.ndarray, float32, shape (H, W)
    kernel  : np.ndarray, float32, shape (k, k)
    tile_w  : int — CUDA thread-block width  (8, 16, or 32)
    tile_h  : int — CUDA thread-block height (8, 16, or 32)

    Returns
    -------
    np.ndarray, float32, shape (H, W)

How tiling works
----------------
Each CUDA thread block cooperatively loads a (tile + halo) patch of the
input image into on-chip shared memory (~10–100× faster than global memory).
Threads then read exclusively from shared memory during the kernel loop,
eliminating the redundant global-memory re-fetches present in conv_naive.

Tile-size guidance (3×3 kernel, from CLAUDE.md benchmarks):
  tile  8×8  — best overall; 10×10 shared tile = 400 bytes, low occupancy cost
  tile 16×16 — good for 1024×1024 images; 18×18 = 1296 bytes
  tile 32×32 — large images only; 34×34 = 4624 bytes, may reduce occupancy

Integration context — batch preprocessing pipeline
---------------------------------------------------
In a training data pipeline (DALI, PyTorch DataLoader) you may want to apply
the same convolution filter (e.g. Gaussian blur, Sobel edge) to every image
in a batch before they reach the model.  Use conv_tiled in a map() transform:

    import kernel_craft_python as kc
    import numpy as np

    gaussian = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float32) / 16.0

    def augment(image_uint8: np.ndarray) -> np.ndarray:
        img = image_uint8.astype(np.float32) / 255.0
        return kc.conv_tiled(img, gaussian, tile_w=8, tile_h=8)

    dataset = dataset.map(augment)

For sustained GPU throughput across a batch, prefer the PyTorch tensor
interface (example_torch_conv_tiled.py) to avoid per-image H2D copies.

Usage
-----
    cd examples/python
    python example_numpy_conv_tiled.py
"""

import sys
import os
import time

_REPO = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_REPO, "src", "python", "build"))

import numpy as np
import kernel_craft_python as kc  # type: ignore


def main() -> None:
    print("=" * 60)
    print("kernel-craft: tiled convolution (NumPy interface)")
    print("=" * 60)

    np.random.seed(1)

    # ------------------------------------------------------------------
    # 512×512 image — large enough to show tile-size differences clearly.
    # ------------------------------------------------------------------
    H, W = 512, 512
    image = np.random.rand(H, W).astype(np.float32)
    print(f"\nInput shape: {image.shape}  dtype={image.dtype}")

    # ------------------------------------------------------------------
    # 3×3 box blur.  All tile sizes produce identical output.
    # ------------------------------------------------------------------
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    print(f"Kernel: 3×3 box blur (all weights = 1/9)")

    # ------------------------------------------------------------------
    # Tile-size sweep with timing.
    # The H2D + D2H round-trip is included in the wall-clock time here;
    # for pure GPU timing, use CUDA events (see example_torch_conv_tiled.py).
    # ------------------------------------------------------------------
    print("\n--- Tile size sweep ---")
    best = None
    for tw, th in [(8, 8), (16, 16), (32, 32)]:
        # Warm-up: first call may include JIT compilation overhead
        kc.conv_tiled(image, kernel, tw, th)

        t0 = time.perf_counter()
        result = kc.conv_tiled(image, kernel, tw, th)
        elapsed_ms = (time.perf_counter() - t0) * 1e3

        print(f"  tile {tw:2d}×{th:2d}  →  {elapsed_ms:.2f} ms  "
              f"output range [{result.min():.4f}, {result.max():.4f}]")

        if best is None or elapsed_ms < best[1]:
            best = ((tw, th), elapsed_ms, result)

    print(f"\nBest tile: {best[0][0]}×{best[0][1]}  ({best[1]:.2f} ms)")

    # ------------------------------------------------------------------
    # Correctness: all tile sizes must match conv_naive.
    # ------------------------------------------------------------------
    print("\n--- Correctness vs conv_naive ---")
    ref = kc.conv_naive(image, kernel)
    for tw, th in [(8, 8), (16, 16), (32, 32)]:
        got     = kc.conv_tiled(image, kernel, tw, th)
        max_err = float(np.max(np.abs(ref - got)))
        print(f"  tile {tw}×{th}  max_err={max_err:.2e}  "
              f"{'PASS' if max_err < 1e-4 else 'FAIL'}")

    # ------------------------------------------------------------------
    # Save best result (optional)
    # ------------------------------------------------------------------
    try:
        from PIL import Image
        out_dir = os.path.join(_REPO, "data", "outputs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "output_tiled.png")
        clipped = np.clip(best[2], 0, 1)
        Image.fromarray((clipped * 255).astype(np.uint8)).save(out_path)
        print(f"\nSaved → {out_path}")
    except ImportError:
        pass

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
