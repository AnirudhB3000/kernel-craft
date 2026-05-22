#!/usr/bin/env python3
"""
example_numpy_conv_naive.py — kernel-craft naive convolution via NumPy arrays.

API overview
------------
``kernel_craft_python.conv_naive(input, kernel)``

    Runs the naive 2-D GPU convolution kernel on a NumPy float32 array.
    The module transparently:
      1. Copies the array to a CUDA device buffer (cudaMalloc + cudaMemcpy).
      2. Launches the naive kernel with a 16×16 thread block.
      3. Copies the result back and returns a new NumPy array of the same shape.

    Parameters
    ----------
    input  : np.ndarray, float32, shape (H, W)
    kernel : np.ndarray, float32, shape (k, k)  — must have odd k

    Returns
    -------
    np.ndarray, float32, shape (H, W)

Integration context — scikit-image / OpenCV replacement
--------------------------------------------------------
This function is a drop-in GPU accelerator for scipy.ndimage.convolve2d.
For batch processing in a data-loading pipeline (e.g. PyTorch DataLoader
workers that pre-process images on the CPU), replacing scipy.ndimage with
this call moves the compute to the GPU without changing the NumPy-based
interface:

    # Before (CPU, scipy)
    from scipy.ndimage import convolve
    result = convolve(image, kernel, mode='constant', cval=0.0)

    # After (GPU, kernel-craft)
    import kernel_craft_python as kc
    result = kc.conv_naive(image.astype(np.float32), kernel.astype(np.float32))

For sustained throughput, prefer the tiled variant (conv_tiled) or the
Python torch bridge (example_torch_ops.py) which avoids the H2D/D2H copies.

Usage
-----
    cd examples/python
    python example_numpy_conv_naive.py
"""

import sys
import os

# Locate the compiled extension (.so) relative to this file.
# Adjust if you installed the package with `pip install -e .`.
_REPO = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_REPO, "src", "python", "build"))

import numpy as np
import kernel_craft_python as kc  # type: ignore


# ---------------------------------------------------------------------------
# CPU reference (used for correctness verification only)
# ---------------------------------------------------------------------------

def _conv_cpu(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Slow but unambiguous CPU convolution used to verify GPU results."""
    H, W = image.shape
    kH, kW = kernel.shape
    ph, pw = kH // 2, kW // 2
    out = np.zeros_like(image)
    for oy in range(H):
        for ox in range(W):
            acc = 0.0
            for ky in range(kH):
                iy = oy + ky - ph
                if iy < 0 or iy >= H:
                    continue
                for kx in range(kW):
                    ix = ox + kx - pw
                    if ix < 0 or ix >= W:
                        continue
                    acc += image[iy, ix] * kernel[ky, kx]
            out[oy, ox] = acc
    return out


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("kernel-craft: naive convolution (NumPy interface)")
    print("=" * 60)

    np.random.seed(0)

    # ------------------------------------------------------------------
    # Synthetic 128×128 grayscale image.
    # Replace with a real image loader (e.g. PIL, cv2) in production.
    # ------------------------------------------------------------------
    H, W = 128, 128
    image = np.random.rand(H, W).astype(np.float32)
    print(f"\nInput shape : {image.shape}  dtype={image.dtype}")

    # ------------------------------------------------------------------
    # 3×3 Laplacian edge-detection kernel.
    # Other common choices:
    #   Sobel-X  : [[-1,0,1],[-2,0,2],[-1,0,1]]
    #   Gaussian : (1/16) * [[1,2,1],[2,4,2],[1,2,1]]
    # ------------------------------------------------------------------
    kernel = np.array([
        [ 0,  1,  0],
        [ 1, -4,  1],
        [ 0,  1,  0],
    ], dtype=np.float32)
    print(f"Kernel shape: {kernel.shape}")
    print(f"Kernel (Laplacian edge detection):\n{kernel}")

    # ------------------------------------------------------------------
    # GPU call — returns a NumPy array (H2D + kernel + D2H round-trip).
    # For repeated calls on the same data, prefer the PyTorch tensor path
    # which avoids redundant copies (see example_torch_conv_naive.py).
    # ------------------------------------------------------------------
    print("\nRunning kc.conv_naive on GPU ...")
    result_gpu = kc.conv_naive(image, kernel)
    print(f"Output shape: {result_gpu.shape}")
    print(f"Output range: [{result_gpu.min():.4f}, {result_gpu.max():.4f}]")

    # ------------------------------------------------------------------
    # Correctness check (on a small tile to keep this fast)
    # ------------------------------------------------------------------
    tile = image[:16, :16].copy()
    ref  = _conv_cpu(tile, kernel)
    got  = kc.conv_naive(tile, kernel)
    max_err = float(np.max(np.abs(ref - got)))
    print(f"\nCorrectness (16×16 tile vs CPU): max_err={max_err:.2e}  ",
          end="")
    print("PASS" if max_err < 1e-4 else "FAIL")

    # ------------------------------------------------------------------
    # Optional: save result as a PNG (requires Pillow)
    # ------------------------------------------------------------------
    try:
        from PIL import Image
        out_dir = os.path.join(_REPO, "data", "outputs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "output_naive.png")
        clipped = np.clip(result_gpu, 0, 1)
        Image.fromarray((clipped * 255).astype(np.uint8)).save(out_path)
        print(f"Saved output image → {out_path}")
    except ImportError:
        print("(Pillow not installed — skipping image save)")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
