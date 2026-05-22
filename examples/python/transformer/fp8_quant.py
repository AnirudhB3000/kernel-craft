#!/usr/bin/env python3
"""
example_fp8_quant.py — FP8 (E4M3) quantization and dequantization.

Background — FP8 for LLM inference
------------------------------------
FP8 E4M3 (4 exponent bits, 3 mantissa bits) provides:
  - 8× memory compression vs FP32  (same as INT8)
  - ~2× throughput over FP16 on NVIDIA H100 / Ada Lovelace Tensor Cores
  - Better dynamic range than INT8 (exponent bits vs fixed scale only)
  - ~5.8% mean relative error (3 mantissa bits → 1/8 relative precision)

Supported on: Ada Lovelace (RTX 40xx, L40S), Hopper (H100), Blackwell (B200).
On older GPUs the kernel runs but uses FP32 compute (no FP8 Tensor Core).

API overview
------------
``kernel_craft_transformer.fp8_quantize(input, per_token)``

    Quantises a float32 tensor to FP8 E4M3 using per-token or per-tensor
    scaling.  Per-token scaling computes one scale per row, bounding error
    tightly at the cost of storing one float32 per token.

    Parameters
    ----------
    input     : np.ndarray float32 [rows, cols]
    per_token : bool — True → one scale per row; False → one global scale

    Returns
    -------
    (quantized, scales)
    quantized : np.ndarray uint8 [rows, cols]  — packed FP8 bit patterns
    scales    : np.ndarray float32 [rows] or scalar

``kernel_craft_transformer.fp8_dequantize(quantized, scales, rows, cols, per_token)``

    Reconstructs float32 from FP8 bit patterns and scales.

    Parameters
    ----------
    quantized : np.ndarray uint8 [rows, cols]
    scales    : np.ndarray float32 [rows] or scalar
    rows, cols: int
    per_token : bool

    Returns
    -------
    np.ndarray float32 [rows, cols]

Integration — LLM inference pipeline
--------------------------------------
FP8 is used for both weight quantisation and activation quantisation in
high-throughput inference:

    # Weight quantisation (done once at model load)
    W_fp8, W_scales = kct.fp8_quantize(W_fp32, per_token=False)  # per-tensor scale

    # Per-request: quantise activations and compute GEMM in FP8
    x_fp8, x_scales = kct.fp8_quantize(x_fp32, per_token=True)   # per-token scale
    y_fp32 = kct.fp8_dequantize(x_fp8 @ W_fp8, combined_scales, rows, cols, per_token=True)

SmoothQuant integration
------------------------
Before quantising activations, apply SmoothQuant channel scaling to migrate
quantisation difficulty from activations to weights (improves accuracy):

    activation_smooth = kct.smoothquant_scale(activation, smooth_scales)
    # smooth_scales: per-channel factors from offline calibration
    x_fp8, x_scales = kct.fp8_quantize(activation_smooth, per_token=True)

Usage
-----
    cd examples/python
    python example_fp8_quant.py
"""

import sys
import os

_REPO = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_REPO, "src", "python", "build"))
sys.path.insert(0, os.path.join(_REPO, "src", "python"))

import numpy as np

try:
    import kernel_craft_transformer as kct  # type: ignore
    _HAS_KCT = True
except ImportError:
    _HAS_KCT = False
    print("Warning: kernel_craft_transformer not found — build first.")


def main() -> None:
    print("=" * 60)
    print("kernel-craft: FP8 E4M3 quantization / dequantization")
    print("=" * 60)

    if not _HAS_KCT:
        return

    rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # Simulate an activation tensor: [batch_tokens, hidden_dim]
    # Typical Llama-3-8B: hidden_dim = 4096
    # ------------------------------------------------------------------
    rows, cols = 64, 512   # (tokens, hidden_dim)
    activations = rng.standard_normal((rows, cols)).astype(np.float32) * 2.5
    print(f"\nInput: {rows} tokens × {cols} hidden  dtype=float32")
    print(f"  Range: [{activations.min():.3f}, {activations.max():.3f}]  "
          f"std={activations.std():.3f}")

    # ------------------------------------------------------------------
    # Per-token FP8 quantization (one scale per row → tighter error bound)
    # ------------------------------------------------------------------
    q_pt, scales_pt = kct.fp8_quantize(activations, per_token=True)
    recon_pt = kct.fp8_dequantize(q_pt, scales_pt, rows, cols, per_token=True)

    rel_err_pt = np.abs(activations - recon_pt) / (np.abs(activations) + 1e-6)
    print(f"\n[Per-token FP8]")
    print(f"  Packed shape    : {q_pt.shape}  dtype={q_pt.dtype}  "
          f"({q_pt.nbytes} bytes vs {activations.nbytes} bytes fp32)")
    print(f"  Compression     : {activations.nbytes / q_pt.nbytes:.0f}× over fp32, "
          f"{activations.nbytes // 2 // q_pt.nbytes}× over fp16")
    print(f"  Max rel error   : {rel_err_pt.max():.4f}  ({rel_err_pt.max()*100:.2f}%)")
    print(f"  Mean rel error  : {rel_err_pt.mean():.4f}  ({rel_err_pt.mean()*100:.2f}%)")
    print(f"  Scales shape    : {scales_pt.shape}  (one scale per token row)")

    # ------------------------------------------------------------------
    # Per-tensor FP8 quantization (single global scale → simpler, more error)
    # ------------------------------------------------------------------
    q_tn, scales_tn = kct.fp8_quantize(activations, per_token=False)
    recon_tn = kct.fp8_dequantize(q_tn, scales_tn, rows, cols, per_token=False)

    rel_err_tn = np.abs(activations - recon_tn) / (np.abs(activations) + 1e-6)
    print(f"\n[Per-tensor FP8]")
    print(f"  Mean rel error  : {rel_err_tn.mean():.4f}  ({rel_err_tn.mean()*100:.2f}%)")
    print(f"  (higher than per-token because outlier rows compress the scale)")

    # ------------------------------------------------------------------
    # Comparison: FP8 vs INT8 error characteristics
    # ------------------------------------------------------------------
    # INT8 simulation: uniform quantise to [-127, 127]
    max_abs = np.max(np.abs(activations))
    scale_int8 = max_abs / 127.0
    q_int8 = np.clip(np.round(activations / scale_int8), -127, 127).astype(np.int8)
    recon_int8 = q_int8.astype(np.float32) * scale_int8
    rel_err_int8 = np.abs(activations - recon_int8) / (np.abs(activations) + 1e-6)

    print(f"\n[Comparison: INT8 per-tensor (simulation)]")
    print(f"  Mean rel error  : {rel_err_int8.mean():.4f}  ({rel_err_int8.mean()*100:.2f}%)")
    print(f"  FP8 advantage: dynamic range (exponent bits) handles outliers better.")

    # ------------------------------------------------------------------
    # SmoothQuant demo
    # ------------------------------------------------------------------
    try:
        smooth_scales = rng.uniform(0.5, 2.0, size=cols).astype(np.float32)
        smoothed = kct.smoothquant_scale(activations, smooth_scales)
        q_smooth, sc_smooth = kct.fp8_quantize(smoothed, per_token=True)
        recon_smooth_raw = kct.fp8_dequantize(q_smooth, sc_smooth, rows, cols, per_token=True)
        # Undo smoothing: divide each column by smooth_scales
        recon_smooth = recon_smooth_raw / smooth_scales[np.newaxis, :]
        rel_err_smooth = np.abs(activations - recon_smooth) / (np.abs(activations) + 1e-6)
        print(f"\n[SmoothQuant + Per-token FP8]")
        print(f"  Mean rel error  : {rel_err_smooth.mean():.4f}  "
              f"({rel_err_smooth.mean()*100:.2f}%)")
        print(f"  SmoothQuant migrates quantisation difficulty from activations to weights,")
        print(f"  improving accuracy on transformer models with large activation outliers.")
    except Exception as e:
        print(f"\n[SmoothQuant]: skipped ({e})")

    print("\n  Integration with vLLM:")
    print("  FP8 quantization is enabled via:")
    print("    llm = LLM(model='meta-llama/Llama-3-8B', quantization='fp8')")
    print("  which uses per-tensor scales calibrated on a small dataset.")
    print("  Per-token (dynamic) quantization requires custom kernel-craft backend.")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
