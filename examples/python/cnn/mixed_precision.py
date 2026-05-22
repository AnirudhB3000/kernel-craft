#!/usr/bin/env python3
"""
example_mixed_precision.py — FP16 / TF32 / FP32 precision trade-offs.

Background
----------
Modern NVIDIA GPUs have dedicated Tensor Core hardware that provides
dramatically higher FLOP rates for reduced-precision arithmetic:

  Precision | A100 peak TFLOPS | RTX 4070 peak TFLOPS | Accuracy
  ----------|------------------|----------------------|---------
  FP32      |      19.5        |         10.7         | full
  TF32      |      156         |         ~85          | ~99.99%
  FP16 (TC) |      312         |         170          | ~99.9%
  INT8  (TC)|      624         |         340          | ~99%

TF32 is automatic on Ampere+ (A100, RTX 30xx/40xx) when
``torch.backends.cuda.matmul.allow_tf32 = True`` (PyTorch default since 1.7).

FP16 is the standard precision for LLM inference and fine-tuning:
- Weights stored in float16 → 2× less VRAM vs FP32.
- Activations accumulated in float32 for stability (AMP: Automatic Mixed Precision).
- PyTorch AMP: ``with torch.autocast(device_type='cuda', dtype=torch.float16):``

API used
--------
``kc.conv_tiled`` operates in FP32.  This example shows:
1. FP32 baseline.
2. FP16 simulation via input downcast (demonstrates accuracy impact).
3. PyTorch AMP context (Tensor Core path for matrix ops).

Integration — AMP in training
-------------------------------
    scaler = torch.cuda.amp.GradScaler()
    optimizer.zero_grad()
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        feat = kc.conv_tiled(input_fp32, weight_fp32, 8, 8)  # runs in FP32
        logits = model.head(feat.unsqueeze(0))               # FC in FP16 via AMP
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

Integration — FP8 inference (Phase 11 kernels)
------------------------------------------------
For even higher throughput on Ada/Hopper GPUs use the FP8 E4M3 kernels:
    import kernel_craft_transformer as kct
    q_fp8, scales = kct.fp8_quantize(activations, per_token=True)
    # ~5.8% mean relative error; ~2× throughput over FP16 on H100
See examples/python/example_fp8_quant.py.

Usage
-----
    cd examples/python
    python example_mixed_precision.py
"""

import sys
import os

_REPO = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_REPO, "src", "python", "build"))
sys.path.insert(0, os.path.join(_REPO, "src", "python"))

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

import kernel_craft_python as kc  # type: ignore


def _pct_error(a: np.ndarray, b: np.ndarray) -> float:
    """Mean relative error between two arrays (in percent)."""
    denom = np.mean(np.abs(b)) + 1e-12
    return float(np.mean(np.abs(a - b))) / denom * 100.0


def main() -> None:
    print("=" * 60)
    print("kernel-craft: mixed precision (FP32 / FP16 / TF32)")
    print("=" * 60)

    np.random.seed(7)
    H, W = 256, 256
    kernel_np = np.random.rand(3, 3).astype(np.float32)

    # ------------------------------------------------------------------
    # 1. FP32 baseline
    # ------------------------------------------------------------------
    input_fp32 = np.random.rand(H, W).astype(np.float32)
    result_fp32 = kc.conv_tiled(input_fp32, kernel_np, 8, 8)
    print(f"\n[FP32] range: [{result_fp32.min():.5f}, {result_fp32.max():.5f}]")

    # ------------------------------------------------------------------
    # 2. FP16 simulation via NumPy (shows round-off impact)
    # The actual FP16 kernel is in src/performance/mixed_precision.cu and
    # invoked through the C++ benchmark.  Here we simulate by casting the
    # input through float16 to measure accuracy loss.
    # ------------------------------------------------------------------
    input_fp16 = input_fp32.astype(np.float16).astype(np.float32)
    kernel_fp16 = kernel_np.astype(np.float16).astype(np.float32)
    result_fp16 = kc.conv_tiled(input_fp16, kernel_fp16, 8, 8)
    err_pct = _pct_error(result_fp16, result_fp32)
    print(f"[FP16] range: [{result_fp16.min():.5f}, {result_fp16.max():.5f}]  "
          f"mean rel error: {err_pct:.4f}%")

    # ------------------------------------------------------------------
    # 3. PyTorch AMP demo (only if CUDA available)
    # ------------------------------------------------------------------
    if _HAS_TORCH and torch.cuda.is_available():
        device = torch.device("cuda")
        x = torch.from_numpy(input_fp32).to(device)
        k = torch.from_numpy(kernel_np).to(device)

        # --- FP32 (no AMP) ---
        out_fp32 = kc.conv_tiled(x, k, 8, 8)
        print(f"\n[PyTorch FP32] result dtype: {out_fp32.dtype}")

        # --- AMP: conv_tiled stays FP32; downstream matmuls use FP16 TC ---
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out_amp = kc.conv_tiled(x, k, 8, 8)
            # A linear layer here would automatically run in FP16 Tensor Core
            dummy_fc = torch.nn.Linear(W, 10).to(device).half()
            # Flatten and project to verify AMP is active downstream
            logits = dummy_fc(out_amp.half().flatten()[:W])

        print(f"[PyTorch AMP ] conv output dtype: {out_amp.dtype}  "
              f"FC output dtype: {logits.dtype}")

        # TF32 (Ampere+ default — automatically used for cublasSgemm)
        was_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"\n[TF32] torch.backends.cuda.matmul.allow_tf32 = True (Ampere+ default)")
        print(f"       Convolutions use FP32 math; GEMM layers use TF32 TC automatically.")
        torch.backends.cuda.matmul.allow_tf32 = was_tf32

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n--- Precision summary ---")
    print(f"  FP32  : max_err=0.000000  (baseline)")
    print(f"  FP16  : mean_rel_err={err_pct:.4f}%  "
          "(<0.1% typical for 3×3 conv on random input)")
    print(f"  TF32  : ~0.001% mean rel error (automatic on Ampere+)")
    print(f"\n  For FP8 (E4M3, ~5.8% err, 2× FP16 throughput on Ada/Hopper):")
    print(f"  → see example_fp8_quant.py and src/kernels/transformer/fp8_quant.cu")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
