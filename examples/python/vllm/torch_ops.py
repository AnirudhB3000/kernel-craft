#!/usr/bin/env python3
"""
example_torch_ops.py — kernel-craft ops via the torch.ops.kernel_craft namespace.

Background
-----------
``src/python/kernel_craft_torch_ops.py`` registers all kernel-craft launchers
as first-class PyTorch custom operators under the ``kernel_craft`` namespace
using ``torch.library``.  This gives them the same composability properties
as built-in PyTorch ops:

  - torch.compile / dynamo tracing
  - torch.export / ONNX export (with custom op support)
  - torch.jit.script (limited)
  - Proper device dispatch (CPU stub falls back gracefully)
  - Device placement in ``torch.nn.Module.to(device)``

API — all ops under torch.ops.kernel_craft
-------------------------------------------
``torch.ops.kernel_craft.flash_attention(Q, K, V, H_kv, causal)``
    FlashAttention prefill.  All tensors must be CUDA float32.
    Q: [B, H, N, d]  K: [B, H_kv, N, d]  V: [B, H_kv, N, d]
    → [B, H, N, d]

``torch.ops.kernel_craft.paged_attention(Q, block_table, K_pool, V_pool, seq_lens, H_kv, block_size)``
    PagedAttention decode.  Q: [B, H, d]  K_pool: [P, bs, H_kv, d]
    → [B, H, d]

``torch.ops.kernel_craft.int4_dequantize(packed, scales, zeros, rows, cols, group_size)``
    Dequantise GPTQ/AWQ INT4 weights to FP32.
    → float32 [rows, cols]

``torch.ops.kernel_craft.int4_gemv(packed, scales, zeros, x, rows, cols, group_size)``
    On-the-fly dequant + matvec:  y = W_int4 @ x
    → float32 [rows]

``torch.ops.kernel_craft.fp8_quantize(x, per_token)``
    → (quantized uint8, scales float32)

``torch.ops.kernel_craft.fp8_dequantize(q, scales, rows, cols, per_token)``
    → float32 [rows, cols]

``torch.ops.kernel_craft.speculative_verify(draft_probs, target_probs,
                                            draft_tokens, rand_vals, rand_vals2)``
    → (accepted int32, corrected int32)

Why use torch.ops instead of direct imports?
---------------------------------------------
1. **Composability**: ops can be used inside torch.compile graphs.
2. **Type safety**: PyTorch validates tensor shapes and dtypes before dispatch.
3. **Fake tensor support**: works in torchdynamo's symbolic tracing mode.
4. **Op registry**: visible to tools like torch.fx, Netron, custom profilers.

The ctypes bridge (kernel_craft_torch_ops.py) also exposes the same functions
as plain Python callables (flash_attention, paged_attention, etc.) for
frameworks that do not need torch.ops composability.

Usage
-----
    cd examples/python
    python example_torch_ops.py

    # Or, in a Python session:
    import kernel_craft_torch_ops          # registers torch.ops.kernel_craft
    O = torch.ops.kernel_craft.flash_attention(Q, K, V, H_kv=8, causal=True)
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
    print("PyTorch not installed — cannot run this example.")

try:
    import kernel_craft_torch_ops as kc_ops  # type: ignore  # noqa: F401
    _HAS_OPS = True
except Exception as e:
    _HAS_OPS = False
    print(f"kernel_craft_torch_ops not available: {e}")
    print("Build libkernels.so first: cd build && make kernels -j$(nproc)")


def _header(name: str) -> None:
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")


def main() -> None:
    print("=" * 60)
    print("kernel-craft: torch.ops.kernel_craft namespace")
    print("=" * 60)

    if not (_HAS_TORCH and _HAS_OPS):
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}  (ops fallback to CPU stubs when CUDA unavailable)")
    print(f"TORCH_OPS_REGISTERED: {getattr(kc_ops, 'TORCH_OPS_REGISTERED', False)}")
    print("\nAvailable ops:")
    for op in dir(torch.ops.kernel_craft):
        if not op.startswith("_"):
            print(f"  torch.ops.kernel_craft.{op}")

    # ------------------------------------------------------------------
    # FlashAttention via torch.ops
    # ------------------------------------------------------------------
    _header("flash_attention (prefill)")
    B, H, H_kv, N, d = 1, 4, 4, 64, 64
    Q = torch.randn(B, H, N, d, device=device) * 0.02
    K = torch.randn(B, H_kv, N, d, device=device) * 0.02
    V = torch.randn(B, H_kv, N, d, device=device) * 0.02

    O = torch.ops.kernel_craft.flash_attention(Q, K, V, H_kv, causal=True)
    print(f"  Q: {tuple(Q.shape)}  →  O: {tuple(O.shape)}  dtype={O.dtype}")
    print(f"  Output finite: {O.isfinite().all().item()}")

    # ------------------------------------------------------------------
    # INT4 dequant via torch.ops
    # ------------------------------------------------------------------
    _header("int4_dequantize (weight decompression)")
    rows, cols, group_size = 256, 512, 128
    n_groups = cols // group_size
    # Packed: 2 int4 per byte → cols/2 bytes per row
    packed  = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, device=device)
    scales  = torch.rand(rows, n_groups, device=device) * 0.1
    zeros   = torch.randint(0, 128, (rows, n_groups // 2), dtype=torch.uint8, device=device)

    W_fp32 = torch.ops.kernel_craft.int4_dequantize(
        packed, scales, zeros, rows, cols, group_size)
    print(f"  packed: {tuple(packed.shape)} uint8  →  W: {tuple(W_fp32.shape)} float32")
    print(f"  Compression: {W_fp32.nbytes}/{packed.nbytes} = {W_fp32.nbytes/packed.nbytes:.1f}×")
    print(f"  W range: [{W_fp32.min():.4f}, {W_fp32.max():.4f}]")

    # ------------------------------------------------------------------
    # INT4 GEMV via torch.ops (on-the-fly dequant + matvec)
    # ------------------------------------------------------------------
    _header("int4_gemv (decode matmul)")
    x_vec = torch.randn(cols, device=device)
    y = torch.ops.kernel_craft.int4_gemv(
        packed, scales, zeros, x_vec, rows, cols, group_size)
    print(f"  x: [{cols}]  W_int4: [{rows},{cols}]  →  y: {tuple(y.shape)}")
    # Verify vs dequant + torch matmul
    y_ref = W_fp32 @ x_vec
    err = (y - y_ref).abs().max().item()
    print(f"  Max error vs W_fp32 @ x: {err:.4f}  (expected ~0 with same dequant)")

    # ------------------------------------------------------------------
    # FP8 round-trip via torch.ops
    # ------------------------------------------------------------------
    _header("fp8_quantize / fp8_dequantize")
    act_rows, act_cols = 32, 256
    activations = torch.randn(act_rows, act_cols, device=device)
    q_fp8, scales_fp8 = torch.ops.kernel_craft.fp8_quantize(activations, True)
    recon = torch.ops.kernel_craft.fp8_dequantize(
        q_fp8, scales_fp8, act_rows, act_cols, True)
    rel_err = ((activations - recon).abs() / (activations.abs() + 1e-6)).mean().item()
    print(f"  Input: {tuple(activations.shape)} float32")
    print(f"  FP8 quantized: {tuple(q_fp8.shape)} uint8")
    print(f"  Mean relative error: {rel_err*100:.2f}%  (expected ~5-6%)")

    # ------------------------------------------------------------------
    # torch.compile compatibility
    # ------------------------------------------------------------------
    _header("torch.compile compatibility (dynamo trace)")
    try:
        @torch.compile(fullgraph=False)
        def compiled_attn(Q, K, V):
            return torch.ops.kernel_craft.flash_attention(Q, K, V, H_kv, causal=False)

        O_compiled = compiled_attn(Q, K, V)
        print(f"  torch.compile succeeded.  O: {tuple(O_compiled.shape)}")
        diff = (O_compiled - O).abs().max().item()
        print(f"  Max diff compiled vs eager: {diff:.2e}")
    except Exception as e:
        print(f"  torch.compile: {e}")
        print(f"  (Custom ops need fake tensor registration for full dynamo support)")

    print("\n  Integration:")
    print("  import kernel_craft_torch_ops   # once at startup")
    print("  # Then use anywhere PyTorch tensors flow:")
    print("  O = torch.ops.kernel_craft.flash_attention(Q, K, V, H_kv=8, causal=True)")
    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
