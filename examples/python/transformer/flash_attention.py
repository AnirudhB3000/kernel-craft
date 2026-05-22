#!/usr/bin/env python3
"""
example_flash_attention.py — FlashAttention via kernel_craft_transformer.

API overview
------------
``kernel_craft_transformer.flash_attention(Q, K, V, H_kv, causal)``

    Tiled online-softmax attention.  No N×N score matrix materialised.
    O(N·d) shared-memory usage vs O(N²·H) for standard attention.

    Parameters
    ----------
    Q      : np.ndarray float32 [B, H,    N, d]  — query
    K      : np.ndarray float32 [B, H_kv, N, d]  — key
    V      : np.ndarray float32 [B, H_kv, N, d]  — value
    H_kv   : int — number of KV heads (H_kv = H: MHA, H_kv < H: GQA)
    causal : bool — apply causal (lower-triangular) mask for decoder

    Returns
    -------
    np.ndarray float32 [B, H, N, d]

    Internally: H2D copy → GPU kernel → D2H copy.
    For zero-copy use the PyTorch ctypes bridge in kernel_craft_torch_ops.py.

When to use FlashAttention vs PagedAttention
--------------------------------------------
  FlashAttention   — Prefill stage: full prompt tokens known simultaneously.
                     Seq len N can be large (512–32768 tokens in one shot).

  PagedAttention   — Decode stage: one new token per step.
                     KV cache lives in non-contiguous physical pages.
                     See example_paged_attention.py.

GQA (Grouped-Query Attention) configuration
--------------------------------------------
Llama-3, Mistral, and Falcon use GQA to reduce KV cache size:
    H=32 Q heads, H_kv=8 KV heads  →  4:1 GQA ratio
    Memory saved: 1 - H_kv/H = 75%

    Q, K, V = ..., K shape: [B, 8, N, d], V shape: [B, 8, N, d]
    O = kct.flash_attention(Q, K, V, H_kv=8, causal=True)

Causal masking and efficiency
-------------------------------
With causal=True the kernel skips KV tiles where all query positions are
earlier than all key positions (tile_max stays -inf → entire tile is zero
contribution).  At N=1024 this halves the number of active tiles, giving
~1.4× speedup over non-causal (see CLAUDE.md §Phase 11 benchmarks).

Performance (RTX 4070 Laptop, CUDA 12.0):
    N=512,  d=64, H=8, causal=No    → ~0.80 ms,  ~675 Gflops
    N=1024, d=64, H=8, causal=No    → ~3.08 ms,  ~698 Gflops
    N=1024, d=64, H=8, causal=Yes   → ~2.18 ms,  ~985 Gflops
    N=512,  d=64, H=8, H_kv=2 GQA  → ~0.67 ms,  ~801 Gflops

Usage
-----
    cd examples/python
    python example_flash_attention.py
"""

import sys
import os
import math

_REPO = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_REPO, "src", "python", "build"))
sys.path.insert(0, os.path.join(_REPO, "src", "python"))

import numpy as np

try:
    import kernel_craft_transformer as kct  # type: ignore
    _HAS_KCT = True
except ImportError:
    _HAS_KCT = False
    print("Warning: kernel_craft_transformer not found — "
          "build the project first with cmake --build build")


# ---------------------------------------------------------------------------
# CPU reference (slow — only used on small configs for correctness)
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _attention_ref(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                   causal: bool = False) -> np.ndarray:
    """Scaled dot-product attention on CPU.  Q: [B,H,N,d]."""
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    O = np.zeros_like(Q)
    for b in range(B):
        for h in range(H):
            h_kv = h % K.shape[1]          # GQA head mapping
            S = Q[b, h] @ K[b, h_kv].T * scale   # [N, N]
            if causal:
                mask = np.triu(np.full((N, N), -1e9), k=1)
                S += mask
            P = _softmax(S)                       # [N, N]
            O[b, h] = P @ V[b, h_kv]             # [N, d]
    return O


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("kernel-craft: FlashAttention (transformer prefill)")
    print("=" * 60)

    if not _HAS_KCT:
        return

    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    # Config 1: MHA, non-causal  (encoder-style: BERT, ViT)
    # ------------------------------------------------------------------
    B, H, H_kv, N, d = 1, 8, 8, 128, 64
    Q = rng.standard_normal((B, H, N, d)).astype(np.float32) * 0.02
    K = rng.standard_normal((B, H_kv, N, d)).astype(np.float32) * 0.02
    V = rng.standard_normal((B, H_kv, N, d)).astype(np.float32) * 0.02

    O_gpu = kct.flash_attention(Q, K, V, H_kv=H_kv, causal=False)
    O_ref = _attention_ref(Q, K, V, causal=False)
    err1 = float(np.max(np.abs(O_gpu - O_ref)))

    print(f"\n[MHA non-causal]  B={B} H={H} H_kv={H_kv} N={N} d={d}")
    print(f"  GPU output shape : {O_gpu.shape}")
    print(f"  Max error vs CPU : {err1:.2e}  {'PASS' if err1 < 1e-2 else 'FAIL'}")

    # ------------------------------------------------------------------
    # Config 2: Causal MHA  (decoder-only: GPT, Llama)
    # ------------------------------------------------------------------
    O_gpu_c = kct.flash_attention(Q, K, V, H_kv=H_kv, causal=True)
    O_ref_c = _attention_ref(Q, K, V, causal=True)
    err2 = float(np.max(np.abs(O_gpu_c - O_ref_c)))

    print(f"\n[MHA causal]      B={B} H={H} H_kv={H_kv} N={N} d={d}")
    print(f"  Max error vs CPU : {err2:.2e}  {'PASS' if err2 < 1e-2 else 'FAIL'}")
    print(f"  Note: causal mask skips ~N/2 KV tiles → faster at same N.")

    # ------------------------------------------------------------------
    # Config 3: GQA (Llama-3 style: H=8, H_kv=2)
    # ------------------------------------------------------------------
    H_gqa, H_kv_gqa = 8, 2
    K_gqa = rng.standard_normal((B, H_kv_gqa, N, d)).astype(np.float32) * 0.02
    V_gqa = rng.standard_normal((B, H_kv_gqa, N, d)).astype(np.float32) * 0.02
    O_gpu_gqa = kct.flash_attention(Q, K_gqa, V_gqa, H_kv=H_kv_gqa, causal=True)
    O_ref_gqa = _attention_ref(Q, K_gqa, V_gqa, causal=True)
    err3 = float(np.max(np.abs(O_gpu_gqa - O_ref_gqa)))

    print(f"\n[GQA causal]      B={B} H={H_gqa} H_kv={H_kv_gqa} N={N} d={d}")
    print(f"  GQA ratio        : {H_gqa}/{H_kv_gqa} = {H_gqa//H_kv_gqa}:1")
    print(f"  KV cache savings : {1 - H_kv_gqa/H_gqa:.0%} vs MHA")
    print(f"  Max error vs CPU : {err3:.2e}  {'PASS' if err3 < 1e-2 else 'FAIL'}")

    # ------------------------------------------------------------------
    # Throughput estimate (simple wall-clock, not CUDA events)
    # ------------------------------------------------------------------
    import time
    ITERS = 10
    # Use larger N for meaningful throughput
    B2, H2, N2, d2 = 1, 8, 512, 64
    Q2 = rng.standard_normal((B2, H2, N2, d2)).astype(np.float32) * 0.02
    K2 = rng.standard_normal((B2, H2, N2, d2)).astype(np.float32) * 0.02
    V2 = rng.standard_normal((B2, H2, N2, d2)).astype(np.float32) * 0.02
    kct.flash_attention(Q2, K2, V2, H_kv=H2, causal=True)  # warm-up
    t0 = time.perf_counter()
    for _ in range(ITERS):
        kct.flash_attention(Q2, K2, V2, H_kv=H2, causal=True)
    ms = (time.perf_counter() - t0) * 1e3 / ITERS
    flops = 4.0 * B2 * H2 * N2 * N2 * d2
    gflops = flops / (ms * 1e-3) / 1e9
    print(f"\n[Throughput]  N={N2} H={H2} causal → {ms:.2f} ms/call  {gflops:.0f} Gflops")
    print("  (includes H2D + D2H; see kernel_craft_torch_ops.py for zero-copy path)")

    print("\n  Integration:")
    print("  - Register vLLM backend: import kernel_craft_vllm_backend; kc_backend.register()")
    print("  - Prefill calls flash_attention; decode calls paged_attention")
    print("  - See examples/python/example_paged_attention.py for decode path")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
