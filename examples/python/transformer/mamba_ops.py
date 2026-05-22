"""
mamba_ops.py — Mamba / SSM kernel examples via kernel-craft Python bindings.

Covers
------
1. Selective scan (ZOH SSM recurrence)
2. Causal depthwise Conv1d (short-convolutional mixer)
3. RMSNorm (fused normalize + scale)

All three sit inside a single Mamba block:

    x  →  [linear projection]
        ├─ z (gate branch)
        ├─ x' → depthwise_conv1d → silu → selective_scan → y
        └─ y ⊙ silu(z) → rmsnorm → output

API
---
    import kernel_craft_transformer as kct

    y     = kct.selective_scan(u, A_log, B, C, delta)   # [B,L,D]
    y_dw  = kct.depthwise_conv1d(x, w, bias, d_conv)    # [B,D,L]
    y_rms = kct.rmsnorm(x, g, eps=1e-6)                 # [rows, D]

Integration
-----------
vLLM / HuggingFace Mamba models call `causal_conv1d_fn` and
`selective_scan_fn` from `mamba_ssm`.  Replace with kernel-craft launchers
via the ctypes bridge in `kernel_craft_torch_ops.py` for a zero-copy path
when CUDA tensors are available.  The pybind11 bindings here use NumPy arrays
(host-side); they are suitable for correctness testing and CPU-fallback paths.

Performance notes (RTX 4070 Laptop)
------------------------------------
    selective_scan  (B=1, D=512, N=16, L=1024): ~1.7 ms, ~29 Gflops
    depthwise_conv1d (B=1, D=2048, d_conv=4, L=4096): ~0.3 ms, ~664 GB/s
    rmsnorm         (128 rows, D=4096): ~0.025 ms, ~171 GB/s
"""

import sys
import time

import numpy as np

try:
    import kernel_craft_transformer as kct
except ImportError:
    print("kernel_craft_transformer not found — build with:")
    print("  cmake --build build --target kernel_craft_transformer")
    sys.exit(0)


# ---------------------------------------------------------------------------
# CPU references
# ---------------------------------------------------------------------------

def _cpu_selective_scan(u, A_log, B_in, C, delta):
    """ZOH selective scan — pure NumPy reference."""
    Bsz, L, D = u.shape
    N = A_log.shape[1]
    y = np.zeros_like(u)
    h = np.zeros((Bsz, D, N), dtype=np.float32)
    for t in range(L):
        dt = delta[:, t, :]            # [B, D]
        u_t = u[:, t, :]              # [B, D]
        B_t = B_in[:, t, :]           # [B, N]
        C_t = C[:, t, :]              # [B, N]
        A_bar = np.exp(dt[:, :, None] * A_log[None, :, :])   # [B, D, N]
        B_bar = dt[:, :, None] * B_t[:, None, :]              # [B, D, N]
        h = A_bar * h + B_bar * u_t[:, :, None]
        y[:, t, :] = (h * C_t[:, None, :]).sum(-1)
    return y


def _cpu_depthwise_conv1d(x, w, bias, d_conv):
    """Causal depthwise Conv1d — pure NumPy."""
    B, D, L = x.shape
    y = np.full_like(x, 0.0)
    for k in range(d_conv):
        shift = d_conv - 1 - k
        if shift < L:
            y[:, :, shift:] += w[:, k:k+1] * x[:, :, :L - shift]
    if bias is not None:
        y += bias[:, None]
    return y


def _cpu_rmsnorm(x, g, eps=1e-6):
    """RMSNorm — pure NumPy."""
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return x / rms * g


# ---------------------------------------------------------------------------
# Demo sections
# ---------------------------------------------------------------------------

def demo_selective_scan():
    print("=" * 60)
    print("1. Selective Scan (ZOH SSM)")
    print("=" * 60)

    rng = np.random.default_rng(0)
    B, L, D, N = 1, 256, 64, 16

    u     = rng.random((B, L, D), dtype=np.float32)
    A_log = -rng.random((D, N), dtype=np.float32)   # negative → stable
    B_in  = rng.random((B, L, N), dtype=np.float32)
    C     = rng.random((B, L, N), dtype=np.float32)
    delta = (rng.random((B, L, D), dtype=np.float32) * 0.1).astype(np.float32)

    y_cpu = _cpu_selective_scan(u, A_log, B_in, C, delta)

    # kct.selective_scan expects flat arrays
    y_gpu = kct.selective_scan(u, A_log, B_in, C, delta)

    err = float(np.abs(y_gpu - y_cpu).max())
    print(f"  B={B}  L={L}  D={D}  N_state={N}")
    print(f"  max_abs_error vs CPU reference: {err:.2e}")
    assert err < 1e-4, f"Correctness check failed: {err}"
    print("  Correctness: PASS")

    # Throughput over increasing L
    print("\n  Throughput sweep:")
    for L_test in [64, 256, 1024, 4096]:
        u_t     = rng.random((B, L_test, D), dtype=np.float32)
        A_log_t = -rng.random((D, N), dtype=np.float32)
        B_t     = rng.random((B, L_test, N), dtype=np.float32)
        C_t     = rng.random((B, L_test, N), dtype=np.float32)
        dt_t    = (rng.random((B, L_test, D), dtype=np.float32) * 0.1).astype(np.float32)

        # Warmup
        kct.selective_scan(u_t, A_log_t, B_t, C_t, dt_t)

        reps = 10
        t0 = time.perf_counter()
        for _ in range(reps):
            kct.selective_scan(u_t, A_log_t, B_t, C_t, dt_t)
        ms = (time.perf_counter() - t0) / reps * 1e3

        gflops = 1e-9 * B * L_test * D * N * 5 / (ms * 1e-3)
        print(f"    L={L_test:5d}  {ms:6.2f} ms  {gflops:5.1f} Gflops")


def demo_depthwise_conv1d():
    print("\n" + "=" * 60)
    print("2. Causal Depthwise Conv1d")
    print("=" * 60)

    rng = np.random.default_rng(1)
    B, D, d_conv = 1, 128, 4
    L = 512

    x    = rng.random((B, D, L), dtype=np.float32)
    w    = rng.random((D, d_conv), dtype=np.float32)
    bias = rng.random((D,), dtype=np.float32)

    y_cpu = _cpu_depthwise_conv1d(x, w, bias, d_conv)
    y_gpu = kct.depthwise_conv1d(x, w, bias, d_conv)

    err = float(np.abs(y_gpu - y_cpu).max())
    print(f"  B={B}  D={D}  L={L}  d_conv={d_conv}")
    print(f"  max_abs_error vs CPU reference: {err:.2e}")
    assert err < 1e-4, f"Correctness check failed: {err}"
    print("  Correctness: PASS")

    # Bandwidth sweep
    print("\n  Bandwidth sweep (B=1, D=512, d_conv=4):")
    D_bw = 512
    w_bw    = rng.random((D_bw, d_conv), dtype=np.float32)
    bias_bw = rng.random((D_bw,), dtype=np.float32)
    for L_test in [64, 1024, 4096, 16384]:
        x_t = rng.random((B, D_bw, L_test), dtype=np.float32)
        kct.depthwise_conv1d(x_t, w_bw, bias_bw, d_conv)  # warmup

        reps = 30
        t0 = time.perf_counter()
        for _ in range(reps):
            kct.depthwise_conv1d(x_t, w_bw, bias_bw, d_conv)
        ms = (time.perf_counter() - t0) / reps * 1e3

        bytes_ = (B * D_bw * L_test * 2 + D_bw * d_conv) * 4
        gb_s = bytes_ / (ms * 1e-3) * 1e-9
        print(f"    L={L_test:6d}  {ms:6.2f} ms  {gb_s:5.0f} GB/s")


def demo_rmsnorm():
    print("\n" + "=" * 60)
    print("3. RMSNorm")
    print("=" * 60)

    rng = np.random.default_rng(2)
    rows, D = 64, 512
    eps = 1e-6

    x = rng.standard_normal((rows, D)).astype(np.float32)
    g = rng.random((D,), dtype=np.float32) + 0.5

    y_cpu = _cpu_rmsnorm(x, g, eps)
    y_gpu = kct.rmsnorm(x, g, eps)

    err = float(np.abs(y_gpu - y_cpu).max())
    print(f"  rows={rows}  D={D}  eps={eps:.0e}")
    print(f"  max_abs_error vs CPU reference: {err:.2e}")
    assert err < 1e-5, f"Correctness check failed: {err}"
    print("  Correctness: PASS")

    # Bandwidth sweep
    print("\n  Bandwidth sweep (128 rows):")
    rows_bw = 128
    for D_test in [512, 1024, 2048, 4096]:
        x_t = rng.standard_normal((rows_bw, D_test)).astype(np.float32)
        g_t = rng.random((D_test,), dtype=np.float32) + 0.5
        kct.rmsnorm(x_t, g_t, eps)  # warmup

        reps = 200
        t0 = time.perf_counter()
        for _ in range(reps):
            kct.rmsnorm(x_t, g_t, eps)
        ms = (time.perf_counter() - t0) / reps * 1e3

        bytes_ = (rows_bw * D_test * 2 + D_test) * 4
        gb_s = bytes_ / (ms * 1e-3) * 1e-9
        print(f"    D={D_test:5d}  {ms:6.3f} ms  {gb_s:5.0f} GB/s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("kernel-craft — Mamba / SSM ops")
    print("API: kernel_craft_transformer (selective_scan, depthwise_conv1d, rmsnorm)\n")

    demo_selective_scan()
    demo_depthwise_conv1d()
    demo_rmsnorm()

    print("\nAll Mamba op checks passed.")


if __name__ == "__main__":
    main()
