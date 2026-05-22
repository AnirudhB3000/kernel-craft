"""
test_mamba_bindings.py
Correctness tests for Phase 14 Mamba SSM Python bindings.

Tests:
  - selective_scan vs numpy sequential reference
  - depthwise_conv1d vs numpy convolution reference
  - rmsnorm vs numpy formula

All tests use numpy-only references (no PyTorch required).
"""

import sys
import os
import math
import pytest
import numpy as np

_REPO = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, os.path.join(_REPO, "src", "python"))
sys.path.insert(0, os.path.join(_REPO, "src", "python", "build"))

# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

try:
    import kernel_craft_transformer as kct
    HAS_KCT = True
    HAS_MAMBA = hasattr(kct, "selective_scan")
except ImportError:
    HAS_KCT = False
    HAS_MAMBA = False

pytestmark = pytest.mark.skipif(
    not HAS_MAMBA,
    reason="kernel_craft_transformer not built or Phase 14 not available"
)

# ---------------------------------------------------------------------------
# CPU references
# ---------------------------------------------------------------------------

def selective_scan_ref(u, A_log, B, C, delta):
    """Sequential Mamba-1 ZOH selective scan on CPU.

    Args:
        u:      [Batch, L, D]
        A_log:  [D, N]
        B:      [Batch, L, N]
        C:      [Batch, L, N]
        delta:  [Batch, L, D]

    Returns:
        y:      [Batch, L, D]
    """
    B_, L, D = u.shape
    N = A_log.shape[1]
    y = np.zeros_like(u)
    for b in range(B_):
        for d in range(D):
            h = np.zeros(N, dtype=np.float32)
            for t in range(L):
                dt = delta[b, t, d]
                a_bar = np.exp(dt * A_log[d])          # [N]
                b_bar = dt * B[b, t]                   # [N]
                h = a_bar * h + b_bar * u[b, t, d]
                y[b, t, d] = float(np.dot(C[b, t], h))
    return y


def depthwise_conv1d_ref(x, w, bias=None):
    """Causal depthwise conv1d reference.

    Args:
        x:    [B, D, L]
        w:    [D, d_conv]
        bias: [D] or None

    Returns:
        y:    [B, D, L]
    """
    B_, D, L = x.shape
    d_conv = w.shape[1]
    y = np.zeros_like(x)
    for b in range(B_):
        for d in range(D):
            for t in range(L):
                val = 0.0
                for k in range(d_conv):
                    src = t - (d_conv - 1 - k)
                    if src >= 0:
                        val += x[b, d, src] * w[d, k]
                if bias is not None:
                    val += bias[d]
                y[b, d, t] = val
    return y


def rmsnorm_ref(x, g, eps=1e-6):
    """RMSNorm reference.

    Args:
        x:   [rows, D]
        g:   [D]
        eps: float

    Returns:
        y:   [rows, D]
    """
    ss = np.mean(x ** 2, axis=-1, keepdims=True)
    rms_inv = 1.0 / np.sqrt(ss + eps)
    return g * x * rms_inv


# ---------------------------------------------------------------------------
# selective_scan tests
# ---------------------------------------------------------------------------

class TestSelectiveScan:

    def _run(self, B_, L, D, N, seed=42):
        rng = np.random.default_rng(seed)
        u     = rng.uniform(-0.5, 0.5, (B_, L, D)).astype(np.float32)
        A_log = rng.uniform(-2.0, -0.01, (D, N)).astype(np.float32)
        B_arr = rng.uniform(0.0, 0.3, (B_, L, N)).astype(np.float32)
        C_arr = rng.uniform(0.0, 0.3, (B_, L, N)).astype(np.float32)
        delta = rng.uniform(0.05, 0.5, (B_, L, D)).astype(np.float32)

        y_ref = selective_scan_ref(u, A_log, B_arr, C_arr, delta)
        y_gpu = kct.selective_scan(u, A_log, B_arr, C_arr, delta)
        return y_ref, y_gpu

    def test_small(self):
        y_ref, y_gpu = self._run(1, 4, 4, 4)
        err = np.max(np.abs(y_ref - y_gpu))
        assert err < 1e-4, f"max_abs_err={err:.2e}"

    def test_typical_mamba(self):
        """B=2, L=64, D=64, N=16 — representative Mamba config."""
        y_ref, y_gpu = self._run(2, 64, 64, 16, seed=7)
        err = np.max(np.abs(y_ref - y_gpu))
        assert err < 1e-4, f"max_abs_err={err:.2e}"

    def test_long_sequence(self):
        """L=1024 — long-context correctness."""
        y_ref, y_gpu = self._run(1, 1024, 32, 16, seed=99)
        err = np.max(np.abs(y_ref - y_gpu))
        assert err < 1e-4, f"max_abs_err={err:.2e}"

    def test_n_state_32(self):
        """Maximum supported N_state=32."""
        y_ref, y_gpu = self._run(1, 32, 16, 32, seed=55)
        err = np.max(np.abs(y_ref - y_gpu))
        assert err < 1e-4, f"max_abs_err={err:.2e}"

    def test_zero_input(self):
        """Zero input → zero output regardless of A/B/C/delta."""
        rng = np.random.default_rng(11)
        B_, L, D, N = 2, 32, 16, 8
        u     = np.zeros((B_, L, D), dtype=np.float32)
        A_log = rng.uniform(-1.0, -0.01, (D, N)).astype(np.float32)
        B_arr = rng.uniform(0.0, 0.5, (B_, L, N)).astype(np.float32)
        C_arr = rng.uniform(0.0, 0.5, (B_, L, N)).astype(np.float32)
        delta = rng.uniform(0.1, 0.5, (B_, L, D)).astype(np.float32)
        y_gpu = kct.selective_scan(u, A_log, B_arr, C_arr, delta)
        assert np.max(np.abs(y_gpu)) < 1e-6, f"Expected zeros, got max={np.max(np.abs(y_gpu)):.2e}"

    def test_output_shape(self):
        """Output shape matches input shape."""
        B_, L, D, N = 3, 16, 8, 8
        rng = np.random.default_rng(0)
        u     = rng.uniform(-1, 1, (B_, L, D)).astype(np.float32)
        A_log = rng.uniform(-1, -0.01, (D, N)).astype(np.float32)
        B_arr = rng.uniform(0, 0.3, (B_, L, N)).astype(np.float32)
        C_arr = rng.uniform(0, 0.3, (B_, L, N)).astype(np.float32)
        delta = rng.uniform(0.1, 0.5, (B_, L, D)).astype(np.float32)
        y = kct.selective_scan(u, A_log, B_arr, C_arr, delta)
        assert y.shape == (B_, L, D), f"Expected {(B_, L, D)}, got {y.shape}"


# ---------------------------------------------------------------------------
# depthwise_conv1d tests
# ---------------------------------------------------------------------------

class TestDepthwiseConv1d:

    def test_basic_single_channel(self):
        """1 channel, d_conv=4, no bias."""
        rng = np.random.default_rng(1)
        B_, D, L, d_conv = 1, 1, 16, 4
        x = rng.uniform(-1, 1, (B_, D, L)).astype(np.float32)
        w = rng.uniform(-1, 1, (D, d_conv)).astype(np.float32)

        y_ref = depthwise_conv1d_ref(x, w)
        y_gpu = kct.depthwise_conv1d(x, w)

        err = np.max(np.abs(y_ref - y_gpu))
        assert err < 1e-5, f"max_abs_err={err:.2e}"

    def test_multichannel_with_bias(self):
        """Multi-channel conv1d with bias."""
        rng = np.random.default_rng(42)
        B_, D, L, d_conv = 2, 32, 64, 4
        x    = rng.uniform(-1, 1, (B_, D, L)).astype(np.float32)
        w    = rng.uniform(-0.5, 0.5, (D, d_conv)).astype(np.float32)
        bias = rng.uniform(-0.1, 0.1, D).astype(np.float32)

        y_ref = depthwise_conv1d_ref(x, w, bias)
        y_gpu = kct.depthwise_conv1d(x, w, bias)

        err = np.max(np.abs(y_ref - y_gpu))
        assert err < 1e-4, f"max_abs_err={err:.2e}"

    def test_causal_boundary(self):
        """At t=0 with d_conv=4, output = x[0]*w[-1] (last weight, causal)."""
        D, L, d_conv = 1, 8, 4
        x = np.ones((1, D, L), dtype=np.float32)
        w = np.arange(1, d_conv + 1, dtype=np.float32).reshape(D, d_conv)

        y_gpu = kct.depthwise_conv1d(x, w)
        # At t=0: only w[d_conv-1]=4 contributes (src=0 ≥ 0); others have src<0
        assert abs(y_gpu[0, 0, 0] - float(w[0, -1])) < 1e-5, (
            f"t=0 expected {w[0,-1]}, got {y_gpu[0,0,0]}")

    def test_kernel_width_1(self):
        """d_conv=1: pointwise multiply per channel."""
        rng = np.random.default_rng(7)
        B_, D, L = 1, 8, 16
        x = rng.uniform(-1, 1, (B_, D, L)).astype(np.float32)
        w = rng.uniform(-1, 1, (D, 1)).astype(np.float32)

        y_ref = depthwise_conv1d_ref(x, w)
        y_gpu = kct.depthwise_conv1d(x, w)

        err = np.max(np.abs(y_ref - y_gpu))
        assert err < 1e-5, f"max_abs_err={err:.2e}"

    def test_output_shape(self):
        rng = np.random.default_rng(0)
        B_, D, L = 3, 16, 32
        x = rng.uniform(-1, 1, (B_, D, L)).astype(np.float32)
        w = rng.uniform(-1, 1, (D, 4)).astype(np.float32)
        y = kct.depthwise_conv1d(x, w)
        assert y.shape == (B_, D, L), f"Expected {(B_, D, L)}, got {y.shape}"


# ---------------------------------------------------------------------------
# rmsnorm tests
# ---------------------------------------------------------------------------

class TestRMSNorm:

    def test_basic(self):
        rng = np.random.default_rng(3)
        rows, D = 8, 64
        x = rng.uniform(-1, 1, (rows, D)).astype(np.float32)
        g = rng.uniform(0.5, 1.5, D).astype(np.float32)

        y_ref = rmsnorm_ref(x, g).astype(np.float32)
        y_gpu = kct.rmsnorm(x, g)

        err = np.max(np.abs(y_ref - y_gpu))
        assert err < 1e-5, f"max_abs_err={err:.2e}"

    def test_large_dim(self):
        rng = np.random.default_rng(77)
        rows, D = 128, 768
        x = rng.uniform(-2, 2, (rows, D)).astype(np.float32)
        g = rng.uniform(0.8, 1.2, D).astype(np.float32)

        y_ref = rmsnorm_ref(x, g).astype(np.float32)
        y_gpu = kct.rmsnorm(x, g)

        err = np.max(np.abs(y_ref - y_gpu))
        assert err < 1e-4, f"max_abs_err={err:.2e}"

    def test_d4096(self):
        """Mamba-2.8B hidden dim."""
        rng = np.random.default_rng(13)
        rows, D = 16, 4096
        x = rng.uniform(-1, 1, (rows, D)).astype(np.float32)
        g = rng.uniform(0.9, 1.1, D).astype(np.float32)

        y_ref = rmsnorm_ref(x, g).astype(np.float32)
        y_gpu = kct.rmsnorm(x, g)

        err = np.max(np.abs(y_ref - y_gpu))
        assert err < 1e-4, f"max_abs_err={err:.2e}"

    def test_constant_input(self):
        """Constant input x=c, g=1 → y=1 everywhere."""
        rows, D = 4, 64
        x = np.full((rows, D), 2.0, dtype=np.float32)
        g = np.ones(D, dtype=np.float32)
        y_gpu = kct.rmsnorm(x, g)
        err = np.max(np.abs(y_gpu - 1.0))
        assert err < 1e-5, f"Expected ones, max_err={err:.2e}"

    def test_unit_norm(self):
        """After RMSNorm with g=1, rms(y) ≈ 1."""
        rng = np.random.default_rng(22)
        rows, D = 32, 256
        x = rng.uniform(-1, 1, (rows, D)).astype(np.float32)
        g = np.ones(D, dtype=np.float32)
        y_gpu = kct.rmsnorm(x, g)
        rms_y = np.sqrt(np.mean(y_gpu ** 2, axis=-1))
        err = np.max(np.abs(rms_y - 1.0))
        assert err < 1e-4, f"rms of output not ≈ 1, max_err={err:.2e}"

    def test_output_shape(self):
        rng = np.random.default_rng(0)
        rows, D = 7, 128
        x = rng.uniform(-1, 1, (rows, D)).astype(np.float32)
        g = np.ones(D, dtype=np.float32)
        y = kct.rmsnorm(x, g)
        assert y.shape == (rows, D), f"Expected {(rows, D)}, got {y.shape}"
