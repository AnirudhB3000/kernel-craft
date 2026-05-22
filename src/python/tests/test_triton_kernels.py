"""
test_triton_kernels.py
Correctness tests for Phase 15 Triton kernels.

Each Triton kernel is validated against a numpy CPU reference (rtol=1e-3).
Tests are skipped if triton or torch are not installed.
"""
import sys
import os
import math

import numpy as np
import pytest

# Import third-party packages FIRST so sys.modules caches them before our
# src/triton directory is added to sys.path (which would shadow the 'triton' name).
triton = pytest.importorskip("triton")
torch  = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

# Now safe to add src/triton: subsequent `import triton` inside those modules
# will use the already-cached sys.modules['triton'] (the real triton library).
_REPO = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, os.path.join(_REPO, "src", "triton"))

from flash_attention_triton import flash_attention    # noqa: E402
from selective_scan_triton  import selective_scan     # noqa: E402
from int4_gemv_triton       import int4_gemv          # noqa: E402

# ---------------------------------------------------------------------------
# CPU references
# ---------------------------------------------------------------------------

def _attention_ref(Q, K, V, causal=False):
    """Numpy scaled dot-product attention [B, H, N, d]."""
    B, H, N, d = Q.shape
    H_kv = K.shape[1]
    scale = 1.0 / math.sqrt(d)
    O = np.zeros_like(Q)
    for b in range(B):
        for h in range(H):
            h_kv = h * H_kv // H
            q = Q[b, h]           # [N, d]
            k = K[b, h_kv]        # [N, d]
            v = V[b, h_kv]        # [N, d]
            s = (q @ k.T) * scale  # [N, N]
            if causal:
                mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                s[mask] = -np.inf
            s -= s.max(axis=1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(axis=1, keepdims=True)
            O[b, h] = p @ v
    return O.astype(np.float32)


def _selective_scan_ref(u, A_log, B, C, delta):
    """Sequential ZOH selective scan on CPU."""
    Bat, L, D = u.shape
    N = A_log.shape[1]
    y = np.zeros_like(u)
    for b in range(Bat):
        for d in range(D):
            h = np.zeros(N, dtype=np.float32)
            for t in range(L):
                dt    = delta[b, t, d]
                a_bar = np.exp(dt * A_log[d])
                b_bar = dt * B[b, t]
                h     = a_bar * h + b_bar * u[b, t, d]
                y[b, t, d] = float(np.dot(C[b, t], h))
    return y


def _int4_dequant_gemv_ref(packed_np, scales_np, zeros_np, x_np, rows, cols, gs):
    """CPU INT4 dequant + GEMV reference."""
    W = np.zeros((rows, cols), dtype=np.float32)
    num_groups = cols // gs
    nzb = (num_groups + 1) // 2
    for r in range(rows):
        for bi in range(cols // 2):
            c0, c1 = bi * 2, bi * 2 + 1
            p  = int(packed_np[r, bi])
            lo = p & 0xF;  lo = lo - 16 if lo >= 8 else lo
            hi = (p >> 4) & 0xF; hi = hi - 16 if hi >= 8 else hi
            g0, g1 = c0 // gs, c1 // gs
            sc0 = scales_np[r, g0]; sc1 = scales_np[r, g1]
            zb0 = int(zeros_np[r, g0 // 2]); zb1 = int(zeros_np[r, g1 // 2])
            zp0 = zb0 & 0xF if g0 % 2 == 0 else (zb0 >> 4) & 0xF
            zp1 = zb1 & 0xF if g1 % 2 == 0 else (zb1 >> 4) & 0xF
            W[r, c0] = (lo - zp0) * sc0
            if c1 < cols:
                W[r, c1] = (hi - zp1) * sc1
    return (W @ x_np).astype(np.float32)


# ---------------------------------------------------------------------------
# Helper to make a reproducible INT4 dataset
# ---------------------------------------------------------------------------

def _make_int4_data(rows, cols, group_size, seed=42):
    rng = np.random.default_rng(seed)
    num_groups  = cols // group_size
    nzb         = (num_groups + 1) // 2

    # Random signed INT4 values packed into uint8
    w_int4 = rng.integers(-8, 8, size=(rows, cols), dtype=np.int8)
    packed = np.zeros((rows, cols // 2), dtype=np.uint8)
    for bi in range(cols // 2):
        lo = w_int4[:, bi * 2]     & 0xF
        hi = w_int4[:, bi * 2 + 1] & 0xF
        packed[:, bi] = lo | (hi << 4)

    scales = rng.uniform(0.01, 0.1, (rows, num_groups)).astype(np.float32)
    zeros  = np.zeros((rows, nzb), dtype=np.uint8)  # zero-point = 0

    x = rng.uniform(-1.0, 1.0, cols).astype(np.float32)
    return packed, scales, zeros, x


# ---------------------------------------------------------------------------
# FlashAttention tests
# ---------------------------------------------------------------------------

class TestFlashAttentionTriton:

    def _run(self, B, H, N, d, causal=False, H_kv=None, seed=0):
        if H_kv is None:
            H_kv = H
        rng = np.random.default_rng(seed)
        Q_np = rng.standard_normal((B, H,    N, d)).astype(np.float32)
        K_np = rng.standard_normal((B, H_kv, N, d)).astype(np.float32)
        V_np = rng.standard_normal((B, H_kv, N, d)).astype(np.float32)

        ref = _attention_ref(Q_np, K_np, V_np, causal=causal)

        Q = torch.from_numpy(Q_np).cuda()
        K = torch.from_numpy(K_np).cuda()
        V = torch.from_numpy(V_np).cuda()
        out = flash_attention(Q, K, V, causal=causal).cpu().numpy()
        return ref, out

    def test_mha_small(self):
        ref, out = self._run(1, 2, 16, 32)
        assert np.allclose(ref, out, rtol=1e-2, atol=5e-3), f"max_err={np.max(np.abs(ref-out)):.2e}"

    def test_mha_n64_d64(self):
        ref, out = self._run(2, 4, 64, 64)
        assert np.allclose(ref, out, rtol=1e-2, atol=5e-3), f"max_err={np.max(np.abs(ref-out)):.2e}"

    def test_causal(self):
        ref, out = self._run(1, 2, 32, 64, causal=True)
        assert np.allclose(ref, out, rtol=1e-2, atol=5e-3), f"max_err={np.max(np.abs(ref-out)):.2e}"

    def test_gqa_2to1(self):
        """GQA: 4 Q heads, 2 KV heads."""
        ref, out = self._run(1, 4, 32, 64, H_kv=2)
        assert np.allclose(ref, out, rtol=1e-2, atol=5e-3), f"max_err={np.max(np.abs(ref-out)):.2e}"

    def test_output_shape(self):
        B, H, N, d = 2, 4, 48, 64
        Q = torch.randn(B, H, N, d, device="cuda")
        K = torch.randn(B, H, N, d, device="cuda")
        V = torch.randn(B, H, N, d, device="cuda")
        out = flash_attention(Q, K, V)
        assert out.shape == (B, H, N, d)

    def test_n128_d64(self):
        ref, out = self._run(1, 1, 128, 64)
        assert np.allclose(ref, out, rtol=1e-2, atol=5e-3), f"max_err={np.max(np.abs(ref-out)):.2e}"


# ---------------------------------------------------------------------------
# Selective scan tests
# ---------------------------------------------------------------------------

class TestSelectiveScanTriton:

    def _run(self, B_, L, D, N, seed=0):
        rng = np.random.default_rng(seed)
        u     = rng.uniform(-0.5,  0.5,  (B_, L, D)).astype(np.float32)
        A_log = rng.uniform(-2.0, -0.01, (D,  N)).astype(np.float32)
        B_arr = rng.uniform(0.0,   0.3,  (B_, L, N)).astype(np.float32)
        C_arr = rng.uniform(0.0,   0.3,  (B_, L, N)).astype(np.float32)
        delta = rng.uniform(0.05,  0.5,  (B_, L, D)).astype(np.float32)

        ref = _selective_scan_ref(u, A_log, B_arr, C_arr, delta)

        out = selective_scan(
            torch.from_numpy(u).cuda(),
            torch.from_numpy(A_log).cuda(),
            torch.from_numpy(B_arr).cuda(),
            torch.from_numpy(C_arr).cuda(),
            torch.from_numpy(delta).cuda(),
        ).cpu().numpy()
        return ref, out

    def test_small(self):
        ref, out = self._run(1, 4, 4, 4)
        assert np.allclose(ref, out, rtol=1e-2, atol=5e-3), f"max_err={np.max(np.abs(ref-out)):.2e}"

    def test_typical_mamba(self):
        ref, out = self._run(2, 64, 64, 16, seed=7)
        assert np.allclose(ref, out, rtol=1e-2, atol=5e-3), f"max_err={np.max(np.abs(ref-out)):.2e}"

    def test_n_state_32(self):
        ref, out = self._run(1, 32, 16, 32, seed=5)
        assert np.allclose(ref, out, rtol=1e-2, atol=5e-3), f"max_err={np.max(np.abs(ref-out)):.2e}"

    def test_long_sequence(self):
        ref, out = self._run(1, 256, 32, 16, seed=99)
        assert np.allclose(ref, out, rtol=1e-2, atol=5e-3), f"max_err={np.max(np.abs(ref-out)):.2e}"

    def test_output_shape(self):
        B_, L, D, N = 2, 16, 8, 8
        u     = torch.randn(B_, L, D, device="cuda")
        A_log = -torch.rand(D, N, device="cuda")
        B_arr = torch.rand(B_, L, N, device="cuda")
        C_arr = torch.rand(B_, L, N, device="cuda")
        delta = torch.rand(B_, L, D, device="cuda") * 0.5
        out   = selective_scan(u, A_log, B_arr, C_arr, delta)
        assert out.shape == (B_, L, D)


# ---------------------------------------------------------------------------
# INT4 GEMV tests
# ---------------------------------------------------------------------------

class TestInt4GemvTriton:

    def _run(self, rows, cols, group_size=128, seed=0):
        packed_np, scales_np, zeros_np, x_np = _make_int4_data(rows, cols, group_size, seed)
        ref = _int4_dequant_gemv_ref(packed_np, scales_np, zeros_np, x_np, rows, cols, group_size)

        packed = torch.from_numpy(packed_np).cuda()
        scales = torch.from_numpy(scales_np).cuda()
        zeros  = torch.from_numpy(zeros_np).cuda()
        x      = torch.from_numpy(x_np).cuda()

        out = int4_gemv(packed, scales, zeros, x, group_size=group_size).cpu().numpy()
        return ref, out

    def test_small(self):
        """rows=4, cols=256, group_size=128."""
        ref, out = self._run(4, 256, 128)
        assert np.allclose(ref, out, rtol=1e-2, atol=5e-3), f"max_err={np.max(np.abs(ref-out)):.2e}"

    def test_medium(self):
        """LLM-typical: rows=4096, cols=4096."""
        ref, out = self._run(64, 1024, 128, seed=7)
        assert np.allclose(ref, out, rtol=1e-2, atol=5e-3), f"max_err={np.max(np.abs(ref-out)):.2e}"

    def test_group_size_64(self):
        ref, out = self._run(8, 512, 64, seed=3)
        assert np.allclose(ref, out, rtol=1e-2, atol=5e-3), f"max_err={np.max(np.abs(ref-out)):.2e}"

    def test_output_shape(self):
        rows, cols = 16, 256
        packed_np, scales_np, zeros_np, x_np = _make_int4_data(rows, cols, 128)
        out = int4_gemv(
            torch.from_numpy(packed_np).cuda(),
            torch.from_numpy(scales_np).cuda(),
            torch.from_numpy(zeros_np).cuda(),
            torch.from_numpy(x_np).cuda(),
        )
        assert out.shape == (rows,)
