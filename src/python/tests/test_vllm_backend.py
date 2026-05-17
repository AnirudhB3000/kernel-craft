"""
test_vllm_backend.py
Tests for the kernel-craft vLLM attention backend.

All tests skip gracefully when vLLM or PyTorch is not installed.
The torch-ops tests exercise the ctypes bridge (kernel_craft_torch_ops)
without needing a full vLLM installation.
"""

import sys
import os
import math
import pytest
import numpy as np

# Ensure we can find the build directory and python source
_REPO = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, os.path.join(_REPO, "src", "python"))
sys.path.insert(0, os.path.join(_REPO, "src", "python", "build"))

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.cuda
    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False

try:
    import kernel_craft_torch_ops as kc_ops
    HAS_OPS = HAS_TORCH
except Exception:
    HAS_OPS = False

try:
    from vllm.v1.attention.backend import AttentionBackend, AttentionImpl
    import kernel_craft_vllm_backend as kc_backend
    HAS_VLLM = kc_backend.HAS_VLLM
except ImportError:
    HAS_VLLM = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch + CUDA required")
requires_ops   = pytest.mark.skipif(not HAS_OPS,   reason="kernel_craft_torch_ops not available")
requires_vllm  = pytest.mark.skipif(not HAS_VLLM,  reason="vLLM not installed")


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def _attention_ref(Q: np.ndarray, K: np.ndarray, V: np.ndarray, causal: bool = False) -> np.ndarray:
    """Naive attention reference: Q/K/V [B, H, N, d] → [B, H, N, d]."""
    B, H, N, d = Q.shape
    H_kv = K.shape[1]
    scale = 1.0 / math.sqrt(d)
    O = np.zeros_like(Q)
    for b in range(B):
        for h in range(H):
            h_kv = h * H_kv // H
            scores = Q[b, h] @ K[b, h_kv].T * scale
            if causal:
                mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                scores[mask] = -1e38
            scores -= scores.max(axis=1, keepdims=True)
            attn = np.exp(scores)
            attn /= attn.sum(axis=1, keepdims=True)
            O[b, h] = attn @ V[b, h_kv]
    return O


def _decode_ref(q: np.ndarray, K: np.ndarray, V: np.ndarray, H_kv: int) -> np.ndarray:
    """Single-step decode attention reference: q [B, H, d], K/V [B, H_kv, N, d]."""
    B, H, d = q.shape
    seq_len = K.shape[2]
    scale = 1.0 / math.sqrt(d)
    O = np.zeros_like(q)
    for b in range(B):
        for h in range(H):
            h_kv = h * H_kv // H
            scores = q[b, h] @ K[b, h_kv].T * scale  # [d] @ [d, N] = [N]
            scores -= scores.max()
            w = np.exp(scores) / np.exp(scores).sum()
            O[b, h] = w @ V[b, h_kv]
    return O


# ---------------------------------------------------------------------------
# Tests: kernel_craft_torch_ops (ctypes bridge)
# ---------------------------------------------------------------------------

class TestTorchOpsFlashAttention:
    """Verify ctypes-bridged FlashAttention produces correct output on GPU tensors."""

    @requires_ops
    def test_mha_basic(self):
        rng = np.random.default_rng(42)
        B, H, N, d = 1, 4, 32, 64
        Q = torch.tensor(rng.uniform(-1, 1, (B, H, N, d)), dtype=torch.float32).cuda()
        K = torch.tensor(rng.uniform(-1, 1, (B, H, N, d)), dtype=torch.float32).cuda()
        V = torch.tensor(rng.uniform(-1, 1, (B, H, N, d)), dtype=torch.float32).cuda()

        out = kc_ops.flash_attention(Q, K, V, H_kv=H, causal=False)
        ref = _attention_ref(Q.cpu().numpy(), K.cpu().numpy(), V.cpu().numpy())

        assert out.shape == (B, H, N, d)
        np.testing.assert_allclose(out.cpu().numpy(), ref, atol=2e-3)

    @requires_ops
    def test_causal_mask(self):
        rng = np.random.default_rng(7)
        B, H, N, d = 1, 2, 48, 64
        Q = torch.tensor(rng.uniform(-1, 1, (B, H, N, d)), dtype=torch.float32).cuda()
        K = torch.tensor(rng.uniform(-1, 1, (B, H, N, d)), dtype=torch.float32).cuda()
        V = torch.tensor(rng.uniform(-1, 1, (B, H, N, d)), dtype=torch.float32).cuda()

        out = kc_ops.flash_attention(Q, K, V, H_kv=H, causal=True)
        ref = _attention_ref(Q.cpu().numpy(), K.cpu().numpy(), V.cpu().numpy(), causal=True)
        np.testing.assert_allclose(out.cpu().numpy(), ref, atol=2e-3)

    @requires_ops
    def test_gqa(self):
        rng = np.random.default_rng(99)
        B, H, H_kv, N, d = 1, 4, 2, 32, 64
        Q = torch.tensor(rng.uniform(-1, 1, (B, H,    N, d)), dtype=torch.float32).cuda()
        K = torch.tensor(rng.uniform(-1, 1, (B, H_kv, N, d)), dtype=torch.float32).cuda()
        V = torch.tensor(rng.uniform(-1, 1, (B, H_kv, N, d)), dtype=torch.float32).cuda()

        out = kc_ops.flash_attention(Q, K, V, H_kv=H_kv, causal=False)
        ref = _attention_ref(Q.cpu().numpy(), K.cpu().numpy(), V.cpu().numpy())
        np.testing.assert_allclose(out.cpu().numpy(), ref, atol=2e-3)

    @requires_ops
    def test_output_device(self):
        Q = torch.ones(1, 4, 16, 64, device="cuda")
        K = torch.ones(1, 4, 16, 64, device="cuda")
        V = torch.ones(1, 4, 16, 64, device="cuda")
        out = kc_ops.flash_attention(Q, K, V, H_kv=4, causal=False)
        assert out.device.type == "cuda"
        assert out.dtype == torch.float32


class TestTorchOpsPagedAttention:
    """Verify ctypes-bridged PagedAttention against decode reference."""

    def _make_paged(self, K_np, V_np, seq_len, bs, H_kv):
        """Pack numpy K/V [B, H_kv, N, d] into paged pool."""
        B, _, _, d = K_np.shape
        max_blocks = (seq_len + bs - 1) // bs
        phys = 0
        bt   = np.zeros((B, max_blocks), dtype=np.int32)
        K_blocks, V_blocks = [], []
        for b in range(B):
            for blk in range(max_blocks):
                bt[b, blk] = phys; phys += 1
                Kb = np.zeros((bs, H_kv, d), dtype=np.float32)
                Vb = np.zeros((bs, H_kv, d), dtype=np.float32)
                for tok in range(bs):
                    sp = blk * bs + tok
                    if sp < seq_len:
                        for hk in range(H_kv):
                            Kb[tok, hk] = K_np[b, hk, sp]
                            Vb[tok, hk] = V_np[b, hk, sp]
                K_blocks.append(Kb); V_blocks.append(Vb)
        K_pool = np.stack(K_blocks)  # [phys, bs, H_kv, d]
        V_pool = np.stack(V_blocks)
        return K_pool, V_pool, bt

    @requires_ops
    def test_single_batch(self):
        rng = np.random.default_rng(3)
        B, H, H_kv, d, N, bs = 1, 4, 4, 64, 32, 16
        Q_np = rng.uniform(-1, 1, (B, H, d)).astype(np.float32)
        K_np = rng.uniform(-1, 1, (B, H_kv, N, d)).astype(np.float32)
        V_np = rng.uniform(-1, 1, (B, H_kv, N, d)).astype(np.float32)

        ref = _decode_ref(Q_np, K_np, V_np, H_kv)
        K_pool, V_pool, bt = self._make_paged(K_np, V_np, N, bs, H_kv)

        Q_t  = torch.tensor(Q_np,   device="cuda")
        Kp_t = torch.tensor(K_pool, device="cuda")
        Vp_t = torch.tensor(V_pool, device="cuda")
        bt_t = torch.tensor(bt,     dtype=torch.int32, device="cuda")
        sl_t = torch.tensor([N],    dtype=torch.int32, device="cuda")

        out = kc_ops.paged_attention(Q_t, bt_t, Kp_t, Vp_t, sl_t, H_kv=H_kv, block_size=bs)
        np.testing.assert_allclose(out.cpu().numpy(), ref, atol=1e-3)

    @requires_ops
    def test_output_shape(self):
        B, H, H_kv, d, N, bs = 2, 4, 2, 64, 16, 8
        rng = np.random.default_rng(0)
        Q_np = rng.uniform(-1, 1, (B, H, d)).astype(np.float32)
        K_np = rng.uniform(-1, 1, (B, H_kv, N, d)).astype(np.float32)
        V_np = rng.uniform(-1, 1, (B, H_kv, N, d)).astype(np.float32)
        K_pool, V_pool, bt = self._make_paged(K_np, V_np, N, bs, H_kv)

        Q_t  = torch.tensor(Q_np,   device="cuda")
        Kp_t = torch.tensor(K_pool, device="cuda")
        Vp_t = torch.tensor(V_pool, device="cuda")
        bt_t = torch.tensor(bt, dtype=torch.int32, device="cuda")
        sl_t = torch.tensor([N, N], dtype=torch.int32, device="cuda")

        out = kc_ops.paged_attention(Q_t, bt_t, Kp_t, Vp_t, sl_t, H_kv=H_kv, block_size=bs)
        assert out.shape == (B, H, d)


class TestTorchOpsFP8:
    """Verify FP8 quantize/dequantize round-trip on CUDA tensors."""

    @requires_ops
    def test_roundtrip_accuracy(self):
        rng = np.random.default_rng(11)
        x_np = rng.uniform(-1, 1, (8, 64)).astype(np.float32)
        x_t  = torch.tensor(x_np, device="cuda")

        q, scales = kc_ops.fp8_quantize(x_t, per_token=True)
        x_hat = kc_ops.fp8_dequantize(q, scales, rows=8, cols=64, per_token=True)

        rel_err = (x_hat.cpu() - torch.tensor(x_np)).abs() / (torch.tensor(x_np).abs() + 1e-6)
        assert rel_err.max().item() < 0.15, f"max relative error {rel_err.max():.4f}"

    @requires_ops
    def test_output_shapes(self):
        x = torch.randn(16, 128, device="cuda")
        q, scales = kc_ops.fp8_quantize(x, per_token=True)
        assert q.shape     == (16, 128)
        assert scales.shape == (16,)
        out = kc_ops.fp8_dequantize(q, scales, rows=16, cols=128, per_token=True)
        assert out.shape == (16, 128)


class TestTorchOpsSpeculative:
    """Verify speculative verify returns correct acceptance / correction on GPU."""

    @requires_ops
    def test_all_accept(self):
        K, V = 4, 8
        dp = torch.zeros(K, V, device="cuda")
        tp = torch.zeros(K, V, device="cuda")
        dp[:, 0] = 0.5
        tp[:, 0] = 1.0
        tokens = torch.zeros(K, dtype=torch.int32, device="cuda")
        rv  = torch.full((K,), 0.4, device="cuda")
        rv2 = torch.full((K,), 0.5, device="cuda")

        acc, _ = kc_ops.speculative_verify(dp, tp, tokens, rv, rv2)
        assert acc.sum().item() == K, f"Expected all accepted, got {acc.tolist()}"

    @requires_ops
    def test_output_shapes(self):
        K, V = 5, 16
        dp = torch.ones(K, V, device="cuda") / V
        tp = torch.ones(K, V, device="cuda") / V
        tokens = torch.zeros(K, dtype=torch.int32, device="cuda")
        rv  = torch.zeros(K, device="cuda")
        rv2 = torch.zeros(K, device="cuda")
        acc, cor = kc_ops.speculative_verify(dp, tp, tokens, rv, rv2)
        assert acc.shape == (K,)
        assert cor.shape == (K,)


# ---------------------------------------------------------------------------
# Tests: vLLM backend interface
# ---------------------------------------------------------------------------

class TestVLLMBackendInterface:
    """Smoke tests for the KernelCraftAttentionBackend class interface (vLLM v1 API)."""

    @requires_vllm
    def test_backend_name(self):
        assert kc_backend.KernelCraftAttentionBackend.get_name() == "kernel_craft"

    @requires_vllm
    def test_impl_cls(self):
        impl_cls = kc_backend.KernelCraftAttentionBackend.get_impl_cls()
        assert issubclass(impl_cls, AttentionImpl)

    @requires_vllm
    def test_kv_cache_shape(self):
        # v1 API: get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size)
        num_blocks = 64
        shape = kc_backend.KernelCraftAttentionBackend.get_kv_cache_shape(
            num_blocks=num_blocks, block_size=16, num_kv_heads=8, head_size=128
        )
        # Expected: [2, num_blocks, block_size, num_kv_heads, head_size]
        assert shape[0] == 2
        assert shape[1] == num_blocks
        assert shape[2] == 16
        assert shape[3] == 8
        assert shape[4] == 128

    @requires_vllm
    def test_instantiate_impl(self):
        impl = kc_backend.KernelCraftAttentionImpl(
            num_heads=32,
            head_size=128,
            scale=1.0 / math.sqrt(128),
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
        )
        assert impl.num_heads == 32
        assert impl.num_kv_heads == 8
        assert impl.head_size == 128

    @requires_vllm
    def test_forward_includes_kv_cache_update_flag(self):
        """In v1, the KV cache update is done by vLLM via slot_mapping, not by forward."""
        assert kc_backend.KernelCraftAttentionBackend.forward_includes_kv_cache_update is False

    @requires_vllm
    def test_kv_cache_shape_2d_index(self):
        """Block index should be at position 1 (num_blocks dimension)."""
        shape = kc_backend.KernelCraftAttentionBackend.get_kv_cache_shape(
            num_blocks=1234567, block_size=16, num_kv_heads=8, head_size=64
        )
        assert shape.index(1234567) == 1  # kv_cache_block_dim == 1

    @requires_vllm
    @requires_ops
    def test_decode_forward_shape(self):
        """Decode forward: [B, H*d] output shape."""
        B, H, H_kv, d, block_size = 2, 4, 2, 64, 16
        num_blocks = 8

        impl = kc_backend.KernelCraftAttentionImpl(
            num_heads=H, head_size=d, scale=1.0 / math.sqrt(d),
            num_kv_heads=H_kv, alibi_slopes=None, sliding_window=None,
            kv_cache_dtype="auto",
        )

        query  = torch.randn(B, H,    d, device="cuda")
        key    = torch.randn(B, H_kv, d, device="cuda")
        value  = torch.randn(B, H_kv, d, device="cuda")
        kv_cache = torch.randn(2, num_blocks, block_size, H_kv, d, device="cuda")
        output   = torch.zeros(B, H * d, device="cuda")

        meta = kc_backend.KernelCraftMetadata(
            num_actual_tokens=B,
            max_query_len=1,  # decode
            max_seq_len=block_size,
            query_start_loc=torch.arange(B + 1, dtype=torch.int32, device="cuda"),
            seq_lens=torch.full((B,), block_size, dtype=torch.int32, device="cuda"),
            block_table=torch.zeros(B, 1, dtype=torch.int32, device="cuda"),
            slot_mapping=torch.arange(B, dtype=torch.int64, device="cuda"),
        )

        out = impl.forward(None, query, key, value, kv_cache, meta, output)
        assert out.shape == (B, H * d)
