"""
test_transformer_bindings.py
Unit tests for Phase 11 transformer kernel Python bindings.
Requires: pip install kernel_craft_transformer (or build from source).
"""

import sys
import os
import math
import pytest
import numpy as np

# Allow importing from build directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))

try:
    import kernel_craft_transformer as kct
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(not HAS_MODULE,
    reason="kernel_craft_transformer not built; run cmake + make")


# ---------------------------------------------------------------------------
# CPU reference implementations
# ---------------------------------------------------------------------------

def attention_ref(Q, K, V, causal=False):
    """Naive attention: Q [B,H,N,d], K/V [B,H_kv,N,d] → [B,H,N,d]."""
    B, H, N, d = Q.shape
    H_kv = K.shape[1]
    scale = 1.0 / math.sqrt(d)
    O = np.zeros_like(Q)
    for b in range(B):
        for h in range(H):
            h_kv = h * H_kv // H
            q = Q[b, h]          # [N, d]
            k = K[b, h_kv]       # [N, d]
            v = V[b, h_kv]       # [N, d]
            scores = q @ k.T * scale  # [N, N]
            if causal:
                mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                scores[mask] = -1e38
            scores = scores - scores.max(axis=1, keepdims=True)
            attn = np.exp(scores)
            attn /= attn.sum(axis=1, keepdims=True)
            O[b, h] = attn @ v
    return O


# ---------------------------------------------------------------------------
# FlashAttention tests
# ---------------------------------------------------------------------------

class TestFlashAttention:
    def _run(self, B, H, H_kv, N, d, causal=False, atol=2e-3):
        rng = np.random.default_rng(42)
        Q = rng.uniform(-1, 1, (B, H, N, d)).astype(np.float32)
        K = rng.uniform(-1, 1, (B, H_kv, N, d)).astype(np.float32)
        V = rng.uniform(-1, 1, (B, H_kv, N, d)).astype(np.float32)
        ref = attention_ref(Q, K, V, causal=causal)
        out = kct.flash_attention(Q, K, V, H_kv=H_kv, causal=causal)
        np.testing.assert_allclose(out, ref, atol=atol, rtol=0,
            err_msg=f"FlashAttention mismatch B={B} H={H} H_kv={H_kv} N={N} d={d}")

    def test_mha_d64(self):
        self._run(B=2, H=4, H_kv=4, N=64, d=64)

    def test_gqa_d64(self):
        self._run(B=1, H=4, H_kv=2, N=32, d=64)

    def test_mqa_d32(self):
        self._run(B=1, H=4, H_kv=1, N=32, d=32)

    def test_causal_d64(self):
        self._run(B=1, H=2, H_kv=2, N=48, d=64, causal=True)

    def test_d128(self):
        self._run(B=1, H=2, H_kv=2, N=32, d=128, atol=3e-3)

    def test_output_shape(self):
        rng = np.random.default_rng(0)
        Q = rng.uniform(-1, 1, (2, 4, 16, 64)).astype(np.float32)
        K = rng.uniform(-1, 1, (2, 4, 16, 64)).astype(np.float32)
        V = rng.uniform(-1, 1, (2, 4, 16, 64)).astype(np.float32)
        out = kct.flash_attention(Q, K, V, H_kv=4)
        assert out.shape == (2, 4, 16, 64)


# ---------------------------------------------------------------------------
# PagedAttention tests
# ---------------------------------------------------------------------------

class TestPagedAttention:
    def _build_paged(self, K, V, seq_len, block_size, H_kv):
        """Pack contiguous K/V into paged pool and return (K_pool, V_pool, block_table)."""
        B = K.shape[0]
        d = K.shape[-1]
        max_blocks = (seq_len + block_size - 1) // block_size
        phys_counter = [0]
        block_table = np.zeros((B, max_blocks), dtype=np.int32)
        all_K, all_V = [], []

        for b in range(B):
            nb = (seq_len + block_size - 1) // block_size
            for blk in range(nb):
                phys = phys_counter[0]; phys_counter[0] += 1
                block_table[b, blk] = phys
                # Build one physical block of [block_size, H_kv, d]
                Kblk = np.zeros((block_size, H_kv, d), dtype=np.float32)
                Vblk = np.zeros((block_size, H_kv, d), dtype=np.float32)
                for tok in range(block_size):
                    sp = blk * block_size + tok
                    if sp < seq_len:
                        for h_kv in range(H_kv):
                            Kblk[tok, h_kv] = K[b, h_kv, sp]
                            Vblk[tok, h_kv] = V[b, h_kv, sp]
                all_K.append(Kblk)
                all_V.append(Vblk)

        num_phys = phys_counter[0]
        K_pool = np.stack(all_K, axis=0)  # [num_phys, block_size, H_kv, d]
        V_pool = np.stack(all_V, axis=0)
        return K_pool, V_pool, block_table, max_blocks

    def _decode_ref(self, q, K, V, seq_len, H_kv):
        """CPU decode attention for single-token query."""
        B, H, d = q.shape
        scale = 1.0 / math.sqrt(d)
        O = np.zeros_like(q)
        for b in range(B):
            for h in range(H):
                h_kv = h * H_kv // H
                scores = q[b, h] @ K[b, h_kv, :seq_len].T * scale  # [d] @ [seq, d]
                scores -= scores.max()
                weights = np.exp(scores) / np.exp(scores).sum()
                O[b, h] = weights @ V[b, h_kv, :seq_len]
        return O

    def test_basic(self):
        B, H, H_kv, d, seq_len, bs = 1, 4, 4, 64, 32, 16
        rng = np.random.default_rng(7)
        Q = rng.uniform(-1, 1, (B, H, d)).astype(np.float32)
        K = rng.uniform(-1, 1, (B, H_kv, seq_len, d)).astype(np.float32)
        V = rng.uniform(-1, 1, (B, H_kv, seq_len, d)).astype(np.float32)

        ref = self._decode_ref(Q, K, V, seq_len, H_kv)
        K_pool, V_pool, block_table, max_blocks = self._build_paged(K, V, seq_len, bs, H_kv)
        seq_lens = np.array([seq_len], dtype=np.int32)

        out = kct.paged_attention(Q, block_table, K_pool, V_pool, seq_lens, H_kv, bs)
        np.testing.assert_allclose(out, ref, atol=1e-3, rtol=0)

    def test_output_shape(self):
        B, H, H_kv, d, seq_len, bs = 2, 4, 2, 64, 16, 8
        rng = np.random.default_rng(0)
        Q = rng.uniform(-1, 1, (B, H, d)).astype(np.float32)
        K = rng.uniform(-1, 1, (B, H_kv, seq_len, d)).astype(np.float32)
        V = rng.uniform(-1, 1, (B, H_kv, seq_len, d)).astype(np.float32)
        K_pool, V_pool, block_table, max_blocks = self._build_paged(K, V, seq_len, bs, H_kv)
        seq_lens = np.array([seq_len, seq_len], dtype=np.int32)
        out = kct.paged_attention(Q, block_table, K_pool, V_pool, seq_lens, H_kv, bs)
        assert out.shape == (B, H, d)


# ---------------------------------------------------------------------------
# INT4 quantization tests
# ---------------------------------------------------------------------------

class TestQuantInt4:
    def _pack(self, weights_i4):
        """Pack int4 values (int8 array with values in [-8,7]) into uint8."""
        rows, cols = weights_i4.shape
        pc = cols // 2
        packed = np.zeros((rows, pc), dtype=np.uint8)
        for r in range(rows):
            for c in range(pc):
                lo = int(weights_i4[r, c*2])   & 0x0F
                hi = int(weights_i4[r, c*2+1]) & 0x0F
                packed[r, c] = lo | (hi << 4)
        return packed

    def test_dequant_roundtrip(self):
        rows, cols, gs = 4, 16, 8
        rng = np.random.default_rng(42)
        weights = rng.integers(-7, 8, (rows, cols), dtype=np.int8)
        packed = self._pack(weights)
        sc = cols // gs
        scales = np.full((rows, sc), 0.125, dtype=np.float32)
        zeros = np.zeros((rows, sc // 2 + sc % 2), dtype=np.uint8)

        result = kct.quant_int4_dequant(packed, scales, zeros, rows, cols, gs)
        expected = weights.astype(np.float32) * 0.125
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_dequant_shape(self):
        rows, cols, gs = 8, 32, 8
        packed = np.zeros((rows, cols // 2), dtype=np.uint8)
        scales = np.ones((rows, cols // gs), dtype=np.float32)
        zeros  = np.zeros((rows, cols // gs // 2 + 1), dtype=np.uint8)
        out = kct.quant_int4_dequant(packed, scales, zeros, rows, cols, gs)
        assert out.shape == (rows, cols)


# ---------------------------------------------------------------------------
# FP8 quantization tests
# ---------------------------------------------------------------------------

class TestFP8Quant:
    def test_per_token_roundtrip(self):
        rng = np.random.default_rng(42)
        inp = rng.uniform(-1, 1, (8, 64)).astype(np.float32)
        q, scales = kct.fp8_quantize(inp, per_token=True)
        out = kct.fp8_dequantize(q, scales, 8, 64, per_token=True)
        # E4M3 relative error ≤ ~12.5%
        nonzero = np.abs(inp) > 1e-6
        rel_err = np.abs(out[nonzero] - inp[nonzero]) / np.abs(inp[nonzero])
        assert rel_err.max() < 0.15, f"max rel error {rel_err.max():.4f}"

    def test_zero_roundtrip(self):
        inp = np.zeros((4, 32), dtype=np.float32)
        q, scales = kct.fp8_quantize(inp, per_token=True)
        out = kct.fp8_dequantize(q, scales, 4, 32, per_token=True)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_per_token_shape(self):
        inp = np.ones((16, 128), dtype=np.float32)
        q, scales = kct.fp8_quantize(inp, per_token=True)
        assert q.shape == (16, 128)
        assert scales.shape == (16,)

    def test_per_tensor_shape(self):
        inp = np.ones((8, 64), dtype=np.float32)
        q, scales = kct.fp8_quantize(inp, per_token=False)
        assert q.shape == (8, 64)
        assert scales.shape == (1,)


# ---------------------------------------------------------------------------
# Speculative decoding tests
# ---------------------------------------------------------------------------

class TestSpeculativeDecoding:
    def test_all_accept(self):
        # target >= draft for all tokens → all accepted
        num_tokens, vocab_size = 4, 8
        draft_probs  = np.zeros((num_tokens, vocab_size), dtype=np.float32)
        target_probs = np.zeros((num_tokens, vocab_size), dtype=np.float32)
        for i in range(num_tokens):
            draft_probs[i, 0]  = 0.5
            target_probs[i, 0] = 1.0
        draft_tokens = np.zeros(num_tokens, dtype=np.int32)
        rand_vals    = np.full(num_tokens, 0.5, dtype=np.float32)
        rand_vals2   = np.full(num_tokens, 0.5, dtype=np.float32)

        accepted, corrected = kct.speculative_decode(
            draft_probs, target_probs, draft_tokens, rand_vals, rand_vals2)
        assert np.all(accepted == 1), f"Expected all accepted, got {accepted}"

    def test_all_reject(self):
        num_tokens, vocab_size = 4, 8
        draft_probs  = np.zeros((num_tokens, vocab_size), dtype=np.float32)
        target_probs = np.zeros((num_tokens, vocab_size), dtype=np.float32)
        for i in range(num_tokens):
            draft_probs[i, 0]  = 0.8
            draft_probs[i, 1]  = 0.2
            target_probs[i, 2] = 1.0  # target mass on token 2, not 0
        draft_tokens = np.zeros(num_tokens, dtype=np.int32)
        rand_vals    = np.full(num_tokens, 0.5, dtype=np.float32)  # > 0 = accept_prob
        rand_vals2   = np.full(num_tokens, 0.1, dtype=np.float32)

        accepted, corrected = kct.speculative_decode(
            draft_probs, target_probs, draft_tokens, rand_vals, rand_vals2)
        assert np.all(accepted == 0), f"Expected all rejected, got {accepted}"
        assert np.all(corrected == 2), f"Expected corrected=2, got {corrected}"

    def test_output_shapes(self):
        num_tokens, vocab_size = 5, 16
        dp = np.ones((num_tokens, vocab_size), dtype=np.float32) / vocab_size
        tp = np.ones((num_tokens, vocab_size), dtype=np.float32) / vocab_size
        tokens = np.zeros(num_tokens, dtype=np.int32)
        rv  = np.zeros(num_tokens, dtype=np.float32)
        rv2 = np.zeros(num_tokens, dtype=np.float32)
        acc, cor = kct.speculative_decode(dp, tp, tokens, rv, rv2)
        assert acc.shape == (num_tokens,)
        assert cor.shape == (num_tokens,)
