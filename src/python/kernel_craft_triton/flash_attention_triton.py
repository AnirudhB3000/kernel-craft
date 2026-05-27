"""
flash_attention_triton.py
Triton FlashAttention-2 style forward pass.

Mirrors: src/kernels/transformer/flash_attention.cu (Phase 11)
Algorithm: online softmax, tiled Q/K/V, causal masking, GQA/MQA support.
Autotuned over BLOCK_M × BLOCK_N tile sizes {16, 32, 64}.
"""
import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    ],
    key=["N", "HEAD_DIM"],
)
@triton.jit
def _flash_attn_fwd(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qn,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_on,
    H, H_kv, N,
    scale,
    causal: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """FlashAttention forward kernel — one program per (Q-tile, head, batch).

    Q tile covers BLOCK_M rows; K/V tiles cover BLOCK_N rows per iteration.
    Online softmax maintains running (m, l, acc) per Q row in registers.
    """
    start_m  = tl.program_id(0)
    head_id  = tl.program_id(1)
    batch_id = tl.program_id(2)

    # GQA/MQA: map Q head → KV head
    kv_head = head_id * H_kv // H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    Q_base = Q + batch_id * stride_qb + head_id * stride_qh
    K_base = K + batch_id * stride_kb + kv_head * stride_kh
    V_base = V + batch_id * stride_vb + kv_head * stride_vh
    O_base = O + batch_id * stride_ob + head_id * stride_oh

    # Load Q [BLOCK_M, HEAD_DIM] — stride_qd = 1 assumed (contiguous last dim)
    q = tl.load(
        Q_base + offs_m[:, None] * stride_qn + offs_d[None, :],
        mask=offs_m[:, None] < N,
        other=0.0,
    ) * scale

    # Running online-softmax state
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Causal: only iterate over KV tiles that overlap this Q tile
    if causal:
        kv_end = tl.cdiv((start_m + 1) * BLOCK_M, BLOCK_N)
    else:
        kv_end = tl.cdiv(N, BLOCK_N)

    for j in range(kv_end):
        offs_n = j * BLOCK_N + tl.arange(0, BLOCK_N)

        # Load K [BLOCK_N, HEAD_DIM]
        k = tl.load(
            K_base + offs_n[:, None] * stride_kn + offs_d[None, :],
            mask=offs_n[:, None] < N,
            other=0.0,
        )

        # S = Q @ K^T → [BLOCK_M, BLOCK_N]
        s = tl.dot(q, tl.trans(k))

        # Mask out-of-bounds KV positions
        s = tl.where(offs_n[None, :] < N, s, float("-inf"))
        # Causal mask: Q row must be >= KV column
        if causal:
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha  = tl.exp(m_i - m_new)
        p      = tl.exp(s - m_new[:, None])
        l_i    = l_i * alpha + tl.sum(p, axis=1)
        acc    = acc * alpha[:, None]

        # Load V [BLOCK_N, HEAD_DIM] and accumulate weighted sum
        v = tl.load(
            V_base + offs_n[:, None] * stride_vn + offs_d[None, :],
            mask=offs_n[:, None] < N,
            other=0.0,
        )
        acc += tl.dot(p.to(tl.float32), v)
        m_i = m_new

    # Normalize and write output
    acc = acc / l_i[:, None]
    tl.store(
        O_base + offs_m[:, None] * stride_on + offs_d[None, :],
        acc,
        mask=offs_m[:, None] < N,
    )


def flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Triton FlashAttention forward pass.

    Args:
        Q:      [B, H, N, d] float32 CUDA tensor.
        K:      [B, H_kv, N, d] float32 CUDA tensor (H_kv ≤ H, H % H_kv == 0).
        V:      [B, H_kv, N, d] float32 CUDA tensor.
        causal: Apply causal lower-triangular mask.

    Returns:
        O: [B, H, N, d] float32 CUDA tensor.
    """
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "inputs must be CUDA tensors"
    assert Q.dtype == torch.float32, "only float32 supported"
    B, H, N, d = Q.shape
    H_kv = K.shape[1]
    assert d in (16, 32, 64, 128), f"HEAD_DIM={d} must be power-of-2 in [16, 32, 64, 128]"
    assert H % H_kv == 0, "H must be divisible by H_kv (GQA requirement)"

    O = torch.empty_like(Q)
    scale = 1.0 / math.sqrt(d)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]), H, B)  # noqa: E731
    _flash_attn_fwd[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        H=H, H_kv=H_kv, N=N,
        scale=scale,
        causal=causal,
        HEAD_DIM=d,
    )
    return O
