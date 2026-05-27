"""
int4_gemv_triton.py
Triton INT4-quantized GEMV: y = dequant(W_int4) @ x.

Mirrors: src/kernels/transformer/quant_int4.cu (Phase 11)
Layout:
  packed_weights: [rows, cols/2] uint8 — two INT4 per byte, low nibble first
  scales:         [rows, cols/group_size] float32 — per-group scale
  zeros:          [rows, ceil(cols/group_size/2)] uint8 — packed zero-points
  x:              [cols] float32
  y:              [rows] float32
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_BYTES": 32},  num_warps=2),
        triton.Config({"BLOCK_BYTES": 64},  num_warps=4),
        triton.Config({"BLOCK_BYTES": 128}, num_warps=8),
    ],
    key=["cols"],
)
@triton.jit
def _int4_gemv_kernel(
    packed_ptr, scales_ptr, zeros_ptr, x_ptr, y_ptr,
    rows, cols, group_size, num_groups, num_zp_bytes_per_row,
    BLOCK_BYTES: tl.constexpr,
):
    """One program per output row; BLOCK_BYTES packed bytes per loop iteration."""
    row = tl.program_id(0)

    num_bytes = cols // 2
    acc = tl.zeros([], dtype=tl.float32)

    num_iters = tl.cdiv(num_bytes, BLOCK_BYTES)
    for i in range(num_iters):
        b_start   = i * BLOCK_BYTES
        byte_offs = b_start + tl.arange(0, BLOCK_BYTES)
        byte_mask = byte_offs < num_bytes

        col_lo = byte_offs * 2       # even column (low nibble)
        col_hi = byte_offs * 2 + 1  # odd  column (high nibble)

        # --- Load packed bytes and unpack nibbles ---
        p = tl.load(
            packed_ptr + row * num_bytes + byte_offs,
            mask=byte_mask, other=0,
        ).to(tl.uint8)

        lo_nibble = (p & tl.full([BLOCK_BYTES], 0x0F, dtype=tl.uint8)).to(tl.int32)
        hi_nibble = ((p >> tl.full([BLOCK_BYTES], 4, dtype=tl.uint8)) & tl.full([BLOCK_BYTES], 0x0F, dtype=tl.uint8)).to(tl.int32)

        # Sign-extend from 4-bit two's complement [-8, 7]
        lo_s = tl.where(lo_nibble >= 8, lo_nibble - 16, lo_nibble)
        hi_s = tl.where(hi_nibble >= 8, hi_nibble - 16, hi_nibble)

        # --- Load scales (one per group) ---
        grp_lo = col_lo // group_size
        grp_hi = col_hi // group_size

        sc_lo = tl.load(
            scales_ptr + row * num_groups + grp_lo,
            mask=byte_mask, other=1.0,
        )
        sc_hi = tl.load(
            scales_ptr + row * num_groups + grp_hi,
            mask=byte_mask, other=1.0,
        )

        # --- Load zero-points (nibble-packed: two groups per byte) ---
        zp_byte_lo = grp_lo // 2
        zp_byte_hi = grp_hi // 2

        zp_raw_lo = tl.load(
            zeros_ptr + row * num_zp_bytes_per_row + zp_byte_lo,
            mask=byte_mask, other=0,
        ).to(tl.uint8)
        zp_raw_hi = tl.load(
            zeros_ptr + row * num_zp_bytes_per_row + zp_byte_hi,
            mask=byte_mask, other=0,
        ).to(tl.uint8)

        zp_lo = tl.where(
            (grp_lo % 2) == 0,
            (zp_raw_lo & tl.full([BLOCK_BYTES], 0x0F, dtype=tl.uint8)).to(tl.int32),
            ((zp_raw_lo >> tl.full([BLOCK_BYTES], 4, dtype=tl.uint8)) & tl.full([BLOCK_BYTES], 0x0F, dtype=tl.uint8)).to(tl.int32),
        )
        zp_hi = tl.where(
            (grp_hi % 2) == 0,
            (zp_raw_hi & tl.full([BLOCK_BYTES], 0x0F, dtype=tl.uint8)).to(tl.int32),
            ((zp_raw_hi >> tl.full([BLOCK_BYTES], 4, dtype=tl.uint8)) & tl.full([BLOCK_BYTES], 0x0F, dtype=tl.uint8)).to(tl.int32),
        )

        # --- Dequantize: w = (int4 - zero_point) * scale ---
        w_lo = (lo_s - zp_lo).to(tl.float32) * sc_lo
        w_hi = (hi_s - zp_hi).to(tl.float32) * sc_hi

        # --- Load x and accumulate ---
        x_lo = tl.load(x_ptr + col_lo, mask=byte_mask & (col_lo < cols), other=0.0)
        x_hi = tl.load(x_ptr + col_hi, mask=byte_mask & (col_hi < cols), other=0.0)

        acc = acc + tl.sum(w_lo * x_lo + w_hi * x_hi, axis=0)

    tl.store(y_ptr + row, acc)


def int4_gemv(
    packed:     torch.Tensor,
    scales:     torch.Tensor,
    zeros:      torch.Tensor,
    x:          torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """INT4 GEMV: y = dequant(packed, scales, zeros) @ x.

    Args:
        packed:     [rows, cols/2] uint8 CUDA tensor.
        scales:     [rows, cols/group_size] float32 CUDA tensor.
        zeros:      [rows, ceil(cols/group_size/2)] uint8 CUDA tensor.
        x:          [cols] float32 CUDA tensor.
        group_size: Elements sharing one scale/zero-point (default 128).

    Returns:
        y: [rows] float32 CUDA tensor.
    """
    assert packed.is_cuda and x.is_cuda, "inputs must be CUDA tensors"
    rows, half_cols = packed.shape
    cols = half_cols * 2
    assert cols % group_size == 0, "cols must be divisible by group_size"

    num_groups         = cols // group_size
    num_zp_bytes_per_row = (num_groups + 1) // 2

    y = torch.empty(rows, dtype=torch.float32, device=packed.device)
    grid = (rows,)

    _int4_gemv_kernel[grid](
        packed, scales, zeros, x, y,
        rows=rows, cols=cols,
        group_size=group_size,
        num_groups=num_groups,
        num_zp_bytes_per_row=num_zp_bytes_per_row,
    )
    return y
