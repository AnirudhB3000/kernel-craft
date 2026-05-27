"""
selective_scan_triton.py
Triton implementation of Mamba-1 ZOH selective scan.

Mirrors: src/kernels/transformer/selective_scan.cu (Phase 14)
Parallelism: one program per (batch, channel) pair; L timesteps sequential.
N_state dimension vectorized with BLOCK_N (next power-of-2 ≥ N_state).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _selective_scan_kernel(
    u_ptr, A_log_ptr, B_ptr, C_ptr, delta_ptr, y_ptr,
    B_, L, D, N_state,
    BLOCK_N: tl.constexpr,
):
    """Selective scan kernel — one program per (b, d) pair.

    ZOH recurrence:
        A_bar = exp(delta * A_log)
        h     = A_bar * h + delta * B * u
        y     = C · h
    """
    bd = tl.program_id(0)
    b  = bd // D
    d  = bd % D

    offs_n   = tl.arange(0, BLOCK_N)
    mask_n   = offs_n < N_state

    # Load A_log[d, :] — shared across all timesteps
    a_log = tl.load(A_log_ptr + d * N_state + offs_n, mask=mask_n, other=0.0)

    # Hidden state initialized to zero
    h = tl.zeros([BLOCK_N], dtype=tl.float32)

    for t in range(L):
        base_ld = (b * L + t) * D + d
        base_bn = (b * L + t) * N_state

        dt    = tl.load(delta_ptr + base_ld)
        u_td  = tl.load(u_ptr     + base_ld)

        B_t = tl.load(B_ptr + base_bn + offs_n, mask=mask_n, other=0.0)
        C_t = tl.load(C_ptr + base_bn + offs_n, mask=mask_n, other=0.0)

        # ZOH update (element-wise over N_state)
        a_bar = tl.exp(dt * a_log)
        b_bar = dt * B_t
        h     = a_bar * h + b_bar * u_td

        # Output: dot product C · h (reduce over N_state)
        y_td = tl.sum(tl.where(mask_n, C_t * h, 0.0), axis=0)
        tl.store(y_ptr + base_ld, y_td)


def selective_scan(
    u:     torch.Tensor,
    A_log: torch.Tensor,
    B:     torch.Tensor,
    C:     torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """Triton Mamba-1 selective scan.

    Args:
        u:     [Batch, L, D] float32 CUDA tensor.
        A_log: [D, N_state]  float32 CUDA tensor.
        B:     [Batch, L, N_state] float32 CUDA tensor.
        C:     [Batch, L, N_state] float32 CUDA tensor.
        delta: [Batch, L, D] float32 CUDA tensor.

    Returns:
        y: [Batch, L, D] float32 CUDA tensor.
    """
    assert u.is_cuda, "inputs must be CUDA tensors"
    B_, L, D = u.shape
    N_state  = A_log.shape[1]
    assert N_state <= 32, f"N_state={N_state} exceeds max 32"

    BLOCK_N = triton.next_power_of_2(N_state)
    y    = torch.empty_like(u)
    grid = (B_ * D,)

    _selective_scan_kernel[grid](
        u, A_log, B, C, delta, y,
        B_=B_, L=L, D=D, N_state=N_state,
        BLOCK_N=BLOCK_N,
    )
    return y
