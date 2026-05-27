"""
benchmark_triton.py
CUDA vs Triton side-by-side throughput comparison for Phase 15 kernels.

Three kernels benchmarked:
  1. FlashAttention  — Gflops (2*B*H*N²*d for full attention)
  2. Selective scan  — Mtokens/s  (throughput vs sequence length L)
  3. INT4 GEMV       — GB/s  (weight bandwidth utilization)

Triton timing:  triton.testing.do_bench() — GPU-only, sub-microsecond accuracy.
CUDA  timing:   torch.cuda.Event pair around pybind11 call; includes numpy↔GPU
                transfer overhead (noted in output).

Run with: python benchmarks/benchmark_triton.py
(from repo root, with venv active)
"""
import sys
import os
import math
import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Triton kernels (third-party triton must be importable before src/triton) ---
import triton
import triton.testing

sys.path.insert(0, os.path.join(_REPO, "src", "triton"))
from flash_attention_triton import flash_attention     as fa_triton
from selective_scan_triton  import selective_scan      as ss_triton
from int4_gemv_triton       import int4_gemv           as gemv_triton

# --- CUDA kernels via pybind11 ---
sys.path.insert(0, os.path.join(_REPO, "src", "python"))
sys.path.insert(0, os.path.join(_REPO, "src", "python", "build"))
try:
    import kernel_craft_transformer as kct
    HAS_KCT = True
except ImportError:
    HAS_KCT = False
    print("WARNING: kernel_craft_transformer not found — CUDA columns will show N/A")


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def bench_triton(fn, warmup=25, rep=100):
    """Return median GPU time in ms using triton.testing.do_bench."""
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep)


def bench_cuda_event(fn, warmup=10, rep=50):
    """Return mean GPU time in ms using CUDA events."""
    # Warm-up
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(rep)
    ]
    for t0, t1 in events:
        t0.record()
        fn()
        t1.record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in events) / rep


# ---------------------------------------------------------------------------
# 1. FlashAttention
# ---------------------------------------------------------------------------

def benchmark_flash_attention():
    print("\n=== FlashAttention: Triton vs CUDA ===")
    print(f"{'Config':<30} {'Triton ms':>10} {'Triton Gflops':>14} "
          f"{'CUDA ms':>10} {'CUDA Gflops':>12}")
    print("-" * 80)

    configs = [
        # (B, H, N, d, causal)
        (1, 8,  512,  64, False),
        (1, 8, 1024,  64, False),
        (1, 8, 2048,  64, False),
        (1, 8, 1024,  64, True),
        (1, 4,  512,  64, False),   # GQA 4:2
    ]

    for B, H, N, d, causal in configs:
        H_kv = H
        Q = torch.randn(B, H,    N, d, device="cuda")
        K = torch.randn(B, H_kv, N, d, device="cuda")
        V = torch.randn(B, H_kv, N, d, device="cuda")

        # Triton
        ms_t = bench_triton(lambda: fa_triton(Q, K, V, causal=causal))
        # Flops: 2 * B * H * N * N * d (Q@K^T) + 2 * B * H * N * N * d (softmax-weighted V)
        flops = 2 * B * H * N * N * d * 2
        if causal:
            flops //= 2  # ~half KV tiles skipped
        gflops_t = flops / (ms_t * 1e6)

        # CUDA
        if HAS_KCT:
            Q_np = Q.cpu().numpy()
            K_np = K.cpu().numpy()
            V_np = V.cpu().numpy()
            def cuda_fn():
                kct.flash_attention(Q_np, K_np, V_np, H_kv, causal)
            ms_c = bench_cuda_event(cuda_fn)
            gflops_c = flops / (ms_c * 1e6)
            cuda_str = f"{ms_c:>10.3f} {gflops_c:>12.1f}"
        else:
            cuda_str = f"{'N/A':>10} {'N/A':>12}"

        tag = f"B={B} H={H} N={N} d={d}{'(causal)' if causal else ''}"
        print(f"{tag:<30} {ms_t:>10.3f} {gflops_t:>14.1f} {cuda_str}")

    print("  Note: CUDA ms includes numpy↔GPU transfer; Triton ms is GPU-only.")


# ---------------------------------------------------------------------------
# 2. Selective Scan
# ---------------------------------------------------------------------------

def benchmark_selective_scan():
    print("\n=== Selective Scan: Triton vs CUDA ===")
    print(f"{'L':>6}  {'Triton ms':>10} {'Triton Mtok/s':>14} "
          f"{'CUDA ms':>10} {'CUDA Mtok/s':>13}")
    print("-" * 60)

    B_, D, N = 1, 512, 16
    Ls = [64, 256, 1024, 4096, 16384]

    for L in Ls:
        u     = torch.randn(B_, L, D, device="cuda")
        A_log = -torch.rand(D,  N,    device="cuda")
        B_arr = torch.rand(B_, L, N,  device="cuda") * 0.3
        C_arr = torch.rand(B_, L, N,  device="cuda") * 0.3
        delta = torch.rand(B_, L, D,  device="cuda") * 0.5

        # Triton
        ms_t   = bench_triton(lambda: ss_triton(u, A_log, B_arr, C_arr, delta))
        mtok_t = B_ * L / (ms_t * 1e-3) / 1e6

        # CUDA
        if HAS_KCT:
            u_np, al_np, b_np, c_np, d_np = (
                u.cpu().numpy(), A_log.cpu().numpy(),
                B_arr.cpu().numpy(), C_arr.cpu().numpy(), delta.cpu().numpy(),
            )
            def cuda_fn():
                kct.selective_scan(u_np, al_np, b_np, c_np, d_np)
            ms_c   = bench_cuda_event(cuda_fn)
            mtok_c = B_ * L / (ms_c * 1e-3) / 1e6
            cuda_str = f"{ms_c:>10.3f} {mtok_c:>13.2f}"
        else:
            cuda_str = f"{'N/A':>10} {'N/A':>13}"

        print(f"{L:>6}  {ms_t:>10.3f} {mtok_t:>14.2f} {cuda_str}")

    print("  Note: CUDA ms includes numpy↔GPU transfer; Triton ms is GPU-only.")


# ---------------------------------------------------------------------------
# 3. INT4 GEMV
# ---------------------------------------------------------------------------

def _make_int4_tensors(rows, cols, group_size, device="cuda", seed=0):
    rng = np.random.default_rng(seed)
    num_groups   = cols // group_size
    nzb          = (num_groups + 1) // 2

    w_int4  = rng.integers(-8, 8, size=(rows, cols), dtype=np.int8)
    packed  = np.zeros((rows, cols // 2), dtype=np.uint8)
    for bi in range(cols // 2):
        lo = w_int4[:, bi * 2]     & 0xF
        hi = w_int4[:, bi * 2 + 1] & 0xF
        packed[:, bi] = lo | (hi << 4)

    scales = rng.uniform(0.01, 0.1, (rows, num_groups)).astype(np.float32)
    zeros  = np.zeros((rows, nzb), dtype=np.uint8)
    x_np   = rng.uniform(-1.0, 1.0, cols).astype(np.float32)

    return (
        torch.from_numpy(packed).to(device),
        torch.from_numpy(scales).to(device),
        torch.from_numpy(zeros).to(device),
        torch.from_numpy(x_np).to(device),
        packed, scales, zeros, x_np,
    )


def benchmark_int4_gemv():
    print("\n=== INT4 GEMV: Triton vs CUDA ===")
    print(f"{'Config':<24} {'Triton ms':>10} {'Triton GB/s':>12} "
          f"{'CUDA ms':>10} {'CUDA GB/s':>11}")
    print("-" * 73)

    configs = [
        # (rows, cols, group_size)
        (256,   4096, 128),
        (1024,  4096, 128),
        (4096,  4096, 128),
        (4096, 11008, 128),
    ]

    for rows, cols, gs in configs:
        packed, scales, zeros, x, p_np, sc_np, z_np, x_np = \
            _make_int4_tensors(rows, cols, gs)

        # Triton
        ms_t = bench_triton(lambda: gemv_triton(packed, scales, zeros, x, gs))
        # Bandwidth: read packed weights + scales + zeros + x; write y
        bytes_io = rows * (cols // 2) + rows * (cols // gs) * 4 + rows * ((cols // gs + 1) // 2) + cols * 4 + rows * 4
        gbps_t = bytes_io / (ms_t * 1e-3) / 1e9

        # CUDA
        if HAS_KCT:
            def cuda_fn():
                kct.quant_int4_dequant(p_np, sc_np, z_np, rows, cols, gs)
            ms_c = bench_cuda_event(cuda_fn)
            gbps_c = bytes_io / (ms_c * 1e-3) / 1e9
            cuda_str = f"{ms_c:>10.3f} {gbps_c:>11.1f}"
        else:
            cuda_str = f"{'N/A':>10} {'N/A':>11}"

        tag = f"r={rows} c={cols} gs={gs}"
        print(f"{tag:<24} {ms_t:>10.3f} {gbps_t:>12.1f} {cuda_str}")

    print("  Note: CUDA ms includes numpy↔GPU transfer; Triton ms is GPU-only.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Phase 15: Triton vs CUDA Kernel Benchmarks ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Triton {triton.__version__}")

    benchmark_flash_attention()
    benchmark_selective_scan()
    benchmark_int4_gemv()

    print("\nDone.")
