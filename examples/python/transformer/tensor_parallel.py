#!/usr/bin/env python3
"""
example_tensor_parallel.py — tensor parallelism with kernel-craft collectives.

Background
-----------
Tensor parallelism (Shoeybi et al. 2019, Megatron-LM) shards large transformer
weight matrices across multiple GPUs so that each GPU holds only 1/R of each
matrix.  Every GPU computes a partial result and communicates to get the full
output:

  Col-parallel linear (first linear in MLP/attention):
    Each rank r holds W_r  [N/R, K]  and  x  [M, K]  (replicated)
    y_r = x @ W_r^T  →  all-gather  →  y  [M, N]

  Row-parallel linear (second linear in MLP block):
    Each rank r holds W_r  [N, K/R]  and  x_r  [M, K/R]  (pre-split)
    p_r = x_r @ W_r^T  →  all-reduce (sum)  →  y  [M, N]

API overview
------------
``kernel_craft_transformer.col_parallel_linear(x, W)``
    x: float32 [M, K],  W: float32 [N_rank, K]
    → float32 [M, N_rank]   (each rank's shard of y)

``kernel_craft_transformer.row_parallel_linear(x_rank, W)``
    x_rank: float32 [M, K_rank],  W: float32 [N, K_rank]
    → float32 [M, N]   (partial sum; caller must all-reduce)

``kernel_craft_transformer.ring_allreduce(list_of_arrays)``
    Simulated ring all-reduce over R virtual ranks.
    Returns same list with all entries set to the element-wise sum.
    In production: replace with ncclAllReduce.

``kernel_craft_transformer.allgather(list_of_chunks)``
    Concatenates R chunks → one array of size R × chunk_size.
    In production: replace with ncclAllGather.

NCCL integration (multi-GPU)
-----------------------------
For real multi-GPU usage the module exposes:
    handles = kct.nccl_comm_init([0, 1, 2, 3])   # one handle per GPU
    kct.nccl_group_start()
    for rank, h in enumerate(handles):
        kct.nccl_allreduce(h, partial_results[rank])
    kct.nccl_group_end()
This requires NCCL installed and --HAVE_NCCL=ON in CMake.
The HAVE_NCCL attribute reports whether NCCL was detected at build time.

Typical model-parallel configuration (Llama-3-70B, 4 GPUs)
------------------------------------------------------------
    N_total = 8192   # hidden dim
    K_total = 8192
    R = 4            # tensor parallel rank
    # Each GPU holds W_r [2048, 8192] (col-parallel: 8192/4 = 2048 cols)
    # All-reduce requires 8192 * batch_size * 4 bytes per step
    # NVLink bandwidth: ~900 GB/s → ~0.1 ms for 8192 × 1024 × float32

Usage
-----
    cd examples/python
    python example_tensor_parallel.py
"""

import sys
import os

_REPO = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_REPO, "src", "python", "build"))
sys.path.insert(0, os.path.join(_REPO, "src", "python"))

import numpy as np

try:
    import kernel_craft_transformer as kct  # type: ignore
    _HAS_KCT = True
except ImportError:
    _HAS_KCT = False
    print("Warning: kernel_craft_transformer not found — build first.")


def main() -> None:
    print("=" * 60)
    print("kernel-craft: tensor parallelism (col/row parallel linear)")
    print("=" * 60)

    if not _HAS_KCT:
        return

    print(f"\n  NCCL available : {getattr(kct, 'HAVE_NCCL', False)}")
    print(f"  (set HAVE_NCCL=True to enable real multi-GPU NCCL collectives)")

    rng = np.random.default_rng(7)

    R = 2    # simulated ranks
    M = 8    # batch × seq_len (decode step)
    N = 256  # total output features (e.g. hidden_dim)
    K = 256  # input features

    # ------------------------------------------------------------------
    # Col-parallel linear  (first linear in MLP/attention projection)
    # ------------------------------------------------------------------
    print(f"\n--- Col-parallel linear  R={R}, M={M}, N={N}, K={K} ---")
    print(f"  W [{N},{K}] split column-wise: each rank holds W_rank [{N//R},{K}]")

    x = rng.standard_normal((M, K)).astype(np.float32)   # replicated on all ranks

    # Each rank holds a different shard of the weight matrix
    W_col = [rng.standard_normal((N//R, K)).astype(np.float32) for _ in range(R)]

    # Per-rank computation: y_rank = x @ W_rank^T
    y_ranks = [kct.col_parallel_linear(x, W_col[r]) for r in range(R)]
    print(f"  y_rank shape per rank: {y_ranks[0].shape}")

    # All-gather: concatenate along the N axis
    y_col = kct.allgather(y_ranks)                # [M, N] via launch_allgather
    print(f"  After allgather: y shape {y_col.shape}")

    # Reference: full matmul W_full [N, K] = cat(W_col, axis=0)
    W_full = np.concatenate(W_col, axis=0)        # [N, K]
    y_ref  = x @ W_full.T                         # [M, N]
    err_col = float(np.max(np.abs(y_col - y_ref)))
    print(f"  Max error vs full matmul: {err_col:.2e}  "
          f"{'PASS' if err_col < 1e-3 else 'FAIL'}")

    # ------------------------------------------------------------------
    # Row-parallel linear  (second linear in each MLP block)
    # ------------------------------------------------------------------
    print(f"\n--- Row-parallel linear  R={R}, M={M}, N={N}, K={K} ---")
    print(f"  W [{N},{K}] split row-wise: each rank holds W_rank [{N},{K//R}]")

    # Each rank has its own input shard and weight shard
    x_ranks = [rng.standard_normal((M, K//R)).astype(np.float32) for _ in range(R)]
    W_row   = [rng.standard_normal((N, K//R)).astype(np.float32) for _ in range(R)]

    # Per-rank: partial = x_rank @ W_rank^T  [M, N]
    partials = [kct.row_parallel_linear(x_ranks[r], W_row[r]) for r in range(R)]
    print(f"  partial shape per rank: {partials[0].shape}")

    # All-reduce (sum) — in production this is ncclAllReduce
    kct.ring_allreduce(partials)                   # in-place on each partial
    y_row = partials[0]                            # all ranks now hold the same result
    print(f"  After allreduce: y shape {y_row.shape}")

    # Reference: full matmul with concatenated x and W
    x_full = np.concatenate(x_ranks, axis=1)      # [M, K]
    W_full2 = np.concatenate(W_row, axis=1)       # [N, K]
    y_ref2  = x_full @ W_full2.T                  # [M, N]
    err_row = float(np.max(np.abs(y_row - y_ref2)))
    print(f"  Max error vs full matmul: {err_row:.2e}  "
          f"{'PASS' if err_row < 1e-3 else 'FAIL'}")

    # ------------------------------------------------------------------
    # End-to-end MLP block: col-parallel → GeLU → row-parallel
    # ------------------------------------------------------------------
    print(f"\n--- MLP block: col-parallel → GeLU → row-parallel ---")
    # In Megatron-LM FFN: hidden=N_out=4*N, input=N, output=N
    N_ff = N * 2  # feed-forward expansion
    W1 = [rng.standard_normal((N_ff//R, K)).astype(np.float32) for _ in range(R)]
    W2 = [rng.standard_normal((K, N_ff//R)).astype(np.float32) for _ in range(R)]

    # Step 1: col-parallel with GeLU (no communication needed between GELU and rank compute)
    h_ranks = []
    for r in range(R):
        h_r = kct.col_parallel_linear(x, W1[r])    # [M, N_ff//R]
        h_r = h_r * (1 / (1 + np.exp(-1.702 * h_r)))   # approximate GELU
        h_ranks.append(h_r)

    # Step 2: row-parallel (each rank gets its GELU output as x_rank)
    W2_row = [rng.standard_normal((K, N_ff//R)).astype(np.float32) for _ in range(R)]
    out_partials = [kct.row_parallel_linear(h_ranks[r], W2_row[r]) for r in range(R)]
    kct.ring_allreduce(out_partials)    # one all-reduce per MLP block
    y_mlp = out_partials[0]
    print(f"  MLP output shape: {y_mlp.shape}  (one all-reduce, no mid-block comm)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n--- Summary ---")
    print(f"  Col-parallel max error: {err_col:.2e}")
    print(f"  Row-parallel max error: {err_row:.2e}")
    print(f"\n  Megatron-LM communication pattern:")
    print(f"  FFN block = col-parallel → GeLU → row-parallel → all-reduce")
    print(f"  Only ONE all-reduce per FFN (not two) — key efficiency insight.")
    print(f"\n  For multi-GPU NCCL usage:")
    print(f"  handles = kct.nccl_comm_init([0, 1, 2, 3])  # one per device")
    print(f"  kct.nccl_allreduce(handles[rank], partial)   # per-rank call")
    print(f"\n  See src/kernels/transformer/tensor_parallel.cu")
    print(f"  See src/kernels/transformer/tensor_parallel_nccl.cu")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
