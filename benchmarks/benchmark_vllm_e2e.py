"""
benchmark_vllm_e2e.py
End-to-end benchmark: kernel-craft backend vs vLLM default backend.

Metrics
-------
- Time to first token (TTFT) — latency for the first generated token
- Throughput (tokens/s) — sustained generation rate
- Peak VRAM (MB) — peak GPU memory during inference

Hardware requirements
---------------------
- GPU with ≥ 8 GB VRAM (fits 8B AWQ-INT4 or 3B FP16)
- CUDA ≥ 12.1 (required by vLLM)
- PyTorch ≥ 2.1

Usage
-----
::

    # Default: run kernel-craft vs default on opt-125m (small, for CI)
    python benchmarks/benchmark_vllm_e2e.py

    # Full run with a quantized 8B model
    python benchmarks/benchmark_vllm_e2e.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --quantization awq \\
        --batch-sizes 1 4 8 \\
        --output-len 256

Kernel-only fallback
--------------------
If vLLM is not installed, the script falls back to benchmarking the raw
kernel-craft CUDA kernels (FlashAttention + PagedAttention) directly, which
lets you characterize kernel performance without a full LLM runtime.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Imports — guarded so the script runs even without vLLM / torch
# ---------------------------------------------------------------------------

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

# Always available (pybind11 module, built by cmake)
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(_REPO_ROOT / "src" / "python" / "build"))

try:
    import kernel_craft_torch_ops as kc_ops
    HAS_KC_OPS = True
except Exception:
    HAS_KC_OPS = False

try:
    import kernel_craft_transformer as _kct
    HAS_KCT = True
except ImportError:
    HAS_KCT = False

import numpy as np


# ---------------------------------------------------------------------------
# Kernel-level benchmarks (fallback when vLLM is absent)
# ---------------------------------------------------------------------------

def _time_ms(fn, warmup: int = 5, iters: int = 50) -> float:
    """Return average kernel time in milliseconds using CUDA events."""
    if HAS_TORCH:
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        for _ in range(warmup):
            fn()
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / iters
    # CPU timing fallback
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000


def bench_flash_attention(
    configs: Optional[List[tuple]] = None,
    output_file=None,
) -> None:
    """
    Benchmark FlashAttention prefill at various (B, H, N, d) configs.

    :param configs   List of (B, H, H_kv, N, d, causal) tuples.
    :param output_file File object to write results to (or None for stdout).
    """
    if not HAS_TORCH:
        print("PyTorch not available — skipping FlashAttention benchmark")
        return

    import kernel_craft_torch_ops as ops

    configs = configs or [
        (1, 32, 8, 512,  128, False),
        (1, 32, 8, 1024, 128, True),
        (4, 32, 8, 512,  128, False),
        (1, 32, 8, 2048, 128, True),
    ]

    hdr = f"{'Config':40s} {'Time(ms)':>10} {'TFLOPS':>10}"
    print("\n=== FlashAttention Prefill ===")
    print(hdr)
    if output_file:
        output_file.write("FlashAttention Prefill\n")

    for (B, H, H_kv, N, d, causal) in configs:
        dev = torch.device("cuda")
        Q = torch.randn(B, H,    N, d, device=dev)
        K = torch.randn(B, H_kv, N, d, device=dev)
        V = torch.randn(B, H_kv, N, d, device=dev)

        ms = _time_ms(lambda: ops.flash_attention(Q, K, V, H_kv=H_kv, causal=causal))

        # FLOPs: 2 * B * H * N^2 * d (Q@K^T + attn@V)
        flops = 2.0 * B * H * N * N * d
        if causal:
            flops /= 2  # ~half tiles skipped
        tflops = flops / (ms * 1e9)

        label = f"B={B} H={H} H_kv={H_kv} N={N} d={d} causal={causal}"
        print(f"  {label:38s} {ms:>10.3f} {tflops:>10.2f}")
        if output_file:
            output_file.write(f"  FlashAttn {label}: {ms:.3f} ms, {tflops:.2f} TFLOPS\n")


def bench_paged_attention(
    configs: Optional[List[tuple]] = None,
    output_file=None,
) -> None:
    """
    Benchmark PagedAttention decode at various (B, H, H_kv, d, seq_len, block_size) configs.

    :param configs   List of (B, H, H_kv, d, seq_len, block_size) tuples.
    :param output_file File object for results.
    """
    if not HAS_TORCH:
        print("PyTorch not available — skipping PagedAttention benchmark")
        return

    import kernel_craft_torch_ops as ops

    configs = configs or [
        (1,  32, 8, 128, 256,  16),
        (1,  32, 8, 128, 1024, 16),
        (8,  32, 8, 128, 256,  16),
        (8,  32, 8, 128, 1024, 32),
        (16, 32, 8, 128, 512,  32),
    ]

    hdr = f"{'Config':50s} {'Time(ms)':>10} {'Eff.BW(GB/s)':>14}"
    print("\n=== PagedAttention Decode ===")
    print(hdr)
    if output_file:
        output_file.write("\nPagedAttention Decode\n")

    for (B, H, H_kv, d, seq_len, bs) in configs:
        dev = torch.device("cuda")
        max_blocks = (seq_len + bs - 1) // bs
        num_phys   = B * max_blocks

        Q          = torch.randn(B, H,    d,        device=dev)
        K_pool     = torch.randn(num_phys, bs, H_kv, d, device=dev)
        V_pool     = torch.randn(num_phys, bs, H_kv, d, device=dev)
        block_table= torch.arange(num_phys, dtype=torch.int32, device=dev).view(B, max_blocks)
        seq_lens   = torch.full((B,), seq_len, dtype=torch.int32, device=dev)

        ms = _time_ms(lambda: ops.paged_attention(
            Q, block_table, K_pool, V_pool, seq_lens,
            H_kv=H_kv, block_size=bs,
        ))

        # Effective bandwidth: read Q + K_pool + V_pool, write O
        bytes_io = (B * H * d + 2 * num_phys * bs * H_kv * d + B * H * d) * 4
        bw_gbs   = bytes_io / (ms * 1e6)

        label = f"B={B} H={H} H_kv={H_kv} d={d} seq={seq_len} bs={bs}"
        print(f"  {label:48s} {ms:>10.3f} {bw_gbs:>14.2f}")
        if output_file:
            output_file.write(f"  PagedAttn {label}: {ms:.3f} ms, {bw_gbs:.2f} GB/s\n")


# ---------------------------------------------------------------------------
# vLLM end-to-end benchmark
# ---------------------------------------------------------------------------

def bench_vllm(
    model: str,
    quantization: Optional[str],
    batch_sizes: List[int],
    input_len: int,
    output_len: int,
    backend: str,
    output_file=None,
) -> None:
    """
    Run vLLM end-to-end benchmark with the specified backend.

    :param model        HuggingFace model identifier.
    :param quantization Quantization method ("awq", "gptq", or None).
    :param batch_sizes  Batch sizes to sweep.
    :param input_len    Number of input tokens per request.
    :param output_len   Number of output tokens to generate.
    :param backend      vLLM attention backend name ("kernel_craft" or default).
    :param output_file  Results file.
    """
    if not HAS_VLLM:
        print("vLLM not installed — skipping end-to-end benchmark.")
        print("Install with: pip install vllm  (requires CUDA ≥ 12.1)")
        return

    if backend == "kernel_craft":
        import kernel_craft_vllm_backend
        kernel_craft_vllm_backend.register()
        os.environ["VLLM_ATTENTION_BACKEND"] = "kernel_craft"
    else:
        os.environ.pop("VLLM_ATTENTION_BACKEND", None)

    print(f"\n=== vLLM E2E Benchmark — model={model} backend={backend} ===")
    if output_file:
        output_file.write(f"\nvLLM E2E: model={model} backend={backend}\n")

    llm = LLM(
        model=model,
        quantization=quantization,
        dtype="float16",
        max_num_batched_tokens=max(batch_sizes) * (input_len + output_len),
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=output_len)

    # Warm-up
    dummy_prompt = "The quick brown fox " * (input_len // 5)
    llm.generate([dummy_prompt], sampling)

    hdr = f"  {'Batch':>6} {'TTFT(ms)':>10} {'Throughput(tok/s)':>18} {'VRAM(MB)':>10}"
    print(hdr)

    for bs in batch_sizes:
        prompts = [dummy_prompt] * bs
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        outputs = llm.generate(prompts, sampling)

        elapsed_s = time.perf_counter() - t0
        total_out  = sum(len(o.outputs[0].token_ids) for o in outputs)
        throughput = total_out / elapsed_s
        vram_mb    = torch.cuda.max_memory_allocated() / (1024 ** 2)
        # TTFT approximation: single-item latency (single-request)
        ttft_ms = elapsed_s * 1000 / max(output_len, 1) if bs == 1 else float("nan")

        print(f"  {bs:>6} {ttft_ms:>10.1f} {throughput:>18.1f} {vram_mb:>10.0f}")
        if output_file:
            output_file.write(
                f"  bs={bs}: ttft={ttft_ms:.1f}ms thpt={throughput:.1f}tok/s vram={vram_mb:.0f}MB\n"
            )

    del llm
    if HAS_TORCH:
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="kernel-craft vs vLLM end-to-end benchmark"
    )
    parser.add_argument(
        "--model", default="facebook/opt-125m",
        help="HuggingFace model ID (default: facebook/opt-125m for CI speed)"
    )
    parser.add_argument(
        "--quantization", default=None, choices=[None, "awq", "gptq"],
        help="Quantization method"
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[1, 4, 8],
        help="Batch sizes to sweep"
    )
    parser.add_argument("--input-len",  type=int, default=128)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument(
        "--kernels-only", action="store_true",
        help="Skip vLLM, only benchmark raw CUDA kernels"
    )
    parser.add_argument(
        "--backends", nargs="+", default=["kernel_craft", "default"],
        help="vLLM backends to compare"
    )
    args = parser.parse_args()

    out_dir = _REPO_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    rpt_path = out_dir / "benchmark_vllm_e2e.txt"

    with open(rpt_path, "w") as rpt:
        rpt.write("kernel-craft vLLM E2E Benchmark\n\n")

        # Always run kernel-level benchmarks
        bench_flash_attention(output_file=rpt)
        bench_paged_attention(output_file=rpt)

        if not args.kernels_only:
            for backend in args.backends:
                bench_vllm(
                    model=args.model,
                    quantization=args.quantization,
                    batch_sizes=args.batch_sizes,
                    input_len=args.input_len,
                    output_len=args.output_len,
                    backend=backend,
                    output_file=rpt,
                )
        else:
            print("\n(--kernels-only: vLLM E2E skipped)")
            rpt.write("\n(kernels-only run: vLLM E2E skipped)\n")

    print(f"\nReport written to {rpt_path}")


if __name__ == "__main__":
    main()
