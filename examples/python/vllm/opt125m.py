#!/usr/bin/env python3
"""
example_vllm_opt125m.py — OPT-125M inference with the kernel-craft vLLM backend.

Overview
--------
This example shows how to register the kernel-craft attention backend with vLLM
and run end-to-end text generation on OPT-125M, a small open-source language
model (125M parameters, ~250 MB in fp16) suitable for development and testing
without requiring large GPU memory.

Architecture of OPT-125M
--------------------------
  Embedding  : vocab_size=50272, hidden=768
  Transformer : 12 layers × (MHA + FFN)
  MHA        : 12 heads, head_dim=64, no GQA (H_kv = H)
  FFN        : hidden=768 → 3072 → 768
  Parameters : 125M

Attention kernel routing in kernel-craft backend
-------------------------------------------------
  Prefill (seq_len > 1):
    → KernelCraftAttentionImpl.forward() calls kc_ops.flash_attention()
       (FlashAttention: online softmax, O(N·d) shmem, causal mask)

  Decode (seq_len = 1, new token per step):
    → KernelCraftAttentionImpl.forward() calls kc_ops.paged_attention()
       (PagedAttention: non-contiguous KV pages via block table)

  KV cache layout:
    [2, num_blocks, block_size, H_kv, d]  — index 0=K, 1=V
    Managed by vLLM's BlockManager; kernel-craft never allocates KV pages.

How to activate the kernel-craft backend
-----------------------------------------
Option A — Environment variable (before starting vLLM process):
    export VLLM_ATTENTION_BACKEND=kernel_craft

Option B — Programmatic (before creating LLM):
    import kernel_craft_vllm_backend
    kernel_craft_vllm_backend.register()

    from vllm import LLM
    llm = LLM(model="facebook/opt-125m")

Option C — Monkey-patch for testing without modifying vLLM internals:
    See the test_vllm_backend.py approach using mock backends.

Requirements
------------
  - vLLM ≥ 0.21.0 (v1 API)         pip install vllm==0.21.0
  - torch ≥ 2.0 with CUDA           pip install torch --index-url .../cu121
  - kernel-craft built               cmake --build build --target kernels
  - ~4 GB VRAM for OPT-125M fp16
  - Model weights downloaded:        huggingface-cli download facebook/opt-125m

Larger models (same backend, different vRAM):
  facebook/opt-1.3b   — 2.6 GB fp16, fits in 8 GB VRAM
  facebook/opt-6.7b   — 13 GB fp16, requires multi-GPU or int4 quantization
  meta-llama/Llama-3.1-8B-Instruct + AWQ — 4 GB, fits on RTX 4070 8 GB

Usage
-----
    cd examples/python
    python example_vllm_opt125m.py

    # With a custom backend (option A):
    VLLM_ATTENTION_BACKEND=kernel_craft python example_vllm_opt125m.py

Benchmark mode:
    python example_vllm_opt125m.py --benchmark

Note on WSL2
------------
WSL2 has no GPU peer-to-peer memory, so tensor parallelism across GPUs is not
testable locally.  Run single-GPU only (tensor_parallel_size=1).
"""

import sys
import os
import time
import argparse

_REPO = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_REPO, "src", "python", "build"))
sys.path.insert(0, os.path.join(_REPO, "src", "python"))


def check_dependencies() -> bool:
    """Verify required libraries are available before importing vLLM."""
    missing = []
    try:
        import torch
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available — inference will run on CPU (very slow).")
    except ImportError:
        missing.append("torch")

    try:
        import vllm  # noqa: F401
    except ImportError:
        missing.append("vllm (pip install vllm==0.21.0)")

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r src/python/requirements.txt")
        return False
    return True


def register_backend() -> bool:
    """
    Register the kernel-craft attention backend with vLLM.

    Returns True if registration succeeded, False if the backend module is
    unavailable (falls back to vLLM's default FlashAttention backend).

    The registration patches vLLM's backend registry to return
    KernelCraftAttentionBackend instead of FlashAttentionBackend when
    VLLM_ATTENTION_BACKEND=kernel_craft or register() was called.
    """
    try:
        import kernel_craft_vllm_backend as kc_backend  # type: ignore
        kc_backend.register()
        print("kernel-craft vLLM backend registered.")
        print(f"  FlashAttention (prefill) : kernel_craft_torch_ops.flash_attention")
        print(f"  PagedAttention (decode)  : kernel_craft_torch_ops.paged_attention")
        return True
    except ImportError as e:
        print(f"kernel_craft_vllm_backend not available ({e})")
        print("Falling back to vLLM's default FlashAttention backend.")
        return False
    except Exception as e:
        print(f"Backend registration failed: {e}")
        return False


def run_generation(model_name: str, prompts: list, max_tokens: int = 50,
                   temperature: float = 0.8, use_kc_backend: bool = True) -> dict:
    """
    Run text generation with vLLM using the kernel-craft backend.

    Parameters
    ----------
    model_name   : HuggingFace model name or local path.
    prompts      : List of prompt strings.
    max_tokens   : Maximum new tokens to generate per prompt.
    temperature  : Sampling temperature (0.0 = greedy, 1.0 = full sampling).
    use_kc_backend: If True, activate kernel-craft attention; else use vLLM default.

    Returns
    -------
    dict with keys: outputs, ttft_ms, throughput_tps, vram_gb
    """
    from vllm import LLM, SamplingParams

    if use_kc_backend:
        use_kc_backend = register_backend()

    print(f"\nLoading model: {model_name}")
    print(f"Backend      : {'kernel-craft' if use_kc_backend else 'vLLM default'}")

    t_load_start = time.perf_counter()
    llm = LLM(
        model=model_name,
        dtype="float16",                   # fp16 inference, fits on 8 GB VRAM
        gpu_memory_utilization=0.85,       # leave 15% for KV cache growth
        max_model_len=2048,                # max context window
        tensor_parallel_size=1,            # single GPU (WSL2 has no GPU P2P)
        enforce_eager=True,                # disable CUDA Graphs in vLLM (use ours)
    )
    t_load = time.perf_counter() - t_load_start
    print(f"Model loaded in {t_load:.1f}s")

    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["\n\n"],                     # stop at paragraph boundary
    )

    # ---- First token latency (TTFT) ----
    # Measure time-to-first-token by generating 1 token on a single short prompt.
    ttft_prompt = [prompts[0]]
    ttft_params  = SamplingParams(temperature=0.0, max_tokens=1)
    t0 = time.perf_counter()
    _ = llm.generate(ttft_prompt, ttft_params)
    ttft_ms = (time.perf_counter() - t0) * 1e3
    print(f"\nTime-to-first-token (TTFT): {ttft_ms:.1f} ms")

    # ---- Full generation ----
    print(f"\nGenerating {max_tokens} tokens for {len(prompts)} prompts ...")
    t_gen_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    t_gen = time.perf_counter() - t_gen_start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput   = total_tokens / t_gen

    # ---- VRAM usage ----
    try:
        import torch
        vram_gb = torch.cuda.max_memory_allocated() / 1024**3
    except Exception:
        vram_gb = 0.0

    return {
        "outputs": outputs,
        "ttft_ms": ttft_ms,
        "throughput_tps": throughput,
        "total_tokens": total_tokens,
        "gen_time_s": t_gen,
        "vram_gb": vram_gb,
    }


def run_benchmark(model_name: str) -> None:
    """
    Compare kernel-craft backend vs vLLM default on representative prompts.

    Measures:
      - TTFT (prefill latency)     → tests FlashAttention kernel
      - Decode throughput (tok/s)  → tests PagedAttention kernel
      - Peak VRAM (GB)
    """
    from vllm import LLM, SamplingParams

    prompts = [
        "The history of artificial intelligence began",
        "The key difference between machine learning and deep learning is",
        "In quantum computing, a qubit differs from a classical bit because",
    ]

    print("\n" + "=" * 60)
    print("Benchmark: kernel-craft vs vLLM default attention backend")
    print("=" * 60)

    for backend_name, use_kc in [("vLLM default", False), ("kernel-craft", True)]:
        print(f"\n--- Backend: {backend_name} ---")
        stats = run_generation(model_name, prompts, max_tokens=64,
                               temperature=0.0, use_kc_backend=use_kc)
        print(f"  TTFT              : {stats['ttft_ms']:.1f} ms")
        print(f"  Throughput        : {stats['throughput_tps']:.1f} tokens/s")
        print(f"  Total tokens      : {stats['total_tokens']}")
        print(f"  Generation time   : {stats['gen_time_s']:.2f} s")
        print(f"  Peak VRAM         : {stats['vram_gb']:.2f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(description="OPT-125M with kernel-craft backend")
    parser.add_argument("--model",     default="facebook/opt-125m",
                        help="HuggingFace model name or local path")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark comparing backends")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--no-backend", action="store_true",
                        help="Use vLLM default backend (skip kernel-craft)")
    args = parser.parse_args()

    print("=" * 60)
    print("kernel-craft vLLM backend — OPT-125M example")
    print("=" * 60)

    if not check_dependencies():
        sys.exit(1)

    # ------------------------------------------------------------------
    # Demo prompts covering different text styles
    # ------------------------------------------------------------------
    prompts = [
        "The key advantage of transformer models over RNNs is",
        "GPU memory bandwidth is the bottleneck in LLM inference because",
        "PagedAttention improves LLM serving efficiency by",
    ]

    if args.benchmark:
        run_benchmark(args.model)
        return

    # ------------------------------------------------------------------
    # Single run with kernel-craft backend
    # ------------------------------------------------------------------
    stats = run_generation(
        model_name=args.model,
        prompts=prompts,
        max_tokens=args.max_tokens,
        temperature=0.8,
        use_kc_backend=not args.no_backend,
    )

    print("\n" + "=" * 60)
    print("Generated outputs:")
    print("=" * 60)
    for i, out in enumerate(stats["outputs"]):
        prompt_text = out.prompt
        generated   = out.outputs[0].text
        n_tokens    = len(out.outputs[0].token_ids)
        print(f"\n[{i+1}] Prompt  : {prompt_text}")
        print(f"     Generated: {generated.strip()}")
        print(f"     Tokens   : {n_tokens}")

    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    print(f"  TTFT (prefill)    : {stats['ttft_ms']:.1f} ms")
    print(f"  Throughput        : {stats['throughput_tps']:.1f} tokens/s")
    print(f"  Total tokens gen  : {stats['total_tokens']}")
    print(f"  Peak VRAM used    : {stats['vram_gb']:.2f} GB")

    print("\n  Kernel routing summary:")
    print("  Prefill → flash_attention  (src/kernels/transformer/flash_attention.cu)")
    print("  Decode  → paged_attention  (src/kernels/transformer/paged_attention.cu)")
    print("\n  To switch to vLLM's built-in backend:")
    print("    python example_vllm_opt125m.py --no-backend")
    print("  To benchmark both:")
    print("    python example_vllm_opt125m.py --benchmark")
    print("\n  For larger models (same backend):")
    print("    python example_vllm_opt125m.py --model facebook/opt-1.3b")
    print("    python example_vllm_opt125m.py --model facebook/opt-6.7b  # needs 16 GB VRAM")
    print("=" * 60)


if __name__ == "__main__":
    main()
