#!/usr/bin/env python3
"""
run_all.py — run every Python example and report pass / skip / fail.

Layout
------
    python/
    ├── cnn/          conv_naive_numpy, conv_tiled_numpy, conv_naive_torch,
    │                 conv_tiled_torch, memory_pool, mixed_precision
    ├── transformer/  flash_attention, fp8_quant, speculative_decoding,
    │                 tensor_parallel
    └── vllm/         torch_ops, opt125m

Usage
-----
    python examples/python/run_all.py              # run all
    python examples/python/run_all.py --filter conv # name contains 'conv'
    python examples/python/run_all.py --filter cnn  # entire cnn group
    python examples/python/run_all.py --list        # list without running

Exit code: 0 if all ran examples passed, 1 if any failed.
"""

import argparse
import importlib.util
import os
import sys
import time
import traceback

# ---------------------------------------------------------------------------
# Example registry
# Each entry: (rel_path_no_ext, description, requires)
# rel_path_no_ext is relative to this file's directory (examples/python/).
# requires: package names that must be importable; example skipped otherwise.
# ---------------------------------------------------------------------------
EXAMPLES = [
    # ---- cnn/ ---------------------------------------------------------------
    ("cnn/conv_naive_numpy",  "Naive conv — NumPy arrays",          ["kernel_craft_python"]),
    ("cnn/conv_tiled_numpy",  "Tiled conv — NumPy, tile-size sweep",["kernel_craft_python"]),
    ("cnn/conv_naive_torch",  "Naive conv — PyTorch CUDA tensors",  ["torch", "kernel_craft_python"]),
    ("cnn/conv_tiled_torch",  "Tiled conv — PyTorch CUDA tensors",  ["torch", "kernel_craft_python"]),
    ("cnn/memory_pool",       "Batch processing & memory reuse",    ["kernel_craft_python"]),
    ("cnn/mixed_precision",   "FP16 / TF32 precision trade-offs",   ["kernel_craft_python"]),
    # ---- transformer/ -------------------------------------------------------
    ("transformer/flash_attention",     "FlashAttention prefill (MHA/GQA/causal)", ["kernel_craft_transformer"]),
    ("transformer/fp8_quant",           "FP8 E4M3 quantize / dequantize",          ["kernel_craft_transformer"]),
    ("transformer/speculative_decoding","Draft token rejection sampling",           ["kernel_craft_transformer"]),
    ("transformer/tensor_parallel",     "Col/row parallel linear + collectives",    ["kernel_craft_transformer"]),
    ("transformer/mamba_ops",           "Selective scan, depthwise Conv1d, RMSNorm",["kernel_craft_transformer"]),
    # ---- vllm/ --------------------------------------------------------------
    ("vllm/torch_ops",  "torch.ops.kernel_craft namespace",          ["torch"]),
    ("vllm/opt125m",    "OPT-125M with kernel-craft vLLM backend",
     ["torch", "vllm", "kernel_craft_torch_ops"]),
]

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "..", "..")

for _p in [
    os.path.join(_REPO, "src", "python", "build"),
    os.path.join(_REPO, "src", "python"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _can_import(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _load_and_run(rel_path: str) -> None:
    """Load examples/python/<rel_path>.py and call its main()."""
    full_path = os.path.join(_HERE, f"{rel_path}.py")
    module_name = rel_path.replace("/", ".")
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "main"):
        mod.main()


def run(filter_str: str = "") -> int:
    results = []

    for rel_path, description, requires in EXAMPLES:
        name = rel_path  # e.g. "cnn/conv_naive_numpy"
        if filter_str and filter_str.lower() not in name.lower():
            continue

        missing = [r for r in requires if not _can_import(r)]
        if missing:
            results.append((name, "SKIP", f"missing: {', '.join(missing)}", 0.0))
            continue

        if "torch" in requires and not _has_cuda():
            results.append((name, "SKIP", "CUDA not available", 0.0))
            continue

        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"  {description}")
        print(f"{'='*60}")

        t0 = time.perf_counter()
        try:
            _load_and_run(rel_path)
            elapsed = time.perf_counter() - t0
            results.append((name, "PASS", "", elapsed))
        except SystemExit as e:
            elapsed = time.perf_counter() - t0
            status = "PASS" if e.code == 0 else "FAIL"
            results.append((name, status, f"sys.exit({e.code})", elapsed))
        except Exception:
            elapsed = time.perf_counter() - t0
            tb = traceback.format_exc()
            results.append((name, "FAIL", tb.strip().splitlines()[-1], elapsed))
            print(tb)

    print(f"\n\n{'='*60}")
    print(f"  Example results")
    print(f"{'='*60}")
    n_pass = n_fail = n_skip = 0
    for name, status, note, elapsed in results:
        tag = {"PASS": "✓", "FAIL": "✗", "SKIP": "–"}.get(status, "?")
        time_str = f"{elapsed:.2f}s" if elapsed > 0 else "  –  "
        note_str = f"  ({note})" if note and status != "PASS" else ""
        print(f"  {tag} {status:<4}  {time_str:>6}  {name}{note_str}")
        if status == "PASS": n_pass += 1
        elif status == "FAIL": n_fail += 1
        else: n_skip += 1

    print(f"\n  {n_pass} passed  {n_skip} skipped  {n_fail} failed")
    return n_fail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filter", default="",
                        help="Only run examples whose path contains this string")
    parser.add_argument("--list", action="store_true",
                        help="List available examples without running")
    args = parser.parse_args()

    if args.list:
        print("Available examples  (examples/python/):")
        prev_group = None
        for rel_path, desc, req in EXAMPLES:
            group = rel_path.split("/")[0]
            if group != prev_group:
                print(f"\n  {group}/")
                prev_group = group
            ok = all(_can_import(r) for r in req)
            status = "ready" if ok else f"needs: {', '.join(r for r in req if not _can_import(r))}"
            filename = rel_path.split("/")[-1]
            print(f"    {filename:<35}  {desc}  [{status}]")
        return

    sys.exit(0 if run(filter_str=args.filter) == 0 else 1)


if __name__ == "__main__":
    main()
