"""
otel_tracing.py — OpenTelemetry inference observability example (Phase 16).

Demonstrates the two-level tracing instrumentation added in Phase 16:

    - kernel_span()   — wraps individual GPU kernel calls; records kernel.cuda_ms
    - setup_tracing() — installs a TracerProvider (OTLP or in-memory for demos)
    - Span hierarchy  — request → prefill/decode → kernel_span children

Instrumented kernels (kernel_craft_torch_ops.py)
-------------------------------------------------
    flash_attention   → span "flash_attention"  attr: batch, heads, seq_len
    paged_attention   → span "paged_attention"  attr: batch, heads, seq_len
    int4_gemv         → span "int4_gemv"        attr: rows, cols
    fp8_quantize      → span "fp8_quantize"     attr: rows, cols
    speculative_verify→ span "speculative_verify" attr: draft_len

vLLM backend hierarchy (kernel_craft_vllm_backend.py)
------------------------------------------------------
    vllm.forward   (request span)
    ├── vllm.prefill
    │   └── flash_attention   (kernel span)
    └── vllm.decode
        └── paged_attention   (kernel span)

Graceful degradation
--------------------
If opentelemetry-api / opentelemetry-sdk are not installed, kernel_span() is a
silent no-op (zero overhead).  This example detects OTEL availability and shows
the appropriate demo:
  - With OTEL: uses InMemorySpanExporter to collect and print spans without
               a running Jaeger / OTLP collector.
  - Without OTEL: shows how kernel_span() degrades transparently.

For a real collector (Jaeger):
    docker run -p 4317:4317 jaegertracing/all-in-one
    from kernel_craft_otel import setup_tracing
    setup_tracing(endpoint="http://localhost:4317", service_name="my-app")

Usage
-----
    pip install opentelemetry-api opentelemetry-sdk   # optional
    python examples/python/observability/otel_tracing.py
"""

import os
import sys

_REPO = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
sys.path.insert(0, os.path.join(_REPO, "src", "python", "build"))
sys.path.insert(0, os.path.join(_REPO, "src", "python"))

# ---------------------------------------------------------------------------
# Check for OTEL availability
# ---------------------------------------------------------------------------
try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    _HAS_OTEL_SDK = True
except ImportError:
    _HAS_OTEL_SDK = False

try:
    import torch
    _HAS_TORCH = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH = False

try:
    import kernel_craft_otel as otel
    _HAS_KC_OTEL = True
except ImportError:
    _HAS_KC_OTEL = False


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------

def _print_spans(spans):
    """Pretty-print a list of finished OTEL spans."""
    for span in spans:
        attrs = dict(span.attributes or {})
        cuda_ms = attrs.pop("kernel.cuda_ms", None)
        attr_str = "  ".join(f"{k}={v}" for k, v in attrs.items())
        cuda_str = f"  cuda_ms={cuda_ms:.3f}" if cuda_ms is not None else ""
        parent = span.parent.span_id if span.parent else None
        parent_str = f"  parent={parent:#x}" if parent else "  (root)"
        print(f"    [{span.name}]{parent_str}  {attr_str}{cuda_str}")


# ---------------------------------------------------------------------------
# Demo 1: kernel_span with InMemorySpanExporter (no external collector)
# ---------------------------------------------------------------------------

def demo_in_memory_spans():
    print("=" * 60)
    print("Demo 1: kernel_span() with InMemorySpanExporter")
    print("=" * 60)

    if not _HAS_OTEL_SDK:
        print("  opentelemetry-sdk not installed.")
        print("  Install: pip install opentelemetry-api opentelemetry-sdk")
        print("  Showing no-op behaviour instead.\n")
        _demo_noop()
        return

    if not _HAS_KC_OTEL:
        print("  kernel_craft_otel not found — add src/python to PYTHONPATH")
        return

    exporter = InMemorySpanExporter()
    provider  = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Patch the module to use our test provider so spans are collected
    import opentelemetry.trace as _ot
    otel._provider = provider
    otel._tracer   = provider.get_tracer("kernel_craft")

    tracer = provider.get_tracer("example")

    # Simulate a request span containing two kernel spans
    with tracer.start_as_current_span("vllm.prefill") as req_span:
        req_span.set_attribute("batch_size", 1)
        req_span.set_attribute("seq_len", 512)

        with otel.kernel_span("flash_attention", batch=1, heads=8, seq_len=512):
            # Simulate the kernel call with a short sleep
            import time
            time.sleep(0.002)

        with otel.kernel_span("fp8_quantize", rows=512, cols=4096):
            time.sleep(0.001)

    with tracer.start_as_current_span("vllm.decode") as dec_span:
        dec_span.set_attribute("batch_size", 1)
        dec_span.set_attribute("step", 1)

        with otel.kernel_span("paged_attention", batch=1, heads=8, seq_len=256):
            time.sleep(0.001)

    spans = exporter.get_finished_spans()
    print(f"  Collected {len(spans)} spans:\n")
    _print_spans(spans)

    # Verify parent-child structure
    kernel_spans = [s for s in spans if s.name in
                    ("flash_attention", "fp8_quantize", "paged_attention")]
    assert all(s.parent is not None for s in kernel_spans), \
        "kernel spans should have a parent"
    print(f"\n  Parent-child hierarchy: PASS ({len(kernel_spans)} kernel spans have parents)")
    print()


def _demo_noop():
    """Show that kernel_span is a safe no-op without OTEL."""
    import contextlib
    # Simulate kernel_span as nullcontext (what kernel_craft_otel does)
    with contextlib.nullcontext() as span:
        pass
    print("  kernel_span() → nullcontext (no-op, zero overhead)")
    print("  No spans collected; no errors raised.")
    print()


# ---------------------------------------------------------------------------
# Demo 2: CUDA timing attribute (kernel.cuda_ms)
# ---------------------------------------------------------------------------

def demo_cuda_timing():
    print("=" * 60)
    print("Demo 2: kernel.cuda_ms (GPU-only timing)")
    print("=" * 60)

    if not _HAS_OTEL_SDK:
        print("  opentelemetry-sdk not installed — skipping.\n")
        return
    if not _HAS_TORCH:
        print("  CUDA not available — kernel.cuda_ms will be absent.\n")
        return
    if not _HAS_KC_OTEL:
        print("  kernel_craft_otel not found — skipping.\n")
        return

    import torch
    import time

    exporter = InMemorySpanExporter()
    provider  = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel._provider = provider
    otel._tracer   = provider.get_tracer("kernel_craft")

    # Warm up CUDA
    _ = torch.randn(128, 128, device="cuda") @ torch.randn(128, 128, device="cuda")
    torch.cuda.synchronize()

    with otel.kernel_span("matmul_demo", rows=1024, cols=1024):
        # Simple matmul as a stand-in for a real kernel call
        A = torch.randn(1024, 1024, device="cuda")
        B = torch.randn(1024, 1024, device="cuda")
        C = A @ B
        torch.cuda.synchronize()

    spans = exporter.get_finished_spans()
    span  = spans[0]
    cuda_ms = (span.attributes or {}).get("kernel.cuda_ms")

    if cuda_ms is not None:
        print(f"  kernel.cuda_ms = {cuda_ms:.3f} ms  (GPU-only; excludes Python overhead)")
        print(f"  PASS — CUDA timing recorded in span attributes")
    else:
        print("  kernel.cuda_ms not present (CUDA event timing unavailable)")
    print()


# ---------------------------------------------------------------------------
# Demo 3: OTLP collector instructions
# ---------------------------------------------------------------------------

def demo_otlp_instructions():
    print("=" * 60)
    print("Demo 3: Production OTLP setup (Jaeger / Grafana Tempo)")
    print("=" * 60)
    print("""
  To send spans to a local Jaeger instance:

    # Start Jaeger all-in-one (OTLP gRPC on :4317, UI on :16686)
    docker run -p 4317:4317 -p 16686:16686 jaegertracing/all-in-one

    # In your inference code:
    from kernel_craft_otel import setup_tracing
    setup_tracing(
        endpoint="http://localhost:4317",
        service_name="my-llm-server",
    )
    # kernel_span() calls now emit spans automatically

  Environment variable alternative:
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 python serve.py

  Span names emitted by kernel-craft:
    flash_attention    — prefill attention (attrs: batch, heads, seq_len)
    paged_attention    — decode attention  (attrs: batch, heads, seq_len)
    int4_gemv          — INT4 dequant+GEMV (attrs: rows, cols)
    fp8_quantize       — FP8 E4M3 quantize (attrs: rows, cols)
    speculative_verify — draft token check  (attrs: draft_len)

  All kernel spans carry  kernel.cuda_ms  — pure GPU execution time.
""")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("kernel-craft — OpenTelemetry inference observability (Phase 16)")
    print("Module: kernel_craft_otel  |  src/python/kernel_craft_otel.py\n")

    if not _HAS_KC_OTEL:
        print("WARNING: kernel_craft_otel not importable.")
        print("  Add src/python to PYTHONPATH or run from the repo root.\n")

    demo_in_memory_spans()
    demo_cuda_timing()
    demo_otlp_instructions()

    print("Done.")


if __name__ == "__main__":
    main()
