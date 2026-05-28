"""
kernel_craft_otel.py
OpenTelemetry instrumentation for kernel-craft GPU kernels.

Central OTEL tracing module: setup, kernel_span context manager, no-op fallback.

Setup
-----
::

    from kernel_craft_otel import setup_tracing, kernel_span
    setup_tracing(endpoint="http://localhost:4317", service_name="kernel-craft")

    with kernel_span("flash_attention", batch=1, heads=8):
        result = kc_ops.flash_attention(Q, K, V, H_kv=8)

Graceful degradation
--------------------
If ``setup_tracing()`` is never called (or ``opentelemetry-api`` is not
installed), ``kernel_span()`` is a no-op context manager with zero overhead.
All imports are guarded so this module is safe to import in environments
without OTEL dependencies.
"""

from __future__ import annotations

import contextlib
from typing import Any, Optional

# ---------------------------------------------------------------------------
# OTEL imports — optional; fall back to no-ops if not installed
# ---------------------------------------------------------------------------

try:
    from opentelemetry import trace as _ot_trace
    _HAS_OTEL_API = True
except ImportError:
    _HAS_OTEL_API = False

_tracer: Any = None
_provider: Any = None


def _get_tracer() -> Any:
    """Return the global tracer (OTEL no-op tracer if setup_tracing() was never called)."""
    global _tracer
    if _tracer is None:
        if _HAS_OTEL_API:
            _tracer = _ot_trace.get_tracer("kernel_craft")
        else:
            _tracer = _NoOpTracer()
    return _tracer


def setup_tracing(
    endpoint: str = "http://localhost:4317",
    service_name: str = "kernel-craft",
    exporter: Any = None,
) -> None:
    """Configure OpenTelemetry tracing for kernel-craft.

    Install a TracerProvider backed by OTLPSpanExporter (or a custom exporter).

    :param endpoint:     OTLP gRPC collector endpoint (e.g. Jaeger). Ignored when
                         ``exporter`` is supplied directly.
    :param service_name: Service name tag on all emitted spans.
    :param exporter:     Optional custom SpanExporter for testing
                         (e.g. ``InMemorySpanExporter``). When ``None`` and
                         ``opentelemetry-exporter-otlp-proto-grpc`` is installed,
                         an ``OTLPSpanExporter`` targeting ``endpoint`` is used.

    Calling ``setup_tracing()`` more than once is safe — subsequent calls are
    no-ops (idempotent guard on ``_provider``).
    """
    global _tracer, _provider

    if not _HAS_OTEL_API:
        return  # silently skip; kernel_span() will be a no-op

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource

    # Idempotent: if already configured, do nothing
    if _provider is not None:
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if exporter is None:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
            # Production path: batch for performance
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except ImportError:
            pass  # OTLP exporter not installed; no exporter attached
    else:
        # Test / custom exporter path: synchronous export so spans are visible immediately
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        provider.add_span_processor(SimpleSpanProcessor(exporter))

    _ot_trace.set_tracer_provider(provider)
    _provider = provider
    _tracer = provider.get_tracer("kernel_craft")


@contextlib.contextmanager
def kernel_span(name: str, **attrs: Any):
    """Context manager: wraps a kernel call in an OTEL span with true GPU timing.

    Start an OTEL child span, record CUDA events, set kernel.cuda_ms on exit.

    :param name:  Span name (e.g. ``"flash_attention"``).
    :param attrs: Arbitrary keyword attributes added to the span
                  (e.g. ``batch=1, heads=8, seq_len=512``).

    Uses ``torch.cuda.Event(enable_timing=True)`` pairs so ``kernel.cuda_ms``
    reflects pure GPU execution time, not Python overhead.

    When ``setup_tracing()`` was never called, or when OTEL/CUDA are
    unavailable, the context manager is a silent no-op.
    """
    tracer = _get_tracer()

    # Capture CUDA timing events before entering the span
    cuda_start = cuda_stop = None
    try:
        import torch
        if torch.cuda.is_available():
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_stop  = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
    except Exception:
        pass

    with tracer.start_as_current_span(name) as span:
        try:
            for k, v in attrs.items():
                try:
                    span.set_attribute(k, v)
                except Exception:
                    pass
            yield span
        finally:
            if cuda_start is not None and cuda_stop is not None:
                try:
                    import torch
                    cuda_stop.record()
                    torch.cuda.synchronize()
                    elapsed_ms = cuda_start.elapsed_time(cuda_stop)
                    try:
                        span.set_attribute("kernel.cuda_ms", elapsed_ms)
                    except Exception:
                        pass
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# No-op fallback objects (used when opentelemetry-api is not installed)
# ---------------------------------------------------------------------------

class _NoOpSpan:
    """Minimal no-op span: silently ignores all attribute and status calls."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    # Support ``span.context`` access used by trace hierarchy checks
    context = None


class _NoOpTracer:
    """Minimal no-op tracer used when opentelemetry-api is not installed."""

    @contextlib.contextmanager
    def start_as_current_span(self, name: str, **kwargs: Any):
        yield _NoOpSpan()
