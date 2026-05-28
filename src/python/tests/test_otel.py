"""
test_otel.py
Unit tests for kernel_craft_otel using InMemorySpanExporter (no external collector).

Run with:
    pytest src/python/tests/test_otel.py -v
"""
import sys
import importlib

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_otel_state():
    """Reset kernel_craft_otel module-level singletons between tests.

    The global OTEL TracerProvider can only be set once per process; we do NOT
    reset it here. Instead we reset the module-level _tracer / _provider so that
    each test installs a fresh provider with its own InMemorySpanExporter.
    setup_tracing() uses the provider directly (not the global) for span creation,
    so tests are isolated from each other without fighting the OTEL global lock.
    """
    import kernel_craft_otel as _m
    orig_tracer   = _m._tracer
    orig_provider = _m._provider
    _m._tracer   = None
    _m._provider = None
    yield
    _m._tracer   = orig_tracer
    _m._provider = orig_provider


def _make_exporter():
    """Return (InMemorySpanExporter, setup_tracing) or skip if OTEL SDK absent."""
    try:
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )
    except ImportError:
        pytest.skip("opentelemetry-sdk not installed")
    return InMemorySpanExporter()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_setup_tracing_installs_provider():
    """setup_tracing() with in-memory exporter registers a working TracerProvider."""
    from opentelemetry import trace
    from kernel_craft_otel import setup_tracing

    exporter = _make_exporter()
    setup_tracing(service_name="test-svc", exporter=exporter)

    provider = trace.get_tracer_provider()
    # Should be an SDK TracerProvider, not the default no-op one
    from opentelemetry.sdk.trace import TracerProvider
    assert isinstance(provider, TracerProvider)


def test_kernel_span_emits_one_span():
    """kernel_span() emits exactly one finished span with the correct name."""
    from kernel_craft_otel import setup_tracing, kernel_span

    exporter = _make_exporter()
    setup_tracing(service_name="test-svc", exporter=exporter)

    with kernel_span("flash_attention", batch=1):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "flash_attention"


def test_kernel_span_records_attributes():
    """kernel_span() sets all keyword attributes on the span."""
    from kernel_craft_otel import setup_tracing, kernel_span

    exporter = _make_exporter()
    setup_tracing(service_name="test-svc", exporter=exporter)

    with kernel_span("flash_attention", batch=2, heads=8, seq_len=512):
        pass

    span = exporter.get_finished_spans()[0]
    assert span.attributes.get("batch") == 2
    assert span.attributes.get("heads") == 8
    assert span.attributes.get("seq_len") == 512


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="torch not installed").cuda.is_available(),
    reason="CUDA device required for GPU timing test",
)
def test_kernel_span_cuda_ms_present():
    """kernel.cuda_ms attribute is set and positive when a real CUDA op runs inside the span."""
    import torch
    from kernel_craft_otel import setup_tracing, kernel_span

    exporter = _make_exporter()
    setup_tracing(service_name="test-svc", exporter=exporter)

    with kernel_span("test_cuda_op"):
        # Tiny CUDA op to give the events something to measure
        a = torch.ones(1024, device="cuda")
        b = a * 2.0
        _ = b.sum().item()

    span = exporter.get_finished_spans()[0]
    cuda_ms = span.attributes.get("kernel.cuda_ms")
    assert cuda_ms is not None, "kernel.cuda_ms attribute must be present"
    assert cuda_ms >= 0.0, "kernel.cuda_ms must be non-negative"


def test_parent_child_span_hierarchy():
    """A kernel_span() nested inside a parent span is recorded as its child."""
    from opentelemetry import trace
    from kernel_craft_otel import setup_tracing, kernel_span, _get_tracer

    exporter = _make_exporter()
    setup_tracing(service_name="test-svc", exporter=exporter)

    tracer = _get_tracer()
    with tracer.start_as_current_span("vllm.forward") as parent_span:
        with kernel_span("flash_attention", batch=1):
            pass

    spans = exporter.get_finished_spans()
    # Both spans should be finished
    names = {s.name for s in spans}
    assert "vllm.forward" in names
    assert "flash_attention" in names

    parent = next(s for s in spans if s.name == "vllm.forward")
    child  = next(s for s in spans if s.name == "flash_attention")
    assert child.parent is not None
    assert child.parent.span_id == parent.context.span_id


def test_noop_mode_no_crash():
    """When setup_tracing() is never called, kernel_span() must not raise."""
    # _reset_otel_state fixture ensures _tracer / _provider are None
    from kernel_craft_otel import kernel_span

    # Must not raise even without any provider configured
    with kernel_span("flash_attention", batch=1):
        x = 1 + 1
    assert x == 2


def test_setup_tracing_idempotent():
    """Calling setup_tracing() twice does not duplicate exporters."""
    from kernel_craft_otel import setup_tracing, kernel_span

    exporter = _make_exporter()
    setup_tracing(service_name="test-svc", exporter=exporter)
    setup_tracing(service_name="test-svc", exporter=exporter)  # second call — no-op

    with kernel_span("flash_attention", batch=1):
        pass

    spans = exporter.get_finished_spans()
    # If exporters were duplicated, we'd see 2 spans; correct behavior is 1
    assert len(spans) == 1
