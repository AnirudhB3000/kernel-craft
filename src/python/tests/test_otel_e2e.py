"""
test_otel_e2e.py
Comprehensive end-to-end tests for kernel-craft OTEL and vLLM integrations.

Covers
------
Kernel spans (kernel_craft_torch_ops)
  - All 5 instrumented ops: flash_attention, paged_attention, int4_gemv,
    fp8_quantize, speculative_verify — span emission, cuda_ms, user attrs
  - All five ops in one combined trace

vLLM impl.forward() spans (kernel_craft_vllm_backend)
  - Prefill: vllm.forward → prefill → flash_attention hierarchy
  - Decode:  vllm.forward → decode  → paged_attention hierarchy
  - Multi-request prefill (B=2): two prefill spans per forward call
  - GQA via impl (H_kv < H)
  - attn_metadata=None: output zeroed, no vllm.forward span
  - alibi_slopes / sliding_window raises NotImplementedError

vLLM backend API
  - register() idempotency
  - get_kv_cache_shape() layout (2, num_blocks, block_size, H_kv, d)
  - forward_includes_kv_cache_update == False

torch.ops.kernel_craft dispatch
  - flash_attention, paged_attention, fp8_quantize, speculative_verify callable

OTEL cross-cutting
  - kernel.cuda_ms >= 0 on all kernel spans
  - No ERROR status on any span
  - Correct parent-child hierarchy throughout

Note: does NOT use vllm.LLM() — vLLM v1 runs inference in a subprocess;
InMemorySpanExporter spans cannot cross process boundaries.

Run with:
    pytest src/python/tests/test_otel_e2e.py -v -m e2e
"""

import math
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

if torch.cuda.device_count() == 0:
    pytest.skip("No CUDA device available", allow_module_level=True)

try:
    import kernel_craft_torch_ops as ops
    HAS_OPS = True
except Exception:
    HAS_OPS = False

try:
    import kernel_craft_vllm_backend as kc_backend
    HAS_VLLM = kc_backend.HAS_VLLM
except Exception:
    HAS_VLLM = False

requires_ops  = pytest.mark.skipif(not HAS_OPS,  reason="kernel_craft_torch_ops not available")
requires_vllm = pytest.mark.skipif(not HAS_VLLM, reason="vLLM not installed")


# ---------------------------------------------------------------------------
# OTEL isolation fixture — reset module singletons before each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_otel_state():
    import kernel_craft_otel as _m
    orig_tracer, orig_provider = _m._tracer, _m._provider
    _m._tracer   = None
    _m._provider = None
    yield
    _m._tracer   = orig_tracer
    _m._provider = orig_provider


def _setup_tracing():
    """Install a fresh InMemorySpanExporter; return (exporter, tracer)."""
    try:
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    except ImportError:
        pytest.skip("opentelemetry-sdk not installed")
    from kernel_craft_otel import setup_tracing, _get_tracer
    exporter = InMemorySpanExporter()
    setup_tracing(service_name="kc-e2e-test", exporter=exporter)
    return exporter, _get_tracer()


# ---------------------------------------------------------------------------
# Input helpers for vLLM impl tests
# ---------------------------------------------------------------------------

def _prefill_inputs(B, H, H_kv, N, d, block_size=16):
    """(query, key, value, kv_cache, output, meta) for a prefill forward call."""
    meta_cls  = kc_backend.KernelCraftMetadata
    total     = B * N
    max_blocks = max((N + block_size - 1) // block_size, 1)
    num_phys  = B * max_blocks

    query    = torch.randn(total, H,    d, device="cuda")
    key      = torch.randn(total, H_kv, d, device="cuda")
    value    = torch.randn(total, H_kv, d, device="cuda")
    kv_cache = torch.randn(2, num_phys, block_size, H_kv, d, device="cuda")
    output   = torch.zeros(total, H * d, device="cuda")

    qsl = torch.zeros(B + 1, dtype=torch.int32, device="cuda")
    for i in range(B):
        qsl[i + 1] = (i + 1) * N

    meta = meta_cls(
        num_actual_tokens=total,
        max_query_len=N,
        max_seq_len=N,
        query_start_loc=qsl,
        seq_lens=torch.full((B,), N, dtype=torch.int32, device="cuda"),
        block_table=torch.zeros(B, max_blocks, dtype=torch.int32, device="cuda"),
        slot_mapping=torch.arange(total, dtype=torch.int64, device="cuda"),
        causal=False,
    )
    return query, key, value, kv_cache, output, meta


def _decode_inputs(B, H, H_kv, N, d, block_size=16):
    """(query, key, value, kv_cache, output, meta) for a decode forward call."""
    meta_cls   = kc_backend.KernelCraftMetadata
    max_blocks = max((N + block_size - 1) // block_size, 1)
    num_phys   = B * max_blocks  # non-overlapping physical blocks per request

    query    = torch.randn(B, H,    d, device="cuda")
    key      = torch.randn(B, H_kv, d, device="cuda")
    value    = torch.randn(B, H_kv, d, device="cuda")
    kv_cache = torch.randn(2, num_phys, block_size, H_kv, d, device="cuda")
    output   = torch.zeros(B, H * d, device="cuda")

    # Each request owns its own physical blocks (no aliasing)
    bt = torch.zeros(B, max_blocks, dtype=torch.int32, device="cuda")
    for b in range(B):
        for blk in range(max_blocks):
            bt[b, blk] = b * max_blocks + blk

    meta = meta_cls(
        num_actual_tokens=B,
        max_query_len=1,
        max_seq_len=N,
        query_start_loc=torch.arange(B + 1, dtype=torch.int32, device="cuda"),
        seq_lens=torch.full((B,), N, dtype=torch.int32, device="cuda"),
        block_table=bt,
        slot_mapping=torch.arange(B, dtype=torch.int64, device="cuda"),
    )
    return query, key, value, kv_cache, output, meta


def _make_impl(H=4, d=64, H_kv=4):
    return kc_backend.KernelCraftAttentionImpl(
        num_heads=H, head_size=d, scale=1.0 / math.sqrt(d),
        num_kv_heads=H_kv, alibi_slopes=None, sliding_window=None,
        kv_cache_dtype="auto",
    )


# ===========================================================================
# Kernel span emission
# ===========================================================================

@pytest.mark.e2e
@requires_ops
def test_flash_attention_span_emitted():
    exporter, _ = _setup_tracing()
    B, H, N, d = 1, 4, 32, 64
    Q = torch.randn(B, H, N, d, device="cuda")
    ops.flash_attention(Q, Q.clone(), Q.clone(), H_kv=H, causal=False)
    assert any(s.name == "flash_attention" for s in exporter.get_finished_spans())


@pytest.mark.e2e
@requires_ops
def test_paged_attention_span_emitted():
    exporter, _ = _setup_tracing()
    B, H, N, d, bs = 1, 4, 32, 64, 16
    max_blocks = (N + bs - 1) // bs
    Q  = torch.randn(B, H, d, device="cuda")
    bt = torch.arange(max_blocks, dtype=torch.int32, device="cuda").unsqueeze(0).expand(B, -1).contiguous()
    Kp = torch.randn(max_blocks, bs, H, d, device="cuda")
    Vp = torch.randn(max_blocks, bs, H, d, device="cuda")
    sl = torch.tensor([N], dtype=torch.int32, device="cuda")
    ops.paged_attention(Q, bt, Kp, Vp, sl, H_kv=H, block_size=bs)
    assert any(s.name == "paged_attention" for s in exporter.get_finished_spans())


@pytest.mark.e2e
@requires_ops
def test_int4_gemv_span_emitted():
    exporter, _ = _setup_tracing()
    rows, cols, gs = 256, 512, 128
    n_groups = cols // gs
    packed = torch.zeros(rows, cols // 2, dtype=torch.uint8, device="cuda")
    scales = torch.ones(rows, n_groups, dtype=torch.float32, device="cuda")
    zeros  = torch.zeros(rows, n_groups // 2, dtype=torch.uint8, device="cuda")
    x      = torch.randn(cols, device="cuda")
    ops.int4_gemv(packed, scales, zeros, x, rows=rows, cols=cols, group_size=gs)
    assert any(s.name == "int4_gemv" for s in exporter.get_finished_spans())


@pytest.mark.e2e
@requires_ops
def test_fp8_quantize_span_emitted():
    exporter, _ = _setup_tracing()
    ops.fp8_quantize(torch.randn(16, 128, device="cuda"), per_token=True)
    assert any(s.name == "fp8_quantize" for s in exporter.get_finished_spans())


@pytest.mark.e2e
@requires_ops
def test_speculative_verify_span_emitted():
    exporter, _ = _setup_tracing()
    K, V = 4, 32
    unif = torch.ones(K, V, device="cuda") / V
    ops.speculative_verify(unif, unif.clone(),
                           torch.zeros(K, dtype=torch.int32, device="cuda"),
                           torch.rand(K, device="cuda"), torch.rand(K, device="cuda"))
    assert any(s.name == "speculative_verify" for s in exporter.get_finished_spans())


@pytest.mark.e2e
@requires_ops
def test_all_five_kernel_spans_in_one_trace():
    """All five instrumented ops emit spans within a single parent trace."""
    exporter, tracer = _setup_tracing()
    B, H, N, d, bs = 1, 4, 32, 64, 16

    with tracer.start_as_current_span("combined_trace"):
        Q = torch.randn(B, H, N, d, device="cuda")
        ops.flash_attention(Q, Q.clone(), Q.clone(), H_kv=H, causal=False)

        max_blocks = (N + bs - 1) // bs
        Qd = torch.randn(B, H, d, device="cuda")
        bt = torch.arange(max_blocks, dtype=torch.int32, device="cuda").unsqueeze(0).expand(B, -1).contiguous()
        Kp = torch.randn(max_blocks, bs, H, d, device="cuda")
        Vp = torch.randn(max_blocks, bs, H, d, device="cuda")
        ops.paged_attention(Qd, bt, Kp, Vp,
                            torch.tensor([N], dtype=torch.int32, device="cuda"),
                            H_kv=H, block_size=bs)

        rows, cols, gs = 128, 256, 128
        n_groups = cols // gs
        packed = torch.zeros(rows, cols // 2, dtype=torch.uint8, device="cuda")
        scales = torch.ones(rows, n_groups, dtype=torch.float32, device="cuda")
        zeros  = torch.zeros(rows, n_groups // 2, dtype=torch.uint8, device="cuda")
        ops.int4_gemv(packed, scales, zeros,
                      torch.randn(cols, device="cuda"), rows=rows, cols=cols, group_size=gs)

        ops.fp8_quantize(torch.randn(8, 64, device="cuda"), per_token=True)

        K2, V2 = 4, 32
        unif = torch.ones(K2, V2, device="cuda") / V2
        ops.speculative_verify(unif, unif.clone(),
                               torch.zeros(K2, dtype=torch.int32, device="cuda"),
                               torch.rand(K2, device="cuda"), torch.rand(K2, device="cuda"))

    emitted  = {s.name for s in exporter.get_finished_spans()}
    expected = {"flash_attention", "paged_attention", "int4_gemv", "fp8_quantize", "speculative_verify"}
    assert expected <= emitted, f"Missing spans: {expected - emitted}"


# ===========================================================================
# Span attribute correctness
# ===========================================================================

@pytest.mark.e2e
@requires_ops
def test_flash_attention_span_attributes():
    """flash_attention span carries batch, heads, kv_heads, seq_len, head_dim, causal."""
    exporter, _ = _setup_tracing()
    B, H, H_kv, N, d = 1, 4, 2, 32, 64
    Q = torch.randn(B, H,    N, d, device="cuda")
    K = torch.randn(B, H_kv, N, d, device="cuda")
    ops.flash_attention(Q, K, K.clone(), H_kv=H_kv, causal=True)
    span = next(s for s in exporter.get_finished_spans() if s.name == "flash_attention")
    a = span.attributes
    assert a["batch"] == B and a["heads"] == H and a["kv_heads"] == H_kv
    assert a["seq_len"] == N and a["head_dim"] == d and a["causal"] is True


@pytest.mark.e2e
@requires_ops
def test_paged_attention_span_attributes():
    """paged_attention span carries batch, heads, kv_heads, seq_len, block_size."""
    exporter, _ = _setup_tracing()
    B, H, N, d, bs = 1, 4, 32, 64, 16
    max_blocks = (N + bs - 1) // bs
    Q  = torch.randn(B, H, d, device="cuda")
    bt = torch.arange(max_blocks, dtype=torch.int32, device="cuda").unsqueeze(0).expand(B, -1).contiguous()
    Kp = torch.randn(max_blocks, bs, H, d, device="cuda")
    ops.paged_attention(Q, bt, Kp, Kp.clone(),
                        torch.tensor([N], dtype=torch.int32, device="cuda"),
                        H_kv=H, block_size=bs)
    span = next(s for s in exporter.get_finished_spans() if s.name == "paged_attention")
    a = span.attributes
    assert a["batch"] == B and a["block_size"] == bs and a["seq_len"] == N


@pytest.mark.e2e
@requires_ops
def test_int4_gemv_span_attributes():
    """int4_gemv span carries rows, cols, group_size."""
    exporter, _ = _setup_tracing()
    rows, cols, gs = 256, 512, 128
    n_groups = cols // gs
    packed = torch.zeros(rows, cols // 2, dtype=torch.uint8, device="cuda")
    scales = torch.ones(rows, n_groups, dtype=torch.float32, device="cuda")
    zeros  = torch.zeros(rows, n_groups // 2, dtype=torch.uint8, device="cuda")
    ops.int4_gemv(packed, scales, zeros,
                  torch.randn(cols, device="cuda"), rows=rows, cols=cols, group_size=gs)
    span = next(s for s in exporter.get_finished_spans() if s.name == "int4_gemv")
    assert span.attributes["rows"] == rows
    assert span.attributes["cols"] == cols
    assert span.attributes["group_size"] == gs


@pytest.mark.e2e
@requires_ops
def test_fp8_quantize_span_attributes():
    """fp8_quantize span carries rows, cols, per_token."""
    exporter, _ = _setup_tracing()
    rows, cols = 16, 128
    ops.fp8_quantize(torch.randn(rows, cols, device="cuda"), per_token=True)
    span = next(s for s in exporter.get_finished_spans() if s.name == "fp8_quantize")
    assert span.attributes["rows"] == rows
    assert span.attributes["cols"] == cols
    assert span.attributes["per_token"] is True


@pytest.mark.e2e
@requires_ops
def test_speculative_verify_span_attributes():
    """speculative_verify span carries num_tokens, vocab_size."""
    exporter, _ = _setup_tracing()
    K, Vs = 5, 50
    unif = torch.ones(K, Vs, device="cuda") / Vs
    ops.speculative_verify(unif, unif.clone(),
                           torch.zeros(K, dtype=torch.int32, device="cuda"),
                           torch.rand(K, device="cuda"), torch.rand(K, device="cuda"))
    span = next(s for s in exporter.get_finished_spans() if s.name == "speculative_verify")
    assert span.attributes["num_tokens"] == K
    assert span.attributes["vocab_size"] == Vs


@pytest.mark.e2e
@requires_ops
def test_all_kernel_spans_have_cuda_ms():
    """kernel.cuda_ms is set and >= 0 on every instrumented kernel span."""
    exporter, _ = _setup_tracing()
    B, H, N, d, bs = 1, 4, 32, 64, 16

    Q = torch.randn(B, H, N, d, device="cuda")
    ops.flash_attention(Q, Q.clone(), Q.clone(), H_kv=H, causal=False)

    max_blocks = (N + bs - 1) // bs
    Qd = torch.randn(B, H, d, device="cuda")
    bt = torch.arange(max_blocks, dtype=torch.int32, device="cuda").unsqueeze(0).expand(B, -1).contiguous()
    Kp = torch.randn(max_blocks, bs, H, d, device="cuda")
    ops.paged_attention(Qd, bt, Kp, Kp.clone(),
                        torch.tensor([N], dtype=torch.int32, device="cuda"),
                        H_kv=H, block_size=bs)

    rows, cols, gs = 128, 256, 128
    n_groups = cols // gs
    packed = torch.zeros(rows, cols // 2, dtype=torch.uint8, device="cuda")
    scales = torch.ones(rows, n_groups, dtype=torch.float32, device="cuda")
    zeros  = torch.zeros(rows, n_groups // 2, dtype=torch.uint8, device="cuda")
    ops.int4_gemv(packed, scales, zeros,
                  torch.randn(cols, device="cuda"), rows=rows, cols=cols, group_size=gs)

    ops.fp8_quantize(torch.randn(8, 64, device="cuda"), per_token=True)

    K2, V2 = 4, 32
    unif = torch.ones(K2, V2, device="cuda") / V2
    ops.speculative_verify(unif, unif.clone(),
                           torch.zeros(K2, dtype=torch.int32, device="cuda"),
                           torch.rand(K2, device="cuda"), torch.rand(K2, device="cuda"))

    kernel_names = {"flash_attention", "paged_attention", "int4_gemv",
                    "fp8_quantize", "speculative_verify"}
    for span in exporter.get_finished_spans():
        if span.name in kernel_names:
            cuda_ms = span.attributes.get("kernel.cuda_ms")
            assert cuda_ms is not None, f"'{span.name}' missing kernel.cuda_ms"
            assert cuda_ms >= 0.0,      f"'{span.name}' cuda_ms={cuda_ms} < 0"


@pytest.mark.e2e
@requires_ops
def test_no_kernel_span_has_error_status():
    """No span emitted by kernel operations has ERROR status."""
    from opentelemetry.trace import StatusCode
    exporter, _ = _setup_tracing()
    Q = torch.randn(1, 4, 32, 64, device="cuda")
    ops.flash_attention(Q, Q.clone(), Q.clone(), H_kv=4, causal=False)
    for span in exporter.get_finished_spans():
        assert span.status.status_code != StatusCode.ERROR, \
            f"Span '{span.name}' has ERROR: {span.status.description}"


# ===========================================================================
# vLLM impl.forward() — prefill
# ===========================================================================

@pytest.mark.e2e
@requires_vllm
@requires_ops
def test_impl_prefill_emits_vllm_forward_span():
    exporter, _ = _setup_tracing()
    impl = _make_impl()
    q, k, v, kv, out, meta = _prefill_inputs(B=1, H=4, H_kv=4, N=32, d=64)
    impl.forward(None, q, k, v, kv, meta, out)
    assert any(s.name == "vllm.forward" for s in exporter.get_finished_spans())


@pytest.mark.e2e
@requires_vllm
@requires_ops
def test_impl_prefill_span_hierarchy():
    """impl prefill: flash_attention ← prefill ← vllm.forward (grandparent)."""
    exporter, _ = _setup_tracing()
    impl = _make_impl()
    q, k, v, kv, out, meta = _prefill_inputs(B=1, H=4, H_kv=4, N=32, d=64)
    impl.forward(None, q, k, v, kv, meta, out)
    spans = exporter.get_finished_spans()
    fwd = next(s for s in spans if s.name == "vllm.forward")
    pre = next(s for s in spans if s.name == "prefill")
    fa  = next(s for s in spans if s.name == "flash_attention")
    assert pre.parent.span_id == fwd.context.span_id, "prefill must be child of vllm.forward"
    assert fa.parent.span_id  == pre.context.span_id, "flash_attention must be child of prefill"


@pytest.mark.e2e
@requires_vllm
@requires_ops
def test_impl_prefill_is_decode_false():
    exporter, _ = _setup_tracing()
    impl = _make_impl()
    q, k, v, kv, out, meta = _prefill_inputs(B=1, H=4, H_kv=4, N=32, d=64)
    impl.forward(None, q, k, v, kv, meta, out)
    fwd = next(s for s in exporter.get_finished_spans() if s.name == "vllm.forward")
    assert fwd.attributes.get("is_decode") is False


@pytest.mark.e2e
@requires_vllm
@requires_ops
def test_impl_prefill_multi_request_two_prefill_spans():
    """B=2 prefill emits exactly two 'prefill' spans (one per sequence)."""
    exporter, _ = _setup_tracing()
    impl = _make_impl()
    q, k, v, kv, out, meta = _prefill_inputs(B=2, H=4, H_kv=4, N=32, d=64)
    impl.forward(None, q, k, v, kv, meta, out)
    prefill_spans = [s for s in exporter.get_finished_spans() if s.name == "prefill"]
    assert len(prefill_spans) == 2, f"Expected 2 prefill spans for B=2, got {len(prefill_spans)}"


@pytest.mark.e2e
@requires_vllm
@requires_ops
def test_impl_prefill_gqa():
    """Prefill with GQA (H_kv=2, H=4) completes and emits flash_attention span."""
    exporter, _ = _setup_tracing()
    H, H_kv, N, d = 4, 2, 32, 64
    impl = _make_impl(H=H, d=d, H_kv=H_kv)
    q, k, v, kv, out, meta = _prefill_inputs(B=1, H=H, H_kv=H_kv, N=N, d=d)
    result = impl.forward(None, q, k, v, kv, meta, out)
    assert any(s.name == "flash_attention" for s in exporter.get_finished_spans())
    assert result.shape == (N, H * d)


@pytest.mark.e2e
@requires_vllm
@requires_ops
def test_impl_prefill_output_shape_and_device():
    _setup_tracing()
    B, H, N, d = 1, 4, 32, 64
    impl = _make_impl(H=H, d=d, H_kv=H)
    q, k, v, kv, out, meta = _prefill_inputs(B=B, H=H, H_kv=H, N=N, d=d)
    result = impl.forward(None, q, k, v, kv, meta, out)
    assert result.shape == (B * N, H * d)
    assert result.device.type == "cuda"


# ===========================================================================
# vLLM impl.forward() — decode
# ===========================================================================

@pytest.mark.e2e
@requires_vllm
@requires_ops
def test_impl_decode_emits_vllm_forward_span():
    exporter, _ = _setup_tracing()
    impl = _make_impl()
    q, k, v, kv, out, meta = _decode_inputs(B=1, H=4, H_kv=4, N=32, d=64)
    impl.forward(None, q, k, v, kv, meta, out)
    assert any(s.name == "vllm.forward" for s in exporter.get_finished_spans())


@pytest.mark.e2e
@requires_vllm
@requires_ops
def test_impl_decode_span_hierarchy():
    """impl decode: paged_attention ← decode ← vllm.forward (grandparent)."""
    exporter, _ = _setup_tracing()
    impl = _make_impl()
    q, k, v, kv, out, meta = _decode_inputs(B=1, H=4, H_kv=4, N=32, d=64)
    impl.forward(None, q, k, v, kv, meta, out)
    spans = exporter.get_finished_spans()
    fwd = next(s for s in spans if s.name == "vllm.forward")
    dec = next(s for s in spans if s.name == "decode")
    pa  = next(s for s in spans if s.name == "paged_attention")
    assert dec.parent.span_id == fwd.context.span_id, "decode must be child of vllm.forward"
    assert pa.parent.span_id  == dec.context.span_id, "paged_attention must be child of decode"


@pytest.mark.e2e
@requires_vllm
@requires_ops
def test_impl_decode_is_decode_true():
    exporter, _ = _setup_tracing()
    impl = _make_impl()
    q, k, v, kv, out, meta = _decode_inputs(B=1, H=4, H_kv=4, N=32, d=64)
    impl.forward(None, q, k, v, kv, meta, out)
    fwd = next(s for s in exporter.get_finished_spans() if s.name == "vllm.forward")
    assert fwd.attributes.get("is_decode") is True


@pytest.mark.e2e
@requires_vllm
@requires_ops
def test_impl_decode_output_shape_and_device():
    _setup_tracing()
    B, H, N, d = 2, 4, 32, 64
    impl = _make_impl(H=H, d=d, H_kv=H)
    q, k, v, kv, out, meta = _decode_inputs(B=B, H=H, H_kv=H, N=N, d=d)
    result = impl.forward(None, q, k, v, kv, meta, out)
    assert result.shape == (B, H * d)
    assert result.device.type == "cuda"


@pytest.mark.e2e
@requires_vllm
@requires_ops
def test_impl_none_metadata_zeroes_output_no_forward_span():
    """attn_metadata=None: output is filled with 0, vllm.forward span not emitted."""
    exporter, _ = _setup_tracing()
    B, H, d = 1, 4, 64
    impl   = _make_impl()
    query  = torch.randn(B, H, d, device="cuda")
    kv     = torch.randn(2, 2, 16, H, d, device="cuda")
    output = torch.ones(B, H * d, device="cuda")  # non-zero sentinel
    result = impl.forward(None, query, query.clone(), query.clone(), kv, None, output)
    assert (result == 0.0).all(), "Expected zeros from None metadata"
    fwd_count = sum(1 for s in exporter.get_finished_spans() if s.name == "vllm.forward")
    assert fwd_count == 0, "No vllm.forward span should be emitted when metadata is None"


# ===========================================================================
# impl constructor validation
# ===========================================================================

@pytest.mark.e2e
@requires_vllm
def test_impl_alibi_slopes_raises():
    with pytest.raises(NotImplementedError, match="ALiBi"):
        kc_backend.KernelCraftAttentionImpl(
            num_heads=4, head_size=64, scale=1.0,
            alibi_slopes=[0.1, 0.2, 0.3, 0.4],
        )


@pytest.mark.e2e
@requires_vllm
def test_impl_sliding_window_raises():
    with pytest.raises(NotImplementedError, match="sliding-window"):
        kc_backend.KernelCraftAttentionImpl(
            num_heads=4, head_size=64, scale=1.0,
            sliding_window=512,
        )


# ===========================================================================
# vLLM backend API
# ===========================================================================

@pytest.mark.e2e
@requires_vllm
def test_backend_register_idempotent():
    """register() completes without raising; calling twice is safe."""
    kc_backend.register()
    kc_backend.register()


@pytest.mark.e2e
@requires_vllm
def test_kv_cache_shape_layout():
    """get_kv_cache_shape returns (2, num_blocks, block_size, H_kv, d)."""
    shape = kc_backend.KernelCraftAttentionBackend.get_kv_cache_shape(
        num_blocks=128, block_size=16, num_kv_heads=8, head_size=64,
    )
    assert len(shape) == 5
    assert shape[0] == 2    # K/V axis
    assert shape[1] == 128  # num_blocks
    assert shape[2] == 16   # block_size
    assert shape[3] == 8    # num_kv_heads
    assert shape[4] == 64   # head_size


@pytest.mark.e2e
@requires_vllm
def test_forward_includes_kv_cache_update_false():
    """vLLM manages KV cache via slot_mapping; forward() must not do it."""
    assert kc_backend.KernelCraftAttentionBackend.forward_includes_kv_cache_update is False


# ===========================================================================
# torch.ops.kernel_craft dispatch (torch.library registration)
# ===========================================================================

@pytest.mark.e2e
@requires_ops
def test_torch_ops_flash_attention_callable():
    assert ops.TORCH_OPS_REGISTERED, "torch.library registration failed at import"
    B, H, N, d = 1, 4, 32, 64
    Q = torch.randn(B, H, N, d, device="cuda")
    out = torch.ops.kernel_craft.flash_attention(Q, Q.clone(), Q.clone(), H, False)
    assert out.shape == (B, H, N, d)
    assert out.device.type == "cuda"


@pytest.mark.e2e
@requires_ops
def test_torch_ops_paged_attention_callable():
    assert ops.TORCH_OPS_REGISTERED
    B, H, N, d, bs = 1, 4, 32, 64, 16
    max_blocks = (N + bs - 1) // bs
    Q  = torch.randn(B, H, d, device="cuda")
    bt = torch.arange(max_blocks, dtype=torch.int32, device="cuda").unsqueeze(0).expand(B, -1).contiguous()
    Kp = torch.randn(max_blocks, bs, H, d, device="cuda")
    sl = torch.tensor([N], dtype=torch.int32, device="cuda")
    out = torch.ops.kernel_craft.paged_attention(Q, bt, Kp, Kp.clone(), sl, H, bs)
    assert out.shape == (B, H, d)


@pytest.mark.e2e
@requires_ops
def test_torch_ops_fp8_quantize_callable():
    assert ops.TORCH_OPS_REGISTERED
    x = torch.randn(8, 64, device="cuda")
    q, scales = torch.ops.kernel_craft.fp8_quantize(x, True)
    assert q.shape == (8, 64) and scales.shape == (8,)


@pytest.mark.e2e
@requires_ops
def test_torch_ops_speculative_verify_callable():
    assert ops.TORCH_OPS_REGISTERED
    K, V = 4, 32
    unif = torch.ones(K, V, device="cuda") / V
    acc, cor = torch.ops.kernel_craft.speculative_verify(
        unif, unif.clone(), torch.zeros(K, dtype=torch.int32, device="cuda"),
        torch.rand(K, device="cuda"), torch.rand(K, device="cuda"),
    )
    assert acc.shape == (K,) and cor.shape == (K,)


# ===========================================================================
# Original comprehensive e2e test (prefill + decode hierarchy via torch_ops)
# ===========================================================================

@pytest.mark.e2e
@requires_ops
def test_e2e_spans_present_and_valid():
    """
    Real GPU kernel calls produce well-formed OTEL span hierarchy.

    - >=2 vllm.forward spans, one prefill, one decode, one flash_attention,
      one paged_attention
    - Every kernel span has kernel.cuda_ms >= 0
    - No span has ERROR status
    - flash_attention / paged_attention are children of prefill / decode
    """
    exporter, tracer = _setup_tracing()
    from kernel_craft_otel import kernel_span

    B, H, N, d = 1, 4, 32, 64
    H_kv = H; block_size = 16; num_phys = 2; max_blocks = 2

    with tracer.start_as_current_span("vllm.forward") as fwd_pre:
        fwd_pre.set_attribute("is_decode", False)
        fwd_pre.set_attribute("batch", B)
        with kernel_span("prefill", seq_len=N):
            Q = torch.randn(B, H, N, d, device="cuda")
            ops.flash_attention(Q, Q.clone(), Q.clone(), H_kv=H_kv, causal=False)

    with tracer.start_as_current_span("vllm.forward") as fwd_dec:
        fwd_dec.set_attribute("is_decode", True)
        fwd_dec.set_attribute("batch", B)
        with kernel_span("decode", batch=B, max_seq_len=N):
            Qd = torch.randn(B, H, d, device="cuda")
            bt = torch.arange(max_blocks, dtype=torch.int32, device="cuda").unsqueeze(0).expand(B, -1).contiguous()
            Kp = torch.randn(num_phys, block_size, H_kv, d, device="cuda")
            sl = torch.tensor([N], dtype=torch.int32, device="cuda")
            ops.paged_attention(Qd, bt, Kp, Kp.clone(), sl, H_kv=H_kv, block_size=block_size)

    spans      = exporter.get_finished_spans()
    span_names = [s.name for s in spans]

    assert sum(n == "vllm.forward"    for n in span_names) >= 2
    assert any(n == "prefill"          for n in span_names), "No prefill span"
    assert any(n == "decode"           for n in span_names), "No decode span"
    assert any(n == "flash_attention"  for n in span_names), "No flash_attention span"
    assert any(n == "paged_attention"  for n in span_names), "No paged_attention span"

    kernel_names = {"flash_attention", "paged_attention", "int4_gemv",
                    "fp8_quantize", "speculative_verify"}
    for span in spans:
        if span.name in kernel_names:
            cuda_ms = span.attributes.get("kernel.cuda_ms")
            assert cuda_ms is not None, f"'{span.name}' missing kernel.cuda_ms"
            assert cuda_ms >= 0.0,      f"'{span.name}' cuda_ms={cuda_ms}"

    from opentelemetry.trace import StatusCode
    for span in spans:
        assert span.status.status_code != StatusCode.ERROR, \
            f"Span '{span.name}' has ERROR: {span.status.description}"

    prefill_decode_ids = {s.context.span_id for s in spans if s.name in ("prefill", "decode")}
    for span in spans:
        if span.name in ("flash_attention", "paged_attention"):
            assert span.parent is not None, f"'{span.name}' has no parent"
            assert span.parent.span_id in prefill_decode_ids, \
                f"'{span.name}' parent is not prefill/decode"
