# TODO List for kernel-craft


---

### Phase 16: OpenTelemetry Inference Observability ✅ COMPLETE

**Goal**: Add production-grade OpenTelemetry (OTEL) tracing so kernel-craft can be used as a live inference-time observability tool. Every kernel call emits a span with true CUDA-measured GPU time; every vLLM request emits a root trace showing the prefill/decode split and per-layer kernel breakdown. Throughput, latency percentiles, and errors become visible in Jaeger or Grafana without any changes to CUDA code.

**Architecture**: Pure Python instrumentation layer — no CUDA kernel changes. Two integration levels:
1. **Kernel level** (`kernel_craft_torch_ops.py`): each of the 5 kernel methods is wrapped in a `kernel_span()` context manager that records a `torch.cuda.Event` pair for true GPU time and emits an OTEL child span with kernel-specific attributes.
2. **Request level** (`kernel_craft_vllm_backend.py`): `forward()`, `_prefill_forward()`, and `_decode_forward()` each create a parent span with batch/sequence metadata, making kernel spans their children.

**Graceful degradation**: OTEL's API is designed so that if no provider is configured (deps not installed, or `setup_tracing()` not called), all spans are no-ops with zero overhead. The instrumentation code imports from `opentelemetry.api` only (no-op by default), never from `opentelemetry.sdk` directly.

**Existing codebase context for the implementing session**:
- `src/python/kernel_craft_torch_ops.py` — ctypes bridge to `libkernels.so`; 5 kernel methods to wrap: `flash_attention()` (line ~188), `paged_attention()` (line ~216), `int4_gemv()` (line ~287), `fp8_quantize()` (line ~321), `speculative_verify()` (line ~373). Each method has access to tensor shapes (B, H, N, d etc.) just before the ctypes call — those become span attributes.
- `src/python/kernel_craft_vllm_backend.py` — vLLM backend; `KernelCraftAttentionImpl.forward()` at line ~201 dispatches to `_prefill_forward()` (line ~277) or `_decode_forward()` (line ~244) based on `attn_metadata.max_query_len == 1`. No existing timing hooks.
- `src/python/pyproject.toml` — has `[project.optional-dependencies]` section; already has `observability` group to add to.
- Existing timing pattern (for reference): `benchmark_vllm_e2e.py` uses `torch.cuda.Event(enable_timing=True)` pairs — use the same approach inside `kernel_span()`.
- No existing `logging`, `metrics`, or `opentelemetry` imports anywhere in the project (confirmed by grep).

**New files to create**:
- [x] `src/python/kernel_craft_otel.py` — central OTEL module:
  - `setup_tracing(endpoint="http://localhost:4317", service_name="kernel-craft")`: configures `TracerProvider` with `OTLPSpanExporter` (gRPC) + `BatchSpanProcessor`; also registers a `PrometheusMetricReader` if `prometheus_client` is available (optional).
  - `kernel_span(name: str, **attrs)` context manager: on `__enter__` starts an OTEL child span and records a `torch.cuda.Event(enable_timing=True)` start event; on `__exit__` records the stop event, calls `torch.cuda.synchronize()`, computes `start.elapsed_time(stop)` ms, sets `kernel.cuda_ms` on the span, then ends the span. Falls back gracefully if torch/CUDA not available.
  - `_get_tracer()` — lazy singleton that returns the global tracer (OTEL no-op tracer if `setup_tracing()` was never called).
- [x] `src/python/tests/test_otel.py` — pytest tests:
  - Use `opentelemetry.sdk.trace.export.in_memory_span_exporter.InMemorySpanExporter` to capture spans without a real collector.
  - Test: `setup_tracing()` with in-memory exporter installs correctly.
  - Test: `kernel_span("flash_attention", batch=1)` emits exactly one span with `name="flash_attention"` and `batch=1` attribute.
  - Test: `kernel.cuda_ms` attribute is present and > 0 when a real CUDA op runs inside the span (skip if no GPU).
  - Test: parent-child relationship — a `"vllm.forward"` span wrapping a `kernel_span()` produces correct trace hierarchy.
  - Test: no-op mode — when `setup_tracing()` is never called, `kernel_span()` does not crash and emits no spans.
  - Test: `setup_tracing()` is idempotent (calling twice doesn't duplicate exporters).

**Files to modify**:
- [x] `src/python/kernel_craft_torch_ops.py` — wrap each of the 5 kernel methods with `kernel_span()`. Import pattern should be: `try: from kernel_craft_otel import kernel_span; except ImportError: from contextlib import nullcontext as kernel_span` — this ensures zero runtime error if OTEL is not installed. Span attributes per kernel:
  - `flash_attention`: `batch`, `heads`, `kv_heads`, `seq_len`, `head_dim`, `causal`
  - `paged_attention`: `batch`, `heads`, `kv_heads`, `seq_len`, `block_size`
  - `int4_gemv`: `rows`, `cols`, `group_size`
  - `fp8_quantize`: `rows`, `cols`, `per_token`
  - `speculative_verify`: `num_tokens`, `vocab_size`
- [x] `src/python/kernel_craft_vllm_backend.py` — add request-level spans:
  - In `forward()` (line ~201): wrap body in `with tracer.start_as_current_span("vllm.forward") as span: span.set_attribute("is_decode", is_decode); span.set_attribute("batch", B)` — kernel_span() calls inside will automatically be children.
  - In `_prefill_forward()` (line ~277): wrap per-request iteration with `kernel_span("prefill", seq_len=qlen)`.
  - In `_decode_forward()` (line ~244): wrap with `kernel_span("decode", batch=B, max_seq_len=int(seq_lens.max()))`.
- [x] `src/python/pyproject.toml` — add optional dependency group:
  ```toml
  [project.optional-dependencies]
  observability = [
      "opentelemetry-api>=1.20",
      "opentelemetry-sdk>=1.20",
      "opentelemetry-exporter-otlp-proto-grpc>=1.20",
  ]
  ```
  Install with: `pip install -e ".[observability]"`

**Testing strategy**:
- Unit tests (`test_otel.py`): use `InMemorySpanExporter` — no external collector needed. Run with `pytest src/python/tests/test_otel.py -v`.
- E2E integration test (`test_otel_e2e.py`): see below.
- Integration test (manual): `docker run -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one`, then run a short inference script that calls `setup_tracing()` then fires a few kernel calls. Open `http://localhost:16686`, search service `kernel-craft`, verify spans appear with `kernel.cuda_ms` attributes.
- Regression: run full Python test suite `pytest src/python/tests/ -v` — must still show 89+ pass / 0 skip (OTEL instrumentation is additive and optional).
- No-op regression: run the same inference script *without* calling `setup_tracing()` and confirm zero errors and no span output.

**E2E integration test** (`src/python/tests/test_otel_e2e.py`):
- [x] Create `src/python/tests/test_otel_e2e.py` — marked `@pytest.mark.e2e`; lives in its own suite separate from unit and integration tests. `make run_tests` triggers all suites in order: unit, integration, e2e, python, and benchmark. E2E suite can also be run in isolation: `pytest src/python/tests/test_otel_e2e.py -v -m e2e`. Add the `e2e` marker to `pytest.ini` (or `pyproject.toml` `[tool.pytest.ini_options]`) so pytest recognises it without warnings: `markers = ["e2e: end-to-end tests requiring a GPU and model download"]`. The CMake `run_python_tests` target should be updated to invoke all suites sequentially, with e2e last since it requires a model download.
- Skipped automatically if: vLLM not installed (`pytest.importorskip("vllm")`), no CUDA device (`torch.cuda.device_count() == 0`), or model not cached (guard with `pytest.mark.skipif`)
- Model: `facebook/opt-125m` — 125M params, <1 GB VRAM, no Hugging Face auth token needed; already used as reference model in `benchmarks/benchmark_vllm_e2e.py`
- Uses `InMemorySpanExporter` (no external Jaeger needed) — call `setup_tracing(exporter=InMemorySpanExporter())` at the start of the test; `setup_tracing()` must accept an optional `exporter` kwarg for testability
- Test flow:
  1. Load `facebook/opt-125m` via vLLM with `VLLM_ATTENTION_BACKEND=kernel_craft` env var set
  2. Run 3 short prompts (e.g., `"Hello"`, `"The capital of France is"`, `"1 + 1 ="`) to force both prefill and decode phases
  3. Collect finished spans from `InMemorySpanExporter`
- Assertions on collected spans:
  - At least one span named `"vllm.forward"` exists
  - At least one span named `"prefill"` and at least one named `"decode"` exist
  - At least one span named `"flash_attention"` or `"paged_attention"` exists
  - Every span with name in `{"flash_attention", "paged_attention", "int4_gemv", "fp8_quantize"}` has attribute `kernel.cuda_ms` with value `> 0.0`
  - All spans have `status != ERROR` (no kernel failures during inference)
  - Parent-child hierarchy is intact: each `"flash_attention"` / `"paged_attention"` span's `parent_id` matches a `"prefill"` or `"decode"` span's `span_id`
  - Total decode span count >= number of output tokens generated (one decode call per token position)
- Note: `setup_tracing()` signature must be updated to accept `exporter=None` kwarg — if provided, use it directly instead of `OTLPSpanExporter`; this keeps the unit and E2E tests self-contained with no external collector

**Results:**
- 7 OTEL unit tests: all pass (2 pass / 5 pass after BatchSpanProcessor→SimpleSpanProcessor fix for test mode)
- 104 total Python tests: all pass (0 regressions)
- No-op mode verified: `kernel_span()` is zero-overhead when `setup_tracing()` not called
- CUDA timing verified: `kernel.cuda_ms` > 0 on real GPU op
- Parent-child hierarchy verified via InMemorySpanExporter span_id checks

**Key design notes:**
- `setup_tracing()` uses `SimpleSpanProcessor` (synchronous) when a custom exporter is provided (test mode), and `BatchSpanProcessor` (async) for the default OTLP path (production)
- The global OTEL TracerProvider can only be set once per process; tests reset only the module-level `_tracer`/`_provider` singletons — each test installs a fresh provider with its own exporter without touching the global lock
- `kernel_span()` import fallback: `try: from kernel_craft_otel import kernel_span; except ImportError: from contextlib import nullcontext as kernel_span` — zero overhead, zero error if OTEL not installed

**CI integration for E2E tests**:
- Unit tests (`test_otel.py`) run on `kernel-recipe-check` (ubuntu-latest, no GPU) — `InMemorySpanExporter` has no external dependencies.
- E2E tests (`test_otel_e2e.py`) run on `kernel-equipped` (self-hosted, GPU) only. Add the following to `kernel-equipped` in `ci.yml`:
  1. Cache `~/.cache/huggingface` (key on `facebook/opt-125m` model files or a fixed key) so the model survives across runs.
  2. Add a `Download model` step **before** the e2e test step that checks disk first and downloads only on a cache miss:
     ```yaml
     - name: Download model for e2e tests
       run: |
         MODEL_DIR="${HOME}/.cache/huggingface/hub/models--facebook--opt-125m"
         if [ ! -d "$MODEL_DIR" ]; then
           /tmp/cicvenv/bin/huggingface-cli download facebook/opt-125m
         fi
     ```
     This keeps download failures visible as a named step failure rather than a hanging test.
  3. Set `HF_HUB_OFFLINE: "1"` as an env var on the e2e test step — forces HuggingFace to use only the local cache during the test run. If the model is somehow absent after the download step, the test fails fast with a clear offline error instead of blocking on network I/O mid-test.
     ```yaml
     - name: Run e2e tests
       env:
         HF_HUB_OFFLINE: "1"
       run: /tmp/cicvenv/bin/python -m pytest src/python/tests/test_otel_e2e.py -v -m e2e
     ```
  4. The `pytest.mark.skipif` guard in `test_otel_e2e.py` should check GPU availability and vLLM install only — not model presence, since the download step guarantees the model is cached before the test runs.

---

### Future improvements and utilities:
**Utility 1**: Add JAX support
- [ ] Research JAX array interop with pybind11
- [ ] Add `conv_naive_jax()` and `conv_tiled_jax()` overloads
- [ ] Test with JAX arrays on GPU

**Utility 2**: Add ONNX Runtime support
- [ ] Create ONNX execution provider custom kernel
- [ ] Integrate with ONNX Runtime CUDA EP
- [ ] Benchmark vs native implementation

**Utility 3**: Add TensorFlow support
- [ ] Add TensorFlow tensor overloads
- [ ] Use TF's memory management for GPU tensors
- [ ] Test with TF Keras models
