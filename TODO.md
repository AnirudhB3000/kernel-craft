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

### Phase 17: Correctness & Safety — ctypes Boundary Hardening

**Goal**: Eliminate the segfault surface at the ctypes bridge. Today, passing a wrong dtype, a CPU tensor, a non-contiguous tensor, or an out-of-range shape to any of the 5 kernel methods in `kernel_craft_torch_ops.py` will silently dereference a bad pointer and crash the entire server process. Every public entry point must raise a Python exception with a clear message before touching `libkernels.so`.

**Existing codebase context**:
- `src/python/kernel_craft_torch_ops.py` — ctypes bridge; 5 kernel entry points: `flash_attention()` (~line 188), `paged_attention()` (~line 216), `int4_gemv()` (~line 287), `fp8_quantize()` (~line 321), `speculative_verify()` (~line 373). Each calls `self._lib.<fn>()` via ctypes after extracting `.data_ptr()`. No input validation exists anywhere in this file.
- `src/python/kernel_craft_vllm_backend.py` — the vLLM backend calls into `kernel_craft_torch_ops.py`; its callers are vLLM internals, so malformed tensors arrive from vLLM metadata, not user code. Still needs guard.
- `src/python/tests/test_vllm_backend.py` — 17 existing tests; regression baseline.
- `src/python/tests/test_transformer_bindings.py` — 17 existing tests; regression baseline.
- No existing `_validate_*` helpers anywhere in the Python layer (confirmed by grep).

**New file to create**:
- [ ] `src/python/kernel_craft_validation.py` — shared validation helpers:
  - `require_cuda_tensor(t, name)`: raises `ValueError` if `t.device.type != "cuda"`.
  - `require_dtype(t, name, *dtypes)`: raises `TypeError` if `t.dtype` not in `dtypes`.
  - `require_contiguous(t, name)`: raises `ValueError` if not `t.is_contiguous()`; include `.contiguous()` hint in message.
  - `require_shape(t, name, ndim=None, **dim_constraints)`: raises `ValueError` if `t.ndim != ndim` or named dim is out of allowed range (e.g., `head_dim` must be in `{16, 32, 64, 128}`).
  - `require_positive(value, name)`: raises `ValueError` if `value <= 0`.
  - `validate_flash_attention(Q, K, V, H_kv, causal)` — composes the above; checks Q/K/V are 4-D CUDA fp16/bf16 contiguous tensors; H_kv divides H; head_dim in {16,32,64,128}; seq_len > 0.
  - `validate_paged_attention(Q, block_table, K_cache, V_cache, seq_lens)` — Q is 3-D; block_table is 2-D int32 CUDA; K/V_cache are 5-D; seq_lens is 1-D int32; shapes internally consistent.
  - `validate_int4_gemv(x, W_packed, scales)` — x is 1-D fp16 CUDA; W_packed is 2-D uint8; scales fp16; cols consistent.
  - `validate_fp8_quantize(x, per_token)` — x is 2-D fp16/bf16 CUDA; rows > 0; cols > 0.
  - `validate_speculative_verify(draft_probs, target_probs, draft_tokens)` — all CUDA; shapes match; vocab_size > 0.

**Files to modify**:
- [ ] `src/python/kernel_craft_torch_ops.py` — call the corresponding `validate_*()` at the top of each of the 5 kernel methods, before any ctypes call. Pattern:
  ```python
  from kernel_craft_validation import validate_flash_attention

  def flash_attention(self, Q, K, V, H_kv=None, causal=False):
      validate_flash_attention(Q, K, V, H_kv, causal)
      # existing ctypes call unchanged
  ```
  Also add OOM guard: wrap the ctypes call in a `try/except OSError` (ctypes raises `OSError` on `cudaMalloc` failure in some configurations); re-raise as `torch.cuda.OutOfMemoryError` with the original message.

- [ ] `src/python/kernel_craft_vllm_backend.py` — the vLLM backend does its own shape math before calling `KernelCraftOps`; add a single `try/except (ValueError, TypeError) as e: raise RuntimeError(f"kernel-craft backend error: {e}") from e` wrapper in `forward()` so vLLM logs a structured error instead of a crash.

- [ ] `src/python/tests/test_validation.py` — new pytest file; tests for every `validate_*` function:
  - Happy path (valid inputs pass through with no exception).
  - CPU tensor → `ValueError`.
  - Wrong dtype → `TypeError`.
  - Non-contiguous → `ValueError` with `.contiguous()` hint in message.
  - Wrong ndim → `ValueError`.
  - Invalid head_dim (e.g., 48) → `ValueError`.
  - H_kv does not divide H → `ValueError`.
  - Zero seq_len → `ValueError`.
  - H_kv > H → `ValueError`.
  - Each validate_* has at least 3 error-path tests and 1 happy-path test.

**Thread safety**:
- [ ] Audit `kernel_craft_torch_ops.py` for shared mutable state (module-level dicts, cached library handles). The `_lib` handle is loaded once at import time (safe). Confirm there is no shared stream object; `torch.cuda.current_stream()` is per-thread in PyTorch, so ctypes calls are safe under GIL. Add a module-level comment documenting this invariant so future contributors don't introduce a shared stream.
- [ ] Add a `threading.Lock` around the `_lib` lazy-load path in `__init__` if one does not already exist, to guard against two threads racing on first import of `libkernels.so`.

**Testing strategy**:
- `pytest src/python/tests/test_validation.py -v` — all validation tests pass.
- `pytest src/python/tests/ -v` — full suite still passes (104+ pass, 0 regressions).
- Manual smoke: call `flash_attention(torch.randn(1,8,64,64))` (CPU tensor) — confirm `ValueError` raised, process does not crash.

---

### Phase 18: Performance — cuBLAS, Tensor Cores, CUDA Graphs

**Goal**: Close the 5–10× performance gap between the custom SGEMM in `tensor_parallel.cu` and what cuBLAS / Tensor Cores can achieve, add a Tensor Core WMMA path for INT4 GEMV, and capture the hot decode path in a CUDA graph to eliminate per-step kernel-launch overhead.

**Existing codebase context**:
- `src/kernels/transformer/tensor_parallel.cu` — `sgemm_nt_kernel` with 16×16 tiles; no cuBLAS call anywhere. Benchmarks show 0.055–0.94 TFLOPS vs cuBLAS theoretical ~20 TFLOPS on RTX 4070.
- `src/kernels/transformer/quant_int4.cu` — `int4_gemv_kernel` uses 256-thread strided loop; no WMMA path.
- `src/python/kernel_craft_vllm_backend.py` — `_decode_forward()` launches `paged_attention` + optional `speculative_verify` on every token step; no CUDA graph capture.
- `CMakeLists.txt` — cuBLAS is not currently linked. Pattern for linking optional libs already exists for NCCL (`find_library(NCCL_LIBRARY nccl)`).
- Benchmark baselines: col-parallel linear M=32 → 0.94 TFLOPS; INT4 GEMV 1024×4096 → 25 GB/s (Triton path).

**Files to modify**:

**cuBLAS fallback for tensor parallel linear** (`src/kernels/transformer/tensor_parallel.cu`):
- [ ] Add `#ifdef HAVE_CUBLAS` guard around a `cublasGemmEx` path in `launch_col_parallel_linear` and `launch_row_parallel_linear`. When `HAVE_CUBLAS=1`: call `cublasGemmEx` with `CUDA_R_16F` / `CUBLAS_COMPUTE_32F` and `CUBLAS_GEMM_DEFAULT_TENSOR_OP` algorithm. When not defined: fall back to the existing `sgemm_nt_kernel`.
- [ ] `CMakeLists.txt` — add cuBLAS detection:
  ```cmake
  find_library(CUBLAS_LIBRARY cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  if(CUBLAS_LIBRARY)
      target_link_libraries(kernels PUBLIC ${CUBLAS_LIBRARY})
      target_compile_definitions(kernels PUBLIC HAVE_CUBLAS=1)
      message(STATUS "cuBLAS found: ${CUBLAS_LIBRARY}")
  else()
      message(STATUS "cuBLAS not found — using custom SGEMM fallback")
  endif()
  ```
- [ ] `src/python/pybind_transformer.cpp` — expose `HAVE_CUBLAS` as `kernel_craft_transformer.HAVE_CUBLAS` boolean (mirror of existing `HAVE_NCCL` pattern).

**INT4 WMMA path** (`src/kernels/transformer/quant_int4.cu`):
- [ ] Add a `wmma_int4_gemv_kernel` that uses `nvcuda::wmma` (requires SM 7.5+ / Turing+). Dequantize a tile of INT4 weights to FP16, then use `wmma::mma_sync` with 16×16×16 fragments. Guard with `#if __CUDA_ARCH__ >= 750`. Keep the existing strided-loop kernel as fallback for older SMs.
- [ ] `launch_int4_gemv` — runtime SM check: `cudaDeviceGetAttribute(&sm, cudaDevAttrComputeCapabilityMajor, 0)` → choose WMMA or strided-loop path.
- [ ] New benchmark target in `benchmarks/benchmark_quant.cpp`: `BenchmarkInt4WmmavsFallback` comparing both paths at rows ∈ {256, 1024, 4096} × cols ∈ {4096, 11008}.

**CUDA graph capture for decode** (`src/python/kernel_craft_vllm_backend.py`):
- [ ] Add `_decode_graph: Optional[torch.cuda.CUDAGraph]` and `_decode_graph_inputs: dict` instance attributes to `KernelCraftAttentionImpl`.
- [ ] In `_decode_forward()`: on first call, capture the `paged_attention` ctypes call inside `torch.cuda.graph(self._decode_graph)`. On subsequent calls with the same batch size and max_seq_len, replay the graph via `self._decode_graph.replay()`. Invalidate and re-capture when batch shape changes.
- [ ] Add `kernel_craft.decode_graph_enabled` module attribute (default `True`); set `KERNEL_CRAFT_NO_DECODE_GRAPH=1` env var to disable (escape hatch for debugging).
- [ ] `src/python/tests/test_vllm_backend.py` — add 2 tests: graph captures on first decode call; graph replays (not re-captures) on identical second decode call. Use a mock that counts ctypes invocations.

**Testing & benchmarking**:
- [ ] `make run_benchmarks` — `benchmark_quant` prints WMMA vs fallback throughput; WMMA should be ≥2× faster on SM 7.5+.
- [ ] `pytest src/python/tests/test_vllm_backend.py -v` — 19+ tests pass.
- [ ] Profile decode step with Nsight Systems before/after graph capture; confirm kernel-launch overhead drops from ~15–30 μs to ~5–10 μs per step.

---

### Phase 19: Compatibility & Portability

**Goal**: Make kernel-craft runnable (with graceful fallbacks) on CUDA SM 6.1 through SM 9.0, across CUDA 11.8 / 12.0 / 12.4, across PyTorch 2.x, and validated against at least two production LLM architectures (Llama-3 8B with GQA, Mistral 7B with sliding window). Today, using FP8 on a pre-Hopper GPU silently produces wrong results; head_dim=128 is untested; H_kv≠H GQA ratios from real models have not been exercised end-to-end.

**Existing codebase context**:
- `src/kernels/transformer/fp8_quant.cu` — uses `__nv_fp8_e4m3`; this type requires CUDA 11.8+ and produces undefined behavior on pre-Ada (SM < 89) hardware. No SM guard.
- `src/kernels/transformer/flash_attention.cu` — hardcoded `HEAD_DIM` template parameter; values outside {16,32,64,128} are uninstantiated. No runtime check.
- `src/python/kernel_craft_torch_ops.py` — no SM detection anywhere. `fp8_quantize()` will silently corrupt results on GTX 1080 (SM 6.1).
- CLAUDE.md: tested only on RTX 4070 (SM 8.9), CUDA 12.0, torch 2.11, vLLM 0.21.
- `pyproject.toml` — `requires-python = ">=3.9"` but no CUDA version constraint expressed.

**Compatibility matrix to support**:

| SM | GPU example | Flash Attn | Paged Attn | INT4 GEMV | FP8 | WMMA INT4 |
|----|-------------|-----------|-----------|-----------|-----|-----------|
| 6.1 | GTX 1080 | ✓ (fallback) | ✓ | ✓ | ✗ (raise) | ✗ |
| 7.0 | V100 | ✓ | ✓ | ✓ | ✗ (raise) | ✗ |
| 7.5 | T4 / RTX 2080 | ✓ | ✓ | ✓ WMMA | ✗ (raise) | ✓ |
| 8.0 | A100 | ✓ | ✓ | ✓ WMMA | ✗ (raise) | ✓ |
| 8.9 | RTX 4070 (current) | ✓ | ✓ | ✓ WMMA | ✓ | ✓ |
| 9.0 | H100 | ✓ | ✓ | ✓ WMMA | ✓ | ✓ |

**Files to modify**:

**SM detection utility** (`src/python/kernel_craft_compat.py` — new file):
- [ ] `get_sm() -> int`: calls `torch.cuda.get_device_capability()` → returns `major * 10 + minor` (e.g., 89 for Ada, 90 for Hopper). Returns 0 if no CUDA device.
- [ ] `require_sm(minimum: int, feature: str)`: raises `RuntimeError(f"{feature} requires SM {minimum//10}.{minimum%10}+; got SM {get_sm()//10}.{get_sm()%10}")` if `get_sm() < minimum`.
- [ ] `CUDA_VERSION: int` — reads `torch.version.cuda` → int (e.g., "12.0" → 1200).

**FP8 SM guard** (`src/python/kernel_craft_torch_ops.py`):
- [ ] In `fp8_quantize()`: call `require_sm(89, "FP8 E4M3")` before the ctypes call. This turns a silent wrong-result bug into an immediate `RuntimeError` on pre-Ada GPUs.

**Flash attention head_dim guard** (`src/python/kernel_craft_validation.py`):
- [ ] `validate_flash_attention` already checks head_dim ∈ {16,32,64,128}; add `head_dim=128` test case to `test_validation.py` (currently untested at 128).

**Multi-architecture CI** (`CMakeLists.txt`):
- [ ] Change `set(CMAKE_CUDA_ARCHITECTURES 89)` (current) to `set(CMAKE_CUDA_ARCHITECTURES 61;75;80;89;90)` so `libkernels.so` is compiled for all supported SMs in one build. Guard SM-specific code (`__nv_fp8_e4m3`, WMMA) with existing `#if __CUDA_ARCH__` patterns.
- Note: multi-arch compilation increases build time; CI can use `CMAKE_CUDA_ARCHITECTURES=native` for dev builds and the full set only on release builds.

**GQA ratio validation with real models**:
- [ ] `src/python/tests/test_gqa_compat.py` — new pytest file; parametrize `flash_attention` tests over real model GQA ratios:
  - `llama3_8b`: H=32, H_kv=8 (ratio 4:1), head_dim=128, seq_len ∈ {128, 512, 2048}
  - `mistral_7b`: H=32, H_kv=8 (ratio 4:1), head_dim=128, seq_len=512 (no sliding window — just verify GQA kernel produces same output as naive reference)
  - `gemma_7b`: H=16, H_kv=16 (MHA), head_dim=256 — should raise `ValueError` (head_dim not supported)
  - Reference: naive `(Q @ K.T / sqrt(d)).softmax(-1) @ V` in torch; max abs error < 1e-3 (fp16 accumulation tolerance)
- [ ] Skip all tests if no CUDA device.

**CUDA / PyTorch version matrix**:
- [ ] `src/python/kernel_craft_compat.py` — `warn_if_untested_stack()`: emits a `warnings.warn` if `torch.version.cuda` not in `{"11.8", "12.0", "12.4"}` or `torch.__version__` major < 2. Called once at import of `kernel_craft_torch_ops`.
- [ ] `pyproject.toml` — document the tested matrix in `[project]` metadata: `"Programming Language :: GPU :: CUDA 12.0"` classifier; add `torch>=2.0` to `[project.dependencies]`.

**Testing strategy**:
- `pytest src/python/tests/test_gqa_compat.py -v` — all GQA ratio tests pass on GPU; gemma 256-head test raises ValueError.
- `pytest src/python/tests/test_validation.py -v` — SM guard tests: mock `get_sm()` to return 75, confirm `fp8_quantize` raises `RuntimeError`.
- Build with `CMAKE_CUDA_ARCHITECTURES=61;75;80;89;90` on CI and confirm no compilation errors in SM-guarded paths.

---

### Phase 20: Deployment

**Goal**: Make kernel-craft deployable as a containerized GPU inference service with a health endpoint, structured logging, graceful shutdown, and production-grade OTEL dashboards. Today there is no `Dockerfile`, no HTTP API, no signal handling, and no Grafana dashboard — the library can only be used by importing it directly.

**Existing codebase context**:
- `src/python/kernel_craft_otel.py` — OTEL tracing already configured; `setup_tracing()` connects to OTLP gRPC endpoint. No Prometheus metrics emitted.
- `src/python/kernel_craft_vllm_backend.py` — vLLM backend; no graceful shutdown hook.
- `benchmarks/benchmark_vllm_e2e.py` — standalone benchmark script; shows pattern for loading vLLM engine.
- `pyproject.toml` — no `[project.scripts]` entry point defined.

**New files to create**:

**Container** (`Dockerfile`):
- [ ] Multi-stage: stage 1 (`builder`) = `nvidia/cuda:12.0-devel-ubuntu22.04`; runs `cmake --build`, produces `libkernels.so`. Stage 2 (`runtime`) = `nvidia/cuda:12.0-runtime-ubuntu22.04`; copies only `libkernels.so` + Python wheel. This keeps the image small (no CUDA dev headers in prod image).
- Base Python: `python:3.11-slim` layer on top of CUDA runtime; install wheel with `pip install kernel-craft[observability]`.
- Exposes port 8080 (HTTP health) and 4317 (OTLP gRPC passthrough for local Jaeger).
- `ENTRYPOINT ["python", "-m", "kernel_craft.server"]`

**Compose stack** (`docker-compose.yml`):
- [ ] Three services: `kernel-craft` (the inference container, `runtime: nvidia`), `jaeger` (`jaegertracing/all-in-one:latest`, ports 16686+4317), `grafana` (`grafana/grafana:latest`, port 3000).
- `kernel-craft` sets `OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317`.
- `grafana` volume-mounts `deploy/grafana/dashboards/` for provisioned dashboards.

**Inference server** (`src/python/kernel_craft/server.py` — new module):
- [ ] `GET /health` → 200 `{"status": "ok", "gpu": "<device name>", "sm": <sm_version>}`. Returns 503 if CUDA not available.
- [ ] `POST /v1/generate` — accepts `{"prompt": str, "max_tokens": int}` JSON; calls vLLM engine; streams or returns generated text. (Simple wrapper — not a production LLM API, just enough to drive the observability stack.)
- [ ] Signal handling: `SIGTERM` → drain in-flight requests (wait up to 5s), call `torch.cuda.synchronize()`, flush OTEL spans (`tracer_provider.force_flush(timeout_millis=5000)`), then exit. `SIGINT` → same.
- [ ] Startup: calls `setup_tracing()` using `OTEL_EXPORTER_OTLP_ENDPOINT` env var (falls back to `http://localhost:4317`).

**Prometheus metrics** (`src/python/kernel_craft_otel.py` — extend):
- [ ] Add `setup_metrics(port=9090)`: starts a Prometheus exposition server via `prometheus_client.start_http_server(port)` if `prometheus_client` is installed. Exposes:
  - `kernel_craft_kernel_duration_ms` (Histogram, labels: `kernel_name`) — populated by `kernel_span()` on `__exit__`.
  - `kernel_craft_requests_total` (Counter, labels: `phase=prefill|decode`, `status=ok|error`).
  - `kernel_craft_oom_total` (Counter) — incremented by the OOM guard added in Phase 17.
- [ ] `kernel_span()` in `kernel_craft_otel.py` — after setting `kernel.cuda_ms` on the OTEL span, also `observe()` the histogram if metrics are enabled (single-call pattern, no duplicate timing).

**Grafana dashboard** (`deploy/grafana/dashboards/kernel_craft.json`):
- [ ] Provisioned JSON dashboard with panels:
  - P50/P95/P99 kernel duration by kernel name (Histogram quantile from Prometheus).
  - Request rate (prefill vs decode) over time.
  - OOM error rate.
  - GPU memory used (from `nvidia_smi_memory_used_bytes` if node-exporter is present, else a panel placeholder).
- Dashboard is version-controlled; `docker-compose up` provisions it automatically via Grafana's dashboard provisioning config in `deploy/grafana/provisioning/`.

**Entry point** (`pyproject.toml`):
- [ ] Add `[project.scripts]` entry: `kernel-craft-server = "kernel_craft.server:main"` so `pip install kernel-craft` gives users a runnable `kernel-craft-server` CLI.

**Testing strategy**:
- [ ] `src/python/tests/test_server.py` — unit tests using `httpx.AsyncClient` against the server (no GPU required; mock `KernelCraftOps`):
  - `GET /health` returns 200 with expected JSON shape.
  - `POST /v1/generate` with valid payload returns generated text.
  - `POST /v1/generate` with missing `prompt` key returns 422.
  - SIGTERM handler is registered at startup (check `signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL`).
- [ ] `docker build .` succeeds; `docker run --gpus all kernel-craft curl localhost:8080/health` returns 200.
- [ ] `docker-compose up` brings all three services up; Jaeger UI at `http://localhost:16686` shows service `kernel-craft`; Grafana at `http://localhost:3000` shows provisioned dashboard.

---

### Phase 21: Distribution — Stable PyPI Release & Compatibility Extras

**Goal**: Publish a versioned, installable PyPI package that pins to specific CUDA + PyTorch versions via extras, ships pre-built wheels for the supported SM targets from Phase 19, and includes a complete compatibility statement so users know immediately if their GPU/stack is supported.

**Existing codebase context**:
- `src/python/pyproject.toml` — `[build-system]` using `setuptools`; `version = "0.1.0"` (or similar). `[project.optional-dependencies]` already has `observability` and `triton` groups (added in Phases 15–16).
- `.github/workflows/release.yml` — workflow exists; triggered by `gh workflow run release.yml -f version_type=patch|minor|major`; pushes to TestPyPI then PyPI. Requires `testpypi_token` and `pypi_token` secrets (documented in CLAUDE.md §8).
- No pre-built wheels currently — users must compile from source (`pip install -e .` with CUDA dev headers). This blocks adoption on machines without NVCC.
- No `CUDA_ARCHITECTURES` baked into the wheel name; `pip install kernel-craft` on CUDA 11.8 silently installs the CUDA 12.0 wheel.

**New files to create**:

**Wheel build matrix** (`.github/workflows/build_wheels.yml`):
- [ ] New GitHub Actions workflow; triggers on `workflow_dispatch` and on tag push (`v*`).
- Strategy matrix: `cuda_version ∈ {11.8, 12.0, 12.4}` × `python_version ∈ {3.9, 3.10, 3.11, 3.12}` × `os = [linux]`. Uses `cibuildwheel` with custom `CIBW_BEFORE_BUILD` to run `cmake --build` with the correct `CMAKE_CUDA_ARCHITECTURES=61;75;80;89;90`.
- Wheel name convention: `kernel_craft-0.x.y+cu120-cp311-cp311-linux_x86_64.whl` (local version tag `+cu120` encodes CUDA version).
- Artifacts uploaded to GitHub release and to PyPI via `twine upload`.

**CUDA-version-pinned extras** (`src/python/pyproject.toml`):
- [ ] Add extras that pin torch to a CUDA-matched version:
  ```toml
  [project.optional-dependencies]
  cu118 = ["torch>=2.0,<2.2 ; platform_machine=='x86_64'"]
  cu120 = ["torch>=2.1,<2.3 ; platform_machine=='x86_64'"]
  cu124 = ["torch>=2.4 ; platform_machine=='x86_64'"]
  observability = [...]  # existing
  triton = [...]          # existing
  all = ["kernel-craft[cu120,observability,triton]"]
  ```
  Install instructions: `pip install "kernel-craft[cu120,observability]"`.

**Compatibility statement** (`src/python/kernel_craft_compat.py` — extend from Phase 19):
- [ ] `print_compat_table()`: prints a human-readable table of which features are available on the current GPU/CUDA/torch stack. Called by `python -m kernel_craft.compat`. Example output:
  ```
  kernel-craft 0.2.0 — compatibility report
  GPU:        NVIDIA RTX 4070 Laptop (SM 8.9)
  CUDA:       12.0
  PyTorch:    2.11.0+cu130
  ─────────────────────────────────────────────
  Flash Attention    ✓ (head_dim: 16/32/64/128)
  Paged Attention    ✓
  INT4 GEMV          ✓ (WMMA path, SM 7.5+)
  FP8 Quantize       ✓ (SM 8.9+)
  CUDA Graphs        ✓
  Triton kernels     ✓
  OTEL tracing       ✓ (opentelemetry-api installed)
  Prometheus metrics ✗ (pip install kernel-craft[observability])
  cuBLAS             ✓
  NCCL               ✗ (single GPU)
  ```

**CHANGELOG** (`CHANGELOG.md`):
- [ ] Create `CHANGELOG.md` following Keep a Changelog format. First entry: `[Unreleased]` with all phases 1–21 summarized under `Added`. The release workflow should bump this file as part of the version bump step.

**Release workflow update** (`.github/workflows/release.yml`):
- [ ] Extend existing workflow: after bumping the version in `pyproject.toml`, auto-update `CHANGELOG.md` (move `[Unreleased]` → `[x.y.z] — YYYY-MM-DD`). Then trigger `build_wheels.yml` via `workflow_dispatch` and wait for wheel artifacts before publishing to PyPI.

**Testing strategy**:
- [ ] `pip install dist/kernel_craft-*.whl --dry-run` on a clean virtualenv (no CUDA dev headers) — must succeed without compilation.
- [ ] Install from TestPyPI: `pip install -i https://test.pypi.org/simple/ "kernel-craft[cu120,observability]"` → `python -m kernel_craft.compat` runs without error.
- [ ] Compatibility table test: `test_compat.py` mocks `torch.cuda.get_device_capability()` to return (6,1), (7,5), (8,9), (9,0) and asserts `print_compat_table()` outputs correct ✓/✗ for FP8 and WMMA in each case.
- [ ] Matrix build: confirm `build_wheels.yml` produces 4 (python) × 3 (cuda) = 12 wheels per release; all install cleanly in matching environments.

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
