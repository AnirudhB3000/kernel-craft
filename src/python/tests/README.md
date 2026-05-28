# tests/ — Python Binding Tests

pytest suite covering the pybind11 extension module and the vLLM backend integration.

## Files

| File | Scope | Count |
|------|-------|-------|
| `conftest.py` | Shared fixtures (module import, device checks) | — |
| `test_bindings.py` | CNN kernels: conv_naive, conv_tiled, INT8, BN folding, Conv+ReLU | 55 tests |
| `test_transformer_bindings.py` | Transformer kernels: flash_attention, paged_attention, INT4/FP8 quant, speculative_decode | 17 tests |
| `test_mamba_bindings.py` | Mamba/SSM kernels: selective_scan, depthwise_conv1d, rmsnorm | 17 tests |
| `test_vllm_backend.py` | `KernelCraftAttentionBackend` — prefill, decode, KV-cache shape, metadata builder | 17 tests |
| `test_triton_kernels.py` | Triton kernels: flash_attention, selective_scan, int4_gemv (correctness + shapes) | 15 tests |
| `test_otel.py` | OpenTelemetry unit tests: span emission, CUDA timing, parent-child hierarchy, no-op mode | 7 tests |
| `test_otel_e2e.py` | OTEL end-to-end: opt-125m via vLLM, span hierarchy + cuda_ms assertions | `@pytest.mark.e2e` |

**Total: 128 tests pass, 0 skip** (unit + integration suites, with torch 2.11.0+cu130 + vLLM 0.21.0 + triton 3.6.0 + opentelemetry-sdk).
E2E suite is run separately — requires GPU and `facebook/opt-125m` cached in `~/.cache/huggingface`.

## Running

### Via CMake (uses local venv automatically)

```bash
cd build && make run_python_tests
```

### Directly with the vLLM venv

```bash
cd src/python
source venv/bin/activate

# CNN + transformer bindings
pytest tests/test_bindings.py tests/test_transformer_bindings.py -v

# Mamba/SSM kernels
pytest tests/test_mamba_bindings.py -v

# vLLM backend (requires vllm installed)
pytest tests/test_vllm_backend.py -v

# Triton kernels (requires triton installed)
pytest tests/test_triton_kernels.py -v

# OpenTelemetry unit tests (no external collector needed)
pytest tests/test_otel.py -v

# OTEL end-to-end (requires GPU + facebook/opt-125m cached)
pytest tests/test_otel_e2e.py -v -m e2e

# Everything except e2e at once
pytest tests/ -v --ignore=tests/test_otel_e2e.py
```

### Without the venv (CNN kernels only)

```bash
cd src/python
pytest tests/test_bindings.py -v
```

## Notes

- `test_vllm_backend.py` imports from `vllm.v1.attention.backend` — requires vLLM 0.21.0 exactly; the v1 API signature changed from 0.5.x
- GPU tests are skipped automatically if no CUDA device is available (controlled by `conftest.py`)
- `test_transformer_bindings.py` tests FP8 with a ~5.8% mean relative error tolerance (3 mantissa bits in E4M3)
- `test_triton_kernels.py` requires `triton>=3.0` and a CUDA GPU; skipped automatically otherwise
- `test_otel.py` uses `InMemorySpanExporter` — no Jaeger or OTLP collector needed; runs on CPU-only machines
- `test_otel_e2e.py` is gated by `@pytest.mark.e2e`; the default `pytest tests/` run excludes it — pass `-m e2e` to include
