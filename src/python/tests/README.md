# tests/ — Python Binding Tests

pytest suite covering the pybind11 extension module and the vLLM backend integration.

## Files

| File | Scope | Count |
|------|-------|-------|
| `conftest.py` | Shared fixtures (module import, device checks) | — |
| `test_bindings.py` | CNN kernels: conv_naive, conv_tiled, INT8, BN folding, Conv+ReLU | 55 tests |
| `test_transformer_bindings.py` | Transformer kernels: flash_attention, paged_attention, INT4/FP8 quant, speculative_decode | 17 tests |
| `test_vllm_backend.py` | `KernelCraftAttentionBackend` — prefill, decode, KV-cache shape, metadata builder | 17 tests |

**Total: 89 tests pass, 0 skip** (with torch 2.11.0+cu130 + vLLM 0.21.0 venv).

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

# vLLM backend (requires vllm installed)
pytest tests/test_vllm_backend.py -v

# Everything at once
pytest tests/ -v
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
