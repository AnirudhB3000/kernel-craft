# kernel-craft Examples

End-to-end usage examples for every kernel family, organised by language and topic.

---

## Quick start

```bash
# Build all C++ examples
cmake --build build --target build_examples

# Run all C++ examples in sequence
cmake --build build --target run_examples

# Run a single example
./build/bin/example_flash_attention

# Run all Python examples (auto-skips unavailable deps)
python examples/python/run_all.py

# Run only the conv examples
python examples/python/run_all.py --filter conv

# List examples and their status (ready / needs: ...)
python examples/python/run_all.py --list
```

---

## C++ examples

All executables land in `build/bin/`.  
Build individually: `cmake --build build --target <name>`

### CNN / convolution

| Target | File | What it demonstrates |
|---|---|---|
| `example_conv_naive` | `example_conv_naive.cpp` | Baseline kernel: one thread per output pixel, global-memory only. Integration with CNN inference engines. |
| `example_conv_tiled` | `example_conv_tiled.cpp` | Shared-memory tiled convolution. Tile-size sweep (8×8, 16×16, 32×32). Why tiling reduces global-memory traffic. |
| `example_fused_pipeline` | `example_fused_pipeline.cpp` | Fused conv + BN-affine + ReLU in one kernel. Memory-traffic analysis, BN folding explanation, speedup measurement. |
| `example_memory_pool` | `example_memory_pool.cpp` | Pre-allocated device buffer pool. Eliminates per-batch `cudaMalloc` overhead; inference-server integration pattern. |
| `example_preprocess` | `example_preprocess.cpp` | GPU preprocessing pipeline: bilinear resize → normalise → horizontal flip. DeepStream / Triton context. |
| `example_cuda_graphs` | `example_cuda_graphs.cpp` | CUDA Graph capture and replay. Actual `cudaStreamBeginCapture` / `cudaGraphLaunch` cycle with timing. vLLM decode context. |

### Transformer / LLM inference

| Target | File | What it demonstrates |
|---|---|---|
| `example_flash_attention` | `example_flash_attention.cpp` | FlashAttention forward pass. Online-softmax algorithm, MHA vs GQA (`H_kv`), causal masking. Prefill routing in LLM serving. |
| `example_paged_attention` | `example_paged_attention.cpp` | PagedAttention decode. Physical-page / block-table design (virtual-memory analogy). vLLM KV-cache integration. |
| `example_tensor_parallel` | `example_tensor_parallel.cpp` | Column-parallel and row-parallel linear layers. Simulated ring all-reduce / all-gather. Megatron-LM FFN pattern. |

### TensorRT (optional — requires `-DHAVE_TENSORRT=ON`)

| Target | File | What it demonstrates |
|---|---|---|
| `example_tensorrt_usage` | `tensorrt/example_plugin_usage.cpp` | Small CNN (Conv→BN→ReLU×2 → MaxPool → FC/1000) using kernel-craft `IPluginV2DynamicExt` plugins. Full engine build, serialise, deserialise, `enqueueV3` inference. INT8 calibration scales. |

```bash
cmake -DHAVE_TENSORRT=ON -DTENSORRT_ROOT=/usr/local/tensorrt ..
cmake --build build --target example_tensorrt_usage
./build/bin/example_tensorrt_usage
```

---

## Python examples

All examples are in `examples/python/`.  
**Prerequisites**: build the native extension first.

```bash
cmake --build build --target kernel_craft_python kernel_craft_transformer
# Then either:
pip install -e src/python          # installs the package
# or:
export PYTHONPATH=$PWD/src/python/build:$PWD/src/python
```

Install Python deps: `pip install -r examples/python/requirements.txt`

### CNN / convolution

| File | API used | What it demonstrates |
|---|---|---|
| `example_numpy_conv_naive.py` | `kc.conv_naive(img, ker)` | NumPy array input/output. GPU kernel round-trip. CPU reference comparison. Drop-in replacement for `scipy.ndimage.convolve`. |
| `example_numpy_conv_tiled.py` | `kc.conv_tiled(img, ker, tw, th)` | Tile-size sweep with wall-clock timing. Correctness vs `conv_naive`. |
| `example_torch_conv_naive.py` | `kc.conv_naive(tensor, tensor)` | CUDA tensor input — zero-copy (no H2D). CUDA event timing. Comparison vs `torch.nn.functional.conv2d`. `autograd.Function` integration pattern. |
| `example_torch_conv_tiled.py` | `kc.conv_tiled(tensor, tensor, tw, th)` | CUDA event timing per tile size. CUDA Graph integration pattern. |
| `example_memory_pool.py` | `kc.conv_tiled` (NumPy + PyTorch) | Batch processing. NumPy vs PyTorch tensor path overhead. C++ pool integration note. |
| `example_mixed_precision.py` | `kc.conv_tiled`, `torch.autocast` | FP32 / FP16 / TF32 accuracy comparison. PyTorch AMP context. FP8 pointer for Phase 11 kernels. |

### Transformer / LLM inference

| File | API used | What it demonstrates |
|---|---|---|
| `example_flash_attention.py` | `kct.flash_attention(Q,K,V,H_kv,causal)` | MHA, causal MHA, GQA. CPU reference. Throughput estimate. vLLM routing note. |
| `example_fp8_quant.py` | `kct.fp8_quantize`, `kct.fp8_dequantize`, `kct.smoothquant_scale` | Per-token vs per-tensor scales. FP8 vs INT8 error characteristics. SmoothQuant. vLLM `quantization='fp8'` pointer. |
| `example_speculative_decoding.py` | `kct.speculative_decode(...)` | Three scenarios: all-accept, all-reject, mixed α≈0.7. Residual distribution correctness. Expected speedup formula. |
| `example_tensor_parallel.py` | `kct.col_parallel_linear`, `kct.row_parallel_linear`, `kct.ring_allreduce`, `kct.allgather` | Megatron-LM MLP block end-to-end. Correctness vs full matmul. NCCL integration note. |

### PyTorch ops bridge

| File | API used | What it demonstrates |
|---|---|---|
| `example_torch_ops.py` | `torch.ops.kernel_craft.*` | All ops via the registered PyTorch namespace. `torch.compile` compatibility. INT4 GEMV, FP8 round-trip, FlashAttention. |

### vLLM integration

| File | What it demonstrates |
|---|---|
| `example_vllm_opt125m.py` | End-to-end OPT-125M text generation with the kernel-craft `AttentionBackend`. Registers `KernelCraftAttentionBackend`, routes prefill → `flash_attention`, decode → `paged_attention`. TTFT / throughput / VRAM measurement. Benchmark mode (`--benchmark`) compares kernel-craft vs vLLM default. |

```bash
# Basic run (downloads ~250 MB model on first run)
python examples/python/example_vllm_opt125m.py

# Benchmark both backends side by side
python examples/python/example_vllm_opt125m.py --benchmark

# Larger model (same backend, needs more VRAM)
python examples/python/example_vllm_opt125m.py --model facebook/opt-1.3b
```

---

## Source map

```
examples/
├── CMakeLists.txt                    ← all C++ example targets (add_subdirectory from root)
├── README.md                         ← this file
│
├── cnn/                              ← convolution and CNN pipeline examples
│   ├── conv_naive.cpp                  baseline: one thread per pixel, global-mem only
│   ├── conv_tiled.cpp                  shared-memory tile sweep (8×8, 16×16, 32×32)
│   ├── fused_pipeline.cpp              conv + BN-affine + ReLU in one kernel
│   ├── memory_pool.cpp                 pre-allocated pool vs per-batch cudaMalloc
│   ├── preprocess.cpp                  resize (bilinear) → normalize → flip
│   └── cuda_graphs.cpp                 graph capture / replay vs separate launches
│
├── transformer/                      ← LLM attention and parallelism examples
│   ├── flash_attention.cpp             MHA / GQA / causal; prefill stage
│   ├── paged_attention.cpp             paged KV-cache; decode stage
│   └── tensor_parallel.cpp             col/row parallel linear + collectives
│
├── tensorrt/                         ← TensorRT plugin (optional, -DHAVE_TENSORRT=ON)
│   └── plugin_cnn.cpp                  small CNN via IPluginV2DynamicExt
│
└── python/
    ├── requirements.txt
    ├── run_all.py                    ← runner with pass/skip/fail report
    │
    ├── cnn/                          ← CNN / convolution
    │   ├── conv_naive_numpy.py         kc.conv_naive — NumPy round-trip
    │   ├── conv_tiled_numpy.py         kc.conv_tiled — tile-size sweep
    │   ├── conv_naive_torch.py         kc.conv_naive — CUDA tensor, zero-copy
    │   ├── conv_tiled_torch.py         kc.conv_tiled — CUDA event timing
    │   ├── memory_pool.py              batch processing, NumPy vs PyTorch overhead
    │   └── mixed_precision.py          FP32 / FP16 / TF32 accuracy comparison
    │
    ├── transformer/                  ← LLM / transformer inference
    │   ├── flash_attention.py          MHA, causal MHA, GQA; CPU reference
    │   ├── fp8_quant.py                FP8 E4M3 quantize/dequantize, SmoothQuant
    │   ├── speculative_decoding.py     reject-sampling verification; 3 scenarios
    │   └── tensor_parallel.py          Megatron-LM MLP block end-to-end
    │
    └── vllm/                         ← PyTorch ops bridge + vLLM integration
        ├── torch_ops.py                torch.ops.kernel_craft namespace + torch.compile
        └── opt125m.py                  OPT-125M end-to-end with KernelCraftAttentionBackend
```
