# TODO List for kernel-craft

## Phase 1: Foundations (Weeks 1‑2)
- [x] **Set up development environment**
  - Install CUDA toolkit, Nsight Systems/Compute, and required compiler versions.
  - Verify `nvcc --version` and `nvidia-smi` detect the GPU.
- [x] **Implement naive 2D convolution kernel** (`src/conv_naive.cu`)
  - CUDA kernel with host launcher added.
- [x] **Create benchmark for naive convolution** (`benchmarks/benchmark_conv.cpp`)
  - Generates synthetic data, runs GPU and CPU versions, reports timing and throughput.
- [x] **Run CPU baseline**
  - CPU reference implementation provided in benchmark and test.
- [x] **Compare CPU vs GPU**
  - Verification and performance comparison included in benchmark.


## Phase 2: Optimization (Weeks 3‑4)
- [x] **Design tiled convolution strategy**
  - Determine tile size (e.g., 16×16) and halo handling.
  - Plan shared‑memory usage and synchronization.
- [x] **Implement tiled convolution kernel** (`src/conv_tiled.cu`)
  - Load input tiles into shared memory, handle edge cases, and use `__syncthreads()`.
- [x] **Add register optimization**
  - Keep frequently used values in registers where possible.
- [x] **Update benchmark to include tiled version**
  - Measure and compare against naive implementation.
- [x] **Profile both kernels** using Nsight Compute/System.
  - Capture metrics: memory coalescing, occupancy, shared‑memory usage, warp divergence.
- [x] **Iterate on tile size / block dimensions**
  - Test multiple configurations (8×8, 16×16, 32×32) - 8×8 best overall

## Phase 3: Advanced Kernels (Weeks 5‑6)
- [x] **Implement custom sparse convolution** (`src/custom_op.cu`)
  - Define sparse kernel format, implement CUDA kernel for sparse convolution.
- [x] **Benchmark custom operation** (`benchmarks/benchmark_custom.cpp`)
  - Compare sparse (4/9 non-zero) vs dense fallback: sparse is ~1.7x faster.
- [x] **Design fused kernel pipeline** (`src/pipeline_fused.cu`)
  - Fuse convolution → batch‑norm → ReLU into a single kernel.
- [x] **Implement individual kernels** (`src/pipeline_separate.cu`)
  - Provide separate implementations for baseline comparison.
- [x] **Benchmark fused vs separate pipelines** (`benchmarks/benchmark_pipeline.cpp`)
  - Measure launch overhead, memory traffic, and total latency.

## Phase 4: End‑to‑End GPU Pipeline (Weeks 7‑8)
- [x] **Implement GPU preprocessing kernels** (`src/preprocess_gpu.cu`)
  - Resize (bilinear interpolation)
  - Normalization
  - Simple augmentation (flip, crop)
- [x] **Integrate preprocessing into full pipeline** (`src/preprocess_gpu.cu` includes all kernels)
  - Chain preprocessing → convolution → post‑processing.
- [x] **Create end‑to‑end benchmark** (`benchmarks/benchmark_full_pipeline.cpp`)
  - Measure total throughput, end‑to‑end latency, and compare with CPU‑+‑GPU hybrid approach.
  - **Results**: Full GPU: 0.505 ms (2077 MPixels/s) vs CPU+GPU: 45.356 ms (23 MPixels/s) = **~90x faster**
- [ ] **Profile complete pipeline**
  - Use Nsight Systems to visualize data movement and kernel overlaps.

## Phase 5: Python Integration (Weeks 9‑10)
### Core Bindings
- [x] **Update CMakeLists.txt** with pybind11 FetchContent
  - Add `FetchContent_Declare(pybind11 ...)` for auto-download
  - Add Python extension target (`pybind11_add_module(kernel_craft ...)`)
- [x] **Create src/python/pybind_cuda.cpp** - pybind11 module (~350 lines)
  - Bind `conv_naive(input, kernel)` for numpy and PyTorch
  - Bind `conv_tiled(input, kernel, tile_w, tile_h)` for numpy and PyTorch
  - Handle memory transfer: host↔device copy, data pointer extraction
- [x] **Build and test Python module**
  - `mkdir build && cd build && cmake .. && make kernel_craft_python`
  - `python3 -c "import kernel_craft_python; print(kernel_craft_python.__version__)"`
- [x] **Verify correctness**
  - numpy arrays produce correct output (match CPU reference)
  - PyTorch tensors produce correct output on GPU (tests skipped when torch not available)
  - `tile_w`/`tile_h` parameters affect performance

### Package Infrastructure
- [x] **Create pyproject.toml**
  - Project name, version, description, author, license
  - Dependencies: numpy (required), torch (optional)
  - Support both CMake and `pip install .` builds
- [x] **Add version management**
  - Expose `__version__` in Python module (e.g., `kernel_craft_python.__version__`)
- [ ] **Build wheel support**
  - Configure for universal wheels across CUDA versions/platforms

### Python Tests
- [x] **Create pytest test suite** (`src/python/tests/test_bindings.py`)
  - Test numpy arrays produce correct output
  - Test PyTorch tensors on GPU (skipped when torch not available)
  - Test error handling and validation
  - 10 tests passed, 5 skipped

### Documentation
- [ ] **Update src/python/README.md**
  - Add `pip install kernel_craft` section
  - Document `__version__` attribute
  - Document error handling and exceptions

### Cross‑Cutting: Release Prep
- [ ] **Add semantic versioning**
  - Define version scheme (e.g., 0.1.0)
- [ ] **Add type hints**
  - For better IDE support in Python bindings
- [ ] **Create PyPI release workflow**
  - GitHub Actions workflow to build and publish
  - Support TestPyPI and PyPI deployments
- [ ] **Create binary distribution**
  - Generate and upload `.whl` files to PyPI

## Cross‑Cutting Tasks (All Phases)
- [ ] **Set up CI (optional)** – compile and run unit tests on each push.
- [ ] **Write unit tests** for each kernel using GoogleTest or a simple harness.
- [ ] **Add documentation** (in‑code comments) explaining kernel parameters and launch configurations.
- [ ] **Maintain `README.md`** – keep it up‑to‑date with new deliverables.
- [ ] **Version control** – commit each milestone with clear messages.

---

**Note:** Follow the methodology from `AGENTS.md`: change one variable at a time, compare against baselines, and document everything rigorously.
