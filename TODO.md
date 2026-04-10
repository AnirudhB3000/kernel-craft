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
- [ ] **Design tiled convolution strategy**
  - Determine tile size (e.g., 16×16) and halo handling.
  - Plan shared‑memory usage and synchronization.
- [ ] **Implement tiled convolution kernel** (`src/conv_tiled.cu`)
  - Load input tiles into shared memory, handle edge cases, and use `__syncthreads()`.
- [ ] **Add register optimization**
  - Keep frequently used values in registers where possible.
- [ ] **Update benchmark to include tiled version**
  - Measure and compare against naive implementation.
- [ ] **Profile both kernels** using Nsight Compute/System.
  - Capture metrics: memory coalescing, occupancy, shared‑memory usage, warp divergence.
- [ ] **Analyze profiling data** (`reports/conv_analysis.md`)
  - Identify which optimizations yield the biggest gains.
- [ ] **Iterate on tile size / block dimensions**
  - Test multiple configurations (e.g., 8×8, 32×32) and record results.

## Phase 3: Advanced Kernels (Weeks 5‑6)
- [ ] **Implement custom sparse convolution** (`src/custom_op.cu`)
  - Define a sparse kernel format and develop a CUDA implementation.
- [ ] **Benchmark custom operation** (`benchmarks/benchmark_custom.cpp`)
  - Compare against dense fallback and report performance.
- [ ] **Design fused kernel pipeline** (`src/pipeline_fused.cu`)
  - Fuse convolution → batch‑norm → ReLU into a single kernel.
- [ ] **Implement individual kernels** (`src/pipeline_separate.cu`)
  - Provide separate implementations for baseline comparison.
- [ ] **Benchmark fused vs separate pipelines** (`benchmarks/benchmark_pipeline.cpp`)
  - Measure launch overhead, memory traffic, and total latency.
- [ ] **Document findings** (`reports/fusion_analysis.md`)
  - Include discussion on kernel launch cost, memory reuse, and overall speed‑up.

## Phase 4: End‑to‑End GPU Pipeline (Weeks 7‑8)
- [ ] **Implement GPU preprocessing kernels** (`src/preprocess_gpu.cu`)
  - Resize (bilinear interpolation)
  - Normalization
  - Simple augmentation (flip, crop)
- [ ] **Integrate preprocessing into full pipeline** (`src/pipeline_full_gpu.cu`)
  - Chain preprocessing → convolution → post‑processing.
- [ ] **Create end‑to‑end benchmark** (`benchmarks/benchmark_full_pipeline.cpp`)
  - Measure total throughput, end‑to‑end latency, and compare with CPU‑+‑GPU hybrid approach.
- [ ] **Profile complete pipeline**
  - Use Nsight Systems to visualize data movement and kernel overlaps.
- [ ] **Write final performance report** (`reports/pipeline_analysis.md`)
  - Summarize speed‑up, bottlenecks, and recommendations for production use.

## Cross‑Cutting Tasks (All Phases)
- [ ] **Set up CI (optional)** – compile and run unit tests on each push.
- [ ] **Write unit tests** for each kernel using GoogleTest or a simple harness.
- [ ] **Add documentation** (in‑code comments) explaining kernel parameters and launch configurations.
- [ ] **Maintain `README.md`** – keep it up‑to‑date with new deliverables.
- [ ] **Version control** – commit each milestone with clear messages.
- [ ] **Archive results** – store benchmark logs and plots in the `reports/` directory.

---

**Note:** Follow the methodology from `CLAUDE.md`: change one variable at a time, compare against baselines, and document everything rigorously.
