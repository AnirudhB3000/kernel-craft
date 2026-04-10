# kernel-craft

## Project Overview

`kernel-craft` provides a ready‑to‑run CUDA codebase for common machine‑learning kernel operations. It implements a naïve 2‑D convolution kernel and, as the project progresses, will add:
- A tiled version of the convolution for memory‑traffic reduction.
- Custom operations such as sparse convolutions.
- Fused kernels that combine convolution, batch‑norm, and ReLU.
- GPU‑accelerated preprocessing (resize, normalize, augment) for end‑to‑end pipelines.

The repository supplies the source, benchmarks, and unit tests needed to evaluate and compare these implementations.


## Directory Structure
```
kernel-craft/
├─ src/                # CUDA kernel implementations
│   └─ conv_naive.cu   # baseline naïve 2‑D convolution kernel
├─ benchmarks/        # Host‑side benchmark programs
│   └─ benchmark_conv.cpp   # benchmark for the naïve convolution
├─ data/
│   └─ sample_images/ # Example inputs for benchmarks
├─ CMakeLists.txt      # Build configuration
└─ README.md          # This document
```

## Build & Run
```bash
mkdir build && cd build
cmake ..
make
# Example: ./kernel_craft benchmarks/benchmark_conv.cpp
```

## Testing
Run all unit tests with a concise summary:

```bash
make run_tests          # builds and runs all tests via CTest
# or, after building, invoke CTest directly:
ctest --output-on-failure
```

The `run_tests` target reports the total number of tests, how many passed, and any failures with their names.

## Development Plan
The project follows a phased execution plan (found in `CLAUDE.md`):
1. **Foundations** – implement naïve convolution and benchmark CPU vs. GPU.
2. **Optimization** – add tiled convolution and profile memory traffic.
3. **Advanced Kernels** – build fused kernels and custom operations.
4. **System Design** – create a full GPU‑accelerated pipeline.

Success is measured by the ability to explain GPU execution, write efficient kernels, identify bottlenecks, and produce reproducible benchmark reports.
