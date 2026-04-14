# TODO List for kernel-craft


## Phase 4: End‑to‑End GPU Pipeline (Weeks 7‑8)
- [ ] **Profile complete pipeline**
  - Use Nsight Systems to visualize data movement and kernel overlaps.

## Phase 5: Python Integration (Weeks 9‑10)


### Package Infrastructure
- [x] **Build wheel support**
  - Configure for universal wheels across CUDA versions/platforms
  - Current: py3-none-any wheel (pre-built .so)
  - Future: Use scikit-build or cibuildwheel for multi-CUDA builds

### Documentation
- [x] **Update src/python/README.md**
  - Add `pip install kernel_craft` section
  - Document `__version__` attribute
  - Document error handling and exceptions

### Cross‑Cutting: Release Prep
- [x] **Add semantic versioning**
  - Define version scheme (e.g., 0.1.0) - v0.1.1 published
- [x] **Add type hints**
  - For better IDE support in Python bindings
  - Created `kernel_craft_python.pyi` with overloads for numpy/PyTorch
- [ ] **Create PyPI release workflow**
  - GitHub Actions workflow to build and publish
  - Support TestPyPI and PyPI deployments
- [x] **Create binary distribution**
  - Generate and upload `.whl` files to PyPI - v0.1.1 done

## Cross‑Cutting Tasks (All Phases)
- [x] **Set up CI (optional)** – compile and run unit tests on each push.
  - Created `.github/workflows/ci.yml` - runs CMake build, C++ tests (ctest), Python tests (pytest)
- [x] **Write unit tests** for each kernel using GoogleTest or a simple harness.
  - Already implemented: 5 test files (conv_naive, conv_tiled, custom_op, pipeline, preprocess)
  - Uses simple assert-based harness with CPU reference comparison
- [x] **Add documentation** (in‑code comments) explaining kernel parameters and launch configurations.
  - All CUDA kernels have Doxygen docs for parameters, launch configs, performance notes
- [x] **Maintain `README.md`** – keep it up‑to‑date with new deliverables.
  - Updated with Python pip install, type stubs, updated directory structure
- [ ] **Version control** – commit each milestone with clear messages.
- [x] **Build wheel support** - Done in Phase 5
- [x] **Add type hints** - Done in Phase 5
- [x] **Create binary distribution** - v0.1.1 published to PyPI

---

**Note:** Follow the methodology from `AGENTS.md`: change one variable at a time, compare against baselines, and document everything rigorously.
