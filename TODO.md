# TODO List for kernel-craft


## Phase 4: End‑to‑End GPU Pipeline (Weeks 7‑8)
- [ ] **Profile complete pipeline**
  - Use Nsight Systems to visualize data movement and kernel overlaps.

## Phase 5: Python Integration (Weeks 9‑10)
- [ ] **Create PyPI release workflow**
  - GitHub Actions workflow to build and publish
  - Support TestPyPI and PyPI deployments

## Phase 6: CI with Local GPU Runner (Optional)

**Option 1**: Use GitHub Actions self-hosted runner on local machine ✓ IN PROGRESS
- [x] CI workflow updated for self-hosted runner with GPU label
- [ ] **Set up self-hosted runner on local machine** (instructions below)
- [ ] Add runner to GitHub repository with `gpu` label

### Setup Instructions for Self-Hosted Runner

1. **Create runner user** (optional but recommended):
```bash
sudo useradd -m -s /bin/bash github-runner
```

2. **Add runner to repository**:
   - Go to: https://github.com/anomalyco/kernel-craft/settings/actions/runners
   - Click "New self-hosted runner"
   - Select "Linux" / "x64"
   - Run the commands shown (download + config)

3. **Configure with GPU label**:
```bash
./config.sh --url https://github.com/anomalyco/kernel-craft \
    --token YOUR_TOKEN \
    --labels self-hosted,linux,gpu \
    --name "local-gpu-runner"
```

4. **Run as service** (optional):
```bash
sudo ./svc.sh install
sudo ./svc.sh start
```

5. **Verify**: Runner should appear in repository settings with "gpu" label

**Option 2**: Use CUDA Docker container with GPU passthrough
- [ ] Configure Docker to passthrough NVIDIA GPU
- [ ] Use `nvidia/cuda:12.4.0-runtime-ubuntu22.04` container
- [ ] Verify GPU visibility inside container

## Phase 7: Additional Framework Support (Optional)

**Option 1**: Add JAX support
- [ ] Research JAX array interop with pybind11
- [ ] Add `conv_naive_jax()` and `conv_tiled_jax()` overloads
- [ ] Test with JAX arrays on GPU

**Option 2**: Add ONNX Runtime support
- [ ] Create ONNX execution provider custom kernel
- [ ] Integrate with ONNX Runtime CUDA EP
- [ ] Benchmark vs native implementation

**Option 3**: Add TensorFlow support
- [ ] Add TensorFlow tensor overloads
- [ ] Use TF's memory management for GPU tensors
- [ ] Test with TF Keras models

---

**Note:** Follow the methodology from `AGENTS.md`: change one variable at a time, compare against baselines, and document everything rigorously.
