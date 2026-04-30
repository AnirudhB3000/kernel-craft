# tests/ - Python Binding Tests

## Overview
Test suite for Python bindings validating numpy and PyTorch tensor interfaces.

## Files
- `conftest.py` - pytest configuration and fixtures
- `test_bindings.py` - 13 test cases covering:
  - numpy array input/output for conv_naive and conv_tiled
  - PyTorch tensor input/output on GPU
  - Runtime tile_w/tile_h parameter dispatch
  - Correctness validation against reference implementations

## Usage
```bash
cd /home/aniru/kernel-craft/src/python
pytest tests/ -v
```
