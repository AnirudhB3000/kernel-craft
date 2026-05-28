"""pytest configuration for kernel_craft Python tests."""

import sys
from pathlib import Path

# src/python/ — top-level modules (kernel_craft_otel, kernel_craft_torch_ops, etc.)
src_python_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_python_dir))

# src/python/build/ — compiled pybind11 extension (.so)
build_dir = src_python_dir / "build"
if build_dir.exists():
    sys.path.insert(0, str(build_dir))