"""pytest configuration for kernel_craft Python tests."""

import sys
from pathlib import Path

# Add build directory to path for module import
build_dir = Path(__file__).parent.parent / "build"
if build_dir.exists():
    sys.path.insert(0, str(build_dir))