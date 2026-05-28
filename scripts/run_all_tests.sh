#!/usr/bin/env bash
# Release gate: runs all test suites in order and exits non-zero on first failure.
# Usage: run_all_tests.sh <build_dir> <source_dir>
set -euo pipefail

BUILD_DIR="${1:?build_dir required}"
SOURCE_DIR="${2:?source_dir required}"
PYTHON_TEST_DIR="${SOURCE_DIR}/src/python/tests"
VENV="${SOURCE_DIR}/src/python/venv/bin/activate"

pass() { echo "[PASS] $*"; }
fail() { echo "[FAIL] $*" >&2; exit 1; }

# ── 1. C++ unit tests ────────────────────────────────────────────────────────
echo "========================================"
echo " Suite 1/7: C++ unit tests"
echo "========================================"
bash "${SOURCE_DIR}/scripts/run_cpp_tests.sh" "${BUILD_DIR}" unit \
    && pass "C++ unit tests" || fail "C++ unit tests"

# ── 2. Python unit tests ──────────────────────────────────────────────────────
echo "========================================"
echo " Suite 2/7: Python unit tests"
echo "========================================"
bash "${SOURCE_DIR}/scripts/run_python_tests.sh" "${SOURCE_DIR}" unit \
    && pass "Python unit tests" || fail "Python unit tests"

# ── 3. C++ integration tests ─────────────────────────────────────────────────
echo "========================================"
echo " Suite 3/7: C++ integration tests"
echo "========================================"
bash "${SOURCE_DIR}/scripts/run_cpp_tests.sh" "${BUILD_DIR}" integration \
    && pass "C++ integration tests" || fail "C++ integration tests"

# ── 4. Python integration tests ──────────────────────────────────────────────
echo "========================================"
echo " Suite 4/7: Python integration tests"
echo "========================================"
bash "${SOURCE_DIR}/scripts/run_python_tests.sh" "${SOURCE_DIR}" integration \
    && pass "Python integration tests" || fail "Python integration tests"

# ── 5. C++ benchmarks ────────────────────────────────────────────────────────
echo "========================================"
echo " Suite 5/7: C++ benchmarks"
echo "========================================"
bash "${SOURCE_DIR}/scripts/run_cpp_tests.sh" "${BUILD_DIR}" benchmark \
    && pass "C++ benchmarks" || fail "C++ benchmarks"

# ── 6. OTEL unit tests ───────────────────────────────────────────────────────
echo "========================================"
echo " Suite 6/7: Python OTEL unit tests"
echo "========================================"
bash "${SOURCE_DIR}/scripts/run_python_tests.sh" "${SOURCE_DIR}" otel \
    && pass "Python OTEL unit tests" || fail "Python OTEL unit tests"

# ── 7. E2E OTEL tests ────────────────────────────────────────────────────────
echo "========================================"
echo " Suite 7/7: Python OTEL e2e tests"
echo "========================================"
bash "${SOURCE_DIR}/scripts/run_python_tests.sh" "${SOURCE_DIR}" e2e \
    && pass "Python OTEL e2e tests" || fail "Python OTEL e2e tests"

echo ""
echo "========================================"
echo " ALL SUITES PASSED"
echo "========================================"
