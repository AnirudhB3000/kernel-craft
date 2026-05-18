#!/usr/bin/env bash
# Run Python tests by suite.
# Usage: run_python_tests.sh <source_dir> <suite>
#   suite: unit | integration | all
set -euo pipefail

SOURCE_DIR="${1:?source_dir required}"
SUITE="${2:-all}"
TEST_DIR="${SOURCE_DIR}/src/python/tests"
VENV="${SOURCE_DIR}/src/python/venv/bin/activate"

if [[ ! -f "${VENV}" ]]; then
    echo "ERROR: venv not found at ${VENV}" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${VENV}"

case "${SUITE}" in
    unit)
        python -m pytest \
            "${TEST_DIR}/test_bindings.py" \
            "${TEST_DIR}/test_transformer_bindings.py" \
            -v
        ;;
    integration)
        python -m pytest \
            "${TEST_DIR}/test_vllm_backend.py" \
            -v
        ;;
    all)
        python -m pytest "${TEST_DIR}" -v
        ;;
    *)
        echo "ERROR: unknown suite '${SUITE}'. Use: unit | integration | all" >&2
        exit 1
        ;;
esac
