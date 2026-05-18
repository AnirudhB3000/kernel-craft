#!/usr/bin/env bash
# Run C++ tests by suite. Called by CMake custom targets and run_all_tests.sh.
# Usage: run_cpp_tests.sh <build_dir> <suite>
#   suite: unit | integration | benchmark
set -euo pipefail

BUILD_DIR="${1:?build_dir required}"
SUITE="${2:?suite required}"

INTEGRATION_PATTERN="pipeline_test|cuda_graphs_test|async_streams_test|unified_memory_test|multi_stream_pipeline_test|tensorrt_integration_test"

cd "${BUILD_DIR}"

case "${SUITE}" in
    unit)
        ctest -R "_test$" -E "${INTEGRATION_PATTERN}" --output-on-failure
        ;;
    integration)
        ctest -R "${INTEGRATION_PATTERN}" --output-on-failure
        ;;
    benchmark)
        ctest -R "^benchmark_" --output-on-failure
        ;;
    *)
        echo "ERROR: unknown suite '${SUITE}'. Use: unit | integration | benchmark" >&2
        exit 1
        ;;
esac
