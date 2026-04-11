cmake_minimum_required(VERSION 3.18)
project(kernel_craft LANGUAGES CXX CUDA)

# ------------------------------------------------------------
# Clean up stale binaries that may have been generated in the
# source directory from previous builds. This runs at CMake
# configure time so that a subsequent "cmake .." call always starts
# from a clean state.
# ------------------------------------------------------------
execute_process(COMMAND ${CMAKE_COMMAND} -E rm -f
    ${CMAKE_SOURCE_DIR}/test_conv_naive
    ${CMAKE_SOURCE_DIR}/conv_naive.o
    RESULT_VARIABLE _clean_res
    OUTPUT_QUIET
    ERROR_QUIET)

# Optionally warn if cleanup failed (non‑zero result)
if(_clean_res)
    message(WARNING "Failed to delete stale binaries during configure")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files - CUDA only
file(GLOB_RECURSE CUDA_SOURCES src/*.cu)

# Link CUDA libraries if needed
find_package(CUDAToolkit REQUIRED)

# Default tile size for tiled convolution (8x8 provides best overall performance)
set(TILE_SIZE_W 8)
set(TILE_SIZE_H 8)

# Add tile size as compile definition for CUDA targets
add_compile_definitions(TILE_W=${TILE_SIZE_W} TILE_H=${TILE_SIZE_H})

# Output directories for binaries and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Test executable for naive convolution
add_executable(test_conv_naive tests/test_conv_naive.cpp src/conv_naive.cu)
target_link_libraries(test_conv_naive PRIVATE CUDA::cudart)

# Test executable for tiled convolution
add_executable(test_conv_tiled tests/test_conv_tiled.cpp src/conv_tiled.cu)
target_link_libraries(test_conv_tiled PRIVATE CUDA::cudart)

# Benchmark executable for convolution
add_executable(benchmark_conv benchmarks/benchmark_conv.cpp src/conv_naive.cu src/conv_tiled.cu)
target_link_libraries(benchmark_conv PRIVATE CUDA::cudart)

# Benchmark executable for custom operations (sparse convolution)
add_executable(benchmark_custom benchmarks/benchmark_custom.cpp src/custom_op.cu)
target_link_libraries(benchmark_custom PRIVATE CUDA::cudart)

# Benchmark executable for pipeline (fused vs separate)
add_executable(benchmark_pipeline benchmarks/benchmark_pipeline.cpp src/pipeline_separate.cu src/pipeline_fused.cu)
target_link_libraries(benchmark_pipeline PRIVATE CUDA::cudart)

# Enable CTest BEFORE registering tests
include(CTest)
enable_testing()

# Register tests with CTest
add_test(NAME conv_naive_test COMMAND test_conv_naive)
add_test(NAME conv_tiled_test COMMAND test_conv_tiled)

# Convenience target that runs all registered tests and shows a concise summary
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_conv_naive test_conv_tiled
    COMMENT "Running all unit tests via CTest"
)

# Register benchmark as a test for CTest integration
add_test(NAME benchmark_conv COMMAND benchmark_conv)

# Convenience target to run only benchmarks via CTest (provides summary)
add_custom_target(run_benchmarks
    COMMAND ${CMAKE_CTEST_COMMAND} -R benchmark_conv --output-on-failure
    DEPENDS benchmark_conv
    COMMENT "Running all benchmarks via CTest"
)