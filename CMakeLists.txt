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

# Source files - CUDA only (exclude venv directory)
file(GLOB_RECURSE CUDA_SOURCES 
    CONFIGURE_DEPENDS 
    "${CMAKE_SOURCE_DIR}/src" 
    "*.cu"
)
# Filter out venv and build directories
set(CUDA_SOURCES_EXCLUDE "")
foreach(src ${CUDA_SOURCES})
    if(src MATCHES "venv|build|__pycache__|\\.pytest")
        list(APPEND CUDA_SOURCES_EXCLUDE ${src})
    endif()
endforeach()
list(REMOVE_DUPLICATES CUDA_SOURCES_EXCLUDE)
list(REMOVE_ITEM CUDA_SOURCES ${CUDA_SOURCES_EXCLUDE})

# Link CUDA libraries if needed
find_package(CUDAToolkit REQUIRED)

# ------------------------------------------------------------
# pybind11 Python bindings (auto-downloaded via FetchContent)
# ------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Python extension target (links against kernels library)
# Output to src/python/build/ for easy import
set(PYTHON_MODULE_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/src/python/build)
pybind11_add_module(kernel_craft_python src/python/pybind_cuda.cpp)
target_link_libraries(kernel_craft_python PRIVATE kernels CUDA::cudart)
set_target_properties(kernel_craft_python PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PYTHON_MODULE_OUTPUT_DIR}
)

# Create shared library from CUDA sources (required for linking into Python module)
add_library(kernels SHARED ${CUDA_SOURCES})
target_link_libraries(kernels PUBLIC CUDA::cudart)
target_include_directories(kernels PUBLIC ${CMAKE_SOURCE_DIR}/src)

# Default tile size for tiled convolution (8x8 provides best overall performance)
set(TILE_SIZE_W 8)
set(TILE_SIZE_H 8)

# Add tile size as compile definition for all targets using kernels
add_compile_definitions(TILE_W=${TILE_SIZE_W} TILE_H=${TILE_SIZE_H})

# Output directories for binaries and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Test executable for naive convolution
add_executable(test_conv_naive tests/test_conv_naive.cpp)
target_link_libraries(test_conv_naive PRIVATE kernels)

# Test executable for tiled convolution
add_executable(test_conv_tiled tests/test_conv_tiled.cpp)
target_link_libraries(test_conv_tiled PRIVATE kernels)

# Test executable for preprocessing kernels
add_executable(test_preprocess tests/test_preprocess.cpp)
target_link_libraries(test_preprocess PRIVATE kernels)

# Test executable for pipeline kernels
add_executable(test_pipeline tests/test_pipeline.cpp)
target_link_libraries(test_pipeline PRIVATE kernels)

# Test executable for custom operations
add_executable(test_custom_op tests/test_custom_op.cpp)
target_link_libraries(test_custom_op PRIVATE kernels)

# Benchmark executable for convolution
add_executable(benchmark_conv benchmarks/benchmark_conv.cpp)
target_link_libraries(benchmark_conv PRIVATE kernels)

# Benchmark executable for custom operations (sparse convolution)
add_executable(benchmark_custom benchmarks/benchmark_custom.cpp)
target_link_libraries(benchmark_custom PRIVATE kernels)

# Benchmark executable for pipeline (fused vs separate)
add_executable(benchmark_pipeline benchmarks/benchmark_pipeline.cpp)
target_link_libraries(benchmark_pipeline PRIVATE kernels)

# Benchmark executable for full GPU pipeline
add_executable(benchmark_full_pipeline benchmarks/benchmark_full_pipeline.cpp)
target_link_libraries(benchmark_full_pipeline PRIVATE kernels)

# Enable CTest BEFORE registering tests
include(CTest)
enable_testing()

# Register tests with CTest
add_test(NAME conv_naive_test COMMAND test_conv_naive)
add_test(NAME conv_tiled_test COMMAND test_conv_tiled)
add_test(NAME preprocess_test COMMAND test_preprocess)
add_test(NAME pipeline_test COMMAND test_pipeline)
add_test(NAME custom_op_test COMMAND test_custom_op)

# Register all benchmarks as tests
add_test(NAME benchmark_conv COMMAND benchmark_conv)
add_test(NAME benchmark_custom COMMAND benchmark_custom)
add_test(NAME benchmark_pipeline COMMAND benchmark_pipeline)
add_test(NAME benchmark_full_pipeline COMMAND benchmark_full_pipeline)

# Convenience target that runs all registered tests and shows a concise summary
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_conv_naive test_conv_tiled test_preprocess test_pipeline test_custom_op
    COMMENT "Running all unit tests via CTest"
)

# Convenience target to run only benchmarks via CTest (provides summary)
add_custom_target(run_benchmarks
    COMMAND ${CMAKE_CTEST_COMMAND} -R benchmark_ --output-on-failure
    DEPENDS benchmark_conv benchmark_custom benchmark_pipeline benchmark_full_pipeline
    COMMENT "Running all benchmarks via CTest"
)

# Example executables
add_executable(example_conv_naive examples/example_conv_naive.cpp)
target_link_libraries(example_conv_naive PRIVATE kernels)

add_executable(example_conv_tiled examples/example_conv_tiled.cpp)
target_link_libraries(example_conv_tiled PRIVATE kernels)

add_executable(example_preprocess examples/example_preprocess.cpp)
target_link_libraries(example_preprocess PRIVATE kernels)

add_executable(example_fused_pipeline examples/example_fused_pipeline.cpp)
target_link_libraries(example_fused_pipeline PRIVATE kernels)

# Target to build and run all examples (use: make examples)
add_custom_target(examples
    COMMAND ${CMAKE_COMMAND} --build . --target example_conv_naive example_conv_tiled example_preprocess example_fused_pipeline
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/example_conv_naive
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/example_conv_tiled
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/example_preprocess
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/example_fused_pipeline
    COMMENT "Building and running all examples"
    DEPENDS example_conv_naive example_conv_tiled example_preprocess example_fused_pipeline
)