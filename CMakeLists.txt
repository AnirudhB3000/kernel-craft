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

# ---------------------------------------------------------------
# NCCL support (optional) — Phase 13 tensor parallelism
# ---------------------------------------------------------------
find_path(NCCL_INCLUDE_DIR nccl.h
    HINTS
        ${CUDAToolkit_INCLUDE_DIRS}
        /usr/include
        /usr/local/cuda/include
        /usr/local/include
)
find_library(NCCL_LIBRARY
    NAMES nccl nccl_static
    HINTS
        ${CUDAToolkit_LIBRARY_DIR}
        /usr/lib/x86_64-linux-gnu
        /usr/local/cuda/lib64
        /usr/local/lib
)
if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
    message(STATUS "Found NCCL: include=${NCCL_INCLUDE_DIR}  lib=${NCCL_LIBRARY}")
    set(HAVE_NCCL TRUE)
else()
    message(STATUS "NCCL not found — NCCL-backed collectives will be stubs (Phase 13)")
    set(HAVE_NCCL FALSE)
endif()

# TensorRT support (optional)
if(DEFINED TENSORRT_ROOT)
    set(TensorRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)
    set(TensorRT_LIBRARY_DIR ${TENSORRT_ROOT}/lib)
    if(EXISTS ${TensorRT_INCLUDE_DIR}/NvInfer.h)
        message(STATUS "Found TensorRT at ${TENSORRT_ROOT}")
        include_directories(${TensorRT_INCLUDE_DIR})
        link_directories(${TensorRT_LIBRARY_DIR})
        add_compile_definitions(HAVE_TENSORRT)
        set(HAVE_TENSORRT TRUE)
    else()
        message(WARNING "TensorRT headers not found at ${TensorRT_INCLUDE_DIR}")
    endif()
endif()

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

# Wire NCCL into kernels (and propagate HAVE_NCCL definition) if found
if(HAVE_NCCL)
    target_include_directories(kernels PUBLIC ${NCCL_INCLUDE_DIR})
    target_link_libraries(kernels PUBLIC ${NCCL_LIBRARY})
    target_compile_definitions(kernels PUBLIC HAVE_NCCL)
endif()

# TensorRT plugin library (optional, requires TensorRT SDK)
if(HAVE_TENSORRT)
    add_library(kernel_craft_tensorrt SHARED
        src/tensorrt/plugin_wrapper.cpp
        src/kernels/inference/conv_int8.cu
        src/kernels/inference/conv_activation_fusion.cu
    )
    target_include_directories(kernel_craft_tensorrt PRIVATE ${TensorRT_INCLUDE_DIR})
    target_link_libraries(kernel_craft_tensorrt PRIVATE ${TensorRT_LIBRARY_DIR}/libnvinfer.so CUDA::cudart)
    target_compile_definitions(kernel_craft_tensorrt PRIVATE HAVE_TENSORRT)
    message(STATUS "TensorRT plugin library target created")
endif()

# Default tile size for tiled convolution (8x8 provides best overall performance)
set(TILE_SIZE_W 8)
set(TILE_SIZE_H 8)

# Add tile size as compile definition for all targets using kernels
add_compile_definitions(TILE_W=${TILE_SIZE_W} TILE_H=${TILE_SIZE_H})

# Output directories for binaries and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Test executable for 3D convolution
add_executable(test_conv3d tests/test_conv3d.cpp)
target_link_libraries(test_conv3d PRIVATE kernels)

# Test executable for dilated convolution
add_executable(test_conv_dilated tests/test_conv_dilated.cpp)
target_link_libraries(test_conv_dilated PRIVATE kernels)

# Test executable for transposed convolution
add_executable(test_conv_transposed tests/test_conv_transposed.cpp)
target_link_libraries(test_conv_transposed PRIVATE kernels)

# Test executable for grouped convolution
add_executable(test_conv_grouped tests/test_conv_grouped.cpp)
target_link_libraries(test_conv_grouped PRIVATE kernels)

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

# Test executable for memory pool
add_executable(test_memory_pool tests/test_memory_pool.cpp)
target_link_libraries(test_memory_pool PRIVATE kernels)

# Test executable for CUDA graphs
add_executable(test_cuda_graphs tests/test_cuda_graphs.cpp)
target_link_libraries(test_cuda_graphs PRIVATE kernels)

# Test executable for mixed precision
add_executable(test_mixed_precision tests/test_mixed_precision.cpp)
target_link_libraries(test_mixed_precision PRIVATE kernels)

# Test executable for persistent kernels
add_executable(test_persistent_kernels tests/test_persistent_kernels.cpp)
target_link_libraries(test_persistent_kernels PRIVATE kernels)

# Test executable for INT8 quantized convolution
add_executable(test_conv_int8 tests/test_conv_int8.cpp)
target_link_libraries(test_conv_int8 PRIVATE kernels)

# Test executable for batch normalization folding
add_executable(test_bn_folding tests/test_bn_folding.cpp)
target_link_libraries(test_bn_folding PRIVATE kernels)

# Test executable for conv + activation fusion
add_executable(test_conv_activation_fusion tests/test_conv_activation_fusion.cpp)
target_link_libraries(test_conv_activation_fusion PRIVATE kernels)

# Test executable for TensorRT integration
add_executable(test_tensorrt_integration tests/test_tensorrt_integration.cpp)
target_link_libraries(test_tensorrt_integration PRIVATE kernels)

# Phase 8: Integration Enhancements tests
add_executable(test_async_streams tests/test_async_streams.cpp)
target_link_libraries(test_async_streams PRIVATE kernels)

add_executable(test_unified_memory tests/test_unified_memory.cpp)
target_link_libraries(test_unified_memory PRIVATE kernels)

add_executable(test_multi_stream_pipeline tests/test_multi_stream_pipeline.cpp)
target_link_libraries(test_multi_stream_pipeline PRIVATE kernels)

# Phase 11: Transformer/LLM inference tests
add_executable(test_flash_attention tests/test_flash_attention.cpp)
target_link_libraries(test_flash_attention PRIVATE kernels)

add_executable(test_paged_attention tests/test_paged_attention.cpp)
target_link_libraries(test_paged_attention PRIVATE kernels)

add_executable(test_quant_int4 tests/test_quant_int4.cpp)
target_link_libraries(test_quant_int4 PRIVATE kernels)

add_executable(test_fp8_quant tests/test_fp8_quant.cpp)
target_link_libraries(test_fp8_quant PRIVATE kernels)

add_executable(test_speculative_decoding tests/test_speculative_decoding.cpp)
target_link_libraries(test_speculative_decoding PRIVATE kernels)

add_executable(test_tensor_parallel tests/test_tensor_parallel.cpp)
target_link_libraries(test_tensor_parallel PRIVATE kernels)

# Test executable for TensorRT plugin (only if TensorRT is available)
if(HAVE_TENSORRT)
    add_executable(test_tensorrt_plugin tests/test_tensorrt_plugin.cpp
        src/tensorrt/plugin_wrapper.cpp
        src/kernels/inference/conv_int8.cu
        src/kernels/inference/conv_activation_fusion.cu)
    target_include_directories(test_tensorrt_plugin PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(test_tensorrt_plugin PRIVATE ${TENSORRT_ROOT}/lib/libnvinfer.so CUDA::cudart)
    target_compile_definitions(test_tensorrt_plugin PRIVATE HAVE_TENSORRT)
endif()

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

# Benchmark executable for memory pool
add_executable(benchmark_memory_pool benchmarks/benchmark_memory_pool.cpp)
target_link_libraries(benchmark_memory_pool PRIVATE kernels)

# Benchmark executable for CUDA graphs
add_executable(benchmark_cuda_graphs benchmarks/benchmark_cuda_graphs.cpp)
target_link_libraries(benchmark_cuda_graphs PRIVATE kernels)

# Benchmark executable for mixed precision
add_executable(benchmark_mixed_precision benchmarks/benchmark_mixed_precision.cpp)
target_link_libraries(benchmark_mixed_precision PRIVATE kernels)

# Benchmark executable for persistent kernels
add_executable(benchmark_persistent_kernels benchmarks/benchmark_persistent_kernels.cpp)
target_link_libraries(benchmark_persistent_kernels PRIVATE kernels)

# Phase 8: Integration Enhancements benchmarks
add_executable(benchmark_async_streams benchmarks/benchmark_async_streams.cpp)
target_link_libraries(benchmark_async_streams PRIVATE kernels)

add_executable(benchmark_unified_memory benchmarks/benchmark_unified_memory.cpp)
target_link_libraries(benchmark_unified_memory PRIVATE kernels)

add_executable(benchmark_multi_stream_pipeline benchmarks/benchmark_multi_stream_pipeline.cpp)
target_link_libraries(benchmark_multi_stream_pipeline PRIVATE kernels)

# Phase 11: Transformer/LLM inference benchmarks
add_executable(benchmark_flash_attention benchmarks/benchmark_flash_attention.cpp)
target_link_libraries(benchmark_flash_attention PRIVATE kernels)

add_executable(benchmark_paged_attention benchmarks/benchmark_paged_attention.cpp)
target_link_libraries(benchmark_paged_attention PRIVATE kernels)

add_executable(benchmark_quant benchmarks/benchmark_quant.cpp)
target_link_libraries(benchmark_quant PRIVATE kernels)

# Phase 13: Tensor parallel benchmark
add_executable(benchmark_tensor_parallel benchmarks/benchmark_tensor_parallel.cpp)
target_link_libraries(benchmark_tensor_parallel PRIVATE kernels)
if(HAVE_NCCL)
    target_compile_definitions(benchmark_tensor_parallel PRIVATE HAVE_NCCL)
endif()

# Phase 11: Python transformer module (separate pybind11 extension)
pybind11_add_module(kernel_craft_transformer src/python/pybind_transformer.cpp)
target_link_libraries(kernel_craft_transformer PRIVATE kernels CUDA::cudart)
set_target_properties(kernel_craft_transformer PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PYTHON_MODULE_OUTPUT_DIR}
)

# Enable CTest BEFORE registering tests
include(CTest)
enable_testing()

# Register tests with CTest
add_test(NAME conv3d_test COMMAND test_conv3d)
add_test(NAME conv_dilated_test COMMAND test_conv_dilated)
add_test(NAME conv_transposed_test COMMAND test_conv_transposed)
add_test(NAME conv_grouped_test COMMAND test_conv_grouped)
add_test(NAME conv_naive_test COMMAND test_conv_naive)
add_test(NAME conv_tiled_test COMMAND test_conv_tiled)
add_test(NAME preprocess_test COMMAND test_preprocess)
add_test(NAME pipeline_test COMMAND test_pipeline)
add_test(NAME custom_op_test COMMAND test_custom_op)
add_test(NAME memory_pool_test COMMAND test_memory_pool)
add_test(NAME cuda_graphs_test COMMAND test_cuda_graphs)
add_test(NAME mixed_precision_test COMMAND test_mixed_precision)
add_test(NAME persistent_kernels_test COMMAND test_persistent_kernels)
add_test(NAME conv_int8_test COMMAND test_conv_int8)
add_test(NAME bn_folding_test COMMAND test_bn_folding)
add_test(NAME conv_activation_fusion_test COMMAND test_conv_activation_fusion)
add_test(NAME tensorrt_integration_test COMMAND test_tensorrt_integration)

# Phase 8: Integration Enhancements tests
add_test(NAME async_streams_test COMMAND test_async_streams)
add_test(NAME unified_memory_test COMMAND test_unified_memory)
add_test(NAME multi_stream_pipeline_test COMMAND test_multi_stream_pipeline)

# Phase 11: Transformer/LLM inference tests
add_test(NAME flash_attention_test COMMAND test_flash_attention)
add_test(NAME paged_attention_test COMMAND test_paged_attention)
add_test(NAME quant_int4_test COMMAND test_quant_int4)
add_test(NAME fp8_quant_test COMMAND test_fp8_quant)
add_test(NAME speculative_decoding_test COMMAND test_speculative_decoding)
add_test(NAME tensor_parallel_test COMMAND test_tensor_parallel)

# Register all benchmarks as tests
add_test(NAME benchmark_conv COMMAND benchmark_conv)
add_test(NAME benchmark_custom COMMAND benchmark_custom)
add_test(NAME benchmark_pipeline COMMAND benchmark_pipeline)
add_test(NAME benchmark_full_pipeline COMMAND benchmark_full_pipeline)
add_test(NAME benchmark_memory_pool COMMAND benchmark_memory_pool)
add_test(NAME benchmark_cuda_graphs COMMAND benchmark_cuda_graphs)
add_test(NAME benchmark_mixed_precision COMMAND benchmark_mixed_precision)
add_test(NAME benchmark_persistent_kernels COMMAND benchmark_persistent_kernels)

# Phase 8 benchmarks
add_test(NAME benchmark_async_streams COMMAND benchmark_async_streams)
add_test(NAME benchmark_unified_memory COMMAND benchmark_unified_memory)
add_test(NAME benchmark_multi_stream_pipeline COMMAND benchmark_multi_stream_pipeline)

# Phase 11 benchmarks
add_test(NAME benchmark_flash_attention COMMAND benchmark_flash_attention)
add_test(NAME benchmark_paged_attention COMMAND benchmark_paged_attention)
add_test(NAME benchmark_quant COMMAND benchmark_quant)

# Phase 13 benchmarks
add_test(NAME benchmark_tensor_parallel COMMAND benchmark_tensor_parallel)

# ---------------------------------------------------------------
# Test suite labels: unit | integration | benchmark
# ---------------------------------------------------------------
set_tests_properties(
    conv3d_test conv_dilated_test conv_transposed_test conv_grouped_test
    conv_naive_test conv_tiled_test
    preprocess_test custom_op_test
    memory_pool_test mixed_precision_test persistent_kernels_test
    conv_int8_test bn_folding_test conv_activation_fusion_test
    flash_attention_test paged_attention_test
    quant_int4_test fp8_quant_test
    speculative_decoding_test tensor_parallel_test
    PROPERTIES LABELS "unit"
)

set_tests_properties(
    pipeline_test
    cuda_graphs_test
    async_streams_test unified_memory_test multi_stream_pipeline_test
    tensorrt_integration_test
    PROPERTIES LABELS "integration"
)

set_tests_properties(
    benchmark_conv benchmark_custom benchmark_pipeline benchmark_full_pipeline
    benchmark_memory_pool benchmark_cuda_graphs benchmark_mixed_precision
    benchmark_persistent_kernels
    benchmark_async_streams benchmark_unified_memory benchmark_multi_stream_pipeline
    benchmark_flash_attention benchmark_paged_attention benchmark_quant
    benchmark_tensor_parallel
    PROPERTIES LABELS "benchmark"
)

# ---------------------------------------------------------------
# Convenience make targets
# ---------------------------------------------------------------

# Run C++ unit tests only
# Matches all *_test names, excludes the six integration tests
add_custom_target(run_unit_tests
    COMMAND bash "${CMAKE_SOURCE_DIR}/scripts/run_cpp_tests.sh" "${CMAKE_BINARY_DIR}" unit
    DEPENDS
        test_conv_naive test_conv_tiled test_conv3d test_conv_dilated
        test_conv_transposed test_conv_grouped
        test_conv_int8 test_bn_folding test_conv_activation_fusion
        test_preprocess test_custom_op
        test_memory_pool test_mixed_precision test_persistent_kernels
        test_flash_attention test_paged_attention
        test_quant_int4 test_fp8_quant
        test_speculative_decoding test_tensor_parallel
    COMMENT "Running C++ unit tests"
)

# Run C++ integration tests only
add_custom_target(run_integration_tests
    COMMAND bash "${CMAKE_SOURCE_DIR}/scripts/run_cpp_tests.sh" "${CMAKE_BINARY_DIR}" integration
    DEPENDS
        test_pipeline test_cuda_graphs
        test_async_streams test_unified_memory test_multi_stream_pipeline
        test_tensorrt_integration
    COMMENT "Running C++ integration tests"
)

# Run all C++ benchmarks
add_custom_target(run_benchmarks
    COMMAND bash "${CMAKE_SOURCE_DIR}/scripts/run_cpp_tests.sh" "${CMAKE_BINARY_DIR}" benchmark
    DEPENDS
        benchmark_conv benchmark_custom benchmark_pipeline benchmark_full_pipeline
        benchmark_memory_pool benchmark_cuda_graphs benchmark_mixed_precision
        benchmark_persistent_kernels
        benchmark_async_streams benchmark_unified_memory benchmark_multi_stream_pipeline
        benchmark_flash_attention benchmark_paged_attention benchmark_quant
        benchmark_tensor_parallel
    COMMENT "Running all C++ benchmarks"
)

# Run all Python tests (unit + integration) via pytest
add_custom_target(run_python_tests
    COMMAND bash "${CMAKE_SOURCE_DIR}/scripts/run_python_tests.sh" "${CMAKE_SOURCE_DIR}" all
    COMMENT "Running all Python tests"
)

# Full release gate: C++ unit → Python unit → C++ integration → Python integration → benchmarks
add_custom_target(run_tests
    COMMAND bash "${CMAKE_SOURCE_DIR}/scripts/run_all_tests.sh" "${CMAKE_BINARY_DIR}" "${CMAKE_SOURCE_DIR}"
    DEPENDS
        test_conv_naive test_conv_tiled test_conv3d test_conv_dilated
        test_conv_transposed test_conv_grouped
        test_conv_int8 test_bn_folding test_conv_activation_fusion
        test_preprocess test_custom_op
        test_memory_pool test_mixed_precision test_persistent_kernels
        test_flash_attention test_paged_attention
        test_quant_int4 test_fp8_quant
        test_speculative_decoding test_tensor_parallel
        test_pipeline test_cuda_graphs
        test_async_streams test_unified_memory test_multi_stream_pipeline
        test_tensorrt_integration
        benchmark_conv benchmark_custom benchmark_pipeline benchmark_full_pipeline
        benchmark_memory_pool benchmark_cuda_graphs benchmark_mixed_precision
        benchmark_persistent_kernels
        benchmark_async_streams benchmark_unified_memory benchmark_multi_stream_pipeline
        benchmark_flash_attention benchmark_paged_attention benchmark_quant
        benchmark_tensor_parallel
    COMMENT "Release gate: unit → python unit → integration → python integration → benchmarks"
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

# TensorRT example (optional, only if TensorRT is available)
if(HAVE_TENSORRT)
    add_executable(example_tensorrt_usage examples/tensorrt/example_plugin_usage.cpp
        src/tensorrt/plugin_wrapper.cpp
        src/kernels/inference/conv_int8.cu
        src/kernels/inference/conv_activation_fusion.cu)
    target_include_directories(example_tensorrt_usage PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(example_tensorrt_usage PRIVATE ${TENSORRT_ROOT}/lib/libnvinfer.so CUDA::cudart)
    target_compile_definitions(example_tensorrt_usage PRIVATE HAVE_TENSORRT)
endif()

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