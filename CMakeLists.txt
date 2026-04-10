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

# Source files
file(GLOB_RECURSE CUDA_SOURCES src/*.cu)
file(GLOB_RECURSE CPP_SOURCES benchmarks/*.cpp)

add_executable(kernel_craft ${CUDA_SOURCES} ${CPP_SOURCES})

# Link CUDA libraries if needed
find_package(CUDAToolkit REQUIRED)
target_link_libraries(kernel_craft PRIVATE CUDA::cudart)

# Output directories for binaries and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Test executable for naive convolution
add_executable(test_conv_naive tests/test_conv_naive.cpp src/conv_naive.cu)
target_link_libraries(test_conv_naive PRIVATE CUDA::cudart)

# Benchmark executable for naive convolution
add_executable(benchmark_conv benchmarks/benchmark_conv.cpp src/conv_naive.cu)
target_link_libraries(benchmark_conv PRIVATE CUDA::cudart)


# Enable CTest integration for scalable test handling
include(CTest)   # Provides CMAKE_CTEST_COMMAND
enable_testing()


# Register the test with CTest (name will appear in summary)
add_test(NAME conv_naive_test COMMAND test_conv_naive)

# Convenience target that runs all registered tests and shows a concise summary
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_conv_naive
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