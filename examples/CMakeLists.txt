# examples/CMakeLists.txt
# Builds all kernel-craft example executables.
#
# Directory layout
#   cnn/         — core convolution and performance examples
#   transformer/ — FlashAttention, PagedAttention, tensor parallelism
#   tensorrt/    — TensorRT IPluginV2DynamicExt CNN example (optional)
#
# Every example links against the `kernels` target which provides all
# CUDA launcher symbols (launch_conv_*, launch_flash_attention, etc.).
#
# Build a specific example:
#   cmake --build build --target conv_naive
#   cmake --build build --target flash_attention
#
# Build all C++ examples:
#   cmake --build build --target build_examples
#
# Build and run all C++ examples:
#   cmake --build build --target run_examples
#
# Run all Python examples:
#   python examples/python/run_all.py

cmake_minimum_required(VERSION 3.18)

# ---------------------------------------------------------------------------
# Helper macro: declare one example target.
# NAME    — CMake target name (also the binary name in build/bin/)
# SUBDIR  — subdirectory under examples/ that contains the source file
# ---------------------------------------------------------------------------
macro(add_example NAME SUBDIR)
    add_executable(${NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR}/${NAME}.cpp
    )
    target_link_libraries(${NAME} PRIVATE kernels)
    set_target_properties(${NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endmacro()

# ---------------------------------------------------------------------------
# examples/cnn/  — convolution and CNN pipeline examples
# ---------------------------------------------------------------------------
add_example(conv_naive      cnn)   # baseline: one thread per pixel, global-mem only
add_example(conv_tiled      cnn)   # shared-memory tile sweep (8×8, 16×16, 32×32)
add_example(fused_pipeline  cnn)   # conv + BN-affine + ReLU in one kernel
add_example(memory_pool     cnn)   # pre-allocated pool vs per-batch cudaMalloc
add_example(preprocess      cnn)   # resize (bilinear) → normalize → flip
add_example(cuda_graphs     cnn)   # graph capture / replay vs separate launches

# ---------------------------------------------------------------------------
# examples/transformer/  — LLM attention and parallelism examples
# ---------------------------------------------------------------------------
add_example(flash_attention  transformer)  # MHA / GQA / causal; prefill stage
add_example(paged_attention  transformer)  # paged KV-cache; decode stage
add_example(tensor_parallel  transformer)  # col/row parallel linear + collectives
add_example(selective_scan   transformer)  # ZOH SSM recurrence (Mamba)
add_example(depthwise_conv1d transformer)  # causal depthwise Conv1d (Mamba mixer)
add_example(rmsnorm          transformer)  # fused RMSNorm (Mamba / LLaMA)

# ---------------------------------------------------------------------------
# examples/tensorrt/  — TensorRT plugin (optional)
# Enable: cmake -DHAVE_TENSORRT=ON -DTENSORRT_ROOT=/usr/local/tensorrt ..
# ---------------------------------------------------------------------------
if(HAVE_TENSORRT)
    add_executable(plugin_cnn
        ${CMAKE_CURRENT_SOURCE_DIR}/tensorrt/plugin_cnn.cpp
        ${CMAKE_SOURCE_DIR}/src/tensorrt/plugin_wrapper.cpp
        ${CMAKE_SOURCE_DIR}/src/kernels/inference/conv_int8.cu
        ${CMAKE_SOURCE_DIR}/src/kernels/inference/conv_activation_fusion.cu
    )
    target_include_directories(plugin_cnn PRIVATE
        ${TENSORRT_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/src/tensorrt
    )
    target_link_libraries(plugin_cnn PRIVATE
        ${TENSORRT_ROOT}/lib/libnvinfer.so
        CUDA::cudart
    )
    target_compile_definitions(plugin_cnn PRIVATE HAVE_TENSORRT)
    set_target_properties(plugin_cnn PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    message(STATUS "TensorRT example enabled  (${TENSORRT_ROOT})")
else()
    message(STATUS "TensorRT example skipped  (-DHAVE_TENSORRT=ON to enable)")
endif()

# ---------------------------------------------------------------------------
# Convenience targets
# ---------------------------------------------------------------------------
set(_CORE_EXAMPLES
    conv_naive
    conv_tiled
    fused_pipeline
    memory_pool
    preprocess
    cuda_graphs
    flash_attention
    paged_attention
    tensor_parallel
    selective_scan
    depthwise_conv1d
    rmsnorm
)

# build_examples — compile all non-TRT examples
add_custom_target(build_examples DEPENDS ${_CORE_EXAMPLES})

# run_examples — compile then run each in sequence
add_custom_target(run_examples
    COMMENT "Building and running all C++ examples"
    DEPENDS ${_CORE_EXAMPLES}
)
foreach(_ex ${_CORE_EXAMPLES})
    add_custom_command(TARGET run_examples POST_BUILD
        COMMAND ${CMAKE_BINARY_DIR}/bin/${_ex}
        COMMENT "  → ${_ex}"
    )
endforeach()

# Legacy alias so `make examples` still works
add_custom_target(examples DEPENDS build_examples)
