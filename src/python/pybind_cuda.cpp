/**
 * \file pybind_cuda.cpp
 * \brief Python bindings for CUDA convolution kernels.
 *
 * This file exposes the conv_naive and conv_tiled CUDA kernels to Python,
 * supporting both numpy arrays and PyTorch tensors as inputs. It handles
 * all memory transfer between host and device automatically.
 *
 * \par Supported Input Types
 * - numpy.ndarray (float32, 2D)
 * - torch.Tensor (float32, 2D, on CUDA device)
 *
 * \par Supported Tile Sizes for conv_tiled
 * - 8x8 (default, best overall performance for 3x3 kernels)
 * - 16x16
 * - 32x32
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>

#include <cstring>
#include <iostream>
#include <sstream>

namespace py = pybind11;

/**
 * \brief Macro for checking CUDA function return values.
 *
 * This macro wraps CUDA API calls and throws a std::runtime_error if
 * the call fails, including the file and line number for debugging.
 *
 * \param call The CUDA API function call to check.
 */
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + \
                cudaGetErrorString(err) + " at " + __FILE__ + ":" + \
                std::to_string(__LINE__)); \
        } \
    } while (0)

// ---------------------------------------------------------------------------
// Forward declarations of CUDA kernels (defined in conv_naive.cu, conv_tiled.cu)
// ---------------------------------------------------------------------------

/**
 * \brief Launch the naive convolution kernel.
 *
 * \param input   Device pointer to input image.
 * \param kernel  Device pointer to convolution kernel.
 * \param output  Device pointer to output buffer.
 * \param width   Image width.
 * \param height  Image height.
 * \param ksize   Kernel size (must be odd).
 * \param block   CUDA block dimensions (default 16x16).
 */
extern "C" void launch_conv_naive(const float* input, const float* kernel,
                                 float* output, int width, int height, int ksize,
                                 dim3 block = dim3(16,16,1));

/**
 * \brief Launch the tiled convolution kernel.
 *
 * \param input   Device pointer to input image.
 * \param kernel  Device pointer to convolution kernel.
 * \param output  Device pointer to output buffer.
 * \param width   Image width.
 * \param height  Image height.
 * \param ksize   Kernel size (must be odd).
 * \param block   CUDA block dimensions (determines tile size).
 */
extern "C" void launch_conv_tiled(const float* input, const float* kernel,
                                   float* output, int width, int height, int ksize,
                                   dim3 block = dim3(8,8,1));

// ---------------------------------------------------------------------------
// Tile size dispatch
// ---------------------------------------------------------------------------

/**
 * \brief Dispatch to tiled convolution kernel with specified tile size.
 *
 * The tiled convolution kernel requires compile-time tile size for shared
 * memory allocation. This function dispatches to the appropriate variant
 * based on the requested tile dimensions. Unsupported sizes fall back
 * to the default 8x8 configuration.
 *
 * \param d_input  Device pointer to input image.
 * \param d_kernel Device pointer to convolution kernel.
 * \param d_output Device pointer to output buffer.
 * \param width    Image width.
 * \param height   Image height.
 * \param ksize    Kernel size (must be odd).
 * \param tile_w   Requested tile width (8, 16, or 32).
 * \param tile_h   Requested tile height (8, 16, or 32).
 */
void launch_conv_tiled_dispatch(const float* d_input, const float* d_kernel,
                                 float* d_output, int width, int height, int ksize,
                                 int tile_w, int tile_h) {
    // Set block dimensions to match requested tile size
    dim3 block(tile_w, tile_h, 1);

    // Dispatch to appropriate compiled variant
    if (tile_w == 8 && tile_h == 8) {
        // 8x8 tile - best for small/medium images with 3x3 kernels
        launch_conv_tiled(d_input, d_kernel, d_output, width, height, ksize, block);
    } else if (tile_w == 16 && tile_h == 16) {
        // 16x16 tile - good balance for medium images
        launch_conv_tiled(d_input, d_kernel, d_output, width, height, ksize, block);
    } else if (tile_w == 32 && tile_h == 32) {
        // 32x32 tile - best for large images but more shared memory overhead
        launch_conv_tiled(d_input, d_kernel, d_output, width, height, ksize, block);
    } else {
        // Fallback to default 8x8 for unsupported sizes
        launch_conv_tiled(d_input, d_kernel, d_output, width, height, ksize, dim3(8,8,1));
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/**
 * \brief Check if PyTorch is available at runtime.
 *
 * This function attempts to import torch to determine if PyTorch is installed.
 * Used to conditionally enable PyTorch tensor support.
 *
 * \return true if PyTorch is available, false otherwise.
 */
bool is_torch_available() {
    try {
        py::module torch = py::module::import("torch");
        return true;
    } catch (...) {
        return false;
    }
}

/**
 * \brief CPU reference implementation for convolution (verification only).
 *
 * This is a simple double-precision CPU implementation used for testing
 * and verification. Not optimized for performance.
 *
 * \param input  Input image (row-major).
 * \param kernel Convolution kernel (row-major).
 * \param output Output image buffer.
 * \param width  Image width.
 * \param height Image height.
 * \param ksize  Kernel size.
 */
void conv_cpu(const float* input, const float* kernel, float* output,
              int width, int height, int ksize) {
    int kHalf = ksize / 2;
    for (int oy = 0; oy < height; ++oy) {
        for (int ox = 0; ox < width; ++ox) {
            double sum = 0.0;
            for (int ky = 0; ky < ksize; ++ky) {
                int iy = oy + ky - kHalf;
                if (iy < 0 || iy >= height) continue;
                for (int kx = 0; kx < ksize; ++kx) {
                    int ix = ox + kx - kHalf;
                    if (ix < 0 || ix >= width) continue;
                    sum += static_cast<double>(input[iy * width + ix]) *
                           static_cast<double>(kernel[ky * ksize + kx]);
                }
            }
            output[oy * width + ox] = static_cast<float>(sum);
        }
    }
}

// ---------------------------------------------------------------------------
// numpy array handling
// ---------------------------------------------------------------------------

/**
 * \brief Naive convolution for numpy arrays.
 *
 * Takes numpy arrays as input, allocates GPU memory, copies data to device,
 * runs the convolution kernel, and copies results back to host.
 *
 * \param input  2D float32 numpy array (HxW), row-major.
 * \param kernel 2D float32 numpy array (kxk), row-major, must be odd-sized.
 * \return 2D float32 numpy array (HxW) with convolution result.
 *
 * \throws std::runtime_error If input/kernel dimensions are invalid.
 */
py::array_t<float> conv_naive_numpy(py::array_t<float, py::array::c_style> input,
                                     py::array_t<float, py::array::c_style> kernel) {
    // ---- Input Validation ----
    // Verify input is 2D
    if (input.ndim() != 2) {
        throw std::runtime_error("input must be a 2D array");
    }
    // Verify kernel is 2D
    if (kernel.ndim() != 2) {
        throw std::runtime_error("kernel must be a 2D array");
    }
    // Verify kernel is square
    if (kernel.shape(0) != kernel.shape(1)) {
        throw std::runtime_error("kernel must be square");
    }
    // Verify kernel has odd dimension (required for center-aligned convolution)
    if (kernel.shape(0) % 2 == 0) {
        throw std::runtime_error("kernel dimension must be odd");
    }

    // ---- Extract dimensions ----
    int height = static_cast<int>(input.shape(0));
    int width = static_cast<int>(input.shape(1));
    int ksize = static_cast<int>(kernel.shape(0));

    // ---- Allocate output buffer ----
    py::array_t<float> output = py::array_t<float>({height, width});
    auto output_buf = output.request();
    float* output_ptr = static_cast<float*>(output_buf.ptr);

    // ---- Allocate GPU memory ----
    float *d_input, *d_kernel, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, height * width * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kernel, ksize * ksize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, height * width * sizeof(float)));

    // ---- Copy input and kernel to device ----
    auto input_buf = input.request();
    auto kernel_buf = kernel.request();
    CHECK_CUDA(cudaMemcpy(d_input, input_buf.ptr, height * width * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, kernel_buf.ptr, ksize * ksize * sizeof(float), cudaMemcpyHostToDevice));

    // ---- Launch CUDA kernel ----
    launch_conv_naive(d_input, d_kernel, d_output, width, height, ksize);

    // ---- Copy result back to host ----
    CHECK_CUDA(cudaMemcpy(output_ptr, d_output, height * width * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- Cleanup GPU memory ----
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_kernel));
    CHECK_CUDA(cudaFree(d_output));

    return output;
}

/**
 * \brief Tiled convolution for numpy arrays.
 *
 * Similar to conv_naive_numpy but uses the tiled kernel which loads
 * image tiles into shared memory for better memory bandwidth utilization.
 *
 * \param input  2D float32 numpy array (HxW), row-major.
 * \param kernel 2D float32 numpy array (kxk), row-major, must be odd-sized.
 * \param tile_w Tile width (8, 16, or 32).
 * \param tile_h Tile height (8, 16, or 32).
 * \return 2D float32 numpy array (HxW) with convolution result.
 *
 * \throws std::runtime_error If input/kernel dimensions are invalid.
 */
py::array_t<float> conv_tiled_numpy(py::array_t<float, py::array::c_style> input,
                                     py::array_t<float, py::array::c_style> kernel,
                                     int tile_w, int tile_h) {
    // ---- Input Validation ----
    if (input.ndim() != 2) {
        throw std::runtime_error("input must be a 2D array");
    }
    if (kernel.ndim() != 2) {
        throw std::runtime_error("kernel must be a 2D array");
    }
    if (kernel.shape(0) != kernel.shape(1)) {
        throw std::runtime_error("kernel must be square");
    }
    if (kernel.shape(0) % 2 == 0) {
        throw std::runtime_error("kernel dimension must be odd");
    }

    // ---- Extract dimensions ----
    int height = static_cast<int>(input.shape(0));
    int width = static_cast<int>(input.shape(1));
    int ksize = static_cast<int>(kernel.shape(0));

    // ---- Allocate output buffer ----
    py::array_t<float> output = py::array_t<float>({height, width});
    auto output_buf = output.request();
    float* output_ptr = static_cast<float*>(output_buf.ptr);

    // ---- Allocate GPU memory ----
    float *d_input, *d_kernel, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, height * width * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kernel, ksize * ksize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, height * width * sizeof(float)));

    // ---- Copy input and kernel to device ----
    auto input_buf = input.request();
    auto kernel_buf = kernel.request();
    CHECK_CUDA(cudaMemcpy(d_input, input_buf.ptr, height * width * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, kernel_buf.ptr, ksize * ksize * sizeof(float), cudaMemcpyHostToDevice));

    // ---- Launch tiled CUDA kernel ----
    // Dispatch to appropriate tile size variant
    launch_conv_tiled_dispatch(d_input, d_kernel, d_output, width, height, ksize, tile_w, tile_h);

    // ---- Copy result back to host ----
    CHECK_CUDA(cudaMemcpy(output_ptr, d_output, height * width * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- Cleanup GPU memory ----
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_kernel));
    CHECK_CUDA(cudaFree(d_output));

    return output;
}

// ---------------------------------------------------------------------------
// PyTorch tensor handling
// ---------------------------------------------------------------------------

/**
 * \brief Naive convolution for PyTorch tensors.
 *
 * Takes PyTorch tensors on CUDA device as input. Extracts device pointers
 * directly to avoid unnecessary memory copies. Creates output tensor on
 * the same device.
 *
 * \param input  2D float32 torch.Tensor on CUDA device.
 * \param kernel 2D float32 torch.Tensor (kxk, must be odd-sized).
 * \return 2D float32 torch.Tensor on same device.
 *
 * \throws std::runtime_error If PyTorch is not available or dimensions invalid.
 *
 * \note This function uses data_ptr() to get raw device pointers, bypassing
 *       pybind11's tensor binding for performance. The caller must ensure
 *       tensors are on CUDA and are float32.
 */
py::object conv_naive_torch(py::object input, py::object kernel) {
    // Check if PyTorch is available
    if (!is_torch_available()) {
        throw std::runtime_error("PyTorch is not available");
    }

    // Import torch module for creating output tensor
    py::module torch = py::module::import("torch");

    // ---- Get input tensor properties ----
    // Note: We skip explicit dtype validation - CUDA will fail for wrong types
    py::object dtype = input.attr("dtype");
    py::object device = input.attr("device");

    // ---- Validate input dimensions ----
    py::tuple shape = input.attr("shape").cast<py::tuple>();
    if (py::len(shape) != 2) {
        throw std::runtime_error("input must be a 2D tensor");
    }

    int height = shape[0].cast<int>();
    int width = shape[1].cast<int>();

    // ---- Validate kernel dimensions ----
    py::tuple kshape = kernel.attr("shape").cast<py::tuple>();
    if (py::len(kshape) != 2) {
        throw std::runtime_error("kernel must be a 2D tensor");
    }
    if (kshape[0].cast<int>() != kshape[1].cast<int>()) {
        throw std::runtime_error("kernel must be square");
    }
    int ksize = kshape[0].cast<int>();
    if (ksize % 2 == 0) {
        throw std::runtime_error("kernel dimension must be odd");
    }

    // ---- Extract raw device pointers ----
    // data_ptr() returns a Python int representing the memory address.
    // We cast to uintptr_t first, then reinterpret to float* for CUDA.
    uintptr_t input_addr = input.attr("data_ptr")().cast<uintptr_t>();
    uintptr_t kernel_addr = kernel.attr("data_ptr")().cast<uintptr_t>();
    float* input_ptr = reinterpret_cast<float*>(input_addr);
    float* kernel_ptr = reinterpret_cast<float*>(kernel_addr);

    // ---- Create output tensor on same device ----
    // Use torch.empty() to allocate uninitialized output, matching input's dtype/device
    py::object input_dtype = input.attr("dtype");
    py::object input_device = input.attr("device");
    py::object output = torch.attr("empty")(
        py::make_tuple(height, width),
        py::arg("dtype") = input_dtype, py::arg("device") = input_device
    );

    // Extract output device pointer
    uintptr_t output_addr = output.attr("data_ptr")().cast<uintptr_t>();
    float* output_ptr = reinterpret_cast<float*>(output_addr);

    // ---- Launch CUDA kernel directly on device pointers ----
    // No memory copies needed - kernel operates directly on GPU memory
    launch_conv_naive(input_ptr, kernel_ptr, output_ptr, width, height, ksize);

    return output;
}

/**
 * \brief Tiled convolution for PyTorch tensors.
 *
 * Similar to conv_naive_torch but uses the tiled kernel with configurable
 * tile size for optimized memory access patterns.
 *
 * \param input  2D float32 torch.Tensor on CUDA device.
 * \param kernel 2D float32 torch.Tensor (kxk, must be odd-sized).
 * \param tile_w Tile width (8, 16, or 32).
 * \param tile_h Tile height (8, 16, or 32).
 * \return 2D float32 torch.Tensor on same device.
 *
 * \throws std::runtime_error If PyTorch is not available or dimensions invalid.
 */
py::object conv_tiled_torch(py::object input, py::object kernel, int tile_w, int tile_h) {
    // Check if PyTorch is available
    if (!is_torch_available()) {
        throw std::runtime_error("PyTorch is not available");
    }

    py::module torch = py::module::import("torch");

    // ---- Validate input dimensions ----
    py::tuple shape = input.attr("shape").cast<py::tuple>();
    if (py::len(shape) != 2) {
        throw std::runtime_error("input must be a 2D tensor");
    }

    int height = shape[0].cast<int>();
    int width = shape[1].cast<int>();

    // ---- Validate kernel dimensions ----
    py::tuple kshape = kernel.attr("shape").cast<py::tuple>();
    if (py::len(kshape) != 2) {
        throw std::runtime_error("kernel must be a 2D tensor");
    }
    int ksize = kshape[0].cast<int>();

    // ---- Extract raw device pointers ----
    uintptr_t input_addr = input.attr("data_ptr")().cast<uintptr_t>();
    uintptr_t kernel_addr = kernel.attr("data_ptr")().cast<uintptr_t>();
    float* input_ptr = reinterpret_cast<float*>(input_addr);
    float* kernel_ptr = reinterpret_cast<float*>(kernel_addr);

    // ---- Create output tensor on same device ----
    py::object input_dtype = input.attr("dtype");
    py::object input_device = input.attr("device");
    py::object output = torch.attr("empty")(
        py::make_tuple(height, width),
        py::arg("dtype") = input_dtype, py::arg("device") = input_device
    );

    uintptr_t output_addr = output.attr("data_ptr")().cast<uintptr_t>();
    float* output_ptr = reinterpret_cast<float*>(output_addr);

    // ---- Launch tiled CUDA kernel ----
    launch_conv_tiled_dispatch(input_ptr, kernel_ptr, output_ptr, width, height, ksize, tile_w, tile_h);

    return output;
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

/**
 * \brief pybind11 module definition for kernel_craft_python.
 *
 * This creates the Python module with version info and exposes all
 * convolution functions with proper docstrings and overload handling.
 */
PYBIND11_MODULE(kernel_craft_python, m) {
    // Module docstring
    m.doc() = "Python bindings for CUDA convolution kernels\n\n"
              "Supports numpy arrays and PyTorch tensors on CUDA device.\n"
              "See conv_naive() and conv_tiled() for usage.";

    // ---- Version ----
    // Expose __version__ attribute for programmatic access
    m.attr("__version__") = "0.1.1";

    // ---- conv_naive - numpy array overload ----
    m.def("conv_naive", &conv_naive_numpy,
          "Naive 2D convolution (numpy arrays)\n\n"
          "Performs convolution using a simple GPU kernel where each thread\n"
          "computes one output pixel by reading from global memory.\n\n"
          "Args:\n"
          "  input (numpy.ndarray): 2D float32 array (HxW), row-major\n"
          "  kernel (numpy.ndarray): 2D float32 array (kxk), must be odd-sized\n\n"
          "Returns:\n"
          "  numpy.ndarray: 2D float32 array (HxW) with convolution result",
          py::arg("input"), py::arg("kernel"));

    // ---- conv_tiled - numpy array overload ----
    m.def("conv_tiled", &conv_tiled_numpy,
          "Tiled 2D convolution (numpy arrays)\n\n"
          "Performs convolution using a tiled GPU kernel that loads image\n"
          "tiles into shared memory for better memory bandwidth utilization.\n\n"
          "Args:\n"
          "  input (numpy.ndarray): 2D float32 array (HxW), row-major\n"
          "  kernel (numpy.ndarray): 2D float32 array (kxk), must be odd-sized\n"
          "  tile_w (int): Tile width for shared memory (8, 16, or 32)\n"
          "  tile_h (int): Tile height for shared memory (8, 16, or 32)\n\n"
          "Returns:\n"
          "  numpy.ndarray: 2D float32 array (HxW) with convolution result",
          py::arg("input"), py::arg("kernel"), py::arg("tile_w") = 8, py::arg("tile_h") = 8);

    // ---- conv_naive - PyTorch tensor overload ----
    m.def("conv_naive", &conv_naive_torch,
          "Naive 2D convolution (PyTorch tensors)\n\n"
          "Performs convolution on PyTorch tensors already on CUDA device.\n"
          "Creates output tensor on same device.\n\n"
          "Args:\n"
          "  input (torch.Tensor): 2D float32 tensor on CUDA\n"
          "  kernel (torch.Tensor): 2D float32 tensor (kxk), must be odd-sized\n\n"
          "Returns:\n"
          "  torch.Tensor: 2D float32 tensor on same CUDA device",
          py::arg("input"), py::arg("kernel"));

    // ---- conv_tiled - PyTorch tensor overload ----
    m.def("conv_tiled", &conv_tiled_torch,
          "Tiled 2D convolution (PyTorch tensors)\n\n"
          "Performs tiled convolution on PyTorch tensors already on CUDA device.\n\n"
          "Args:\n"
          "  input (torch.Tensor): 2D float32 tensor on CUDA\n"
          "  kernel (torch.Tensor): 2D float32 tensor (kxk), must be odd-sized\n"
          "  tile_w (int): Tile width (8, 16, or 32)\n"
          "  tile_h (int): Tile height (8, 16, or 32)\n\n"
          "Returns:\n"
          "  torch.Tensor: 2D float32 tensor on same CUDA device",
          py::arg("input"), py::arg("kernel"), py::arg("tile_w") = 8, py::arg("tile_h") = 8);

    // ---- Error handling ----
    // Register translator to convert C++ exceptions to Python errors
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });
}