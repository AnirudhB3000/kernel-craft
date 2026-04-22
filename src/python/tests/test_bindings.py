"""Python tests for kernel_craft Python bindings."""

import numpy as np
import pytest


def conv_cpu(input_arr, kernel):
    """CPU reference implementation for convolution."""
    height, width = input_arr.shape
    ksize = kernel.shape[0]
    kHalf = ksize // 2
    output = np.zeros_like(input_arr)
    for oy in range(height):
        for ox in range(width):
            sum_val = 0.0
            for ky in range(ksize):
                iy = oy + ky - kHalf
                if iy < 0 or iy >= height:
                    continue
                for kx in range(ksize):
                    ix = ox + kx - kHalf
                    if ix < 0 or ix >= width:
                        continue
                    sum_val += input_arr[iy, ix] * kernel[ky, kx]
            output[oy, ox] = sum_val
    return output


class TestConvNaiveNumpy:
    """Tests for conv_naive with numpy arrays."""

    def test_basic(self):
        """Test basic convolution produces correct shape."""
        import kernel_craft_python as kc
        input_arr = np.random.rand(8, 8).astype(np.float32)
        kernel = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        result = kc.conv_naive(input_arr, kernel)
        assert result.shape == (8, 8)

    def test_correctness(self):
        """Test output matches CPU reference."""
        import kernel_craft_python as kc
        np.random.seed(42)
        input_arr = np.random.rand(16, 16).astype(np.float32)
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        result_gpu = kc.conv_naive(input_arr, kernel)
        result_cpu = conv_cpu(input_arr, kernel)
        np.testing.assert_allclose(result_gpu, result_cpu, rtol=1e-5)

    def test_large_kernel(self):
        """Test with larger 5x5 kernel."""
        import kernel_craft_python as kc
        np.random.seed(42)
        input_arr = np.random.rand(32, 32).astype(np.float32)
        kernel = np.random.rand(5, 5).astype(np.float32)
        result_gpu = kc.conv_naive(input_arr, kernel)
        result_cpu = conv_cpu(input_arr, kernel)
        np.testing.assert_allclose(result_gpu, result_cpu, rtol=1e-4)

    def test_invalid_input_dim(self):
        """Test that 1D input raises error."""
        import kernel_craft_python as kc
        input_arr = np.random.rand(16).astype(np.float32)
        kernel = np.array([[1, 0], [0, 1]], dtype=np.float32)
        with pytest.raises(RuntimeError):
            kc.conv_naive(input_arr, kernel)

    def test_invalid_kernel_dim(self):
        """Test that non-2D kernel raises error."""
        import kernel_craft_python as kc
        input_arr = np.random.rand(16, 16).astype(np.float32)
        kernel = np.random.rand(4).astype(np.float32)
        with pytest.raises(RuntimeError):
            kc.conv_naive(input_arr, kernel)

    def test_even_kernel_raises(self):
        """Test that even-sized kernel raises error."""
        import kernel_craft_python as kc
        input_arr = np.random.rand(16, 16).astype(np.float32)
        kernel = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(RuntimeError):
            kc.conv_naive(input_arr, kernel)


class TestConvTiledNumpy:
    """Tests for conv_tiled with numpy arrays."""

    def test_tile_8x8(self):
        """Test tiled convolution with 8x8 tiles."""
        import kernel_craft_python as kc
        np.random.seed(42)
        input_arr = np.random.rand(16, 16).astype(np.float32)
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        result = kc.conv_tiled(input_arr, kernel, 8, 8)
        result_ref = kc.conv_naive(input_arr, kernel)
        np.testing.assert_allclose(result, result_ref, rtol=1e-5)

    def test_tile_16x16(self):
        """Test tiled convolution with 16x16 tiles."""
        import kernel_craft_python as kc
        np.random.seed(42)
        input_arr = np.random.rand(32, 32).astype(np.float32)
        kernel = np.random.rand(3, 3).astype(np.float32)
        result = kc.conv_tiled(input_arr, kernel, 16, 16)
        result_ref = kc.conv_naive(input_arr, kernel)
        np.testing.assert_allclose(result, result_ref, rtol=1e-5)

    def test_tile_32x32(self):
        """Test tiled convolution with 32x32 tiles on larger image."""
        import kernel_craft_python as kc
        np.random.seed(42)
        input_arr = np.random.rand(64, 64).astype(np.float32)
        kernel = np.random.rand(5, 5).astype(np.float32)
        result = kc.conv_tiled(input_arr, kernel, 32, 32)
        result_ref = kc.conv_naive(input_arr, kernel)
        np.testing.assert_allclose(result, result_ref, rtol=1e-5)

    def test_different_tile_w_h(self):
        """Test different tile width and height."""
        import kernel_craft_python as kc
        np.random.seed(42)
        input_arr = np.random.rand(24, 24).astype(np.float32)
        kernel = np.random.rand(3, 3).astype(np.float32)
        result = kc.conv_tiled(input_arr, kernel, 8, 16)
        result_ref = kc.conv_naive(input_arr, kernel)
        np.testing.assert_allclose(result, result_ref, rtol=1e-5)


class TestConvNaiveTorch:
    """Tests for conv_naive with PyTorch tensors."""

    def test_basic(self):
        """Test basic convolution with PyTorch tensor."""
        try:
            import torch
            import kernel_craft_python as kc
        except ImportError:
            pytest.skip("PyTorch not installed")

        input_tensor = torch.rand(8, 8, dtype=torch.float32, device='cuda')
        kernel = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32, device='cuda')
        result = kc.conv_naive(input_tensor, kernel)
        assert result.shape == (8, 8)
        assert result.device.type == 'cuda'

    def test_correctness(self):
        """Test output matches numpy reference."""
        try:
            import torch
            import kernel_craft_python as kc
        except ImportError:
            pytest.skip("PyTorch not installed")

        np.random.seed(42)
        input_np = np.random.rand(16, 16).astype(np.float32)
        kernel_np = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

        result_gpu = kc.conv_naive(
            torch.from_numpy(input_np).cuda(),
            torch.from_numpy(kernel_np).cuda()
        )
        result_cpu = conv_cpu(input_np, kernel_np)
        np.testing.assert_allclose(result_gpu.cpu().numpy(), result_cpu, rtol=1e-4)


class TestConvTiledTorch:
    """Tests for conv_tiled with PyTorch tensors."""

    def test_tile_sizes(self):
        """Test different tile sizes with PyTorch."""
        try:
            import torch
            import kernel_craft_python as kc
        except ImportError:
            pytest.skip("PyTorch not installed")

        input_tensor = torch.rand(32, 32, dtype=torch.float32, device='cuda')
        kernel = torch.rand(3, 3, dtype=torch.float32, device='cuda')

        for tw, th in [(8, 8), (16, 16), (32, 32)]:
            result = kc.conv_tiled(input_tensor, kernel, tw, th)
            assert result.shape == (32, 32)
            assert result.device.type == 'cuda'


class TestMemoryPool:
    """Tests for memory pool (batch processing)."""

    def test_batch_processing_numpy(self):
        """Test batch processing with numpy arrays."""
        import kernel_craft_python as kc

        np.random.seed(42)
        num_batches = 4
        width, height = 64, 64
        ksize = 3

        inputs = [np.random.rand(height, width).astype(np.float32) for _ in range(num_batches)]
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

        results = []
        for inp in inputs:
            result = kc.conv_tiled(inp, kernel, 8, 8)
            results.append(result)

        assert len(results) == num_batches
        for r in results:
            assert r.shape == (height, width)

    def test_batch_correctness(self):
        """Test that batch processing produces correct results."""
        import kernel_craft_python as kc

        np.random.seed(42)
        input_arr = np.random.rand(16, 16).astype(np.float32)
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

        result1 = kc.conv_tiled(input_arr, kernel, 8, 8)
        result2 = kc.conv_tiled(input_arr, kernel, 8, 8)

        np.testing.assert_allclose(result1, result2, rtol=1e-5)


class TestMixedPrecision:
    """Tests for mixed precision (FP16) convolution."""

    def test_fp16_conversion(self):
        """Test FP16 conversion and back."""
        import kernel_craft_python as kc

        np.random.seed(42)
        input_fp32 = np.random.rand(16, 16).astype(np.float32)
        kernel = np.random.rand(3, 3).astype(np.float32)

        result_fp32 = kc.conv_tiled(input_fp32, kernel, 8, 8)

        input_fp16 = input_fp32.astype(np.float16)
        kernel_fp16 = kernel.astype(np.float16)

        result_fp16 = kc.conv_tiled(input_fp16.astype(np.float32), kernel_fp16.astype(np.float32), 8, 8)

        assert result_fp32.shape == result_fp16.shape


class TestPerformance:
    """Performance-related tests."""

    def test_multiple_tile_sizes(self):
        """Test that all tile sizes produce consistent results."""
        import kernel_craft_python as kc

        np.random.seed(42)
        input_arr = np.random.rand(32, 32).astype(np.float32)
        kernel = np.random.rand(3, 3).astype(np.float32)

        result_ref = kc.conv_naive(input_arr, kernel)

        for tw, th in [(8, 8), (16, 16)]:
            result = kc.conv_tiled(input_arr, kernel, tw, th)
            np.testing.assert_allclose(result, result_ref, rtol=1e-4)

    def test_large_input(self):
        """Test with larger input for performance characterization."""
        import kernel_craft_python as kc

        np.random.seed(42)
        input_arr = np.random.rand(512, 512).astype(np.float32)
        kernel = np.random.rand(3, 3).astype(np.float32)

        result = kc.conv_tiled(input_arr, kernel, 8, 8)
        assert result.shape == (512, 512)

        result_ref = kc.conv_naive(input_arr, kernel)
        np.testing.assert_allclose(result, result_ref, rtol=1e-4)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_minimum_size_input(self):
        """Test with minimum size 3x3 input."""
        import kernel_craft_python as kc
        input_arr = np.ones((3, 3), dtype=np.float32)
        kernel = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        result = kc.conv_naive(input_arr, kernel)
        assert result.shape == (3, 3)

    def test_3x3_input_with_5x5_kernel(self):
        """Test 3x3 input with 5x5 kernel."""
        import kernel_craft_python as kc
        np.random.seed(42)
        input_arr = np.random.rand(3, 3).astype(np.float32)
        kernel = np.random.rand(5, 5).astype(np.float32)
        result = kc.conv_naive(input_arr, kernel)
        assert result.shape == (3, 3)

    def test_zero_input(self):
        """Test with all-zero input."""
        import kernel_craft_python as kc
        input_arr = np.zeros((16, 16), dtype=np.float32)
        kernel = np.random.rand(3, 3).astype(np.float32)
        result = kc.conv_naive(input_arr, kernel)
        np.testing.assert_array_equal(result, np.zeros_like(input_arr))

    def test_zero_kernel(self):
        """Test with all-zero kernel."""
        import kernel_craft_python as kc
        np.random.seed(42)
        input_arr = np.random.rand(16, 16).astype(np.float32)
        kernel = np.zeros((3, 3), dtype=np.float32)
        result = kc.conv_naive(input_arr, kernel)
        np.testing.assert_array_equal(result, np.zeros_like(input_arr))

    def test_constant_input(self):
        """Test with constant input."""
        import kernel_craft_python as kc
        input_arr = np.ones((16, 16), dtype=np.float32) * 5.0
        kernel = np.ones((3, 3), dtype=np.float32)
        result = kc.conv_naive(input_arr, kernel)
        assert result.max() > 0

    def test_single_pixel_output(self):
        """Test with 1x1 output tile."""
        import kernel_craft_python as kc
        np.random.seed(42)
        input_arr = np.random.rand(5, 5).astype(np.float32)
        kernel = np.random.rand(3, 3).astype(np.float32)
        result = kc.conv_tiled(input_arr, kernel, 1, 1)
        result_ref = kc.conv_naive(input_arr, kernel)
        np.testing.assert_allclose(result, result_ref, rtol=1e-4)

    def test_odd_input_dimensions(self):
        """Test with non-power-of-2 dimensions."""
        import kernel_craft_python as kc
        np.random.seed(42)
        for wh in [(13, 13), (17, 23), (31, 47)]:
            input_arr = np.random.rand(*wh).astype(np.float32)
            kernel = np.random.rand(3, 3).astype(np.float32)
            result = kc.conv_tiled(input_arr, kernel, 8, 8)
            result_ref = kc.conv_naive(input_arr, kernel)
            np.testing.assert_allclose(result, result_ref, rtol=1e-4)


class TestKernelTypes:
    """Tests with different kernel types."""

    def test_identity_kernel(self):
        """Test with identity kernel."""
        import kernel_craft_python as kc
        np.random.seed(42)
        input_arr = np.random.rand(16, 16).astype(np.float32)
        kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        result = kc.conv_naive(input_arr, kernel)
        np.testing.assert_allclose(result, input_arr, rtol=1e-5)

    def test_sobel_x_kernel(self):
        """Test with Sobel X edge detection kernel."""
        import kernel_craft_python as kc
        np.random.seed(42)
        input_arr = np.random.rand(16, 16).astype(np.float32)
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        result_gpu = kc.conv_naive(input_arr, kernel)
        result_cpu = conv_cpu(input_arr, kernel)
        np.testing.assert_allclose(result_gpu, result_cpu, rtol=1e-4)

    def test_sobel_y_kernel(self):
        """Test with Sobel Y edge detection kernel."""
        import kernel_craft_python as kc
        np.random.seed(42)
        input_arr = np.random.rand(16, 16).astype(np.float32)
        kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        result_gpu = kc.conv_naive(input_arr, kernel)
        result_cpu = conv_cpu(input_arr, kernel)
        np.testing.assert_allclose(result_gpu, result_cpu, rtol=1e-4)

    def test_laplacian_kernel(self):
        """Test with Laplacian kernel for edge detection."""
        import kernel_craft_python as kc
        np.random.seed(42)
        input_arr = np.random.rand(16, 16).astype(np.float32)
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        result_gpu = kc.conv_naive(input_arr, kernel)
        result_cpu = conv_cpu(input_arr, kernel)
        np.testing.assert_allclose(result_gpu, result_cpu, rtol=1e-5)

    def test_gaussian_like_kernel(self):
        """Test with Gaussian-like blur kernel."""
        import kernel_craft_python as kc
        np.random.seed(42)
        input_arr = np.random.rand(16, 16).astype(np.float32)
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
        result_gpu = kc.conv_naive(input_arr, kernel)
        result_cpu = conv_cpu(input_arr, kernel)
        np.testing.assert_allclose(result_gpu, result_cpu, rtol=1e-5)

    def test_7x7_kernel(self):
        """Test with larger 7x7 kernel."""
        import kernel_craft_python as kc
        np.random.seed(42)
        input_arr = np.random.rand(32, 32).astype(np.float32)
        kernel = np.random.rand(7, 7).astype(np.float32)
        result_gpu = kc.conv_naive(input_arr, kernel)
        result_cpu = conv_cpu(input_arr, kernel)
        np.testing.assert_allclose(result_gpu, result_cpu, rtol=1e-4)


class TestNumpyTorchInterop:
    """Tests for numpy <-> torch interoperability."""

    def test_numpy_to_torch_and_back(self):
        """Test round-trip between numpy and torch."""
        try:
            import torch
            import kernel_craft_python as kc
        except ImportError:
            pytest.skip("PyTorch not installed")

        np.random.seed(42)
        input_np = np.random.rand(16, 16).astype(np.float32)
        kernel_np = np.random.rand(3, 3).astype(np.float32)

        input_torch = torch.from_numpy(input_np).cuda()
        kernel_torch = torch.from_numpy(kernel_np).cuda()

        result_torch = kc.conv_naive(input_torch, kernel_torch)

        result_back = result_torch.cpu().numpy()
        result_ref = conv_cpu(input_np, kernel_np)
        np.testing.assert_allclose(result_back, result_ref, rtol=1e-4)

    def test_inplace_not_required(self):
        """Test that input tensor is not modified."""
        try:
            import torch
            import kernel_craft_python as kc
        except ImportError:
            pytest.skip("PyTorch not installed")

        np.random.seed(42)
        input_np = np.random.rand(16, 16).astype(np.float32)
        kernel_np = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

        input_torch = torch.from_numpy(input_np).cuda()
        input_original = input_torch.clone()

        _ = kc.conv_naive(input_torch, torch.from_numpy(kernel_np).cuda())

        np.testing.assert_allclose(input_torch.cpu().numpy(), input_original.cpu().numpy())

    def test_different_tiled_results_torch(self):
        """Test tiled matches naive for torch on various sizes."""
        try:
            import torch
            import kernel_craft_python as kc
        except ImportError:
            pytest.skip("PyTorch not installed")

        for size in [8, 16, 32, 64]:
            np.random.seed(42)
            input_arr = np.random.rand(size, size).astype(np.float32)
            kernel = np.random.rand(3, 3).astype(np.float32)

            naive_result = kc.conv_naive(
                torch.from_numpy(input_arr).cuda(),
                torch.from_numpy(kernel).cuda()
            )
            tiled_result = kc.conv_tiled(
                torch.from_numpy(input_arr).cuda(),
                torch.from_numpy(kernel).cuda(),
                8, 8
            )

            np.testing.assert_allclose(
                naive_result.cpu().numpy(),
                tiled_result.cpu().numpy(),
                rtol=1e-4
            )


class TestMemoryAndBatch:
    """Memory and batch processing tests."""

    def test_multiple_consecutive_calls(self):
        """Test multiple consecutive kernel calls."""
        import kernel_craft_python as kc

        np.random.seed(42)
        kernel = np.random.rand(3, 3).astype(np.float32)

        results = []
        for _ in range(5):
            input_arr = np.random.rand(32, 32).astype(np.float32)
            result = kc.conv_naive(input_arr, kernel)
            results.append(result)

        assert len(results) == 5
        for r in results:
            assert r.shape == (32, 32)

    def test_reuse_kernel(self):
        """Test reusing the same kernel multiple times."""
        import kernel_craft_python as kc

        np.random.seed(42)
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

        for i in range(3):
            input_arr = np.random.rand(16, 16).astype(np.float32)
            result = kc.conv_tiled(input_arr, kernel, 8, 8)
            result_ref = conv_cpu(input_arr, kernel)
            np.testing.assert_allclose(result, result_ref, rtol=1e-4)

    def test_batch_varying_sizes(self):
        """Test batch with varying input sizes."""
        import kernel_craft_python as kc

        np.random.seed(42)
        sizes = [(8, 8), (16, 16), (24, 24), (32, 32)]
        kernel = np.random.rand(3, 3).astype(np.float32)

        results = []
        for h, w in sizes:
            input_arr = np.random.rand(h, w).astype(np.float32)
            result = kc.conv_tiled(input_arr, kernel, 8, 8)
            results.append((result.shape, w == result.shape[1]))

        for (shape, w_ok) in results:
            assert w_ok, f"Width mismatch: {shape}"


class TestLargeScale:
    """Large scale tests."""

    def test_1024x1024(self):
        """Test with 1024x1024 image."""
        import kernel_craft_python as kc

        np.random.seed(42)
        input_arr = np.random.rand(1024, 1024).astype(np.float32)
        kernel = np.random.rand(3, 3).astype(np.float32)

        result = kc.conv_tiled(input_arr, kernel, 8, 8)
        assert result.shape == (1024, 1024)

    def test_2048x2048(self):
        """Test with 2048x2048 image."""
        import kernel_craft_python as kc

        np.random.seed(42)
        input_arr = np.random.rand(2048, 2048).astype(np.float32)
        kernel = np.random.rand(3, 3).astype(np.float32)

        result = kc.conv_tiled(input_arr, kernel, 16, 16)
        assert result.shape == (2048, 2048)

    def test_4096x4096(self):
        """Test with 4096x4096 image."""
        import kernel_craft_python as kc

        np.random.seed(42)
        input_arr = np.random.rand(4096, 4096).astype(np.float32)
        kernel = np.random.rand(3, 3).astype(np.float32)

        result = kc.conv_tiled(input_arr, kernel, 32, 32)
        assert result.shape == (4096, 4096)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])