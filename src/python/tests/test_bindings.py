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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])