"""
kernel_craft.triton — Triton GPU kernels for ML inference (Phase 15).

Exports:
  flash_attention  — FlashAttention-2 style attention
  selective_scan   — Mamba-1 ZOH selective scan
  int4_gemv        — INT4-quantized GEMV with per-group dequant
"""
from .flash_attention_triton import flash_attention
from .selective_scan_triton import selective_scan
from .int4_gemv_triton import int4_gemv

__all__ = ["flash_attention", "selective_scan", "int4_gemv"]
