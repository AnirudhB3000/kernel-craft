"""
kernel_craft_triton — Triton GPU kernels for ML inference (Phase 15).

Install: pip install kernel-craft[triton]

Exports:
  flash_attention  — FlashAttention-2 style tiled online-softmax attention (GQA/causal)
  selective_scan   — Mamba-1 ZOH selective scan
  int4_gemv        — INT4-quantized GEMV with per-group scale/zero-point dequantization
"""
from .flash_attention_triton import flash_attention
from .selective_scan_triton import selective_scan
from .int4_gemv_triton import int4_gemv

__all__ = ["flash_attention", "selective_scan", "int4_gemv"]
