"""
kernel_craft_vllm_backend.py
vLLM v1 AttentionBackend implementation backed by kernel-craft CUDA kernels.

Compatible with vLLM ≥ 0.21.0 (v1 API: vllm.v1.attention.backend).

Activation
----------
Set the environment variable before launching vLLM::

    export VLLM_ATTENTION_BACKEND=kernel_craft

Or call ``register()`` programmatically before creating the vLLM engine::

    import kernel_craft_vllm_backend
    kernel_craft_vllm_backend.register()

    from vllm import LLM
    llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", quantization="awq")

Design
------
- **Prefill** uses ``launch_flash_attention`` (tiled online-softmax, O(N·d·H) memory).
- **Decode** uses ``launch_paged_attention`` (non-contiguous KV-cache, block table).
- Tensor layout mirrors FlashAttentionBackend: [2, num_blocks, block_size, H_kv, d].
- ``forward_includes_kv_cache_update = False`` — vLLM updates the KV cache via
  slot_mapping before calling forward.

vLLM version compatibility
---------------------------
Written against vLLM ≥ 0.21.0 (v1 API at vllm.v1.attention.backend).
Pin your vLLM version in pyproject.toml before deploying.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Tuple, Type

# ---------------------------------------------------------------------------
# Guard: only functional when vLLM is installed
# ---------------------------------------------------------------------------

try:
    import torch
    from vllm.v1.attention.backend import (
        AttentionBackend,
        AttentionImpl,
        AttentionImplBase,
        AttentionMetadata,
        AttentionMetadataBuilder,
        AttentionType,
        CommonAttentionMetadata,
    )
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

try:
    from kernel_craft_torch_ops import (
        flash_attention as _flash_attn,
        paged_attention as _paged_attn,
    )
    HAS_OPS = True
except Exception:
    HAS_OPS = False

_BACKEND_NAME = "kernel_craft"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

if HAS_VLLM:
    @dataclass
    class KernelCraftMetadata:
        """
        Per-layer attention metadata for kernel-craft backend.

        Built from ``CommonAttentionMetadata`` by ``KernelCraftMetadataBuilder``.

        Prefill fields
        ``query_start_loc`` — cumulative query token offsets [B+1].
        ``seq_lens`` — total KV context length per request [B] int32.
        ``max_query_len`` — maximum query length across requests.

        Decode fields
        ``block_table`` — [B, max_blocks] int32 physical block indices.
        ``slot_mapping`` — [num_tokens] int64 KV slot for each token.
        """

        num_actual_tokens: int
        max_query_len: int
        max_seq_len: int
        query_start_loc: "torch.Tensor"    # [B+1] int32
        seq_lens:         "torch.Tensor"   # [B] int32
        block_table:      "torch.Tensor"   # [B, max_blocks] int32
        slot_mapping:     "torch.Tensor"   # [num_tokens] int64
        causal: bool = True

    class KernelCraftMetadataBuilder(AttentionMetadataBuilder):
        """
        Builds ``KernelCraftMetadata`` from vLLM's ``CommonAttentionMetadata``.

        :param kv_cache_spec  Attention spec (block size, dtype).
        :param layer_names    Layer names managed by this builder.
        :param vllm_config    Global vLLM config.
        :param device         Target device.
        """

        def __init__(
            self,
            kv_cache_spec: Any,
            layer_names: list,
            vllm_config: Any,
            device: "torch.device",
        ) -> None:
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: "CommonAttentionMetadata",
            fast_build: bool = False,
        ) -> KernelCraftMetadata:
            cam = common_attn_metadata
            return KernelCraftMetadata(
                num_actual_tokens=cam.num_actual_tokens,
                max_query_len=cam.max_query_len,
                max_seq_len=cam.max_seq_len,
                query_start_loc=cam.query_start_loc,
                seq_lens=cam.seq_lens,
                block_table=cam.block_table_tensor,
                slot_mapping=cam.slot_mapping,
                causal=cam.causal,
            )


# ---------------------------------------------------------------------------
# Attention implementation
# ---------------------------------------------------------------------------

if HAS_VLLM:
    class KernelCraftAttentionImpl(AttentionImpl[KernelCraftMetadata]):
        """
        Attention implementation using kernel-craft FlashAttention + PagedAttention.

        Prefill path
        For each sequence: Q/K/V slices → ``launch_flash_attention`` (causal, GQA).

        Decode path
        Single-token queries + paged KV-cache → ``launch_paged_attention``.

        KV-cache update
        ``forward_includes_kv_cache_update = False``: vLLM writes K/V into the
        paged cache via slot_mapping before calling forward.  Our forward reads
        directly from ``kv_cache`` without an additional write.

        :param num_heads      Query head count.
        :param head_size      Head dimension.
        :param scale          Attention scale (1/√d); passed through to kernels.
        :param num_kv_heads   KV head count for GQA/MQA.
        :param alibi_slopes   ALiBi slopes — not supported; raises if set.
        :param sliding_window Sliding-window attention — not supported.
        :param kv_cache_dtype KV-cache dtype string (e.g. "auto", "float16").
        :param logits_soft_cap Logit capping — not used by current kernels.
        :param attn_type      Attention type string (DECODER / ENCODER etc.).
        :param kv_sharing_target_layer_name Cross-layer KV sharing — not used.
        """

        def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: Optional[int] = None,
            alibi_slopes: Optional[list] = None,
            sliding_window: Optional[int] = None,
            kv_cache_dtype: str = "auto",
            logits_soft_cap: Optional[float] = None,
            attn_type: str = AttentionType.DECODER,
            kv_sharing_target_layer_name: Optional[str] = None,
        ) -> None:
            self.num_heads    = num_heads
            self.head_size    = head_size
            self.scale        = float(scale)
            self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
            self.kv_cache_dtype = kv_cache_dtype

            if alibi_slopes is not None:
                raise NotImplementedError(
                    "kernel_craft backend does not support ALiBi slopes."
                )
            if sliding_window is not None:
                raise NotImplementedError(
                    "kernel_craft backend does not support sliding-window attention."
                )

        def forward(
            self,
            layer: "torch.nn.Module",
            query:  "torch.Tensor",
            key:    "torch.Tensor",
            value:  "torch.Tensor",
            kv_cache: "torch.Tensor",
            attn_metadata: "KernelCraftMetadata",
            output: "torch.Tensor",
            output_scale: Optional["torch.Tensor"] = None,
            output_block_scale: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            """
            Compute attention and write result into ``output``.

            :param layer         Parent attention layer (unused by our kernels).
            :param query         [num_tokens, H, d] float32.
            :param key           [num_tokens, H_kv, d] float32 (already cached by vLLM).
            :param value         [num_tokens, H_kv, d] float32 (already cached by vLLM).
            :param kv_cache      [2, num_blocks, block_size, H_kv, d] — paged KV pool.
            :param attn_metadata KernelCraftMetadata for this batch.
            :param output        [num_tokens, H * d] — pre-allocated output buffer.
            :return:              ``output`` filled with attention results.
            """
            if not HAS_OPS:
                raise RuntimeError(
                    "kernel_craft_torch_ops not available — "
                    "ensure libkernels.so is built (cd build && make kernels)."
                )

            if attn_metadata is None:
                return output.fill_(0)

            H, d = self.num_heads, self.head_size
            is_decode = (attn_metadata.max_query_len == 1)

            if is_decode:
                self._decode_forward(query, kv_cache, attn_metadata, output)
            else:
                self._prefill_forward(query, key, value, attn_metadata, output)

            return output

        def _decode_forward(
            self,
            query:    "torch.Tensor",
            kv_cache: "torch.Tensor",
            meta:     "KernelCraftMetadata",
            output:   "torch.Tensor",
        ) -> None:
            """
            Decode step: one query token per request.

            ``kv_cache`` layout: [2, num_blocks, block_size, H_kv, d].
            ``query`` shape:     [B, H, d] (B == num_actual_tokens for decode).
            ``output`` shape:    [B, H * d].
            """
            B = meta.seq_lens.shape[0]
            H, d = self.num_heads, self.head_size

            q = query[:B].contiguous().float()  # [B, H, d]
            K_pool = kv_cache[0].float().contiguous()  # [num_blocks, bs, H_kv, d]
            V_pool = kv_cache[1].float().contiguous()
            bt     = meta.block_table[:B].contiguous()
            sl     = meta.seq_lens[:B].to(torch.int32).contiguous()

            block_size = K_pool.shape[1]

            O = _paged_attn(
                q, bt, K_pool, V_pool, sl,
                H_kv=self.num_kv_heads,
                block_size=block_size,
            )  # [B, H, d]

            output[:B] = O.to(output.dtype).view(B, H * d)

        def _prefill_forward(
            self,
            query: "torch.Tensor",
            key:   "torch.Tensor",
            value: "torch.Tensor",
            meta:  "KernelCraftMetadata",
            output: "torch.Tensor",
        ) -> None:
            """
            Prefill step: full-sequence attention per request.

            Iterates over requests using ``query_start_loc`` offsets,
            slices each request's tokens, reshapes to [1, H, seq_len, d],
            runs flash_attention, and writes into output.
            """
            H, H_kv, d = self.num_heads, self.num_kv_heads, self.head_size
            qsl = meta.query_start_loc  # [B+1]
            B   = qsl.shape[0] - 1

            offset = 0
            for i in range(B):
                q_start = int(qsl[i].item())
                q_end   = int(qsl[i + 1].item())
                qlen    = q_end - q_start

                # Slice this request's tokens
                qi = query[q_start:q_end].float()  # [qlen, H, d]
                ki = key  [q_start:q_end].float()  # [qlen, H_kv, d]
                vi = value[q_start:q_end].float()

                # Reshape to [1, H, qlen, d] for our kernel
                qi = qi.permute(1, 0, 2).unsqueeze(0)  # [1, H, qlen, d]
                ki = ki.permute(1, 0, 2).unsqueeze(0)  # [1, H_kv, qlen, d]
                vi = vi.permute(1, 0, 2).unsqueeze(0)

                Oi = _flash_attn(qi, ki, vi, H_kv=H_kv, causal=meta.causal)
                # Oi: [1, H, qlen, d] → [qlen, H * d]
                Oi = Oi.squeeze(0).permute(1, 0, 2).reshape(qlen, H * d)
                output[q_start:q_end] = Oi.to(output.dtype)


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

if HAS_VLLM:
    class KernelCraftAttentionBackend(AttentionBackend):
        """
        vLLM v1 AttentionBackend routing to kernel-craft CUDA kernels.

        Activate via::

            VLLM_ATTENTION_BACKEND=kernel_craft  python serve.py ...

        KV-cache format
        Matches FlashAttentionBackend: [2, num_blocks, block_size, H_kv, d].
        Index 0 → K blocks; index 1 → V blocks.

        Limitations (current)
        - FP32 attention computation (fp16 inputs cast before calling kernels)
        - No ALiBi or sliding-window attention
        - Single-GPU only (tensor parallelism via torch.distributed)
        - block_size must be a multiple of 16 (inherited from flash_attention tiling)
        """

        # Match FlashAttentionBackend: KV cache update happens via slot_mapping
        forward_includes_kv_cache_update: bool = False

        supported_dtypes: ClassVar[list] = [torch.float16, torch.bfloat16]

        @staticmethod
        def get_name() -> str:
            return _BACKEND_NAME

        @staticmethod
        def get_impl_cls() -> Type[KernelCraftAttentionImpl]:
            return KernelCraftAttentionImpl

        @staticmethod
        def get_builder_cls() -> Type[KernelCraftMetadataBuilder]:
            return KernelCraftMetadataBuilder

        @staticmethod
        def get_kv_cache_shape(
            num_blocks: int,
            block_size: int,
            num_kv_heads: int,
            head_size: int,
            cache_dtype_str: str = "auto",
        ) -> Tuple[int, ...]:
            # Mirror FlashAttentionBackend layout: [2, num_blocks, bs, H_kv, d]
            return (2, num_blocks, block_size, num_kv_heads, head_size)


# ---------------------------------------------------------------------------
# Registration hook
# ---------------------------------------------------------------------------

def register() -> None:
    """
    Register ``KernelCraftAttentionBackend`` with vLLM's backend registry.

    Call this before constructing the vLLM ``LLM`` or ``AsyncLLMEngine``.
    Alternatively, set ``VLLM_ATTENTION_BACKEND=kernel_craft`` before import.

    Raises ``ImportError`` if vLLM is not installed.
    """
    if not HAS_VLLM:
        raise ImportError(
            "vLLM is not installed. Install with: pip install vllm\n"
            "Note: requires CUDA ≥ 12.1 and PyTorch ≥ 2.1."
        )

    try:
        # vLLM 0.21.x uses a registry dict or env-var lookup
        import vllm.v1.attention.backends.registry as _reg
        if hasattr(_reg, "ATTENTION_BACKENDS"):
            _reg.ATTENTION_BACKENDS[_BACKEND_NAME] = KernelCraftAttentionBackend
        else:
            os.environ["VLLM_ATTENTION_BACKEND"] = _BACKEND_NAME
    except (ImportError, AttributeError):
        os.environ["VLLM_ATTENTION_BACKEND"] = _BACKEND_NAME


# Auto-register when env var is set
if os.environ.get("VLLM_ATTENTION_BACKEND") == _BACKEND_NAME:
    try:
        register()
    except Exception:
        pass
