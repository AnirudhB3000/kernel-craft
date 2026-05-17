"""
kernel_craft_torch_ops.py
Exposes kernel-craft CUDA kernels as PyTorch-compatible operators via ctypes.

Calling convention
------------------
All functions accept and return ``torch.Tensor`` objects on the *same* CUDA
device.  Internally the module loads ``libkernels.so`` (the shared library
produced by the CMake build) through ctypes and calls the ``extern "C"``
launcher functions directly — no GPU→CPU roundtrip, no recompilation needed.

If torch is available the module also registers each op under the
``kernel_craft`` namespace via ``torch.library`` so callers can use
``torch.ops.kernel_craft.<name>(...)``.

Usage
-----
::

    import kernel_craft_torch_ops as kc_ops

    # FlashAttention prefill
    O = kc_ops.flash_attention(Q, K, V, H_kv=4, causal=True)

    # PagedAttention decode
    O = kc_ops.paged_attention(Q, block_table, K_pool, V_pool, seq_lens,
                               H_kv=4, block_size=16)

    # INT4 weight GEMV (on-the-fly dequant)
    y = kc_ops.int4_gemv(packed, scales, zeros, x, rows, cols, group_size=128)

    # FP8 quantization / dequantization
    q, scales = kc_ops.fp8_quantize(x, per_token=True)
    x_hat     = kc_ops.fp8_dequantize(q, scales, rows, cols, per_token=True)

    # Speculative decoding token verification
    accepted, corrected = kc_ops.speculative_verify(
        draft_probs, target_probs, draft_tokens, rand_vals, rand_vals2)

Building libkernels.so
----------------------
::

    mkdir -p build && cd build
    cmake .. && make kernels -j$(nproc)
    # Produces build/libkernels.so

Environment
-----------
Set ``KERNEL_CRAFT_LIB`` to override the default library search path.
"""

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------

def _find_lib() -> str:
    """Return path to libkernels.so, searching standard locations."""
    candidates = []
    if "KERNEL_CRAFT_LIB" in os.environ:
        candidates.append(os.environ["KERNEL_CRAFT_LIB"])

    # Relative to this file: ../../build/libkernels.so
    here = Path(__file__).resolve().parent
    candidates += [
        here.parent.parent / "build" / "libkernels.so",
        here.parent.parent / "libkernels.so",
        Path("/usr/local/lib/kernel_craft/libkernels.so"),
    ]
    for p in candidates:
        if Path(p).exists():
            return str(p)
    raise FileNotFoundError(
        "libkernels.so not found. Build with: cd build && make kernels\n"
        f"Searched: {[str(c) for c in candidates]}"
    )


_LIB: Optional[ctypes.CDLL] = None


def _get_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(_find_lib(), mode=ctypes.RTLD_GLOBAL)
        _setup_prototypes(_LIB)
    return _LIB


def _setup_prototypes(lib: ctypes.CDLL) -> None:
    """Set ctypes argtypes/restype for every kernel launcher."""
    vp = ctypes.c_void_p

    # launch_flash_attention(Q, K, V, O, B, H, H_kv, N, d, causal, stream)
    lib.launch_flash_attention.argtypes = [
        vp, vp, vp, vp,                    # Q, K, V, O  (float*)
        ctypes.c_int, ctypes.c_int,         # B, H
        ctypes.c_int, ctypes.c_int,         # H_kv, N
        ctypes.c_int,                       # d
        ctypes.c_bool,                      # causal
        vp,                                 # cudaStream_t
    ]
    lib.launch_flash_attention.restype = None

    # launch_paged_attention(Q, block_table, K_pool, V_pool, O, seq_lens,
    #                        B, H, H_kv, d, block_size, max_blocks, stream)
    lib.launch_paged_attention.argtypes = [
        vp, vp, vp, vp, vp, vp,            # Q, block_table, K_pool, V_pool, O, seq_lens
        ctypes.c_int, ctypes.c_int,         # B, H
        ctypes.c_int, ctypes.c_int,         # H_kv, d
        ctypes.c_int, ctypes.c_int,         # block_size, max_blocks
        vp,                                 # stream
    ]
    lib.launch_paged_attention.restype = None

    # launch_dequantize_int4(packed, scales, zeros, output, rows, cols, gs, stream)
    lib.launch_dequantize_int4.argtypes = [
        vp, vp, vp, vp,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        vp,
    ]
    lib.launch_dequantize_int4.restype = None

    # launch_gemv_int4(packed, scales, zeros, x, y, rows, cols, gs, stream)
    lib.launch_gemv_int4.argtypes = [
        vp, vp, vp, vp, vp,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        vp,
    ]
    lib.launch_gemv_int4.restype = None

    # launch_quantize_fp8(input, output, scales, rows, cols, per_token, stream)
    lib.launch_quantize_fp8.argtypes = [
        vp, vp, vp,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_bool,
        vp,
    ]
    lib.launch_quantize_fp8.restype = None

    # launch_dequantize_fp8(input, scales, output, rows, cols, per_token, stream)
    lib.launch_dequantize_fp8.argtypes = [
        vp, vp, vp,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_bool,
        vp,
    ]
    lib.launch_dequantize_fp8.restype = None

    # launch_verify_draft_tokens(draft_probs, target_probs, draft_tokens,
    #   rand_vals, rand_vals2, accepted, corrected, num_tokens, vocab_size, stream)
    lib.launch_verify_draft_tokens.argtypes = [
        vp, vp, vp, vp, vp, vp, vp,
        ctypes.c_int, ctypes.c_int,
        vp,
    ]
    lib.launch_verify_draft_tokens.restype = None


# ---------------------------------------------------------------------------
# Helper: get current CUDA stream pointer
# ---------------------------------------------------------------------------

def _stream() -> ctypes.c_void_p:
    """Return the current CUDA stream as a ctypes void pointer."""
    try:
        import torch
        return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    except Exception:
        return ctypes.c_void_p(0)  # default stream


def _ptr(t) -> ctypes.c_void_p:
    """Return data_ptr of a torch tensor as ctypes void pointer."""
    return ctypes.c_void_p(t.data_ptr())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def flash_attention(
    Q,
    K,
    V,
    H_kv: int,
    causal: bool = False,
):
    """
    FlashAttention prefill: O = attention(Q, K, V).

    :param Q      torch.Tensor [B, H, N, d] float32 CUDA
    :param K      torch.Tensor [B, H_kv, N, d] float32 CUDA
    :param V      torch.Tensor [B, H_kv, N, d] float32 CUDA
    :param H_kv   Number of KV heads (H_kv ≤ H; MHA: H_kv=H, MQA: H_kv=1)
    :param causal Apply causal mask
    :return:       torch.Tensor [B, H, N, d] float32 CUDA
    """
    import torch
    Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
    B, H, N, d = Q.shape
    O = torch.zeros(B, H, N, d, dtype=torch.float32, device=Q.device)
    _get_lib().launch_flash_attention(
        _ptr(Q), _ptr(K), _ptr(V), _ptr(O),
        B, H, H_kv, N, d, causal, _stream(),
    )
    return O


def paged_attention(
    Q,
    block_table,
    K_pool,
    V_pool,
    seq_lens,
    H_kv: int,
    block_size: int,
):
    """
    PagedAttention decode: O = attention(Q, paged KV-cache).

    :param Q           torch.Tensor [B, H, d] float32 CUDA — single query token
    :param block_table torch.Tensor [B, max_blocks] int32 CUDA
    :param K_pool      torch.Tensor [num_phys, block_size, H_kv, d] float32 CUDA
    :param V_pool      torch.Tensor [num_phys, block_size, H_kv, d] float32 CUDA
    :param seq_lens    torch.Tensor [B] int32 CUDA
    :param H_kv        Number of KV heads
    :param block_size  Tokens per physical page
    :return:            torch.Tensor [B, H, d] float32 CUDA
    """
    import torch
    Q         = Q.contiguous()
    block_table = block_table.contiguous()
    K_pool    = K_pool.contiguous()
    V_pool    = V_pool.contiguous()
    seq_lens  = seq_lens.contiguous()

    B, H, d   = Q.shape
    max_blocks = block_table.shape[1]
    O = torch.zeros(B, H, d, dtype=torch.float32, device=Q.device)

    _get_lib().launch_paged_attention(
        _ptr(Q), _ptr(block_table), _ptr(K_pool), _ptr(V_pool),
        _ptr(O), _ptr(seq_lens),
        B, H, H_kv, d, block_size, max_blocks, _stream(),
    )
    return O


def int4_dequantize(
    packed,
    scales,
    zeros,
    rows: int,
    cols: int,
    group_size: int = 128,
):
    """
    Dequantize INT4 packed weights to FP32.

    :param packed     torch.Tensor [rows, cols/2] uint8 CUDA
    :param scales     torch.Tensor [rows, cols/group_size] float32 CUDA
    :param zeros      torch.Tensor [rows, cols/group_size/2] uint8 CUDA
    :param rows       Weight matrix rows
    :param cols       Weight matrix columns (even)
    :param group_size Elements per quantization group
    :return:           torch.Tensor [rows, cols] float32 CUDA
    """
    import torch
    packed = packed.contiguous()
    scales = scales.contiguous()
    zeros  = zeros.contiguous()
    output = torch.zeros(rows, cols, dtype=torch.float32, device=packed.device)
    _get_lib().launch_dequantize_int4(
        _ptr(packed), _ptr(scales), _ptr(zeros), _ptr(output),
        rows, cols, group_size, _stream(),
    )
    return output


def int4_gemv(
    packed,
    scales,
    zeros,
    x,
    rows: int,
    cols: int,
    group_size: int = 128,
):
    """
    INT4 weight GEMV: y = W_int4 * x (dequantize on the fly).

    :param packed     torch.Tensor [rows, cols/2] uint8 CUDA
    :param scales     torch.Tensor [rows, cols/group_size] float32 CUDA
    :param zeros      torch.Tensor [rows, cols/group_size/2] uint8 CUDA
    :param x          torch.Tensor [cols] float32 CUDA — input vector
    :param rows       Weight rows
    :param cols       Weight cols (even)
    :param group_size Elements per quantization group
    :return:           torch.Tensor [rows] float32 CUDA
    """
    import torch
    packed = packed.contiguous()
    scales = scales.contiguous()
    zeros  = zeros.contiguous()
    x      = x.contiguous()
    y = torch.zeros(rows, dtype=torch.float32, device=packed.device)
    _get_lib().launch_gemv_int4(
        _ptr(packed), _ptr(scales), _ptr(zeros), _ptr(x), _ptr(y),
        rows, cols, group_size, _stream(),
    )
    return y


def fp8_quantize(
    x,
    per_token: bool = True,
) -> Tuple:
    """
    Quantize FP32 activations to FP8 E4M3.

    :param x         torch.Tensor [rows, cols] float32 CUDA
    :param per_token Scale per row (True) or single global scale (False)
    :return:          (quantized: int8 [rows, cols], scales: float32 [rows] or [1])
    """
    import torch
    x = x.contiguous()
    rows, cols = x.shape
    q      = torch.zeros(rows, cols, dtype=torch.int8, device=x.device)
    n_scales = rows if per_token else 1
    scales = torch.zeros(n_scales, dtype=torch.float32, device=x.device)
    _get_lib().launch_quantize_fp8(
        _ptr(x), _ptr(q), _ptr(scales),
        rows, cols, per_token, _stream(),
    )
    return q, scales


def fp8_dequantize(
    q,
    scales,
    rows: int,
    cols: int,
    per_token: bool = True,
):
    """
    Dequantize FP8 E4M3 activations back to FP32.

    :param q         torch.Tensor [rows, cols] int8 CUDA
    :param scales    torch.Tensor [rows] or [1] float32 CUDA
    :param rows      Tensor rows
    :param cols      Tensor cols
    :param per_token Per-token (True) or per-tensor (False) scaling
    :return:          torch.Tensor [rows, cols] float32 CUDA
    """
    import torch
    q      = q.contiguous()
    scales = scales.contiguous()
    output = torch.zeros(rows, cols, dtype=torch.float32, device=q.device)
    _get_lib().launch_dequantize_fp8(
        _ptr(q), _ptr(scales), _ptr(output),
        rows, cols, per_token, _stream(),
    )
    return output


def speculative_verify(
    draft_probs,
    target_probs,
    draft_tokens,
    rand_vals,
    rand_vals2,
) -> Tuple:
    """
    Verify draft tokens via rejection sampling (Chen et al., 2023).

    :param draft_probs   torch.Tensor [K, V] float32 CUDA — draft model distributions
    :param target_probs  torch.Tensor [K, V] float32 CUDA — target model distributions
    :param draft_tokens  torch.Tensor [K] int32 CUDA — sampled draft token ids
    :param rand_vals     torch.Tensor [K] float32 CUDA — uniform random for accept/reject
    :param rand_vals2    torch.Tensor [K] float32 CUDA — uniform random for residual sampling
    :return:              (accepted: int32 [K], corrected: int32 [K])
                         accepted[i]=1 means draft_tokens[i] accepted;
                         corrected[i] holds resampled token when rejected
    """
    import torch
    draft_probs  = draft_probs.contiguous()
    target_probs = target_probs.contiguous()
    draft_tokens = draft_tokens.contiguous()
    rand_vals    = rand_vals.contiguous()
    rand_vals2   = rand_vals2.contiguous()

    K, vocab = draft_probs.shape
    accepted  = torch.zeros(K, dtype=torch.int32, device=draft_probs.device)
    corrected = torch.zeros(K, dtype=torch.int32, device=draft_probs.device)

    _get_lib().launch_verify_draft_tokens(
        _ptr(draft_probs), _ptr(target_probs),
        _ptr(draft_tokens), _ptr(rand_vals), _ptr(rand_vals2),
        _ptr(accepted), _ptr(corrected),
        K, vocab, _stream(),
    )
    return accepted, corrected


# ---------------------------------------------------------------------------
# torch.library registration (optional — requires PyTorch ≥ 2.0)
# ---------------------------------------------------------------------------

def _register_torch_library() -> None:
    """Register kernel_craft ops in the torch.ops namespace."""
    import torch

    lib = torch.library.Library("kernel_craft", "DEF")
    lib.define(
        "flash_attention(Tensor Q, Tensor K, Tensor V, int H_kv, bool causal) -> Tensor"
    )
    lib.define(
        "paged_attention(Tensor Q, Tensor block_table, Tensor K_pool, "
        "Tensor V_pool, Tensor seq_lens, int H_kv, int block_size) -> Tensor"
    )
    lib.define(
        "int4_dequantize(Tensor packed, Tensor scales, Tensor zeros, "
        "int rows, int cols, int group_size) -> Tensor"
    )
    lib.define(
        "int4_gemv(Tensor packed, Tensor scales, Tensor zeros, "
        "Tensor x, int rows, int cols, int group_size) -> Tensor"
    )
    lib.define(
        "fp8_quantize(Tensor x, bool per_token) -> (Tensor, Tensor)"
    )
    lib.define(
        "fp8_dequantize(Tensor q, Tensor scales, int rows, int cols, bool per_token) -> Tensor"
    )
    lib.define(
        "speculative_verify(Tensor draft_probs, Tensor target_probs, "
        "Tensor draft_tokens, Tensor rand_vals, Tensor rand_vals2) -> (Tensor, Tensor)"
    )

    impl = torch.library.Library("kernel_craft", "IMPL")
    for device in ("CPU", "CUDA"):
        impl.impl("flash_attention",      flash_attention,      device)
        impl.impl("paged_attention",      paged_attention,      device)
        impl.impl("int4_dequantize",      int4_dequantize,      device)
        impl.impl("int4_gemv",            int4_gemv,            device)
        impl.impl("fp8_quantize",         fp8_quantize,         device)
        impl.impl("fp8_dequantize",       fp8_dequantize,       device)
        impl.impl("speculative_verify",   speculative_verify,   device)

    # Keep references alive for the process lifetime
    globals()["_torch_lib_def"]  = lib
    globals()["_torch_lib_impl"] = impl


try:
    import torch  # noqa: F401
    _register_torch_library()
    TORCH_OPS_REGISTERED = True
except Exception:
    TORCH_OPS_REGISTERED = False
