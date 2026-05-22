#!/usr/bin/env python3
"""
example_speculative_decoding.py — draft token verification for speculative decoding.

Background
-----------
Speculative decoding (Leviathan et al. 2023) accelerates LLM autoregressive
generation by running a small fast "draft" model to speculatively generate K
tokens, then verifying all K tokens with the large "target" model in a single
forward pass.  This delivers target-model quality at draft-model speed when the
draft and target distributions are well-aligned (high acceptance rate α).

Expected speedup:
    E[tokens per step] = (1 - α^(K+1)) / (1 - α)
    α=0.8, K=4 → ~3.3 tokens per step instead of 1 → ~3.3× throughput.

Acceptance-rejection sampling
------------------------------
For each draft token t_i with draft probability p(t_i) and target probability
q(t_i):

  Accept  (with prob min(1, q(t_i)/p(t_i))):
      Use t_i as the output token.

  Reject  (with prob 1 - min(1, q(t_i)/p(t_i))):
      Sample a correction token from the residual distribution:
          residual(x) = max(0, q(x) - p(x)) / Z
      This ensures the joint output distribution exactly matches the target.

The verification kernel processes all K draft tokens in parallel, so the
target forward pass is amortised over multiple accepted tokens.

API overview
------------
``kernel_craft_transformer.speculative_decode(draft_probs, target_probs,
                                              draft_tokens, rand_vals, rand_vals2)``

    Vectorised rejection sampling for K draft tokens.

    Parameters
    ----------
    draft_probs  : np.ndarray float32 [K, V]  — draft model's full vocab distribution
    target_probs : np.ndarray float32 [K, V]  — target model's full vocab distribution
    draft_tokens : np.ndarray int32   [K]     — tokens selected by the draft model
    rand_vals    : np.ndarray float32 [K]     — uniform samples for accept/reject (u ~ U[0,1])
    rand_vals2   : np.ndarray float32 [K]     — uniform samples for correction sampling

    Returns
    -------
    (accepted, corrected)
    accepted  : np.ndarray int32 [K]  — accepted tokens (-1 if rejected)
    corrected : np.ndarray int32 [K]  — correction tokens sampled from residual

Integration in a serving engine
---------------------------------
A typical speculative decoding loop with this kernel:

    draft_model  = DraftModel("TinyLlama-1.1B")   # fast, small
    target_model = TargetModel("Llama-3-70B")       # quality, slow

    context = tokenize(prompt)
    output  = list(context)

    while not done:
        # 1. Draft K tokens auto-regressively with the small model
        draft_tokens, draft_probs = draft_model.generate(context, K=4)

        # 2. One target forward pass over context + K draft tokens
        target_probs = target_model.forward(context + draft_tokens)

        # 3. Verify with rejection sampling
        accepted, corrected = kct.speculative_decode(
            draft_probs, target_probs, draft_tokens, rand_vals, rand_vals2)

        # 4. Find how many draft tokens were accepted (prefix)
        # The kernel computes prefix_length: first rejection index
        n_accepted = int((accepted != -1).argmin())  # first -1
        output.extend(accepted[:n_accepted])
        output.append(corrected[n_accepted])          # one correction always added
        context = output

Theoretical properties
-----------------------
- Preserves the target distribution exactly (not an approximation).
- When draft = target: all K tokens accepted → K+1 tokens per step.
- When draft and target are uncorrelated: α ≈ 0 → ~1 token per step (same as baseline).
- Correction token from residual ensures the step always produces at least 1 new token.

Usage
-----
    cd examples/python
    python example_speculative_decoding.py
"""

import sys
import os

_REPO = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_REPO, "src", "python", "build"))
sys.path.insert(0, os.path.join(_REPO, "src", "python"))

import numpy as np

try:
    import kernel_craft_transformer as kct  # type: ignore
    _HAS_KCT = True
except ImportError:
    _HAS_KCT = False
    print("Warning: kernel_craft_transformer not found — build first.")


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def main() -> None:
    print("=" * 60)
    print("kernel-craft: speculative decoding token verification")
    print("=" * 60)

    if not _HAS_KCT:
        return

    rng = np.random.default_rng(123)
    V = 32000   # vocabulary size (Llama tokenizer)
    K = 4       # draft tokens per step

    # ------------------------------------------------------------------
    # Scenario 1: Perfect alignment  (draft ≈ target → all accepted)
    # ------------------------------------------------------------------
    print(f"\n[Scenario 1] Perfect alignment (draft = target)")
    logits = rng.standard_normal((K, V)).astype(np.float32)
    draft_probs  = _softmax(logits)
    target_probs = _softmax(logits)   # identical distributions

    draft_tokens = np.argmax(draft_probs, axis=-1).astype(np.int32)
    rand_vals    = rng.uniform(0, 0.1, size=K).astype(np.float32)  # small → easy accept
    rand_vals2   = rng.uniform(size=K).astype(np.float32)

    accepted, corrected = kct.speculative_decode(
        draft_probs, target_probs, draft_tokens, rand_vals, rand_vals2)

    n_accepted = int(np.sum(accepted != -1))
    print(f"  Draft tokens   : {draft_tokens}")
    print(f"  Accepted       : {accepted}  ({n_accepted}/{K} accepted)")
    print(f"  Corrected      : {corrected}")
    print(f"  Effective toks/step: {n_accepted + 1}  (ideal: {K+1}={K}+correction)")

    # ------------------------------------------------------------------
    # Scenario 2: Zero overlap  (draft always wrong → all rejected)
    # ------------------------------------------------------------------
    print(f"\n[Scenario 2] No overlap (target assigns 0 prob to draft tokens)")
    draft_probs2 = np.zeros((K, V), dtype=np.float32)
    for i in range(K):
        draft_probs2[i, i] = 1.0          # draft is certain about token i

    target_probs2 = np.zeros((K, V), dtype=np.float32)
    for i in range(K):
        target_probs2[i, V - 1 - i] = 1.0  # target prefers opposite end of vocab

    rand_vals2_b   = rng.uniform(size=K).astype(np.float32)
    rand_vals2_c   = rng.uniform(size=K).astype(np.float32)
    draft_tokens2  = np.arange(K, dtype=np.int32)
    rand_force_rej = np.ones(K, dtype=np.float32)  # u=1 → always reject (q/p = 0 < u)

    accepted2, corrected2 = kct.speculative_decode(
        draft_probs2, target_probs2, draft_tokens2, rand_force_rej, rand_vals2_c)

    print(f"  All accepted   : {accepted2}  (all -1 = rejected)")
    print(f"  Corrected      : {corrected2}")
    # Correction should sample from residual = max(0, target - draft) / Z = target here
    expected_corrections = [V - 1 - i for i in range(K)]
    print(f"  Expected corr  : {expected_corrections}")
    print(f"  Correction OK  : {list(corrected2) == expected_corrections}")

    # ------------------------------------------------------------------
    # Scenario 3: Realistic mixed acceptance (α ≈ 0.7)
    # ------------------------------------------------------------------
    print(f"\n[Scenario 3] Realistic mixed (α ≈ 0.7 per token)")
    # Draft is mostly right but occasionally wrong
    target_logits = rng.standard_normal((K, V)).astype(np.float32)
    target_probs3 = _softmax(target_logits)
    draft_logits  = target_logits + rng.standard_normal((K, V)).astype(np.float32) * 0.5
    draft_probs3  = _softmax(draft_logits)

    draft_tokens3 = np.argmax(draft_probs3, axis=-1).astype(np.int32)
    rand3  = rng.uniform(size=K).astype(np.float32)
    rand3b = rng.uniform(size=K).astype(np.float32)

    accepted3, corrected3 = kct.speculative_decode(
        draft_probs3, target_probs3, draft_tokens3, rand3, rand3b)

    n_acc3 = int(np.sum(accepted3 != -1))
    print(f"  Accepted tokens: {n_acc3}/{K}")
    print(f"  accepted       : {accepted3}")
    print(f"  Effective toks/step: {n_acc3 + 1}")
    # Theoretical for α=0.7, K=4: E[tokens] = (1-0.7^5)/(1-0.7) ≈ 3.07
    alpha_est = 0.7
    expected_toks = (1 - alpha_est**(K+1)) / (1 - alpha_est)
    print(f"  Theoretical E[toks] for α=0.7, K={K}: {expected_toks:.2f}")

    print("\n  Key insight: the kernel runs in O(K×V) time (single GPU pass),")
    print("  while the target model verification runs in O(1) forward pass.")
    print("  The speedup is realised because the target forward pass is a")
    print("  batch forward over K+1 positions, not K+1 sequential decodes.")
    print("\n  See src/kernels/transformer/speculative_decoding.cu")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
