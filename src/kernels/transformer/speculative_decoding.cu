/**
 * \file speculative_decoding.cu
 * \brief Speculative decoding draft-token verification kernels.
 *
 * Implements the token-verification step of speculative decoding
 * (Chen et al., 2023; Leviathan et al., 2023):
 *
 * 1. A small draft model proposes K tokens with probabilities q(x).
 * 2. The target model evaluates all K+1 positions in parallel, yielding p(x).
 * 3. Each draft token is accepted with probability min(1, p(x)/q(x)).
 *    On rejection, a corrected token is sampled from the residual distribution
 *    max(0, p - q) / Z.
 *
 * \par Memory layout
 * - draft_probs:  [num_tokens, vocab_size] — draft model softmax distribution
 * - target_probs: [num_tokens, vocab_size] — target model softmax distribution
 * - draft_tokens: [num_tokens]             — token IDs proposed by draft model
 * - rand_vals:    [num_tokens]             — uniform random values in [0,1)
 * - accepted:     [num_tokens]             — 1 if token accepted, 0 otherwise
 * - corrected:    [num_tokens]             — corrected token if rejected
 *
 * \par Key insight
 * This rejection-sampling scheme preserves the target distribution exactly
 * while achieving speedup proportional to the draft acceptance rate α:
 * expected tokens generated per step = (1 - α^(K+1)) / (1 - α).
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <float.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

/**
 * \brief Sample from the residual distribution max(0, p - q) via CDF inversion.
 *
 * \param[in] target_probs   Target distribution for this token position [vocab_size].
 * \param[in] draft_probs    Draft distribution for this token position [vocab_size].
 * \param[in] rand_val       Uniform random value in [0, 1).
 * \param[in] vocab_size     Vocabulary size.
 * \return Corrected token ID sampled from residual distribution.
 */
__device__ int sample_residual(
    const float* target_probs,
    const float* draft_probs,
    float rand_val,
    int vocab_size)
{
    // Compute normalizer Z = sum(max(0, p-q))
    float Z = 0.0f;
    for (int v = 0; v < vocab_size; ++v)
        Z += fmaxf(0.0f, target_probs[v] - draft_probs[v]);

    if (Z <= 0.0f) {
        // Degenerate: sample from target directly (CDF inversion)
        float cdf = 0.0f;
        for (int v = 0; v < vocab_size; ++v) {
            cdf += target_probs[v];
            if (rand_val < cdf) return v;
        }
        return vocab_size - 1;
    }

    // CDF inversion over residual distribution
    float cdf = 0.0f;
    float threshold = rand_val * Z;
    for (int v = 0; v < vocab_size; ++v) {
        float r = fmaxf(0.0f, target_probs[v] - draft_probs[v]);
        cdf += r;
        if (cdf >= threshold) return v;
    }
    return vocab_size - 1;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/**
 * \brief Verify draft tokens via rejection sampling.
 *
 * One thread per draft token. Each thread:
 * 1. Looks up p = target_probs[token] and q = draft_probs[token].
 * 2. Accepts with probability min(1, p/q) by comparing to rand_vals[i].
 * 3. On rejection, samples a corrected token from the residual distribution.
 *
 * \param[in]  draft_probs   Draft distributions [num_tokens, vocab_size].
 * \param[in]  target_probs  Target distributions [num_tokens, vocab_size].
 * \param[in]  draft_tokens  Draft token IDs [num_tokens].
 * \param[in]  rand_vals     Uniform random values [num_tokens] for acceptance test.
 * \param[in]  rand_vals2    Uniform random values [num_tokens] for residual sampling.
 * \param[out] accepted      Acceptance mask [num_tokens]: 1=accept, 0=reject.
 * \param[out] corrected     Corrected tokens [num_tokens] (valid when accepted=0).
 * \param[in]  num_tokens    Number of draft tokens.
 * \param[in]  vocab_size    Vocabulary size.
 */
__global__ void verify_draft_tokens_kernel(
    const float* __restrict__ draft_probs,
    const float* __restrict__ target_probs,
    const int*   __restrict__ draft_tokens,
    const float* __restrict__ rand_vals,
    const float* __restrict__ rand_vals2,
    int* __restrict__ accepted,
    int* __restrict__ corrected,
    int num_tokens, int vocab_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_tokens) return;

    int token_id = draft_tokens[i];
    const float* dp = draft_probs  + i * vocab_size;
    const float* tp = target_probs + i * vocab_size;

    float q = dp[token_id];
    float p = tp[token_id];

    // Clamp q to avoid division by zero; if q=0 and p=0 → accept (both agree)
    float accept_prob = (q > 1e-9f) ? fminf(1.0f, p / q) : 1.0f;

    if (rand_vals[i] < accept_prob) {
        accepted[i]  = 1;
        corrected[i] = token_id;  // no change needed
    } else {
        accepted[i]  = 0;
        corrected[i] = sample_residual(tp, dp, rand_vals2[i], vocab_size);
    }
}

/**
 * \brief Compute the prefix length of accepted tokens (first rejection point).
 *
 * Scans accepted[] and writes the index of the first rejection to output[0].
 * Tokens after the first rejection are invalid.
 *
 * \param[in]  accepted    Acceptance mask [num_tokens].
 * \param[out] prefix_len  Device scalar: number of accepted tokens (0-indexed stop).
 * \param[in]  num_tokens  Number of draft tokens.
 */
__global__ void compute_prefix_length_kernel(
    const int* __restrict__ accepted,
    int* __restrict__ prefix_len,
    int num_tokens)
{
    // Single-thread scan (for small num_tokens typical in speculative decoding)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int len = 0;
        for (int i = 0; i < num_tokens; ++i) {
            if (accepted[i]) len++;
            else break;
        }
        *prefix_len = len;
    }
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

/**
 * \brief Verify draft tokens and compute acceptance mask + corrections.
 *
 * \param[in]  d_draft_probs   Device draft distributions [num_tokens, vocab_size].
 * \param[in]  d_target_probs  Device target distributions [num_tokens, vocab_size].
 * \param[in]  d_draft_tokens  Device draft token IDs [num_tokens].
 * \param[in]  d_rand_vals     Device uniform random values [num_tokens] for accept test.
 * \param[in]  d_rand_vals2    Device uniform random values [num_tokens] for sampling.
 * \param[out] d_accepted      Device acceptance mask [num_tokens].
 * \param[out] d_corrected     Device corrected token IDs [num_tokens].
 * \param[in]  num_tokens      Number of draft tokens.
 * \param[in]  vocab_size      Vocabulary size.
 * \param[in]  stream          CUDA stream (0 = default).
 */
extern "C" void launch_verify_draft_tokens(
    const float* d_draft_probs,
    const float* d_target_probs,
    const int*   d_draft_tokens,
    const float* d_rand_vals,
    const float* d_rand_vals2,
    int*         d_accepted,
    int*         d_corrected,
    int num_tokens, int vocab_size,
    cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((num_tokens + 255) / 256);
    verify_draft_tokens_kernel<<<grid, block, 0, stream>>>(
        d_draft_probs, d_target_probs, d_draft_tokens,
        d_rand_vals, d_rand_vals2,
        d_accepted, d_corrected,
        num_tokens, vocab_size);
    cudaStreamSynchronize(stream);
}

/**
 * \brief Compute the length of the accepted prefix from the acceptance mask.
 *
 * \param[in]  d_accepted   Device acceptance mask [num_tokens].
 * \param[out] d_prefix_len Device scalar output (number of consecutive accepted tokens).
 * \param[in]  num_tokens   Number of draft tokens.
 * \param[in]  stream       CUDA stream (0 = default).
 */
extern "C" void launch_compute_prefix_length(
    const int* d_accepted,
    int*       d_prefix_len,
    int num_tokens,
    cudaStream_t stream)
{
    compute_prefix_length_kernel<<<1, 1, 0, stream>>>(
        d_accepted, d_prefix_len, num_tokens);
    cudaStreamSynchronize(stream);
}
