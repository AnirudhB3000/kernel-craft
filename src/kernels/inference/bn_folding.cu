/**
 * \file bn_folding.cu
 * \brief Batch Normalization folding for CNN inference optimization.
 *
 * Batch Normalization folding fuses BN parameters into the preceding
 * convolution weights and bias, eliminating the BN layer during inference.
 *
 * \par Mathematical Basis
 * During inference, BN computes:
 *   y = (x - μ) / sqrt(σ² + ε) * γ + β
 *     = x * (γ / sqrt(σ² + ε)) + (β - μ * γ / sqrt(σ² + ε))
 *
 * For a conv layer: y = w * x + b
 * After folding: 
 *   w' = w * (γ / sqrt(σ² + ε))
 *   b' = b * (γ / sqrt(σ² + ε)) + (β - μ * γ / sqrt(σ² + ε))
 *
 * \par Benefits
 * - Eliminates BN computation during inference
 * - Reduces memory accesses
 * - Fuses two layers into one
 */

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

/**
 * \brief Fold batch normalization parameters into convolution weights and bias.
 *
 * This kernel processes all output channels in parallel to fold BN parameters
 * into conv weights and bias.
 *
 * \param[in] conv_weights Original convolution weights (format: [C_out, C_in, K_h, K_w]).
 * \param[in] conv_bias Original convolution bias (can be nullptr if no bias).
 * \param[out] folded_weights Output folded weights (same size as conv_weights).
 * \param[out] folded_bias Output folded bias (size: C_out).
 * \param[in] bn_mean Batch normalization mean (size: C_out).
 * \param[in] bn_variance Batch normalization variance (size: C_out).
 * \param[in] bn_gamma Batch normalization scale parameter (size: C_out).
 * \param[in] bn_beta Batch normalization shift parameter (size: C_out).
 * \param[in] epsilon Small constant for numerical stability.
 * \param[in] C_out Number of output channels.
 * \param[in] C_in Number of input channels.
 * \param[in] K_h Kernel height.
 * \param[in] K_w Kernel width.
 */
extern "C" __global__ void bn_folding_kernel(const float* __restrict__ conv_weights,
                                              const float* __restrict__ conv_bias,
                                              float* __restrict__ folded_weights,
                                              float* __restrict__ folded_bias,
                                              const float* __restrict__ bn_mean,
                                              const float* __restrict__ bn_variance,
                                              const float* __restrict__ bn_gamma,
                                              const float* __restrict__ bn_beta,
                                              float epsilon,
                                              int C_out, int C_in, int K_h, int K_w) {
    // Each thread processes one output channel
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= C_out) return;

    // Compute scaling factor and new bias term for this channel
    float inv_std = 1.0f / sqrtf(bn_variance[oc] + epsilon);
    float scale = bn_gamma[oc] * inv_std;
    float bias_term = bn_beta[oc] - bn_mean[oc] * scale;

    // Fold into weights: multiply all weights for this output channel
    int channel_weights = C_in * K_h * K_w;
    for (int i = 0; i < channel_weights; ++i) {
        folded_weights[oc * channel_weights + i] = conv_weights[oc * channel_weights + i] * scale;
    }

    // Fold into bias
    if (conv_bias != nullptr) {
        float bias_val = conv_bias[oc];
        printf("DEBUG KERNEL: oc=%d, conv_bias[oc]=%f, scale=%f, bias_term=%f, result=%f\n",
               oc, bias_val, scale, bias_term, bias_val * scale + bias_term);
        folded_bias[oc] = conv_bias[oc] * scale + bias_term;
    } else {
        folded_bias[oc] = bias_term;
    }
}

/**
 * \brief Host launcher for BN folding.
 *
 * Folds batch normalization parameters into convolution weights and bias.
 * This should be called once after training and before inference deployment.
 *
 * \param[in] d_conv_weights Device pointer to original conv weights.
 * \param[in] d_conv_bias Device pointer to original conv bias (can be nullptr).
 * \param[out] d_folded_weights Device pointer to output folded weights.
 * \param[out] d_folded_bias Device pointer to output folded bias.
 * \param[in] d_bn_mean Device pointer to BN mean.
 * \param[in] d_bn_variance Device pointer to BN variance.
 * \param[in] d_bn_gamma Device pointer to BN gamma.
 * \param[in] d_bn_beta Device pointer to BN beta.
 * \param[in] epsilon Small constant for numerical stability.
 * \param[in] C_out Number of output channels.
 * \param[in] C_in Number of input channels.
 * \param[in] K_h Kernel height.
 * \param[in] K_w Kernel width.
 */
extern "C" void launch_bn_folding(const float* d_conv_weights,
                                   const float* d_conv_bias,
                                   float* d_folded_weights,
                                   float* d_folded_bias,
                                   const float* d_bn_mean,
                                   const float* d_bn_variance,
                                   const float* d_bn_gamma,
                                   const float* d_bn_beta,
                                   float epsilon,
                                   int C_out, int C_in, int K_h, int K_w) {
    int threads_per_block = 256;
    int num_blocks = (C_out + threads_per_block - 1) / threads_per_block;
    
    bn_folding_kernel<<<num_blocks, threads_per_block>>>(
        d_conv_weights, d_conv_bias, d_folded_weights, d_folded_bias,
        d_bn_mean, d_bn_variance, d_bn_gamma, d_bn_beta,
        epsilon, C_out, C_in, K_h, K_w
    );
    cudaDeviceSynchronize();
}

/**
 * \brief CPU reference for BN folding (for verification).
 *
 * \param[in] conv_weights Original conv weights (row-major).
 * \param[in] conv_bias Original conv bias (can be nullptr).
 * \param[out] folded_weights Output folded weights.
 * \param[out] folded_bias Output folded bias.
 * \param[in] bn_mean BN mean array.
 * \param[in] bn_variance BN variance array.
 * \param[in] bn_gamma BN gamma array.
 * \param[in] bn_beta BN beta array.
 * \param[in] epsilon Numerical stability constant.
 * \param[in] C_out Number of output channels.
 * \param[in] C_in Number of input channels.
 * \param[in] K_h Kernel height.
 * \param[in] K_w Kernel width.
 */
extern "C" void bn_folding_cpu_ref(const float* conv_weights,
                        const float* conv_bias,
                        float* folded_weights,
                        float* folded_bias,
                        const float* bn_mean,
                        const float* bn_variance,
                        const float* bn_gamma,
                        const float* bn_beta,
                        float epsilon,
                        int C_out, int C_in, int K_h, int K_w) {
    int channel_weights = C_in * K_h * K_w;
    
    for (int oc = 0; oc < C_out; ++oc) {
        float inv_std = 1.0f / sqrtf(bn_variance[oc] + epsilon);
        float scale = bn_gamma[oc] * inv_std;
        float bias_term = bn_beta[oc] - bn_mean[oc] * scale;
        
        for (int i = 0; i < channel_weights; ++i) {
            folded_weights[oc * channel_weights + i] = conv_weights[oc * channel_weights + i] * scale;
        }
        
        if (conv_bias != nullptr) {
            folded_bias[oc] = conv_bias[oc] * scale + bias_term;
        } else {
            folded_bias[oc] = bias_term;
        }
    }
}
