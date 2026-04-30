/**
 * \file conv_grouped.cu
 * \brief Grouped convolution implementation.
 *
 * This file provides CUDA kernels for performing grouped convolution,
 * as used in ResNeXt and other efficient architectures. The input and
 * output channels are split into groups, reducing parameters and compute.
 */

#include <cuda_runtime.h>

/**
 * \brief Naïve grouped convolution kernel.
 *
 * Each thread computes one output pixel across all groups. The convolution
 * is split such that each group operates on inChannels/groups input channels
 * and produces outChannels/groups output channels.
 *
 * \param[in] input  Pointer to the input image (inChannels × height × width).
 * \param[in] kernel Pointer to the grouped convolution kernel.
 * \param[out] output Pointer to the output image buffer.
 * \param[in] width  Width of the images.
 * \param[in] height Height of the images.
 * \param[in] inChannels  Number of input channels.
 * \param[in] outChannels Number of output channels.
 * \param[in] groups Number of groups (must divide inChannels and outChannels).
 * \param[in] ksize Kernel size (must be odd).
 */
extern "C" __global__ void conv_grouped_naive(const float* __restrict__ input,
                                             const float* __restrict__ kernel,
                                             float* __restrict__ output,
                                             int width, int height,
                                             int inChannels, int outChannels,
                                             int groups, int ksize) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ox >= width || oy >= height) return;

    int channelsPerGroup = outChannels / groups;
    int inChannelsPerGroup = inChannels / groups;

    for (int oc = 0; oc < channelsPerGroup; ++oc) {
        double sum = 0.0;
        int kHalf = ksize / 2;

        for (int ic = 0; ic < inChannelsPerGroup; ++ic) {
            for (int ky = 0; ky < ksize; ++ky) {
                int iy = oy + ky - kHalf;
                if (iy < 0 || iy >= height) continue;
                for (int kx = 0; kx < ksize; ++kx) {
                    int ix = ox + kx - kHalf;
                    if (ix < 0 || ix >= width) continue;
                    float inVal = input[(ic * height + iy) * width + ix];
                    float kVal = kernel[((oc * groups + 0) * inChannelsPerGroup + ic) * ksize * ksize + ky * ksize + kx];
                    sum += static_cast<double>(inVal) * static_cast<double>(kVal);
                }
            }
        }
        output[(oc * height + oy) * width + ox] = static_cast<float>(sum);
    }
}

/**
 * \brief Optimized grouped convolution with one thread per output.
 *
 * Each thread handles one output pixel for one output channel.
 *
 * \param[in] input  Pointer to the input image.
 * \param[in] kernel Pointer to the grouped convolution kernel.
 * \param[out] output Pointer to the output image buffer.
 * \param[in] width  Width of the images.
 * \param[in] height Height of the images.
 * \param[in] inChannels  Number of input channels.
 * \param[in] outChannels Number of output channels.
 * \param[in] groups Number of groups.
 * \param[in] ksize Kernel size.
 */
extern "C" __global__ void conv_grouped_opt(const float* __restrict__ input,
                                            const float* __restrict__ kernel,
                                            float* __restrict__ output,
                                            int width, int height,
                                            int inChannels, int outChannels,
                                            int groups, int ksize) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z * blockDim.z + threadIdx.z;

    if (ox >= width || oy >= height || oc >= outChannels) return;

    int channelsPerGroup = outChannels / groups;
    int inChannelsPerGroup = inChannels / groups;
    int group = oc / channelsPerGroup;
    int ocInGroup = oc % channelsPerGroup;
    int kHalf = ksize / 2;

    double sum = 0.0;
    for (int ic = 0; ic < inChannelsPerGroup; ++ic) {
        int srcIc = group * inChannelsPerGroup + ic;
        int kernelOffset = ((group * channelsPerGroup + ocInGroup) * inChannelsPerGroup + ic) * ksize * ksize;

        for (int ky = 0; ky < ksize; ++ky) {
            int iy = oy + ky - kHalf;
            if (iy < 0 || iy >= height) continue;
            for (int kx = 0; kx < ksize; ++kx) {
                int ix = ox + kx - kHalf;
                if (ix < 0 || ix >= width) continue;
                float inVal = input[(srcIc * height + iy) * width + ix];
                float kVal = kernel[kernelOffset + ky * ksize + kx];
                sum += static_cast<double>(inVal) * static_cast<double>(kVal);
            }
        }
    }
    output[(oc * height + oy) * width + ox] = static_cast<float>(sum);
}

/**
 * \brief Host‑side launcher for the naïve grouped convolution kernel.
 *
 * \param[in] input  Device pointer to the input image.
 * \param[in] kernel Device pointer to the grouped kernel.
 * \param[out] output Device pointer to the output buffer.
 * \param[in] width  Image width.
 * \param[in] height Image height.
 * \param[in] inChannels  Number of input channels.
 * \param[in] outChannels Number of output channels.
 * \param[in] groups Number of groups.
 * \param[in] ksize Kernel size.
 * \param[in] block Block dimensions.
 */
extern "C" void launch_conv_grouped_naive(const float* input,
                                          const float* kernel,
                                          float* output,
                                          int width, int height,
                                          int inChannels, int outChannels,
                                          int groups, int ksize,
                                          dim3 block = dim3(16,16,1)) {
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    conv_grouped_naive<<<grid, block>>>(input, kernel, output, width, height, inChannels, outChannels, groups, ksize);
    cudaDeviceSynchronize();
}

/**
 * \brief Host‑side launcher for the optimized grouped convolution kernel.
 *
 * \param[in] input  Device pointer to the input image.
 * \param[in] kernel Device pointer to the grouped kernel.
 * \param[out] output Device pointer to the output buffer.
 * \param[in] width  Image width.
 * \param[in] height Image height.
 * \param[in] inChannels  Number of input channels.
 * \param[in] outChannels Number of output channels.
 * \param[in] groups Number of groups.
 * \param[in] ksize Kernel size.
 * \param[in] block Block dimensions.
 */
extern "C" void launch_conv_grouped_opt(const float* input,
                                         const float* kernel,
                                         float* output,
                                         int width, int height,
                                         int inChannels, int outChannels,
                                         int groups, int ksize,
                                         dim3 block = dim3(16,16,1)) {
    int channelsPerGroup = outChannels / groups;
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y,
              (outChannels + block.z - 1) / block.z);
    conv_grouped_opt<<<grid, block>>>(input, kernel, output, width, height, inChannels, outChannels, groups, ksize);
    cudaDeviceSynchronize();
}