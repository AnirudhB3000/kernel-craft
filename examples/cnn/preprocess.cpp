/**
 * \file cnn/preprocess.cpp
 * \brief GPU preprocessing pipeline: bilinear resize → normalize → flip.
 *
 * \par Purpose
 * Demonstrates the preprocessing kernels that transform a raw input image
 * into the normalised tensor expected by a CNN inference engine.  Running
 * preprocessing on the GPU (rather than on the CPU) eliminates a host→device
 * copy of the full-resolution input and keeps the data on-device throughout
 * the entire inference pipeline.
 *
 * \par Typical CNN preprocessing pipeline
 * \code
 *   Raw camera frame (e.g. 1920×1080, uint8, BGR)
 *     │
 *     ▼  launch_resize_bilinear   — scales to model input size (e.g. 224×224)
 *     │
 *     ▼  launch_normalize         — (pixel - mean) / std  → float32 in [-1, 1]
 *     │
 *     ▼  launch_flip_horizontal   — optional data-augmentation at inference time
 *     │
 *     ▼  CNN first layer (conv + BN + ReLU)
 * \endcode
 *
 * \par Integration — NVIDIA DeepStream / Triton Inference Server
 * DeepStream's GStreamer pipeline calls nvvideoconvert (GPU resize + colour
 * conversion) before feeding frames to a TensorRT engine.  Our kernels provide
 * the same capability without the DeepStream dependency, making them suitable
 * for custom inference runtimes (e.g. direct RTSP → CUDA → TensorRT pipelines).
 *
 * In a multi-stream setup (see src/performance/multi_stream_pipeline.cu) the
 * preprocessing and the first CNN layer can execute concurrently on separate
 * CUDA streams, hiding the preprocessing latency behind the previous frame's
 * inference.
 *
 * \par Normalization conventions
 * ImageNet models expect:
 *   mean = [0.485, 0.456, 0.406]  (per channel, after dividing uint8 by 255)
 *   std  = [0.229, 0.224, 0.225]
 * Pass these per-channel values when wrapping this kernel for a multi-channel
 * input.  The kernel shown here uses a single mean/std for a grayscale input.
 *
 * \par Compile and run
 * \code
 * cmake --build build --target example_preprocess
 * ./build/examples/example_preprocess
 * \endcode
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

/**
 * \brief Bilinear upscale/downscale (src/pipelines/preprocess_gpu.cu).
 *
 * Each output pixel samples the input at the corresponding back-projected
 * coordinate using bilinear interpolation.  Supports arbitrary scale factors.
 *
 * \param[in]  d_input  Row-major float32 device array [in_h × in_w].
 * \param[out] d_output Row-major float32 device array [out_h × out_w].
 * \param[in]  in_w, in_h   Input spatial dimensions.
 * \param[in]  out_w, out_h Output spatial dimensions (target model input size).
 */
extern "C" void launch_resize_bilinear(const float* d_input, float* d_output,
                                       int in_w, int in_h, int out_w, int out_h);

/**
 * \brief Per-pixel normalisation: out = (in - mean) / std (src/pipelines/preprocess_gpu.cu).
 *
 * Applied after resize to bring pixel values into the range expected by the
 * model's first layer.  For ImageNet models, call once per channel with its
 * specific mean and std values.
 *
 * \param[in]  d_input  float32 device array [height × width].
 * \param[out] d_output float32 device array [height × width] (may alias d_input).
 * \param[in]  width, height  Spatial dimensions.
 * \param[in]  mean  Per-channel pixel mean (e.g. 0.485 for ImageNet R channel).
 * \param[in]  std   Per-channel pixel std  (e.g. 0.229 for ImageNet R channel).
 */
extern "C" void launch_normalize(const float* d_input, float* d_output,
                                 int width, int height, float mean, float std);

/**
 * \brief Horizontal flip (src/pipelines/preprocess_gpu.cu).
 *
 * Used for test-time augmentation (TTA): run inference on both the original
 * and the flipped frame, then average the logits for improved accuracy.
 *
 * \param[in]  d_input  float32 device array [height × width].
 * \param[out] d_output float32 device array [height × width] (may alias d_input).
 * \param[in]  width, height  Spatial dimensions.
 */
extern "C" void launch_flip_horizontal(const float* d_input, float* d_output,
                                       int width, int height);

int main() {
    /* Simulate a small 4×4 input being resized to 8×8 for the model. */
    const int IN_W = 4, IN_H = 4;
    const int OUT_W = 8, OUT_H = 8;

    float h_in[IN_W * IN_H];
    float h_out[OUT_W * OUT_H];

    /* Fill with a gradient: pixel (r,c) = r*IN_W + c. */
    for (int i = 0; i < IN_W * IN_H; ++i) h_in[i] = (float)i;

    float *d_in, *d_out;
    cudaMalloc(&d_in,  sizeof(float) * IN_W  * IN_H);
    cudaMalloc(&d_out, sizeof(float) * OUT_W * OUT_H);
    cudaMemcpy(d_in, h_in, sizeof(float) * IN_W * IN_H, cudaMemcpyHostToDevice);

    /* --- Step 1: resize 4×4 → 8×8 via bilinear interpolation --- */
    launch_resize_bilinear(d_in, d_out, IN_W, IN_H, OUT_W, OUT_H);

    /* --- Step 2: normalise (ImageNet grayscale: mean≈0.45, std≈0.23) --- */
    launch_normalize(d_out, d_out, OUT_W, OUT_H, /*mean=*/0.0f, /*std=*/15.0f);

    /* --- Step 3: horizontal flip for test-time augmentation --- */
    launch_flip_horizontal(d_out, d_out, OUT_W, OUT_H);

    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, sizeof(float) * OUT_W * OUT_H, cudaMemcpyDeviceToHost);

    printf("Preprocessing Pipeline Example\n");
    printf("  Input  : %dx%d   Output : %dx%d\n", IN_W, IN_H, OUT_W, OUT_H);
    printf("  Steps  : resize (bilinear) → normalize → flip_horizontal\n");
    printf("  h_out[ 0] = %.4f  (top-left after flip)\n",    h_out[0]);
    printf("  h_out[%d] = %.4f  (bottom-right after flip)\n",
           OUT_W * OUT_H - 1, h_out[OUT_W * OUT_H - 1]);
    printf("\nNext step: pass d_out directly to launch_conv_tiled or a TensorRT engine.\n");
    printf("See src/performance/multi_stream_pipeline.cu for concurrent preprocess+infer.\n");

    cudaFree(d_in); cudaFree(d_out);
    return EXIT_SUCCESS;
}
