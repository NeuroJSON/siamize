/***************************************************************************//**
**  \mainpage siamize - native C++/ONNX port of SIAM v0.3 brain segmentation
**
**  \author    Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2026
**
**  \section sref Reference:
**  \li \c (\b Valabregue2026) Romain Valabregue, Ikram Khemir, Eric Bardinet,
**         Francois Rousseau, Guillaume Auzias, Reuben Dorent, "SIAM: Head and
**         Brain MRI Segmentation from Few High-Quality Templates via Synthetic
**         Training," arXiv:2605.02737 (2026).
**         <a href="https://arxiv.org/abs/2605.02737">arxiv.org/abs/2605.02737</a>
**  \li \c (\b ORT) Microsoft ONNX Runtime,
**         <a href="https://onnxruntime.ai">onnxruntime.ai</a>
**
**  \section slicense License
**          Apache License 2.0, see LICENSE for details
*******************************************************************************/

/***************************************************************************//**
\file    sliding.h

@brief   Sliding-window inference public interface + CUDA tuning knobs

Declares the entry point that runs the SIAM ResEnc-UNet over a
canonical (Z, Y, X) volume using ONNX Runtime, plus the EngineTuning
struct that lets callers override the CUDA Execution Provider's
memory/algorithm defaults on tight-VRAM GPUs.

The implementation in sliding.cpp handles execution-provider
selection (CPU / CUDA / TensorRT), tile generation, ORT session
caching, and accumulated multi-fold logit averaging.
*******************************************************************************/

#ifndef SIAMIZE_SLIDING_H
#define SIAMIZE_SLIDING_H

#include "siam.h"
#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace siam {

/**
 * \struct EngineTuning
 * \brief  Per-EP memory + algorithm knobs (CUDA + CPU)
 *
 * Most CUDA knobs are relevant on tight-VRAM GPUs where the default
 * cuDNN algorithm picks blow ORT's BFC arena past the GPU's usable
 * contiguous free memory. The `cpu_arena` field controls ORT's CPU
 * memory arena + memory-pattern optimizer (default ON; turning it
 * off keeps RSS smaller at the cost of substantial wall-time
 * regression, see sliding.cpp for the profiling notes).
 * All fields are optional; defaults mean "let ORT decide".
 */
/**
 * \enum  CoreMLUnits
 * \brief Which Apple Silicon hardware the CoreML EP targets
 *
 * Maps 1:1 to ORT's CoreML EP "MLComputeUnits" option. Default ALL
 * lets Core ML route per-op across CPU + Metal GPU + Neural Engine
 * (ANE). On a 3D U-Net the ANE typically claims the conv-heavy
 * decoder stages and the GPU/CPU handle the rest.
 */
enum class CoreMLUnits {
    ALL,           /**< CPU + Metal GPU + Neural Engine, Core ML routes per op (default) */
    CPU_AND_ANE,   /**< CPU + Neural Engine only (no Metal GPU) */
    CPU_AND_GPU,   /**< CPU + Metal GPU only (no Neural Engine) */
    CPU_ONLY,      /**< CPU only (debug; no Core ML acceleration) */
};

struct EngineTuning {
    int cudnn_max_workspace = 1;     /**< 0 = pick small-workspace cuDNN algos */
    int arena_same_as_req   = 0;     /**< 1 = kSameAsRequested arena extend strategy
                                          (vs the default kNextPowerOfTwo) */
    std::string algo_search;         /**< "", "DEFAULT", "HEURISTIC", or "EXHAUSTIVE" */
    size_t gpu_mem_limit_bytes = 0;  /**< 0 = no explicit memory cap */
    int gpuid = 0;                   /**< CUDA EP device_id (0 = first visible GPU).
                                          For the MNN backend's OpenCL / Vulkan
                                          devices, this maps to MNNDeviceContext::
                                          deviceId (device index within the chosen
                                          platform). */
    int gpu_platform = 0;            /**< OpenCL / Vulkan platform index (MNN only).
                                          Inert for ORT EPs. Mostly relevant on
                                          hosts with multiple OpenCL ICDs loaded
                                          (e.g. PoCL + NVIDIA), where platform 0
                                          may be the CPU runtime and the real GPU
                                          lives on platform 1. CLI `-G P:D`
                                          populates both fields. */
    bool mnn_fp16 = false;           /**< MNN BackendConfig::Precision_Low when true,
                                          else Precision_High. Inert for ORT EPs.
                                          fp16 GPU compute (Tensor Cores on
                                          Volta+, half-throughput SIMD on AMD/
                                          Intel) typically gives 1.5-2x speedup
                                          with a small accuracy cost; silently
                                          falls back to fp32 on devices that
                                          report fp16:0 (PoCL CPU-OpenCL, older
                                          GPUs without native fp16). CLI
                                          `--mnn-fp16`. */
    bool mnn_buffer = false;         /**< MNN OpenCL BUFFER memory mode when true,
                                          else IMAGE (MNN's default on NVIDIA /
                                          AMD / Adreno; Mali / Intel default to
                                          BUFFER regardless). Opt-in because the
                                          BUFFER-mode `conv_2d_buf` kernel
                                          source fails to compile on some
                                          NVIDIA OpenCL drivers (CL build
                                          error -9999). On PoCL CPU-OpenCL the
                                          BUFFER path is ~5x faster than IMAGE
                                          on SIAM-class 3D U-Nets; on real
                                          NVIDIA the gain is uncertain and the
                                          compile-failure risk is real. CLI
                                          `--mnn-buffer`. */
    bool cpu_arena = true;           /**< true = ORT CPU memory arena + memory-pattern ON
                                          (default, fast path); false = both disabled
                                          (lower RSS, much slower) */

    // TensorRT EP cache (Linux only with ORT-GPU build). Inert otherwise.
    std::string trt_cache_dir;       /**< empty => $HOME/.cache/siamize/trt. Populated
                                          by sliding_window() from the separate
                                          trt_cache_dir parameter for backward compat. */

    // CoreML EP knobs (macOS only). Inert on Linux / Windows builds.
    CoreMLUnits coreml_units = CoreMLUnits::ALL;  /**< hardware target */
    std::string coreml_cache_dir;                  /**< empty => $HOME/.cache/siamize/coreml */
    bool coreml_static_shapes = true;              /**< Pass MLProgram + RequireStaticInputShapes=1
                                                        when paired with the fixed-shape ONNX
                                                        (recommended); set false for the
                                                        dynamic-shape ONNX (more CPU fallbacks). */
};

/**
 * @brief Run sliding-window inference over a canonical volume
 *
 * Runs each .onnx file in \a model_paths over a regular grid of
 * `patch_size`-sized tiles of \a data (step = `step_ratio * patch`),
 * accumulating logits into a (num_classes, Z, Y, X) volume using
 * the cosine-window blending nnUNet established. Per-tile output
 * is averaged across folds; per-tile execution is delegated to
 * ORT with the selected Execution Provider.
 *
 * @param  data           canonical (Z, Y, X) input volume
 * @param  model_paths    one .onnx file per fold (logits averaged)
 * @param  patch_size     per-tile (Z, Y, X) size
 * @param  num_classes    number of output channels
 * @param  intra_threads  forwarded to ORT's SessionOptions::SetIntraOpNumThreads
 * @param  step_ratio     tile stride as a fraction of patch_size (typ. 0.5)
 * @param  verbose        prints per-tile progress to stderr when true
 * @param  device         "auto" (CoreML on macOS, else CUDA-if-available, else CPU),
 *                        "cpu", "cuda", "tensorrt", or "coreml"
 * @param  trt_cache_dir  TensorRT engine cache directory (only used
 *                        when \a device == "tensorrt"); empty resolves
 *                        to `$HOME/.cache/siamize/trt`
 * @param  engine_tuning    CUDA EP overrides (see EngineTuning)
 * @return                logits at the same grid as \a data
 */
LogitsVolume sliding_window(const Volume& data,
                            const std::vector<std::string>& model_paths,
                            std::array<int64_t, 3> patch_size,
                            int64_t num_classes,
                            int intra_threads,
                            float step_ratio,
                            bool verbose,
                            const std::string& device = "auto",
                            const std::string& trt_cache_dir = "",
                            const EngineTuning& engine_tuning = {});

}  // namespace siam

#endif  // SIAMIZE_SLIDING_H
