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
\file    sliding.cpp
\brief   Sliding-window inference: tile generation, ORT plumbing, fold averaging

Mirrors the Python reference in py/siam_ref.py / siam_ort.py:

  - Build a Gaussian-weighted importance map sized to the patch
    (matches scipy.ndimage.gaussian_filter(sigma=patch/8) on a delta,
    normalized so the max is 10x and zero entries are floored to the
    smallest nonzero weight to avoid voxels with no coverage).
  - Generate evenly-spaced tile starts per axis at `step_ratio *
    patch_size` stride, with the last tile snug against the far edge.
  - Run one ORT session per model fold; per tile, feed in (1, 1, Z, Y, X)
    fp32 patches, accumulate logits * Gaussian into the accumulator,
    track per-voxel weight, then divide at the end.
  - Execution provider selection: CPU (always available), CUDA EP
    (compile-time SIAMIZE_GPU=cuda + runtime cuDNN/cuBLAS), TensorRT
    EP (compile-time SIAMIZE_GPU=tensorrt + runtime libnvinfer). The
    "auto" device silently falls back from CUDA to CPU.

fp32 accumulators throughout; only one fold's session lives in ORT
at any moment so peak VRAM stays bounded.
*******************************************************************************/

#include "sliding.h"
#include "siam.h"
#include "siam_log.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#endif

namespace siam {

namespace {

#ifdef _WIN32
/*******************************************************************************/
/*! \fn    std::wstring widen_path(const std::string& s)
    \brief Convert UTF-8 to UTF-16 for the Windows ORT Session constructor
    \param  s  UTF-8 path
    \return    UTF-16 path
*/
std::wstring widen_path(const std::string& s) {
    if (s.empty()) {
        return std::wstring();
    }

    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()),
                                nullptr, 0);

    if (n <= 0) {
        throw std::runtime_error("MultiByteToWideChar failed for path: " + s);
    }

    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()),
                        w.data(), n);
    return w;
}
#endif

/*******************************************************************************/
/*! \fn    std::vector<float> compute_gaussian(std::array<int64_t, 3> patch,
                                               float sigma_scale,
                                               float value_scaling)
    \brief Build the per-tile Gaussian importance map

    Equivalent to `scipy.ndimage.gaussian_filter(delta, sigma=patch/8)`
    -- the filter response to a delta IS a Gaussian, so we evaluate
    the closed form directly. After normalize-by-max, scipy's
    truncation and edge handling become irrelevant. Zero entries
    (well outside the Gaussian's mass) are floored to the smallest
    nonzero weight so every voxel in the tile contributes some
    coverage to the accumulator.

    \param  patch          tile shape in (Z, Y, X)
    \param  sigma_scale    sigma = sigma_scale * patch_size, default 1/8
    \param  value_scaling  peak after normalization (default 10x)
    \return                flat (Z*Y*X) importance map
*/
std::vector<float> compute_gaussian(std::array<int64_t, 3> patch,
                                    float sigma_scale = 1.0f / 8.0f,
                                    float value_scaling = 10.0f) {
    std::vector<float> g(patch[0] * patch[1] * patch[2]);
    const float cz = patch[0] * 0.5f - 0.5f;     // matches scipy 'center = patch // 2' for even sizes
    const float cy = patch[1] * 0.5f - 0.5f;
    const float cx = patch[2] * 0.5f - 0.5f;
    const float sz = patch[0] * sigma_scale;
    const float sy = patch[1] * sigma_scale;
    const float sx = patch[2] * sigma_scale;
    // Note: siam_ref/siam_ort use `s // 2` as center → integer center.
    // Use the integer center to match Python exactly.
    const float icz = static_cast<float>(patch[0] / 2);
    const float icy = static_cast<float>(patch[1] / 2);
    const float icx = static_cast<float>(patch[2] / 2);
    (void)cz;
    (void)cy;
    (void)cx;

    double gmax = 0.0;

    for (int64_t z = 0; z < patch[0]; ++z) {
        float dz = (z - icz) / sz;

        for (int64_t y = 0; y < patch[1]; ++y) {
            float dy = (y - icy) / sy;

            for (int64_t x = 0; x < patch[2]; ++x) {
                float dx = (x - icx) / sx;
                double v = std::exp(-0.5 * (dx * dx + dy * dy + dz * dz));
                g[z * patch[1] * patch[2] + y * patch[2] + x] = static_cast<float>(v);

                if (v > gmax) {
                    gmax = v;
                }
            }
        }
    }

    const double inv_max = (gmax > 0.0) ? (value_scaling / gmax) : 1.0;
    double nz_min = std::numeric_limits<double>::infinity();

    for (auto& v : g) {
        v = static_cast<float>(v * inv_max);

        if (v > 0.0f && v < nz_min) {
            nz_min = v;
        }
    }

    if (!std::isfinite(nz_min)) {
        nz_min = 1e-6;
    }

    for (auto& v : g) if (v == 0.0f) {
            v = static_cast<float>(nz_min);
        }

    return g;
}

/*******************************************************************************/
/*! \fn    std::vector<int64_t> compute_steps(int64_t image_size,
                                              int64_t patch,
                                              float step_ratio)
    \brief Generate tile start indices along one axis

    Picks the smallest number of evenly-spaced positions whose stride
    is at most `patch * step_ratio` and which cover the full range
    `[0, image_size - patch]`. With one tile, no spacing is needed;
    otherwise the actual stride is recomputed to divide the span
    exactly. Matches nnUNet's `compute_steps_for_sliding_window`.

    \param  image_size  image extent along this axis
    \param  patch       patch extent along this axis
    \param  step_ratio  fraction of patch_size used as the target stride
    \return             monotonically-increasing tile start indices
*/
std::vector<int64_t> compute_steps(int64_t image_size, int64_t patch, float step_ratio) {
    if (image_size < patch) {
        throw std::runtime_error("image smaller than patch (pad first)");
    }

    double target_step = patch * step_ratio;
    int64_t num_steps = static_cast<int64_t>(std::ceil((image_size - patch) / target_step)) + 1;

    if (num_steps < 1) {
        num_steps = 1;
    }

    double actual = (num_steps > 1)
                    ? static_cast<double>(image_size - patch) / (num_steps - 1)
                    : 1e12;
    std::vector<int64_t> steps;
    steps.reserve(num_steps);

    for (int64_t i = 0; i < num_steps; ++i) {
        steps.push_back(static_cast<int64_t>(std::llround(actual * i)));
    }

    return steps;
}

}  // anonymous namespace

/*******************************************************************************/
/*! \fn    LogitsVolume sliding_window(const Volume& data,
                                       const std::vector<std::string>& model_paths,
                                       std::array<int64_t, 3> patch_size,
                                       int64_t num_classes,
                                       int intra_threads,
                                       float step_ratio,
                                       bool verbose,
                                       const std::string& device,
                                       const std::string& trt_cache_dir,
                                       const CudaTuning& cuda_tuning)
    \brief Run sliding-window inference over one canonical volume

    Implementation outline:

      -# Decide which Execution Provider chain to install based on
         \a device, the compile-time SIAMIZE_GPU value, and runtime
         provider-library availability. CUDA EP is attempted before
         CPU on "auto" / "cuda"; TensorRT EP is attempted only when
         requested explicitly.
      -# Pre-compute the Gaussian importance map (compute_gaussian)
         and per-axis tile start indices (compute_steps).
      -# Allocate the (num_classes, Z, Y, X) accumulator and a
         per-voxel weight accumulator.
      -# Loop over folds, creating one ORT session per fold;
         feed (1, 1, Z, Y, X) fp32 tile tensors into Run(); multiply
         the per-class outputs by the Gaussian and add into the
         accumulator. Track per-voxel weight in parallel.
      -# After all folds, divide the per-channel accumulator by the
         weight buffer to obtain final logits.

    \param  data, model_paths, patch_size, num_classes, intra_threads,
            step_ratio, verbose, device, trt_cache_dir, cuda_tuning
            - see sliding.h for parameter semantics
    \return Per-class logits at the same grid as \a data
*/
LogitsVolume sliding_window(const Volume& data,
                            const std::vector<std::string>& model_paths,
                            std::array<int64_t, 3> patch_size,
                            int64_t num_classes,
                            int intra_threads,
                            float step_ratio,
                            bool verbose,
                            const std::string& device,
                            const std::string& trt_cache_dir,
                            const CudaTuning& cuda_tuning) {
    if (model_paths.empty()) {
        throw std::runtime_error("no model paths provided");
    }

    const bool trt_required  = (device == "tensorrt");
    const bool trt_try       = trt_required;
    const bool cuda_required = (device == "cuda");
    const bool cuda_try      = (device == "cuda" || device == "auto"
                                || device == "tensorrt");

#ifndef SIAMIZE_HAS_CUDA

    if (cuda_required || trt_required) {
        throw std::runtime_error(
            "device=" + device + " requested but siamize was built without GPU "
            "support. Rebuild with -DSIAMIZE_GPU=cuda (or =tensorrt) and the "
            "onnxruntime-gpu prebuilt.");
    }

#endif
#ifndef SIAMIZE_HAS_TENSORRT

    if (trt_required) {
        throw std::runtime_error(
            "device=tensorrt requested but siamize was built without TensorRT. "
            "Rebuild with -DSIAMIZE_GPU=tensorrt.");
    }

#endif

    // Resolve the TRT engine cache directory (used only when TRT is enabled).
    std::string trt_cache = trt_cache_dir;

    if (trt_try && trt_cache.empty()) {
        const char* home = std::getenv("HOME");

        if (home == nullptr) {
            home = ".";
        }

        trt_cache = std::string(home) + "/.cache/siamize/trt";
    }

#ifdef SIAMIZE_HAS_TENSORRT

    if (trt_try) {
        try {
            std::filesystem::create_directories(trt_cache);
        } catch (const std::exception& e) {
            if (trt_required) {
                throw std::runtime_error(
                    "failed to create TRT engine cache dir " + trt_cache + ": " + e.what());
            }
        }
    }

#endif

    const int64_t Z = data.shape[0], Y = data.shape[1], X = data.shape[2];
    // Pad if input is smaller than patch in any dim.
    int64_t padZ = std::max<int64_t>(patch_size[0], Z);
    int64_t padY = std::max<int64_t>(patch_size[1], Y);
    int64_t padX = std::max<int64_t>(patch_size[2], X);
    Volume padded;

    if (padZ == Z && padY == Y && padX == X) {
        padded = data;
    } else {
        padded.resize(padZ, padY, padX);
        int64_t lz = (padZ - Z) / 2, ly = (padY - Y) / 2, lx = (padX - X) / 2;

        for (int64_t z = 0; z < Z; ++z) {
            for (int64_t y = 0; y < Y; ++y) {
                std::copy_n(&data.data[z * data.stride(0) + y * data.stride(1)],
                            X,
                            &padded.data[(z + lz) * padded.stride(0) + (y + ly) * padded.stride(1) + lx]);
            }
        }
    }

    int64_t spatZ = padded.shape[0], spatY = padded.shape[1], spatX = padded.shape[2];

    auto stepsZ = compute_steps(spatZ, patch_size[0], step_ratio);
    auto stepsY = compute_steps(spatY, patch_size[1], step_ratio);
    auto stepsX = compute_steps(spatX, patch_size[2], step_ratio);
    int64_t n_tiles = stepsZ.size() * stepsY.size() * stepsX.size();

    siam::log_tag("infer",
                  "image (Z,Y,X) (%" PRId64 ",%" PRId64 ",%" PRId64
                  ")  patch (%" PRId64 ",%" PRId64 ",%" PRId64
                  ")  step %.2f  tiles %" PRId64 "  folds %zu",
                  spatZ, spatY, spatX,
                  patch_size[0], patch_size[1], patch_size[2],
                  step_ratio, n_tiles, model_paths.size());
    (void)verbose;

    auto gauss = compute_gaussian(patch_size);

    LogitsVolume logits;
    logits.resize(num_classes, spatZ, spatY, spatX);
    std::vector<float> weights(static_cast<size_t>(spatZ) * spatY * spatX, 0.0f);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "siam");

    const int64_t patchN = patch_size[0] * patch_size[1] * patch_size[2];
    std::vector<float> input_buf(patchN);

    const std::array<int64_t, 5> input_shape = {1, 1, patch_size[0], patch_size[1], patch_size[2]};
    const std::array<int64_t, 5> output_shape_expected = {1, num_classes, patch_size[0], patch_size[1], patch_size[2]};
    (void)output_shape_expected;

    Ort::AllocatorWithDefaultOptions allocator;
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Decide on the execution providers once. On the first fold, try to
    // register TRT (if requested) then CUDA (if available). Remember the
    // outcome for subsequent folds.
    bool use_trt   = false;
    bool use_cuda  = false;
    bool ep_probed = false;

    for (size_t fi = 0; fi < model_paths.size(); ++fi) {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(intra_threads);
        opts.SetInterOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        const bool will_use_gpu = (use_trt || use_cuda || (!ep_probed && (trt_try || cuda_try)));
        (void)will_use_gpu;

        // Leave ORT's CPU memory arena and memory-pattern optimizer ON
        // (the ORT defaults). An earlier revision disabled both to keep
        // RSS smaller, but profiling on a 64-core Zen2 showed that the
        // disabled path produced ~75M minor page faults and a 43% dTLB
        // miss rate -- per-op mmap/munmap churn that dragged ~88% of
        // system time into the kernel and capped scaling at ~18 cores.
        // With the arena enabled, the same run drops to ~7M page faults,
        // ~27 cores of utilization, and 1.5x wall-time speedup at the
        // cost of ~2x peak RSS (12 GB -> 28 GB on the 18-class SIAM
        // network). Output is byte-identical.

#ifdef SIAMIZE_HAS_TENSORRT

        // TRT EP first so it gets to claim subgraphs it can fuse; CUDA EP
        // (registered below) picks up whatever TRT declines.
        if (trt_try && (!ep_probed || use_trt)) {
            try {
                OrtTensorRTProviderOptionsV2* trt_opts = nullptr;
                Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&trt_opts));

                std::vector<const char*> tkeys;
                std::vector<const char*> tvals;
                tkeys.push_back("trt_engine_cache_enable");
                tvals.push_back("1");
                tkeys.push_back("trt_engine_cache_path");
                tvals.push_back(trt_cache.c_str());
                tkeys.push_back("trt_fp16_enable");
                tvals.push_back("1");
                Ort::ThrowOnError(Ort::GetApi().UpdateTensorRTProviderOptions(
                                      trt_opts, tkeys.data(), tvals.data(), tkeys.size()));
                opts.AppendExecutionProvider_TensorRT_V2(*trt_opts);
                Ort::GetApi().ReleaseTensorRTProviderOptions(trt_opts);

                if (!ep_probed) {
                    use_trt = true;
                    siam::log_tag("trt", "enabled (cache: %s)", trt_cache.c_str());
                    siam::log_cont("first run on a new GPU/TRT version builds engines "
                                   "(~1-5 min/fold)");
                }
            } catch (const Ort::Exception& e) {
                if (trt_required) {
                    throw;   // user explicitly asked for TensorRT
                }

                if (!ep_probed) {
                    siam::log_tag("trt", "unavailable (%s); falling back", e.what());
                }

                use_trt = false;
            }
        }

#endif
#ifdef SIAMIZE_HAS_CUDA

        if (cuda_try && (!ep_probed || use_cuda || use_trt)) {
            try {
                OrtCUDAProviderOptionsV2* cuda_opts = nullptr;
                Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda_opts));

                // Apply CUDA EP tuning overrides. These keys all default
                // to ORT's standard values; we only override the ones the
                // caller asked for. Empty/zero fields are skipped so the
                // common case (no flags) stays bit-identical to before.
                std::vector<std::string> kbuf, vbuf;
                auto add = [&](const char* k, const std::string & v) {
                    kbuf.emplace_back(k);
                    vbuf.emplace_back(v);
                };

                if (cuda_tuning.cudnn_max_workspace == 0) {
                    add("cudnn_conv_use_max_workspace", "0");
                }

                if (cuda_tuning.arena_same_as_req == 1) {
                    add("arena_extend_strategy", "kSameAsRequested");
                }

                if (!cuda_tuning.algo_search.empty()) {
                    add("cudnn_conv_algo_search", cuda_tuning.algo_search);
                }

                if (cuda_tuning.gpu_mem_limit_bytes > 0) {
                    add("gpu_mem_limit", std::to_string(cuda_tuning.gpu_mem_limit_bytes));
                }

                if (cuda_tuning.gpuid != 0) {
                    add("device_id", std::to_string(cuda_tuning.gpuid));
                }

                if (!kbuf.empty()) {
                    std::vector<const char*> kp, vp;

                    for (auto& s : kbuf) {
                        kp.push_back(s.c_str());
                    }

                    for (auto& s : vbuf) {
                        vp.push_back(s.c_str());
                    }

                    Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(
                                          cuda_opts, kp.data(), vp.data(), kp.size()));

                    siam::log_tag("cuda", "tuning:");

                    for (size_t i = 0; i < kbuf.size(); ++i) {
                        siam::log_cont("%s = %s",
                                       kbuf[i].c_str(), vbuf[i].c_str());
                    }
                }

                opts.AppendExecutionProvider_CUDA_V2(*cuda_opts);
                Ort::GetApi().ReleaseCUDAProviderOptions(cuda_opts);

                if (!ep_probed) {
                    use_cuda = true;

                    if (!use_trt) {
                        siam::log_tag("cuda", "enabled");
                    }
                }
            } catch (const Ort::Exception& e) {
                if (cuda_required) {
                    throw;   // user explicitly asked for CUDA
                }

                if (!ep_probed) {
                    siam::log_tag("cuda", "unavailable (%s); using CPU", e.what());
                }

                use_cuda = false;
                // CPU fallback path: keep the arena enabled. See the
                // note at the top of this loop for the profiling
                // analysis behind that choice.
            }
        }

#endif
        ep_probed = true;

#ifdef _WIN32
        auto wpath = widen_path(model_paths[fi]);
        Ort::Session sess(env, wpath.c_str(), opts);
#else
        Ort::Session sess(env, model_paths[fi].c_str(), opts);
#endif

        Ort::AllocatedStringPtr in_name_ptr = sess.GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr out_name_ptr = sess.GetOutputNameAllocated(0, allocator);
        const char* in_name = in_name_ptr.get();
        const char* out_name = out_name_ptr.get();
        const char* in_names[] = {in_name};
        const char* out_names[] = {out_name};

        siam::log_tag("fold", "%zu/%zu  %s",
                      fi + 1, model_paths.size(), model_paths[fi].c_str());

        int64_t tile_idx = 0;

        for (int64_t sz : stepsZ) {
            for (int64_t sy : stepsY) {
                for (int64_t sx : stepsX) {
                    // copy patch into contiguous input buffer
                    for (int64_t z = 0; z < patch_size[0]; ++z) {
                        for (int64_t y = 0; y < patch_size[1]; ++y) {
                            std::copy_n(
                                &padded.data[(sz + z) * padded.stride(0) +
                                                      (sy + y) * padded.stride(1) +
                                                      sx],
                                patch_size[2],
                                &input_buf[z * patch_size[1] * patch_size[2] + y * patch_size[2]]);
                        }
                    }

                    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
                                               mem_info, input_buf.data(), input_buf.size(),
                                               input_shape.data(), input_shape.size());
                    auto outputs = sess.Run(Ort::RunOptions{nullptr},
                                            in_names, &in_tensor, 1,
                                            out_names, 1);
                    const float* pred = outputs[0].GetTensorData<float>();

                    // accumulate: logits[c, z, y, x] += pred[c, ...] * gauss
                    // weights[z, y, x] += gauss   (only first fold)
                    for (int64_t c = 0; c < num_classes; ++c) {
                        float* logit_ch = logits.channel_ptr(c);
                        const float* pred_ch = pred + c * patchN;

                        for (int64_t z = 0; z < patch_size[0]; ++z) {
                            for (int64_t y = 0; y < patch_size[1]; ++y) {
                                float* dst = logit_ch + (sz + z) * spatY * spatX + (sy + y) * spatX + sx;
                                const float* src = pred_ch + z * patch_size[1] * patch_size[2] + y * patch_size[2];
                                const float* gs = gauss.data() + z * patch_size[1] * patch_size[2] + y * patch_size[2];

                                for (int64_t x = 0; x < patch_size[2]; ++x) {
                                    dst[x] += src[x] * gs[x];
                                }
                            }
                        }
                    }

                    if (fi == 0) {
                        for (int64_t z = 0; z < patch_size[0]; ++z) {
                            for (int64_t y = 0; y < patch_size[1]; ++y) {
                                float* wdst = weights.data() + (sz + z) * spatY * spatX + (sy + y) * spatX + sx;
                                const float* gs = gauss.data() + z * patch_size[1] * patch_size[2] + y * patch_size[2];

                                for (int64_t x = 0; x < patch_size[2]; ++x) {
                                    wdst[x] += gs[x];
                                }
                            }
                        }
                    }

                    ++tile_idx;

                    if (tile_idx % 4 == 0 || tile_idx == n_tiles) {
                        siam::log_tag("tile", "%" PRId64 "/%" PRId64,
                                      tile_idx, n_tiles);
                    }
                }
            }
        }
    }

    // Normalize
    const float num_folds = static_cast<float>(model_paths.size());

    for (int64_t c = 0; c < num_classes; ++c) {
        float* lp = logits.channel_ptr(c);
        const int64_t N = spatZ * spatY * spatX;

        for (int64_t i = 0; i < N; ++i) {
            float w = weights[i] * num_folds;
            lp[i] /= w;
        }
    }

    // Crop back if we padded.
    if (spatZ != Z || spatY != Y || spatX != X) {
        int64_t lz = (spatZ - Z) / 2, ly = (spatY - Y) / 2, lx = (spatX - X) / 2;
        LogitsVolume out;
        out.resize(num_classes, Z, Y, X);

        for (int64_t c = 0; c < num_classes; ++c) {
            const float* src = logits.channel_ptr(c);
            float* dst = out.channel_ptr(c);

            for (int64_t z = 0; z < Z; ++z) {
                for (int64_t y = 0; y < Y; ++y) {
                    std::copy_n(src + (z + lz) * spatY * spatX + (y + ly) * spatX + lx,
                                X,
                                dst + z * Y * X + y * X);
                }
            }
        }

        return out;
    }

    return logits;
}

}  // namespace siam
