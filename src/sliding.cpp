/***************************************************************************//**
**  \mainpage siamize - native C++ port of SIAM v0.3 brain segmentation
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
**
**  \section slicense License
**          Apache License 2.0, see LICENSE for details
*******************************************************************************/

/***************************************************************************//**
\file    sliding.cpp
\brief   Sliding-window inference: tile generation, fold averaging

Mirrors the Python reference in py/siam_ref.py / siam_ort.py:

  - Build a Gaussian-weighted importance map sized to the patch
    (matches scipy.ndimage.gaussian_filter(sigma=patch/8) on a delta,
    normalized so the max is 10x and zero entries are floored to the
    smallest nonzero weight to avoid voxels with no coverage).
  - Generate evenly-spaced tile starts per axis at `step_ratio *
    patch_size` stride, with the last tile snug against the far edge.
  - For each fold: construct an Engine (ORT or MNN, depending on
    build-time SIAMIZE_BACKEND), feed (1, 1, Z, Y, X) fp32 patches via
    Engine::run_tile(), accumulate logits * Gaussian, track per-voxel
    weight, then divide at the end.

The backend selection (CPU, CUDA, TensorRT, CoreML for ORT; CPU,
OpenCL, Vulkan, Metal for MNN) is encapsulated in the Engine layer.
This file knows nothing about either runtime -- it just allocates
buffers, tiles, and accumulates.

fp32 accumulators throughout; only one fold's Engine lives in memory
at any moment so peak working-set stays bounded.
*******************************************************************************/

#include "sliding.h"
#include "engine.h"
#include "siam.h"
#include "siam_log.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace siam {

namespace {

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
*/
std::vector<float> compute_gaussian(std::array<int64_t, 3> patch,
                                    float sigma_scale = 1.0f / 8.0f,
                                    float value_scaling = 10.0f) {
    std::vector<float> g(patch[0] * patch[1] * patch[2]);
    const float sz = patch[0] * sigma_scale;
    const float sy = patch[1] * sigma_scale;
    const float sx = patch[2] * sigma_scale;
    // siam_ref/siam_ort use `s // 2` as center -> integer center.
    const float icz = static_cast<float>(patch[0] / 2);
    const float icy = static_cast<float>(patch[1] / 2);
    const float icx = static_cast<float>(patch[2] / 2);

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

    for (auto& v : g) {
        if (v == 0.0f) {
            v = static_cast<float>(nz_min);
        }
    }

    return g;
}

/*******************************************************************************/
/*! \fn    std::vector<int64_t> compute_steps(int64_t image_size,
                                              int64_t patch,
                                              float step_ratio)
    \brief Generate tile start indices along one axis (matches nnUNet)
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
/*! \fn    LogitsVolume sliding_window(...)
    \brief Run sliding-window inference over one canonical volume

    Backend-agnostic. The per-fold inference is delegated to an Engine
    (siam::make_engine() picks the build-time backend: ORT or MNN).
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
                            const EngineTuning& engine_tuning) {
    if (model_paths.empty()) {
        throw std::runtime_error("no model paths provided");
    }

    // Carry the legacy trt_cache_dir parameter through EngineTuning so
    // make_engine() sees one consistent options struct. trt_cache_dir
    // is only meaningful to OrtEngine; MnnEngine ignores it.
    EngineTuning tuning = engine_tuning;

    if (tuning.trt_cache_dir.empty()) {
        tuning.trt_cache_dir = trt_cache_dir;
    }

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
                  ")  step %.2f  tiles %" PRId64 "  weights %zu  backend %s",
                  spatZ, spatY, spatX,
                  patch_size[0], patch_size[1], patch_size[2],
                  step_ratio, n_tiles, model_paths.size(),
                  siam::backend_name());
    (void)verbose;

    auto gauss = compute_gaussian(patch_size);

    LogitsVolume logits;
    logits.resize(num_classes, spatZ, spatY, spatX);
    std::vector<float> weights(static_cast<size_t>(spatZ) * spatY * spatX, 0.0f);

    const int64_t patchN = patch_size[0] * patch_size[1] * patch_size[2];
    std::vector<float> input_buf(patchN);
    std::vector<float> pred_buf(num_classes * patchN);

    for (size_t fi = 0; fi < model_paths.size(); ++fi) {
        siam::log_tag("weight", "%zu/%zu  %s",
                      fi + 1, model_paths.size(), model_paths[fi].c_str());

        // One Engine per fold. The Engine owns the session (ORT
        // Ort::Session / MNN MNN::Session) for this fold's weights.
        // Going out of scope at the end of each fold iteration
        // releases the session memory before the next fold loads.
        auto _me0 = std::chrono::steady_clock::now();
        std::unique_ptr<Engine> engine = siam::make_engine(
                                             model_paths[fi], patch_size, intra_threads, device, tuning, verbose);

        if (std::getenv("SIAMIZE_PHASE") || std::getenv("SIAMIZE_OP_PROFILE")) {
            double me = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - _me0).count();
            fprintf(stderr, "[sliding] fold %zu make_engine (model load + session + resize): %.1f ms\n",
                    fi, me);
        }

        // Defensive: the model's declared num_classes must agree with
        // the caller's. If it doesn't, the accumulator math silently
        // misaligns.
        if (engine->num_classes() != num_classes) {
            throw std::runtime_error(
                "fold " + std::to_string(fi) + " (" + model_paths[fi] +
                ") declares num_classes=" + std::to_string(engine->num_classes()) +
                " but caller passed num_classes=" + std::to_string(num_classes));
        }

        int64_t tile_idx = 0;
        const bool prof = std::getenv("SIAMIZE_PHASE") != nullptr ||
                          std::getenv("SIAMIZE_OP_PROFILE") != nullptr;
        double t_gather = 0, t_run = 0, t_accum = 0;
        using sclk = std::chrono::steady_clock;

        // Opt into GPU-side accumulation when the engine supports it (MNN
        // OpenCL, fp32, BUFFER mode): each tile's Gaussian-weighted logits are
        // accumulated on the GPU into a full-volume device buffer, so the
        // per-tile device->host readback and the host accumulate loop are
        // replaced by one per-fold readback in gpu_accum_finish(). Falls back
        // to the host path if begin() fails (e.g. device alloc too big).
        const bool use_gpu_accum =
            engine->gpu_accum_supported() &&
            engine->gpu_accum_begin({spatZ, spatY, spatX}, gauss.data());

        if (prof && engine->gpu_accum_supported()) {
            fprintf(stderr, "[sliding] fold %zu GPU accumulation: %s\n",
                    fi, use_gpu_accum ? "ON" : "begin() failed -> host path");
        }

        for (int64_t sz : stepsZ) {
            for (int64_t sy : stepsY) {
                for (int64_t sx : stepsX) {
                    auto _g0 = sclk::now();

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

                    auto _r0 = sclk::now();

                    if (use_gpu_accum) {
                        // Forward + accumulate on the GPU; no readback here.
                        engine->gpu_accum_tile(input_buf.data(), sz, sy, sx);
                    } else {
                        engine->run_tile(input_buf.data(), pred_buf.data());
                    }

                    auto _a0 = sclk::now();

                    // accumulate: logits[c, z, y, x] += pred[c, ...] * gauss
                    // weights[z, y, x] += gauss   (only first fold)
                    //
                    // Parallelized across output channels: each channel writes
                    // a disjoint region of `logits` (distinct channel_ptr), so
                    // the threads never touch the same memory -- no locking and
                    // bit-identical results vs the serial version. The inner x
                    // loop is a contiguous fma the compiler auto-vectorizes.
                    //
                    // Skipped under GPU accumulation: gpu_accum_tile() already
                    // folded this tile's weighted logits into the device buffer.
                    if (!use_gpu_accum) {
                        const auto accum_channels = [&](int64_t c0, int64_t c1) {
                            for (int64_t c = c0; c < c1; ++c) {
                                float* logit_ch = logits.channel_ptr(c);
                                const float* pred_ch = pred_buf.data() + c * patchN;

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
                        };

                        int nthreads = intra_threads > 0
                                       ? intra_threads
                                       : static_cast<int>(std::thread::hardware_concurrency());
                        nthreads = std::max(1, std::min<int>(nthreads, static_cast<int>(num_classes)));

                        if (nthreads == 1) {
                            accum_channels(0, num_classes);
                        } else {
                            std::vector<std::thread> pool;
                            pool.reserve(nthreads - 1);
                            const int64_t per = (num_classes + nthreads - 1) / nthreads;

                            for (int t = 1; t < nthreads; ++t) {
                                int64_t c0 = std::min<int64_t>(num_classes, t * per);
                                int64_t c1 = std::min<int64_t>(num_classes, c0 + per);

                                if (c0 < c1) {
                                    pool.emplace_back(accum_channels, c0, c1);
                                }
                            }

                            accum_channels(0, std::min<int64_t>(num_classes, per));

                            for (auto& th : pool) {
                                th.join();
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

                    auto _e0 = sclk::now();

                    if (prof) {
                        t_gather += std::chrono::duration<double, std::milli>(_r0 - _g0).count();
                        t_run    += std::chrono::duration<double, std::milli>(_a0 - _r0).count();
                        t_accum  += std::chrono::duration<double, std::milli>(_e0 - _a0).count();
                    }

                    ++tile_idx;

                    // TTY-aware progress bar (self-redrawing on a terminal,
                    // one-line-per-tile when piped / inside a MEX).
                    siam::log_progress("tile",
                                       static_cast<long long>(tile_idx),
                                       static_cast<long long>(n_tiles));
                }
            }
        }

        // Under GPU accumulation, pull this fold's full-volume weighted logits
        // back once and add them into the cross-fold host accumulator.
        if (use_gpu_accum) {
            auto _f0 = sclk::now();
            engine->gpu_accum_finish(logits.data.data());

            if (prof) {
                t_run += std::chrono::duration<double, std::milli>(
                             sclk::now() - _f0).count();
            }
        }

        if (prof) {
            fprintf(stderr,
                    "\n[sliding] fold %zu CPU/GPU split over %lld tiles:\n"
                    "  patch gather (CPU)      : %8.1f ms  (%6.1f ms/tile)\n"
                    "  run_tile (GPU+xfer)     : %8.1f ms  (%6.1f ms/tile)\n"
                    "  gaussian accumulate(CPU): %8.1f ms  (%6.1f ms/tile)\n",
                    fi, (long long)tile_idx,
                    t_gather, t_gather / std::max<int64_t>(1, tile_idx),
                    t_run,    t_run    / std::max<int64_t>(1, tile_idx),
                    t_accum,  t_accum  / std::max<int64_t>(1, tile_idx));
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
