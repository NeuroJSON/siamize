// Sliding-window driver with Gaussian-blended accumulator, mirroring
// siam_ort.py. fp32 accumulators; one fold loaded into ORT at a time.

#include "sliding.h"
#include "siam.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace siam {

namespace {

// Produce a [Z, Y, X] Gaussian importance map matching scipy.ndimage.gaussian_filter
// with sigma = patch / 8 acting on a delta, then normalized so max == value_scaling.
// The simplest exact match is to evaluate the Gaussian closed form (the filter on
// a delta IS a Gaussian); after normalize-by-max the truncation/edge effects of
// scipy's filter become irrelevant.
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

}  // namespace

LogitsVolume sliding_window(const Volume& data,
                            const std::vector<std::string>& model_paths,
                            std::array<int64_t, 3> patch_size,
                            int64_t num_classes,
                            int intra_threads,
                            float step_ratio,
                            bool verbose) {
    if (model_paths.empty()) {
        throw std::runtime_error("no model paths provided");
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

    if (verbose) {
        std::fprintf(stderr,
                     "sliding window: image (%ld, %ld, %ld), patch (%ld, %ld, %ld), step %.2f, %ld tiles, %zu fold(s)\n",
                     spatZ, spatY, spatX, patch_size[0], patch_size[1], patch_size[2],
                     step_ratio, n_tiles, model_paths.size());
    }

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

    for (size_t fi = 0; fi < model_paths.size(); ++fi) {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(intra_threads);
        opts.SetInterOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        opts.DisableCpuMemArena();
        opts.DisableMemPattern();
        Ort::Session sess(env, model_paths[fi].c_str(), opts);

        Ort::AllocatedStringPtr in_name_ptr = sess.GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr out_name_ptr = sess.GetOutputNameAllocated(0, allocator);
        const char* in_name = in_name_ptr.get();
        const char* out_name = out_name_ptr.get();
        const char* in_names[] = {in_name};
        const char* out_names[] = {out_name};

        if (verbose) {
            std::fprintf(stderr, "  fold %zu/%zu: %s\n", fi + 1, model_paths.size(),
                         model_paths[fi].c_str());
        }

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

                    if (verbose && (tile_idx % 4 == 0 || tile_idx == n_tiles)) {
                        std::fprintf(stderr, "    tile %ld/%ld\n", tile_idx, n_tiles);
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
