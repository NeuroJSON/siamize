// siamize: end-to-end SIAM v0.3 brain segmentation in C++.
//
// CLI mirrors the Python siam-pred:
//   siamize -i input.nii.gz -o output.nii.gz --models fold_0.onnx[,fold_1.onnx,...]
//   Optional: --threads N, -v

#include "siam.h"
#include "nifti_io.h"
#include "preprocess.h"
#include "sliding.h"
#include "weights.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using siam::LogitsVolume;
using siam::NiftiImage;
using siam::Volume;

namespace {

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::string acc;

    for (char c : s) {
        if (c == ',') {
            if (!acc.empty()) {
                out.push_back(acc);
            }

            acc.clear();
        } else {
            acc.push_back(c);
        }
    }

    if (!acc.empty()) {
        out.push_back(acc);
    }

    return out;
}

// "0".."9" expands to "fold_<d>_fp16.onnx"; anything else passes through.
std::string expand_fold_shortcut(const std::string& tok) {
    if (tok.size() == 1 && tok[0] >= '0' && tok[0] <= '9') {
        return "fold_" + tok + "_fp16.onnx";
    }

    return tok;
}

void usage(const char* exe) {
    std::fprintf(stderr,
                 "Usage: %s -i input.nii(.gz) -o output.nii.gz --models fold_0.onnx[,fold_1.onnx,...]\n"
                 "          [--threads N] [-v]\n"
                 "\n"
                 "  -i, --input         input NIfTI (.nii or .nii.gz, 3D)\n"
                 "  -o, --output        output label NIfTI (.nii.gz)\n"
                 "      --models        comma-separated .onnx files (one per fold), logits are averaged.\n"
                 "                      Defaults to single-fold 'fold_0_fp16.onnx'. Each entry can be\n"
                 "                      a full path, a basename, or a single digit shortcut (e.g.\n"
                 "                      --models 0,1,2,3,4 expands to fold_<N>_fp16.onnx). Bare\n"
                 "                      basenames are looked up under $SIAMIZE_CACHE_DIR\n"
                 "                      (default $HOME/.cache/siamize/models/) and auto-downloaded\n"
                 "                      from $SIAMIZE_WEIGHTS_BASE_URL on miss.\n"
                 "      --device D      execution provider: auto|cpu|cuda|tensorrt (default auto).\n"
                 "                      auto tries CUDA (if compiled in) then falls back to CPU.\n"
                 "                      tensorrt tries TRT > CUDA > CPU (first run builds engines).\n"
                 "      --trt-cache-dir P  TensorRT engine cache dir (default ~/.cache/siamize/trt).\n"
                 "                         Engines are GPU- and TRT-version-specific; cached on first run.\n"
                 "      --cudnn-max-workspace 0|1  CUDA EP: 0 picks small-workspace cuDNN algos.\n"
                 "                         Use 0 on tight-VRAM GPUs (e.g. 8 GB laptop cards) that\n"
                 "                         OOM at the default workspace ceiling. Default 1 (ORT default).\n"
                 "      --arena-extend power|same  CUDA EP arena_extend_strategy. `same` is\n"
                 "                         kSameAsRequested (grows the BFC arena by exactly what's\n"
                 "                         needed, less wasteful); default is kNextPowerOfTwo.\n"
                 "      --cudnn-algo default|heuristic|exhaustive  cuDNN algorithm search mode.\n"
                 "                         `heuristic` is the smallest-workspace option.\n"
                 "      --gpu-mem-limit N  CUDA EP gpu_mem_limit in bytes (suffix K/M/G accepted,\n"
                 "                         e.g. 6G). Default 0 = no explicit cap.\n"
                 "      --threads N     ORT intra-op threads (default 0 = all available cores; ignored for GPU EPs)\n"
                 "      --patch ZxYxX   patch size, default 256x256x192 (matches SIAM v0.3 plans)\n"
                 "      --spacing v     target isotropic spacing in mm, default 0.75 (SIAM v0.3 training)\n"
                 "      --classes N     number of output classes, default 18 (SIAM v0.3)\n"
                 "  -v, --verbose       print progress\n"
                 "  -h, --help\n",
                 exe);
}

}  // namespace

int main(int argc, char** argv) {
    std::string input_path, output_path, models_csv;
    std::string device = "auto";       // auto | cpu | cuda | tensorrt
    std::string trt_cache_dir;         // empty => $HOME/.cache/siamize/trt
    int threads = 0;     // 0 = auto: std::thread::hardware_concurrency()
    bool verbose = false;
    std::array<int64_t, 3> patch = {256, 256, 192};
    float target_spacing = 0.75f;
    int64_t num_classes = 18;
    siam::CudaTuning cuda_tuning;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&]() {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires an argument\n", a.c_str());
                std::exit(2);
            }

            return std::string(argv[++i]);
        };

        if      (a == "-i" || a == "--input") {
            input_path = need();
        } else if (a == "-o" || a == "--output") {
            output_path = need();
        } else if (a == "--models") {
            models_csv = need();
        } else if (a == "--device") {
            device = need();

            // Allow `trt` as an alias for `tensorrt`.
            if (device == "trt") {
                device = "tensorrt";
            }

            if (device != "auto" && device != "cpu" && device != "cuda"
                    && device != "tensorrt") {
                std::fprintf(stderr,
                             "--device must be auto|cpu|cuda|tensorrt (got '%s')\n",
                             device.c_str());
                return 2;
            }
        } else if (a == "--trt-cache-dir") {
            trt_cache_dir = need();
        } else if (a == "--threads") {
            threads = std::stoi(need());
        } else if (a == "--spacing") {
            target_spacing = std::stof(need());
        } else if (a == "--classes") {
            num_classes = std::stoll(need());
        } else if (a == "--patch") {
            std::string p = need();
            char x;
            std::stringstream ss(p);
            ss >> patch[0] >> x >> patch[1] >> x >> patch[2];
        } else if (a == "--cudnn-max-workspace") {
            cuda_tuning.cudnn_max_workspace = std::stoi(need());
        } else if (a == "--arena-extend") {
            std::string s = need();

            if      (s == "same"  || s == "kSameAsRequested") {
                cuda_tuning.arena_same_as_req = 1;
            } else if (s == "power" || s == "kNextPowerOfTwo") {
                cuda_tuning.arena_same_as_req = 0;
            } else                                            {
                std::fprintf(stderr,
                             "--arena-extend must be power|same (got '%s')\n",
                             s.c_str());
                return 2;
            }
        } else if (a == "--cudnn-algo") {
            std::string s = need();

            // Accept lowercase shortcuts; canonicalize to ORT's string form.
            if      (s == "default"    || s == "DEFAULT")    {
                cuda_tuning.algo_search = "DEFAULT";
            } else if (s == "heuristic"  || s == "HEURISTIC")  {
                cuda_tuning.algo_search = "HEURISTIC";
            } else if (s == "exhaustive" || s == "EXHAUSTIVE") {
                cuda_tuning.algo_search = "EXHAUSTIVE";
            } else {
                std::fprintf(stderr,
                             "--cudnn-algo must be default|heuristic|exhaustive (got '%s')\n",
                             s.c_str());
                return 2;
            }
        } else if (a == "--gpu-mem-limit") {
            // Accept either a raw byte count or a suffix (K/M/G).
            std::string s = need();
            size_t mult = 1;
            char back = s.empty() ? '\0' : s.back();

            if (back == 'K' || back == 'k') {
                mult = 1024ull;
                s.pop_back();
            } else if (back == 'M' || back == 'm') {
                mult = 1024ull * 1024;
                s.pop_back();
            } else if (back == 'G' || back == 'g') {
                mult = 1024ull * 1024 * 1024;
                s.pop_back();
            }

            cuda_tuning.gpu_mem_limit_bytes =
                static_cast<size_t>(std::stoull(s)) * mult;
        } else if (a == "-v" || a == "--verbose") {
            verbose = true;
        } else if (a == "-h" || a == "--help")   {
            usage(argv[0]);
            return 0;
        } else                                   {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            usage(argv[0]);
            return 2;
        }
    }

    if (input_path.empty() || output_path.empty()) {
        usage(argv[0]);
        return 2;
    }

    // Default to single-fold fold_0 when --models isn't given. resolve_model_path
    // below will look in $HOME/.cache/siamize/models/ and, on a miss, curl the
    // weight from https://neurojson.org/siamize/weights/siam_v03/ (overridable
    // via SIAMIZE_WEIGHTS_BASE_URL / SIAMIZE_CACHE_DIR).
    if (models_csv.empty()) {
        models_csv = "fold_0_fp16.onnx";

        if (verbose) {
            std::fprintf(stderr,
                         "--models not given; defaulting to single-fold "
                         "fold_0_fp16.onnx (auto-downloaded if missing).\n");
        }
    }

    auto model_paths = split_csv(models_csv);

    // Resolve every model spec: existing path, cache lookup, or auto-fetch.
    for (auto& m : model_paths) {
        m = expand_fold_shortcut(m);

        try {
            m = siam::resolve_model_path(m, verbose);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "siamize: %s\n", e.what());
            return 3;
        }
    }

    if (threads <= 0) {
        unsigned hc = std::thread::hardware_concurrency();
        threads = (hc > 0) ? static_cast<int>(hc) : 4;

        if (verbose) {
            std::fprintf(stderr,
                         "intra-op threads: %d (auto-detected from "
                         "std::thread::hardware_concurrency())\n",
                         threads);
        }
    } else if (verbose) {
        std::fprintf(stderr, "intra-op threads: %d (user-supplied)\n", threads);
    }

    auto t_start = std::chrono::steady_clock::now();

    if (verbose) {
        std::fprintf(stderr, "input: %s\n", input_path.c_str());
    }

    NiftiImage img = siam::load_nifti_ras(input_path);

    if (verbose) {
        std::fprintf(stderr,
                     "  canon shape (Z, Y, X): (%" PRId64 ", %" PRId64 ", %" PRId64 ")"
                     ", zooms canon (X, Y, Z): (%.3f, %.3f, %.3f)\n",
                     img.volume.shape[0], img.volume.shape[1], img.volume.shape[2],
                     img.zooms_canon[0], img.zooms_canon[1], img.zooms_canon[2]);
    }

    std::array<int64_t, 3> shape_canon = {img.volume.shape[0], img.volume.shape[1], img.volume.shape[2]};

    // crop to nonzero
    auto crop = siam::crop_to_nonzero(img.volume);

    if (verbose) {
        std::fprintf(stderr,
                     "  cropped to (Z, Y, X) (%" PRId64 ", %" PRId64 ", %" PRId64 ")\n",
                     crop.cropped.shape[0], crop.cropped.shape[1], crop.cropped.shape[2]);
    }

    // z-score
    siam::zscore_inplace(crop.cropped);

    // resample to (target_spacing)^3 mm
    // zooms_canon is in (X, Y, Z); for our (Z, Y, X) layout we reverse.
    std::array<float, 3> spacing_zyx = {img.zooms_canon[2], img.zooms_canon[1], img.zooms_canon[0]};
    std::array<float, 3> new_spacing = {target_spacing, target_spacing, target_spacing};
    auto new_shape = siam::compute_new_shape(
    {crop.cropped.shape[0], crop.cropped.shape[1], crop.cropped.shape[2]},
    spacing_zyx, new_spacing);

    if (verbose) {
        std::fprintf(stderr,
                     "  resample to (%" PRId64 ", %" PRId64 ", %" PRId64 ") @ %.3f mm\n",
                     new_shape[0], new_shape[1], new_shape[2], target_spacing);
    }

    // Forward path: cubic Catmull-Rom (matches scipy order=3 to within ~0.1% Dice).
    Volume resampled = siam::resample_cubic(crop.cropped, new_shape[0], new_shape[1], new_shape[2]);
    crop.cropped = Volume{};  // free

    // sliding window. If --device auto chose CUDA/TensorRT and the
    // session aborts mid-Run with an allocation failure (tight VRAM,
    // contention with other GPU clients, etc.), fall back to CPU and
    // retry. --device cuda / --device tensorrt stay strict -- the
    // exception re-raises so the user sees the explicit failure they
    // asked for.
    LogitsVolume logits;

    try {
        logits = siam::sliding_window(
                     resampled, model_paths, patch, num_classes,
                     threads, 0.5f, verbose, device, trt_cache_dir, cuda_tuning);
    } catch (const Ort::Exception& e) {
        std::string msg = e.what();
        bool oom = msg.find("allocate") != std::string::npos
                   || msg.find("Allocate") != std::string::npos
                   || msg.find("out of memory") != std::string::npos
                   || msg.find("CUDA_ERROR_OUT_OF_MEMORY") != std::string::npos;

        if (device == "auto" && oom) {
            std::fprintf(stderr,
                         "  GPU allocation failed: %s\n"
                         "  --device auto falling back to CPU. To force CPU from the\n"
                         "  start, pass `--device cpu`; for tight-VRAM GPUs try also\n"
                         "  `--cudnn-max-workspace 0 --arena-extend same` before\n"
                         "  giving up on GPU.\n",
                         e.what());
            logits = siam::sliding_window(
                         resampled, model_paths, patch, num_classes,
                         threads, 0.5f, verbose, std::string("cpu"), trt_cache_dir, cuda_tuning);
        } else {
            throw;
        }
    }

    resampled = Volume{};  // free

    // resample logits back to cropped (pre-resample) shape, per channel, with trilinear.
    LogitsVolume logits_back;
    logits_back.resize(num_classes,
                       crop.bbox[0][1] - crop.bbox[0][0],
                       crop.bbox[1][1] - crop.bbox[1][0],
                       crop.bbox[2][1] - crop.bbox[2][0]);
    {
        Volume tmp_in;
        tmp_in.shape = {logits.spat[0], logits.spat[1], logits.spat[2]};
        tmp_in.data.resize(static_cast<size_t>(logits.cstride()));

        for (int64_t c = 0; c < num_classes; ++c) {
            std::copy_n(logits.channel_ptr(c), logits.cstride(), tmp_in.data.data());
            Volume v = siam::resample_trilinear(tmp_in,
                                                logits_back.spat[0],
                                                logits_back.spat[1],
                                                logits_back.spat[2]);
            std::copy_n(v.data.data(), v.numel(), logits_back.channel_ptr(c));
        }
    }
    logits = LogitsVolume{};  // free

    // argmax → uint8 labels at cropped shape
    int64_t cZ = logits_back.spat[0], cY = logits_back.spat[1], cX = logits_back.spat[2];
    std::vector<uint8_t> labels_crop(static_cast<size_t>(cZ) * cY * cX, 0);

    for (int64_t z = 0; z < cZ; ++z) {
        for (int64_t y = 0; y < cY; ++y) {
            for (int64_t x = 0; x < cX; ++x) {
                int64_t i = z * cY * cX + y * cX + x;
                int best = 0;
                float bestv = logits_back.channel_ptr(0)[i];

                for (int64_t c = 1; c < num_classes; ++c) {
                    float v = logits_back.channel_ptr(c)[i];

                    if (v > bestv) {
                        bestv = v;
                        best = static_cast<int>(c);
                    }
                }

                labels_crop[i] = static_cast<uint8_t>(best);
            }
        }
    }

    logits_back = LogitsVolume{};

    // Un-crop: paste into canonical shape with zero outside the bbox.
    std::vector<uint8_t> labels_canon(static_cast<size_t>(shape_canon[0]) * shape_canon[1] * shape_canon[2], 0);

    for (int64_t z = 0; z < cZ; ++z) {
        for (int64_t y = 0; y < cY; ++y) {
            std::copy_n(&labels_crop[z * cY * cX + y * cX],
                        cX,
                        &labels_canon[(z + crop.bbox[0][0]) * shape_canon[1] * shape_canon[2] +
                                                            (y + crop.bbox[1][0]) * shape_canon[2] +
                                                            crop.bbox[2][0]]);
        }
    }

    // Save: nifti_io reorients back to original axis order.
    siam::save_nifti_labels(output_path, img, labels_canon.data());

    auto t_end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(t_end - t_start).count();

    if (verbose) {
        std::fprintf(stderr, "saved %s in %.1fs\n", output_path.c_str(), seconds);
    }

    return 0;
}
