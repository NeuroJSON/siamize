// siamize: end-to-end SIAM v0.3 brain segmentation in C++.
//
// CLI mirrors the Python siam-pred:
//   siamize -i input.nii.gz -o output.nii.gz --models fold_0.onnx[,fold_1.onnx,...]
//   Optional: --threads N, -v

#include "siam.h"
#include "nifti_io.h"
#include "preprocess.h"
#include "sliding.h"

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

void usage(const char* exe) {
    std::fprintf(stderr,
                 "Usage: %s -i input.nii(.gz) -o output.nii.gz --models fold_0.onnx[,fold_1.onnx,...]\n"
                 "          [--threads N] [-v]\n"
                 "\n"
                 "  -i, --input         input NIfTI (.nii or .nii.gz, 3D)\n"
                 "  -o, --output        output label NIfTI (.nii.gz)\n"
                 "      --models        comma-separated .onnx files (one per fold), logits are averaged\n"
                 "      --device D      execution provider: auto|cpu|cuda|tensorrt (default auto).\n"
                 "                      auto tries CUDA (if compiled in) then falls back to CPU.\n"
                 "                      tensorrt tries TRT > CUDA > CPU (first run builds engines).\n"
                 "      --trt-cache-dir P  TensorRT engine cache dir (default ~/.cache/siamize/trt).\n"
                 "                         Engines are GPU- and TRT-version-specific; cached on first run.\n"
                 "      --threads N     ORT intra-op threads (default 8; ignored for GPU EPs)\n"
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
    int threads = 8;
    bool verbose = false;
    std::array<int64_t, 3> patch = {256, 256, 192};
    float target_spacing = 0.75f;
    int64_t num_classes = 18;

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

    if (input_path.empty() || output_path.empty() || models_csv.empty()) {
        usage(argv[0]);
        return 2;
    }

    auto model_paths = split_csv(models_csv);

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

    // sliding window
    LogitsVolume logits = siam::sliding_window(
                              resampled, model_paths, patch, num_classes,
                              threads, 0.5f, verbose, device, trt_cache_dir);
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
