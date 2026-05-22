#pragma once
#include "siam.h"
#include <array>
#include <string>
#include <vector>

namespace siam {

// Run sliding-window inference over `data` using each .onnx file in
// `model_paths`, averaging logits across folds. Patch size = (Z, Y, X).
//
// `intra_threads` is forwarded to ORT's SessionOptions.
// `verbose` prints per-tile progress.
// `device` is "auto", "cpu", "cuda", or "tensorrt":
//    "auto"     -> try CUDA if compiled in, fall back to CPU silently.
//    "cpu"      -> never try GPU.
//    "cuda"     -> require CUDA; throws if it can't be registered.
//    "tensorrt" -> register TRT EP (with CUDA + CPU as fallbacks within ORT).
//
// `trt_cache_dir` is used only when device == "tensorrt"; an empty value
// resolves to $HOME/.cache/siamize/trt.
//
// Returns logits with shape (num_classes, Z, Y, X) at the same grid as input.
LogitsVolume sliding_window(const Volume& data,
                            const std::vector<std::string>& model_paths,
                            std::array<int64_t, 3> patch_size,
                            int64_t num_classes,
                            int intra_threads,
                            float step_ratio,
                            bool verbose,
                            const std::string& device = "auto",
                            const std::string& trt_cache_dir = "");

}  // namespace siam
