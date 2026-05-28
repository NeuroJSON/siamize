/***************************************************************************//**
\file    engine_ort.cpp

@brief   ONNX Runtime backend implementation of the Engine interface

Compiles only when SIAMIZE_HAS_ORT is defined (CMake's
SIAMIZE_BACKEND=ort, the default). Holds everything ORT-specific that
used to live inline in sliding.cpp:

  - Custom ORT log sink (siam_ort_logging_func) that demotes the loud
    "[E:onnxruntime:...]" red-text default to "[warn] ORT: ..." while
    keeping the failure-mode information ORT writes (e.g. "Failed to
    load library libonnxruntime_providers_cuda.so") visible to the
    user.
  - widen_path() for the UTF-16 path argument Windows ORT's Session
    constructor wants.
  - Execution-provider selection: CPU (always), CUDA EP
    (SIAMIZE_HAS_CUDA + runtime cuDNN/cuBLAS), TensorRT EP
    (SIAMIZE_HAS_TENSORRT + runtime libnvinfer), CoreML EP
    (SIAMIZE_HAS_COREML + macOS). Each EP can be required (-c cuda
    fails hard if CUDA isn't available) or attempted (-c auto silently
    falls back).
  - The OrtEngine class that owns one Ort::Session per fold and
    services run_tile() requests.

Threading: OrtEngine sets SetIntraOpNumThreads(intra_threads) and
SetInterOpNumThreads(1). Inter-op stays at 1 because nnUNet-style 3D
U-Nets are deeply serial -- letting ORT parallelize across op
boundaries adds overhead without exposing parallelism. Per-op
parallelism (intra-op) is the only knob that helps.

ORT Env: a single Ort::Env lives for the process lifetime (constructed
lazily on the first make_engine() call). ORT's Env owns the global
thread pool + the log sink; creating multiple Env's in one process is
legal but wasteful. We keep one and reuse it across all folds.
*******************************************************************************/

#ifdef SIAMIZE_HAS_ORT

#include "engine.h"
#include "siam_log.h"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#endif

namespace siam {

namespace {

/*******************************************************************************/
/*! \fn    void ORT_API_CALL siam_ort_logging_func(...)
    \brief Custom ORT log sink that filters/relays ORT's internal messages

    By default ORT writes its own errors and warnings straight to stderr
    in a "[E:onnxruntime:siam, provider_bridge_ort.cc:NNNN] ..." format
    that terminals colour red. Two of those are routine and not actually
    failure conditions:

      - "Failed to load library .../libonnxruntime_providers_cuda.so"
      - "cannot open shared object file" (cuDNN / cuBLAS missing)

    They fire whenever -c auto probes the CUDA EP on a host without
    CUDA drivers installed. siamize handles this case explicitly with
    a user-friendly [cuda] unavailable (...); using CPU line. The
    extra ORT error message just confuses users into thinking the
    run failed.

    This sink suppresses those known-routine messages and routes
    everything else through siam::log_warn so they remain visible
    but not visually alarming.
*/
void ORT_API_CALL siam_ort_logging_func(void* /*param*/,
                                        OrtLoggingLevel severity,
                                        const char* /*category*/,
                                        const char* /*logid*/,
                                        const char* /*code_location*/,
                                        const char* message) {
    if (!message) {
        return;
    }

    // ERROR-severity messages get re-routed through siam::log_warn so
    // they appear as "[warn] ORT: ..." instead of the loud
    // "[E:onnxruntime:siam,...]" default ORT format. The full message
    // body is preserved -- including "Failed to load library
    // libXXX.so" / "cannot open shared object file" lines from the
    // EP-probe path -- so users can see which dependency is missing
    // when an EP fails. INFO/WARNING/VERBOSE chatter is dropped.
    if (severity >= ORT_LOGGING_LEVEL_ERROR) {
        siam::log_warn("ORT: %s", message);
    }
}

#ifdef _WIN32
/*******************************************************************************/
/*! \fn    std::wstring widen_path(const std::string& s)
    \brief Convert UTF-8 to UTF-16 for the Windows ORT Session constructor
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
/*! \fn    Ort::Env& get_env()
    \brief Lazy process-wide singleton Ort::Env

    The ORT C++ API allows multiple Env's, but each one allocates its
    own thread pool + log sink. We construct one with the custom
    siam log sink on first use and keep it for process lifetime.
    Thread-safe via a one-shot call_once.
*/
Ort::Env& get_env() {
    static std::once_flag once;
    static std::unique_ptr<Ort::Env> g_env;
    std::call_once(once, []() {
        g_env.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "siam",
                                 siam_ort_logging_func,
                                 /*logging_param=*/nullptr));
    });
    return *g_env;
}

/// ORT-backed Engine. One Session per fold. EP probe runs once per
/// Engine construction -- with multi-fold runs, each fold incurs the
/// probe again. That's a few-ms cost per fold and negligible against
/// per-fold inference wall time.
class OrtEngine final : public Engine {
  public:
    OrtEngine(const std::string& model_path,
              std::array<int64_t, 3> patch_size,
              int intra_threads,
              const std::string& device,
              const EngineTuning& tuning,
              bool verbose);
    ~OrtEngine() override = default;

    void run_tile(const float* tile, float* logits) override;
    int64_t num_classes() const override {
        return mNumClasses;
    }

  private:
    std::unique_ptr<Ort::Session> mSession;
    Ort::AllocatorWithDefaultOptions mAllocator;
    Ort::MemoryInfo mMemInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Own the input / output names as std::string. ORT's
    // AllocatedStringPtr is unique_ptr<char, Ort::detail::AllocatedFree>
    // whose deleter is not default-constructible, so we can't declare
    // it as an uninitialized member. Copy the name into a std::string
    // at construction time and keep a const char* into it for Run().
    std::string mInName;
    std::string mOutName;
    const char* mInNames[1]  = {nullptr};
    const char* mOutNames[1] = {nullptr};

    std::array<int64_t, 5> mInputShape{};
    int64_t mPatchVoxels = 0;
    int64_t mNumClasses = 0;
};

OrtEngine::OrtEngine(const std::string& model_path,
                     std::array<int64_t, 3> patch_size,
                     int intra_threads,
                     const std::string& device,
                     const EngineTuning& tuning,
                     bool verbose) {
    // Construct (or fetch) the singleton Ort::Env BEFORE any other ORT
    // call. The CUDA / TensorRT / CoreML EP-probe code that follows
    // logs via the default logger; without an Env in place that
    // produces "Attempt to use DefaultLogger but none has been
    // registered" messages instead of the user-friendly "[cuda]
    // unavailable (...)" output we want.
    (void)get_env();

    const bool trt_required    = (device == "tensorrt");
    const bool trt_try         = trt_required;
    const bool cuda_required   = (device == "cuda");
    const bool cuda_try        = (device == "cuda" || device == "auto"
                                  || device == "tensorrt");
    const bool coreml_required = (device == "coreml");
    const bool coreml_try      = (device == "coreml" || device == "auto");

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
#ifndef SIAMIZE_HAS_COREML

    if (coreml_required) {
        throw std::runtime_error(
            "device=coreml requested but siamize was built without CoreML "
            "support. Rebuild on macOS with -DSIAMIZE_GPU=coreml.");
    }

#endif

    // Resolve the TRT engine cache directory (used only when TRT is enabled).
    std::string trt_cache = tuning.trt_cache_dir;

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

    // Resolve the CoreML model-compile cache directory.
    std::string coreml_cache = tuning.coreml_cache_dir;

    if (coreml_try && coreml_cache.empty()) {
        const char* home = std::getenv("HOME");

        if (home == nullptr) {
            home = ".";
        }

        coreml_cache = std::string(home) + "/.cache/siamize/coreml";
    }

#ifdef SIAMIZE_HAS_COREML

    if (coreml_try) {
        try {
            std::filesystem::create_directories(coreml_cache);
        } catch (const std::exception& e) {
            if (coreml_required) {
                throw std::runtime_error(
                    "failed to create CoreML cache dir " + coreml_cache + ": " + e.what());
            }
        }
    }

#endif

    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(intra_threads);
    opts.SetInterOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (!tuning.cpu_arena) {
        // See sliding.cpp pre-refactor notes for the perf trade-off
        // analysis behind this knob: disabling the CPU arena halves
        // peak RSS but inflates wall time ~5x on a 64-core Zen2 due
        // to per-op mmap/munmap churn.
        opts.DisableCpuMemArena();
        opts.DisableMemPattern();
    }

    bool use_trt    = false;
    bool use_cuda   = false;
    bool use_coreml = false;

#ifdef SIAMIZE_HAS_TENSORRT

    if (trt_try) {
        try {
            OrtTensorRTProviderOptionsV2* trt_opts = nullptr;
            Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&trt_opts));
            std::vector<const char*> tkeys, tvals;
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
            use_trt = true;

            if (verbose) {
                siam::log_tag("trt", "enabled (gpuid=%d, cache: %s)",
                              tuning.gpuid, trt_cache.c_str());
                siam::log_cont("first run on a new GPU/TRT version builds engines "
                               "(~1-5 min/fold)");
            }
        } catch (const Ort::Exception& e) {
            if (trt_required) {
                throw;
            }

            if (verbose) {
                siam::log_tag("trt", "unavailable (%s); falling back", e.what());
            }

            use_trt = false;
        }
    }

#endif

#ifdef SIAMIZE_HAS_CUDA

    if (cuda_try && (use_trt || cuda_required || device == "cuda" || device == "auto")) {
        try {
            OrtCUDAProviderOptionsV2* cuda_opts = nullptr;
            Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda_opts));

            std::vector<std::string> kbuf, vbuf;
            auto add = [&](const char* k, const std::string & v) {
                kbuf.emplace_back(k);
                vbuf.emplace_back(v);
            };

            if (tuning.cudnn_max_workspace == 0) {
                add("cudnn_conv_use_max_workspace", "0");
            }

            if (tuning.arena_same_as_req == 1) {
                add("arena_extend_strategy", "kSameAsRequested");
            }

            if (!tuning.algo_search.empty()) {
                add("cudnn_conv_algo_search", tuning.algo_search);
            }

            if (tuning.gpu_mem_limit_bytes > 0) {
                add("gpu_mem_limit", std::to_string(tuning.gpu_mem_limit_bytes));
            }

            if (tuning.gpuid != 0) {
                add("device_id", std::to_string(tuning.gpuid));
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

                if (verbose) {
                    siam::log_tag("cuda", "tuning:");

                    for (size_t i = 0; i < kbuf.size(); ++i) {
                        siam::log_cont("%s = %s", kbuf[i].c_str(), vbuf[i].c_str());
                    }
                }
            }

            opts.AppendExecutionProvider_CUDA_V2(*cuda_opts);
            Ort::GetApi().ReleaseCUDAProviderOptions(cuda_opts);
            use_cuda = true;

            if (verbose && !use_trt) {
                siam::log_tag("cuda", "enabled (gpuid=%d)", tuning.gpuid);
            }
        } catch (const Ort::Exception& e) {
            if (cuda_required) {
                throw;
            }

            if (verbose) {
                siam::log_tag("cuda", "unavailable (%s); using CPU", e.what());
            }

            use_cuda = false;
        }
    }

#endif

#ifdef SIAMIZE_HAS_COREML

    if (coreml_try) {
        try {
            std::unordered_map<std::string, std::string> co_opts;

            switch (tuning.coreml_units) {
                case CoreMLUnits::CPU_ONLY:
                    co_opts["MLComputeUnits"] = "CPUOnly";
                    break;

                case CoreMLUnits::CPU_AND_GPU:
                    co_opts["MLComputeUnits"] = "CPUAndGPU";
                    break;

                case CoreMLUnits::CPU_AND_ANE:
                    co_opts["MLComputeUnits"] = "CPUAndNeuralEngine";
                    break;

                case CoreMLUnits::ALL:
                default:
                    co_opts["MLComputeUnits"] = "ALL";
                    break;
            }

            co_opts["ModelFormat"] = "MLProgram";
            co_opts["ModelCacheDirectory"] = coreml_cache;
            co_opts["RequireStaticInputShapes"] = tuning.coreml_static_shapes ? "1" : "0";
            co_opts["EnableOnSubgraphs"] = "1";
            opts.AppendExecutionProvider("CoreML", co_opts);
            use_coreml = true;

            if (verbose) {
                const char* units_str =
                    tuning.coreml_units == CoreMLUnits::CPU_ONLY    ? "cpu" :
                    tuning.coreml_units == CoreMLUnits::CPU_AND_GPU ? "cpugpu" :
                    tuning.coreml_units == CoreMLUnits::CPU_AND_ANE ? "cpune" :
                    "all";
                siam::log_tag("coreml",
                              "enabled (units=%s, cache=%s, static_shapes=%s)",
                              units_str, coreml_cache.c_str(),
                              tuning.coreml_static_shapes ? "yes" : "no");
                siam::log_cont("first run compiles ONNX -> .mlmodelc "
                               "(~10-30 s, cached after)");
            }
        } catch (const Ort::Exception& e) {
            if (coreml_required) {
                throw;
            }

            if (verbose) {
                siam::log_tag("coreml", "unavailable (%s); using CPU", e.what());
            }

            use_coreml = false;
        }
    }

#endif
    (void)use_trt;
    (void)use_cuda;
    (void)use_coreml;  // some configs unused

#ifdef _WIN32
    auto wpath = widen_path(model_path);
    mSession.reset(new Ort::Session(get_env(), wpath.c_str(), opts));
#else
    mSession.reset(new Ort::Session(get_env(), model_path.c_str(), opts));
#endif

    {
        Ort::AllocatedStringPtr in_name_ptr  = mSession->GetInputNameAllocated(0, mAllocator);
        Ort::AllocatedStringPtr out_name_ptr = mSession->GetOutputNameAllocated(0, mAllocator);
        mInName  = in_name_ptr.get();
        mOutName = out_name_ptr.get();
    }
    mInNames[0]  = mInName.c_str();
    mOutNames[0] = mOutName.c_str();

    mInputShape = {1, 1, patch_size[0], patch_size[1], patch_size[2]};
    mPatchVoxels = patch_size[0] * patch_size[1] * patch_size[2];

    // Read the model's declared output shape for num_classes. The
    // C-channel comes from index 1 of the output's symbolic shape.
    // For SIAM v0.3 it's a static 18; for the dynshape ONNX variant
    // all spatial dims are -1 but the channel dim stays concrete.
    //
    // Hold the TypeInfo + TensorTypeAndShapeInfo as locals through the
    // GetShape() call. ORT C++ wrappers store pointers into the parent
    // C-API object, and the parent is destroyed when the temporary
    // TypeInfo goes out of scope. Chaining the calls produces a
    // dangling vector whose `.size()` reads garbage.
    Ort::TypeInfo out_type_info = mSession->GetOutputTypeInfo(0);
    auto out_tensor_info = out_type_info.GetTensorTypeAndShapeInfo();
    auto out_shape = out_tensor_info.GetShape();

    if (out_shape.size() >= 2 && out_shape[1] > 0) {
        mNumClasses = out_shape[1];
    } else {
        throw std::runtime_error(
            "ORT: cannot determine output num_classes from " + model_path +
            " (output shape rank " + std::to_string(out_shape.size()) +
            "). The model must declare a static channel dim.");
    }
}

void OrtEngine::run_tile(const float* tile, float* logits) {
    // ORT's CreateTensor wants a non-const pointer because it can be
    // used for in-place writes. We don't write to it here; the const
    // cast is safe.
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
                               mMemInfo, const_cast<float*>(tile), static_cast<size_t>(mPatchVoxels),
                               mInputShape.data(), mInputShape.size());

    auto outputs = mSession->Run(Ort::RunOptions{nullptr},
                                 mInNames, &in_tensor, 1,
                                 mOutNames, 1);
    const float* pred = outputs[0].GetTensorData<float>();
    std::copy_n(pred, mNumClasses * mPatchVoxels, logits);
}

}  // anonymous namespace

std::unique_ptr<Engine>
make_engine(const std::string& model_path,
            std::array<int64_t, 3> patch_size,
            int intra_threads,
            const std::string& device,
            const EngineTuning& tuning,
            bool verbose) {
    return std::unique_ptr<Engine>(
               new OrtEngine(model_path, patch_size, intra_threads, device, tuning, verbose));
}

const char* backend_name() {
    return "onnxruntime";
}

}  // namespace siam

#endif  // SIAMIZE_HAS_ORT
