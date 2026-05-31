/***************************************************************************//**
\file    engine_mnn.cpp

@brief   MNN backend implementation of the Engine interface

Compiles only when SIAMIZE_HAS_MNN is defined (CMake's SIAMIZE_BACKEND=mnn).
The MNN runtime must be the patched build from
tools/mnn_probe/patches/mnn-int64-fixes-for-siam.patch -- stock MNN 3.5.0
silently produces wrong logits on SIAM-class workloads (>2 GB
intermediate tensors).

Threading
---------
numThread is capped at 2 on Linux x86_64 because numThread>=4 crashes
in CPURaster::onResize::lambda#4 on this workload (see patches/README.md).
The cap is a workaround until the upstream MT bug is fixed.

Device support
--------------
  "auto"   -> "opencl" if MNN was built with MNN_OPENCL=ON, else "cpu"
  "cpu"    -> MNN_FORWARD_CPU
  "opencl" -> MNN_FORWARD_OPENCL  (NVIDIA via OpenCL ICD, AMD, Intel iGPU)
  "vulkan" -> MNN_FORWARD_VULKAN  (if MNN built with MNN_VULKAN=ON)
  "metal"  -> MNN_FORWARD_METAL   (macOS only; MNN_METAL=ON)

OpenCL on SIAM is functionally correct (verified 99.67 % voxel
agreement vs ORT on a real T1, see tools/mnn_probe/patches/README.md)
but currently ~6x slower than ORT-CUDA per warm tile due to MNN
OpenCL's per-op kernel-launch overhead from the Conv3D-to-Conv2D
decomposition. Document this limitation in your docs.
*******************************************************************************/

#ifdef SIAMIZE_HAS_MNN

#include "engine.h"
#include "siam_log.h"

// Define MNN_USER_SET_DEVICE before including MNNSharedContext.h so the
// MNNDeviceContext struct (used for OpenCL/Vulkan device selection via
// `BackendConfig::sharedContext`) becomes visible. The OpenCL backend
// itself already defines this in OpenCLBackend.hpp, but we need the
// definition here too to construct the struct at our end.
#define MNN_USER_SET_DEVICE
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/MNNForwardType.h>
#include <MNN/MNNSharedContext.h>

// GPU-side sliding accumulator (see gpu_accum_* below). Reaches into MNN's
// OpenCL backend internals to run a custom accumulate kernel on the same
// command queue / context that ran inference, against the session output's
// NC4DHW4 device buffer. Compiled only when CMake locates an MNN source tree
// (SIAMIZE_MNN_SRC) to supply these non-public headers; otherwise the engine
// just reports gpu_accum_supported()==false and the host path is used.
#ifdef SIAMIZE_MNN_GPU_ACCUM
    #include "core/OpenCLBackend.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace siam {

namespace {

/// Map a device string to an MNN forward type, honoring build-time
/// backend availability. Anything we can't resolve falls back to CPU.
MNNForwardType resolve_forward_type(const std::string& device, bool verbose) {
    if (device == "cpu") {
        return MNN_FORWARD_CPU;
    }

    if (device == "opencl") {
        return MNN_FORWARD_OPENCL;
    }

    if (device == "vulkan") {
        return MNN_FORWARD_VULKAN;
    }

    if (device == "metal") {
#ifdef __APPLE__
        return MNN_FORWARD_METAL;
#else

        if (verbose) {
            siam::log_warn("[mnn] device=metal requested on non-Apple host; "
                           "falling back to cpu");
        }

        return MNN_FORWARD_CPU;
#endif
    }

    if (device == "auto" || device.empty()) {
        // Prefer OpenCL when the user didn't specify; we built with
        // MNN_OPENCL=ON. The OpenCL backend gracefully reports
        // unavailability if no ICD is loaded, and MNN will then auto-
        // fall back to CPU at createSession time.
        return MNN_FORWARD_OPENCL;
    }

    if (verbose) {
        siam::log_warn("[mnn] unknown device='%s'; falling back to cpu",
                       device.c_str());
    }

    return MNN_FORWARD_CPU;
}

/// Resolve a per-host path for MNN's auto-tuning cache file.
///
/// MNN's OpenCL / Vulkan / Metal backends benchmark multiple kernel
/// variants (workgroup size, vectorization, memory-access pattern)
/// per op the first time they encounter it, then persist the winners
/// to a binary cache file. Subsequent runs skip the tuning step and
/// jump straight to the optimal kernel. Typical 2-3x speedup on
/// OpenCL backends, with no precision impact.
///
/// We bucket the cache by (forward-type, precision) so swapping
/// `--device opencl` for `--device vulkan` or flipping precision
/// doesn't reuse stale tuning data. The cache key MNN uses
/// internally also encodes per-op input shapes, so different patch
/// sizes get their own entries inside the same file.
///
/// Returns the empty string if the cache directory cannot be created
/// (caller treats this as "tuning disabled, just run as before").
///
/// The filename is keyed by device + precision + a model-architecture tag
/// (the .mnn file size rounded to MB). Rationale: MNN's compiled-kernel
/// cache depends on the op graph / shapes / precision / device, NOT on the
/// weight values -- so the five SIAM folds (identical architecture, hence
/// identical .mnn byte size) share one cache and never recompile when
/// switching folds, while a different model family (e.g. mnn_n3d ~540 MB vs
/// mnn_i8a ~150 MB) gets a separate file. This fixes two coupled MNN issues
/// at once:
///   1. The tune/binary cache being keyed only by device+precision, so a
///      different architecture's entries contaminate ours (BUG-COMMON-4).
///   2. Interpreter::updateCacheFile's only-grow heuristic
///      (core/Interpreter.cpp: `buffer.second > lastCacheSize`), which lets
///      a stale-but-larger cache permanently block a valid-but-smaller
///      rewrite -- observed as a persistent ~3.5 s/fold kernel recompile on
///      an RTX 5090. With per-architecture filenames, each distinct input
///      gets its own file that grows monotonically from empty, and any
///      residual collision self-heals via MNN's `!valid` rewrite path
///      (loadCache rejects a mismatched blob and writes a fresh one).
std::string default_mnn_cache_path(MNNForwardType type,
                                   MNN::BackendConfig::PrecisionMode prec,
                                   const std::string& model_path = std::string()) {
    namespace fs = std::filesystem;

    fs::path base;
    const char* env = std::getenv("SIAMIZE_CACHE_DIR");

    if (env && *env) {
        base = env;
    } else {
        const char* home = std::getenv("HOME");

        if (home && *home) {
            base = fs::path(home) / ".cache" / "siamize";
        } else {
            return "";
        }
    }

    fs::path dir = base / "mnn-tune";
    std::error_code ec;
    fs::create_directories(dir, ec);

    if (ec) {
        return "";
    }

    const char* dev_tag = "cpu";

    switch (type) {
        case MNN_FORWARD_OPENCL:
            dev_tag = "opencl";
            break;

        case MNN_FORWARD_VULKAN:
            dev_tag = "vulkan";
            break;

        case MNN_FORWARD_METAL:
            dev_tag = "metal";
            break;

        default:
            break;
    }

    const char* prec_tag = (prec == MNN::BackendConfig::Precision_Low)
                           ? "fp16" : "fp32";

    // Model-architecture tag: .mnn size in MB. Folds share it (identical
    // size); distinct model families differ. Empty/absent model_path falls
    // back to the un-keyed name for backward compatibility.
    std::string arch_tag;

    if (!model_path.empty()) {
        std::error_code mec;
        std::uintmax_t bytes = fs::file_size(model_path, mec);

        if (!mec) {
            arch_tag = "-" + std::to_string(bytes / (1024 * 1024)) + "m";
        }
    }

    return (dir / (std::string(dev_tag) + "-" + prec_tag + arch_tag + ".cache")).string();
}


/// MNN-backed Engine. One Interpreter + one Session per fold. The
/// patch shape is fixed at construction; sliding.cpp always calls
/// run_tile() with that same shape.
class MnnEngine final : public Engine {
  public:
    MnnEngine(const std::string& model_path,
              std::array<int64_t, 3> patch_size,
              int intra_threads,
              const std::string& device,
              int gpu_platform,
              int gpuid,
              bool mnn_fp16,
              bool mnn_buffer,
              bool verbose);
    ~MnnEngine() override;

    void run_tile(const float* tile, float* logits) override;
    int64_t num_classes() const override {
        return mNumClasses;
    }

    bool gpu_accum_supported() const override;
    bool gpu_accum_begin(const std::array<int64_t, 3>& full,
                         const float* gauss) override;
    void gpu_accum_tile(const float* tile,
                        int64_t sz, int64_t sy, int64_t sx) override;
    void gpu_accum_finish(float* logits_out) override;

  private:
    // MNN's Interpreter manages model storage; Session is per-execution
    // state (allocated buffers, op graphs). One per fold.
    std::shared_ptr<MNN::Interpreter> mInterpreter;
    MNN::Session* mSession = nullptr;   ///< owned by mInterpreter

    MNN::Tensor* mInput = nullptr;      ///< owned by mInterpreter
    MNN::Tensor* mOutput = nullptr;     ///< owned by mInterpreter

    // For copyFromHostTensor / copyToHostTensor we use these scratch
    // host-side Tensors (no device-host copies happen if backend is
    // CPU; the OpenCL backend handles the H<->D transfer).
    std::unique_ptr<MNN::Tensor> mHostInput;
    std::unique_ptr<MNN::Tensor> mHostOutput;

    std::array<int64_t, 3> mPatch;
    int64_t mNumClasses = 0;
    size_t mTileFloats = 0;
    size_t mLogitFloats = 0;

    // Path to the MNN auto-tuning cache file (empty when caching is
    // disabled / not set up successfully). Populated by the ctor;
    // dtor calls updateCacheFile() before tearing down the session
    // so that the cumulative tuning gathered during this run lands
    // on disk for the next process to reuse.
    std::string mCachePath;

    // Config captured at construction, used by gpu_accum_supported().
    MNNForwardType mForwardType = MNN_FORWARD_CPU;
    MNN::BackendConfig::PrecisionMode mPrecision =
        MNN::BackendConfig::Precision_High;
    bool mBufferMode = false;

#ifdef SIAMIZE_MNN_GPU_ACCUM
    // GPU sliding-accumulator state (one fold). All device objects live on
    // the OpenCL backend's own context/queue so the accumulate kernel runs
    // in-order right after each runSession, reading the output's NC4DHW4
    // device buffer directly -- no per-tile readback.
    MNN::OpenCLRuntime* mClRuntime = nullptr;  // borrowed from backend
    std::shared_ptr<MNN::KernelWrap> mAccumKernel;
    std::unique_ptr<::cl::Buffer> mAccumBuf;   // [C * channelStride] fp32
    std::unique_ptr<::cl::Buffer> mGaussBuf;   // [Z*Y*X] fp32
    std::unique_ptr<::cl::Buffer> mPinnedBuf;  // CL_MEM_ALLOC_HOST_PTR staging
    float* mPinnedPtr = nullptr;               // persistent map of mPinnedBuf
    std::array<int64_t, 3> mFull{0, 0, 0};     // padded (Z,Y,X)
    int64_t mChannelStride = 0;                // spatZ*spatY*spatX
    bool mAccumActive = false;
#endif
};

MnnEngine::MnnEngine(const std::string& model_path,
                     std::array<int64_t, 3> patch_size,
                     int intra_threads,
                     const std::string& device,
                     int gpu_platform,
                     int gpuid,
                     bool mnn_fp16,
                     bool mnn_buffer,
                     bool verbose)
    : mPatch(patch_size) {
    // Force MNN's geometry layer to skip Convolution3D / ConvTranspose3D
    // decomposition (Conv3D -> ~27 Conv2D + im2col + Eltwise + Raster).
    // The libMNN we link against carries our native Conv3D / Deconv3D
    // executors for both OpenCL (Conv3DBufExecution) and CPU
    // (CPUConvolution3D), and those bypass the ~400 GB workspace
    // explosion the geometry path produces on SIAM-class workloads.
    // Set the env vars at engine construction time so callers don't have
    // to remember to export them. setenv with overwrite=0 lets a user
    // still override by setting SIAM_DISABLE_GEOM_*=0 explicitly.
    ::setenv("SIAM_DISABLE_GEOM_CONV3D",   "1", 0);
    ::setenv("SIAM_DISABLE_GEOM_DECONV3D", "1", 0);

    const bool _prof = std::getenv("SIAMIZE_PHASE") || std::getenv("SIAMIZE_OP_PROFILE");
    auto _c0 = std::chrono::steady_clock::now();
    mInterpreter.reset(MNN::Interpreter::createFromFile(model_path.c_str()),
                       MNN::Interpreter::destroy);

    if (!mInterpreter) {
        throw std::runtime_error("MNN: failed to load model: " + model_path);
    }

    if (_prof) {
        fprintf(stderr, "[mnn-ctor] createFromFile (parse %s): %.1f ms\n",
                model_path.c_str(),
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - _c0).count());
    }

    MNN::ScheduleConfig cfg;
    cfg.type = resolve_forward_type(device, verbose);
    const bool gpu_dev = (cfg.type == MNN_FORWARD_OPENCL ||
                          cfg.type == MNN_FORWARD_VULKAN ||
                          cfg.type == MNN_FORWARD_METAL);

    // ScheduleConfig::numThread is a UNION with `mode` and means very
    // different things depending on the backend:
    //
    //   CPU   : literal host thread count for MNN's thread pool.
    //   GPU   : bitfield of MNN_GPU_TUNING_* flags (Interpreter.h).
    //           Default is MNN_GPU_TUNING_WIDE (= 4) which MNN's own
    //           header recommends. Setting it to 2 silently selects
    //           MNN_GPU_TUNING_HEAVY which the same header documents
    //           as "usually not suggested" -- worse perf AND triggers
    //           "Error tunning info" cache-load spam from configurations
    //           that fail to tune cleanly. Picking WIDE here was the
    //           single biggest siamize-MNN bug.
    if (gpu_dev) {
        // OR-ed bitfield: tuning mode + (optional) memory layout.
        //
        // MNN_GPU_TUNING_FAST is the default tuning level. Empirical
        // sweep on SIAM v0.3 + RTX 5090 + RTX 2080 SUPER + Titan V
        // (NeuroJSON/MNN siam-opencl-conv3d branch):
        //
        //   tune         5090 256^3   5090 192^3   2080S 256^3   2080S 192^3
        //   FAST (this)  12.7 s warm  16.0 s cold  44.8 s warm   61.6 s warm
        //   WIDE         12.7 s warm  49.2 s warm  44.7 s warm   185  s warm
        //   NONE         18.2 s       23.1 s       75.8 s        114  s
        //   NORMAL       --           --           --            138  s
        //   HEAVY        --           --           --            186  s
        //
        // FAST is tied with WIDE on the patch sizes where WIDE wins,
        // and 3 x faster on the patch sizes where WIDE silently
        // regresses, and it cuts the cold-cache tuning cost from
        // ~5 min (WIDE) to ~10 s. WIDE's wider LWS search picks
        // winners that benchmark fast in isolation but lose under
        // the streaming kernel-launch pattern of SIAM's 14 k
        // launches/tile -- FAST's narrower search avoids that
        // overfitting.
        //
        // Override via SIAMIZE_TUNE=none|fast|normal|wide|heavy.
        //
        // MNN_GPU_MEMORY_BUFFER (opt-in via --mnn-buffer): force
        // BUFFER memory layout instead of MNN's NVIDIA / AMD /
        // Adreno default of IMAGE. The native Conv3D / Deconv3D
        // OpenCL path (NeuroJSON/MNN siam-opencl-conv3d branch)
        // requires this flag -- the IMAGE creator for those ops
        // isn't implemented.
        int mode = MNN_GPU_TUNING_FAST;
        const char* tune_env = std::getenv("SIAMIZE_TUNE");

        if (tune_env) {
            if      (!strcmp(tune_env, "none")) {
                mode = MNN_GPU_TUNING_NONE;
            } else if (!strcmp(tune_env, "fast")) {
                mode = MNN_GPU_TUNING_FAST;
            } else if (!strcmp(tune_env, "normal")) {
                mode = MNN_GPU_TUNING_NORMAL;
            } else if (!strcmp(tune_env, "wide")) {
                mode = MNN_GPU_TUNING_WIDE;
            } else if (!strcmp(tune_env, "heavy")) {
                mode = MNN_GPU_TUNING_HEAVY;
            }
        }

        if (mnn_buffer) {
            mode |= MNN_GPU_MEMORY_BUFFER;
        }

        cfg.numThread = mode;
    } else {
        // CPU path: previously capped at 2 because numThread >= 4 used to
        // segfault in CPURaster::lambda#4 on the geometry-decomposed
        // Conv3D path (see tools/mnn_probe/patches/README.md). With our
        // native-Conv3D CPU executor (SIAM_DISABLE_GEOM_CONV3D=1, set
        // below) the model bypasses CPURaster entirely for Conv3D, so
        // the cap is no longer needed.
        cfg.numThread = std::max(1, intra_threads);
    }

    // "high" -> fp32 accumulators; "low" -> fp16. We default to high
    // because SIAM cascades fp16 noise across 76 layers and we already
    // see ~6 % logit drift even at high precision; low would compound.
    MNN::BackendConfig bcfg;
    // Precision_High = fp32 accumulators (safe default). Precision_Low
    // = fp16 compute -- engages Tensor Cores on Volta+ NVIDIA OpenCL,
    // half-throughput SIMD on AMD / Intel iGPU, and falls back to fp32
    // on devices that report `fp16:0`. Toggle is plumbed all the way
    // out to the CLI's --mnn-fp16 flag so users can measure the
    // speedup/accuracy trade-off on their own hardware.
    bcfg.precision = mnn_fp16
                     ? MNN::BackendConfig::Precision_Low
                     : MNN::BackendConfig::Precision_High;

    // SIAMIZE_PRECISION=high|normal|low override (debug/measurement).
    // Precision_Normal is fp16 storage + fp32 accumulator (the cuDNN
    // Tensor-Core-style mode); Precision_Low is full fp16. The --mnn-fp16
    // flag picks Low; this env var exposes Normal so we can measure the
    // bandwidth-only gain without the accuracy hit from fp16 accumulation.
    const char* prec_env = std::getenv("SIAMIZE_PRECISION");

    if (prec_env) {
        if      (!strcmp(prec_env, "high")) {
            bcfg.precision = MNN::BackendConfig::Precision_High;
        } else if (!strcmp(prec_env, "normal")) {
            bcfg.precision = MNN::BackendConfig::Precision_Normal;
        } else if (!strcmp(prec_env, "low")) {
            bcfg.precision = MNN::BackendConfig::Precision_Low;
        }
    }

    cfg.backendConfig = &bcfg;

    // GPU device selection. MNN's OpenCL backend reads
    // platformId / deviceId from BackendConfig::sharedContext
    // (an MNNDeviceContext*) when info.user->sharedContext != nullptr.
    // We expose gpuid as deviceId only; the platform stays at 0, which
    // matches the typical single-vendor ICD layout (NVIDIA-only,
    // AMD-only, or Intel-only). Users with multiple platforms can set
    // a platform via the OpenCL ICD env vars (OCL_ICD_VENDORS,
    // CUDA_VISIBLE_DEVICES, etc.) -- adding a second flag for
    // platformId would be the next step if anyone asks. The Vulkan
    // backend uses the same struct with deviceId pointing at a
    // VkPhysicalDevice index; CUDA / Metal ignore the struct.
    //
    // The struct must outlive createSession(). createSession reads
    // sharedContext synchronously and the OpenCL backend copies
    // platformId / deviceId into its own state, so a stack-local
    // works -- but we still keep it alive for the full session
    // lifetime to be safe against future MNN changes that might
    // retain the pointer. See OpenCLBackend.cpp:38-43.
    MNNDeviceContext dev_ctx{};
    // gpu_dev above (set near ScheduleConfig::numThread) already covers
    // OpenCL / Vulkan / Metal; reuse it here to keep the device-set
    // decision in one place.

    if (gpu_dev && (gpu_platform > 0 || gpuid > 0)) {
        dev_ctx.platformId = static_cast<uint32_t>(gpu_platform);
        dev_ctx.deviceId = static_cast<uint32_t>(gpuid);
        bcfg.sharedContext = &dev_ctx;
    }

    // Wire the auto-tuning cache BEFORE createSession. MNN reads
    // setCacheFile + setSessionMode at session-creation time:
    //   - setCacheFile(path) tells MNN where to load/save tuned
    //     kernel parameters. Missing file is fine -- MNN populates
    //     it during the first session/run.
    //   - Session_Backend_Fix pins the backend to the one we chose
    //     in cfg.type instead of letting MNN auto-fall-back, which
    //     would invalidate cache entries that were tuned for the
    //     pinned backend.
    // The matching updateCacheFile() call lives in the destructor.
    // Capture the resolved config for gpu_accum_supported().
    mForwardType = cfg.type;
    mPrecision   = bcfg.precision;
    mBufferMode  = mnn_buffer;

    mCachePath = default_mnn_cache_path(cfg.type, bcfg.precision, model_path);

    if (!mCachePath.empty()) {
        mInterpreter->setCacheFile(mCachePath.c_str());
        mInterpreter->setSessionMode(MNN::Interpreter::Session_Backend_Fix);

        if (verbose) {
            siam::log_tag("mnn", "tuning cache: %s", mCachePath.c_str());
        }
    }

    auto _s0 = std::chrono::steady_clock::now();
    mSession = mInterpreter->createSession(cfg);

    if (!mSession) {
        throw std::runtime_error("MNN: createSession failed for " + model_path);
    }

    if (_prof) {
        fprintf(stderr, "[mnn-ctor] createSession: %.1f ms\n",
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - _s0).count());
    }

    // Rebind input tensor to the patch shape. SIAM's exported ONNX has
    // dynamic spatial dims; we always feed (1, 1, Z, Y, X).
    mInput = mInterpreter->getSessionInput(mSession, nullptr);

    if (!mInput) {
        throw std::runtime_error("MNN: no input tensor in " + model_path);
    }

    std::vector<int> tile_shape = {
        1, 1,
        static_cast<int>(patch_size[0]),
        static_cast<int>(patch_size[1]),
        static_cast<int>(patch_size[2]),
    };
    auto _r0 = std::chrono::steady_clock::now();
    mInterpreter->resizeTensor(mInput, tile_shape);
    mInterpreter->resizeSession(mSession);

    if (_prof) {
        fprintf(stderr, "[mnn-ctor] resizeSession (buffer alloc + kernel build): %.1f ms\n",
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - _r0).count());
    }

    mOutput = mInterpreter->getSessionOutput(mSession, nullptr);

    if (!mOutput) {
        throw std::runtime_error("MNN: no output tensor in " + model_path);
    }

    // Output shape from MNN is (1, C, Z, Y, X). Derive num_classes.
    auto out_dims = mOutput->shape();

    if (out_dims.size() != 5 || out_dims[0] != 1) {
        std::ostringstream msg;
        msg << "MNN: unexpected output rank/batch for " << model_path
            << " (got [";

        for (size_t i = 0; i < out_dims.size(); ++i) {
            msg << out_dims[i] << (i + 1 == out_dims.size() ? "" : ",");
        }

        msg << "]; expected [1, C, Z, Y, X])";
        throw std::runtime_error(msg.str());
    }

    mNumClasses = out_dims[1];

    // Pre-allocate the host Tensors used to bounce data into/out of
    // the (possibly device-side) session tensors. Reused across tiles.
    mHostInput.reset(MNN::Tensor::create<float>(
                         tile_shape, nullptr, MNN::Tensor::CAFFE));
    mHostOutput.reset(MNN::Tensor::create<float>(
                          out_dims, nullptr, MNN::Tensor::CAFFE));

    if (!mHostInput || !mHostOutput) {
        throw std::runtime_error("MNN: failed to allocate host scratch tensors");
    }

    mTileFloats = 1ULL * patch_size[0] * patch_size[1] * patch_size[2];
    mLogitFloats = static_cast<size_t>(mNumClasses) * mTileFloats;

    if (verbose) {
        const char* dev_name = "cpu";

        switch (cfg.type) {
            case MNN_FORWARD_OPENCL:
                dev_name = "opencl";
                break;

            case MNN_FORWARD_VULKAN:
                dev_name = "vulkan";
                break;

            case MNN_FORWARD_METAL:
                dev_name = "metal";
                break;

            default:
                break;
        }

        // For the GPU backends the same field is a tuning-mode bitfield,
        // not a thread count. Print it under the right label so the log
        // line doesn't claim "numThread=4" for a one-thread dispatch.
        const char* sched_label = gpu_dev ? "gpu_mode" : "numThread";

        if (gpu_dev && (gpu_platform > 0 || gpuid > 0)) {
            siam::log_tag("mnn", "%s device=%s[platform=%d,device=%d] %s=%d num_classes=%lld",
                          model_path.c_str(), dev_name, gpu_platform, gpuid,
                          sched_label, cfg.numThread,
                          static_cast<long long>(mNumClasses));
        } else {
            siam::log_tag("mnn", "%s device=%s %s=%d num_classes=%lld",
                          model_path.c_str(), dev_name, sched_label,
                          cfg.numThread, static_cast<long long>(mNumClasses));
        }
    }
}

MnnEngine::~MnnEngine() {
    // Persist any tuning data accumulated during this session BEFORE
    // tearing down. MNN updates an in-memory cache during runSession
    // calls; updateCacheFile() flushes it to the file we registered
    // via setCacheFile(). flag=0 = write-back mode. Best-effort: if
    // the write fails (full disk, permission), just silently drop --
    // the inference results are unaffected, only the next run loses
    // the tuning headstart.
    if (mSession && !mCachePath.empty()) {
        mInterpreter->updateCacheFile(mSession, 0);
    }

    // Session lifetime is owned by the Interpreter; releasing the
    // Interpreter releases the Session. Host Tensors are unique_ptr's.
}

// Forward-declarations of the siamize-side CL-error-tracking helpers
// added to NeuroJSON/MNN's siam-opencl-conv3d branch. Defined in
// MNN's source/backend/opencl/core/runtime/OpenCLWrapper.cpp. We
// forward-declare here so we don't need to vendor MNN's internal
// OpenCLWrapper.hpp into siamize's include path. On an upstream MNN
// build without our patch, the linker will fail loudly -- in that
// case the user should pin MNN_REF=siam-opencl-conv3d in fetch_mnn.sh.
extern "C" int  siam_mnn_get_last_cl_error();
extern "C" void siam_mnn_clear_last_cl_error();

void MnnEngine::run_tile(const float* tile, float* logits) {
    // Reset MNN's per-thread CL-error tracker before the call so we
    // can detect a failure that's specific to THIS tile. MNN's
    // MNN_CHECK_CL_SUCCESS macro normally only prints CL errors; we
    // patched it (in NeuroJSON/MNN's siam-opencl-conv3d branch) to
    // also stash the code into ::MNN::gLastCLError so callers like
    // this one can react.
    siam_mnn_clear_last_cl_error();

    // Opt-in per-op-type GPU profiler (SIAMIZE_OP_PROFILE=1). Uses MNN's
    // callback API; in the "after" callback we force a sync on the op's
    // output tensor (Tensor::wait with finish=true) so the measured
    // interval reflects real GPU execution rather than just async enqueue.
    // Aggregated by op type and dumped at process exit. Diagnostic only;
    // the wait() serialization makes absolute numbers ~upper bounds, but
    // the relative attribution across op types is what we want.
    static int s_op_prof = -1;

    if (s_op_prof < 0) {
        const char* e = std::getenv("SIAMIZE_OP_PROFILE");
        s_op_prof = (e && e[0] == '1') ? 1 : 0;
    }

    // Coarse phase timers (same env gate): attribute run_tile wall time
    // across host->device copy, runSession (forced-synced), and the
    // device->host readback.
    using pclk = std::chrono::steady_clock;
    static double s_t_copyin = 0, s_t_run = 0, s_t_copyout = 0;
    static long s_t_n = 0;
    static bool s_phase_reg = false;

    if (s_op_prof && !s_phase_reg) {
        s_phase_reg = true;
        std::atexit([]() {
            fprintf(stderr,
                    "\n[phase] run_tile breakdown over %ld tiles:\n"
                    "  copyFromHostTensor (H->D): %8.1f ms  (%6.1f ms/tile)\n"
                    "  runSession (GPU, synced) : %8.1f ms  (%6.1f ms/tile)\n"
                    "  copyToHostTensor   (D->H): %8.1f ms  (%6.1f ms/tile)\n",
                    s_t_n,
                    s_t_copyin,  s_t_copyin  / std::max<long>(1, s_t_n),
                    s_t_run,     s_t_run     / std::max<long>(1, s_t_n),
                    s_t_copyout, s_t_copyout / std::max<long>(1, s_t_n));
        });
    }

    // Stage input via the host scratch Tensor. copyFromHostTensor
    // bridges to device memory for non-CPU backends; it's a memcpy
    // for CPU.
    auto _pt0 = pclk::now();
    std::memcpy(mHostInput->host<float>(), tile, mTileFloats * sizeof(float));

    if (!mInput->copyFromHostTensor(mHostInput.get())) {
        throw std::runtime_error("MNN: copyFromHostTensor failed");
    }

    auto _pt_run0 = pclk::now();

    MNN::ErrorCode err;

    if (s_op_prof) {
        using clk = std::chrono::steady_clock;
        // total_us, n_calls keyed by op type string
        static std::map<std::string, std::pair<double, long>> s_acc;
        static bool s_registered = false;

        if (!s_registered) {
            s_registered = true;
            std::atexit([]() {
                fprintf(stderr, "\n[op-profile] per-op-type GPU time (wait-bracketed)\n");
                std::vector<std::pair<std::string, std::pair<double, long>>>
                items(s_acc.begin(), s_acc.end());
                std::sort(items.begin(), items.end(),
                [](const auto & a, const auto & b) {
                    return a.second.first > b.second.first;
                });
                double grand = 0;

                for (auto& it : items) {
                    grand += it.second.first;
                }

                for (auto& it : items) {
                    fprintf(stderr,
                            "  %-28s n=%-5ld total=%8.1f ms  avg=%8.1f us  pct=%5.1f%%\n",
                            it.first.c_str(), it.second.second,
                            it.second.first / 1000.0,
                            it.second.first / std::max<long>(1, it.second.second),
                            100.0 * it.second.first / std::max(1.0, grand));
                }

                fprintf(stderr, "  [op-profile total]: %.1f ms\n", grand / 1000.0);
            });
        }

        auto t_start = std::make_shared<clk::time_point>();
        MNN::TensorCallBackWithInfo before =
            [t_start](const std::vector<MNN::Tensor*>&,
        const MNN::OperatorInfo*) -> bool {
            *t_start = clk::now();
            return true;
        };
        MNN::TensorCallBackWithInfo after =
            [t_start](const std::vector<MNN::Tensor*>& outs,
        const MNN::OperatorInfo * info) -> bool {
            if (!outs.empty() && outs[0] != nullptr) {
                outs[0]->wait(MNN::Tensor::MAP_TENSOR_READ, true);
            }

            double dt = std::chrono::duration<double, std::micro>(
                clk::now() - *t_start).count();
            s_acc[info->type()].first += dt;
            s_acc[info->type()].second += 1;
            return true;
        };
        err = mInterpreter->runSessionWithCallBackInfo(mSession, before, after);
    } else {
        err = mInterpreter->runSession(mSession);
    }

    if (err != MNN::NO_ERROR) {
        std::ostringstream msg;
        msg << "MNN: runSession failed with code " << static_cast<int>(err);
        throw std::runtime_error(msg.str());
    }

    // Catch the case where MNN's OpenCL backend swallowed a buffer-
    // allocation failure (CL_MEM_OBJECT_ALLOCATION_FAILURE = -4 is
    // typical on low-VRAM GPUs) and continued with a corrupt buffer.
    // Without this check, runSession returns NO_ERROR, the output
    // contains a mix of valid and zero/garbage logits, and siamize
    // writes a wrong labelmap to disk with exit=0.
    if (const int cl_err = siam_mnn_get_last_cl_error(); cl_err != 0) {
        std::ostringstream msg;
        msg << "MNN: OpenCL allocation/runtime error during runSession "
            << "(CL error " << cl_err << "). The output tensor is "
            << "likely corrupt. On low-VRAM GPUs try a smaller patch "
            << "(e.g. `-P 64x64x64`), or fall back to `-c cpu`. "
            << "Earlier stderr lines beginning with 'CL ERROR CODE : "
            << cl_err << ", info:...' identify the failing op.";
        throw std::runtime_error(msg.str());
    }

    // Force GPU completion so the runSession phase reflects real compute
    // and the copyToHostTensor phase reflects pure transfer.
    if (s_op_prof) {
        mOutput->wait(MNN::Tensor::MAP_TENSOR_READ, true);
    }

    auto _pt_copyout0 = pclk::now();

    // Read the device logits straight into the caller's buffer. Wrapping
    // `logits` in a non-owning Tensor lets copyToHostTensor DMA directly
    // into it, avoiding a redundant full-volume (mLogitFloats * 4 B, ~905
    // MB/tile at the default patch) host-to-host memcpy that used to bounce
    // through mHostOutput. The wrap is a cheap descriptor; no large alloc.
    std::unique_ptr<MNN::Tensor> outWrap(
        MNN::Tensor::create<float>(mHostOutput->shape(), logits, MNN::Tensor::CAFFE));

    if (!mOutput->copyToHostTensor(outWrap.get())) {
        throw std::runtime_error("MNN: copyToHostTensor failed");
    }

    if (s_op_prof) {
        auto _pt_end = pclk::now();
        s_t_copyin  += std::chrono::duration<double, std::milli>(_pt_run0 - _pt0).count();
        s_t_run     += std::chrono::duration<double, std::milli>(_pt_copyout0 - _pt_run0).count();
        s_t_copyout += std::chrono::duration<double, std::milli>(_pt_end - _pt_copyout0).count();
        s_t_n += 1;
    }
}

// ---- GPU-side sliding accumulator ---------------------------------------

#ifdef SIAMIZE_MNN_GPU_ACCUM

// Accumulate one tile's Gaussian-weighted logits into a full-volume device
// buffer. MNN's session-output device buffer is plain channel-major --
// [Cpad][D][H][W], one contiguous D*H*W block per channel (Cpad rounded up to
// a multiple of 4; we only touch the C real channels) -- verified at runtime
// against copyToHostTensor. The accumulator uses the same channel-major
// layout over the full volume, [C][spatZ*spatY*spatX], so finish() is a plain
// add. One work-item per (channel, voxel); neighbouring work-items touch
// adjacent addresses in both `out` and `accum` (fully coalesced). The per-tile
// += is race-free: the driver runs these kernels in queue order (one per
// tile), and within a tile each (channel,voxel) cell is written once.
//
//   out   index : c*(D*H*W)        + (d*H + h)*W + w                  [patch]
//   accum index : c*channelStride  + ((sz+d)*spatY + sy+h)*spatX + sx+w  [vol]
static const char* kSiamAccumSource = R"CLC(
__kernel void siam_zero(__global float* buf, const int n) {
    const int i = get_global_id(0);
    if (i < n) buf[i] = 0.0f;
}
__kernel void siam_accum(
    __global const float* out,
    __global const float* gauss,
    __global float*       accum,
    const int  C, const int D, const int H, const int W,
    const int  spatZ, const int spatY, const int spatX,
    const int  channelStride, const int sz, const int sy, const int sx) {
    const int w  = get_global_id(0);   // 0..W-1
    const int h  = get_global_id(1);   // 0..H-1
    const int g2 = get_global_id(2);   // 0..C*D-1  (channel-major over d)
    if (w >= W || h >= H || g2 >= C * D) return;
    const int d = g2 % D;
    const int c = g2 / D;
    const long sp = (long)D * H * W;
    const long vox = ((long)d * H + h) * W + w;
    const float g = gauss[vox];
    const long oidx = (long)c * sp + vox;
    const long aidx = (long)c * channelStride +
                      ((long)(sz + d) * spatY + (sy + h)) * spatX + (sx + w);
    accum[aidx] += out[oidx] * g;
}
)CLC";

bool MnnEngine::gpu_accum_supported() const {
    // Requires OpenCL + fp32 (Precision_High) + BUFFER mode (the session
    // output device buffer is plain channel-major fp32 there).
    //
    // OPT-IN (SIAMIZE_GPU_ACCUM=1), default OFF. Measured rationale: moving
    // the sliding accumulation onto the GPU eliminates the per-tile readback
    // and the host accumulate loop, but it adds a per-fold setup cost
    // (~1.1 s: a full-volume device accumulator + a pinned host staging
    // buffer must be allocated and zeroed on each fold's fresh OpenCL
    // context) plus one full-volume readback. On a strong host (this box:
    // 64-core, the parallelized host accumulate is only ~0.5 s/fold) that
    // setup makes it a net ~0.6 s/fold LOSS. It is expected to WIN on hosts
    // where the host accumulate dominates (few CPU cores) or readback is
    // cheap (unified-memory mobile GPUs). The unconditional win needs a
    // cross-fold shared context so the setup + readback amortize over all
    // folds -- see notes in sliding.cpp / the status doc.
    if (mForwardType != MNN_FORWARD_OPENCL) return false;
    if (mPrecision != MNN::BackendConfig::Precision_High) return false;
    if (!mBufferMode) return false;
    const char* e = std::getenv("SIAMIZE_GPU_ACCUM");
    return e && e[0] == '1';
}

bool MnnEngine::gpu_accum_begin(const std::array<int64_t, 3>& full,
                                const float* gauss) {
    if (!gpu_accum_supported()) return false;
    const bool tprof = std::getenv("SIAMIZE_GPU_ACCUM_TIME") != nullptr;
    auto _b0 = std::chrono::steady_clock::now();
    const auto bmark = [&](const char* what) {
        if (tprof) fprintf(stderr, "[gpu-accum-time] begin %s: %.1f ms\n", what,
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - _b0).count());
    };

    auto* bk = static_cast<MNN::OpenCL::OpenCLBackend*>(
        const_cast<MNN::Backend*>(mInterpreter->getBackend(mSession, mOutput)));
    if (!bk) return false;
    mClRuntime = bk->getOpenCLRuntime();
    if (!mClRuntime) return false;

    mFull = full;
    mChannelStride = full[0] * full[1] * full[2];
    // Channel-major accumulator: C x full spatial (matches the session
    // output's channel-major device layout, so finish() is a plain add).
    const size_t accumFloats =
        static_cast<size_t>(mNumClasses) * mChannelStride;
    const size_t accumBytes = accumFloats * sizeof(float);
    const size_t gaussBytes = mTileFloats * sizeof(float);

    cl_int err = CL_SUCCESS;
    mAccumBuf.reset(new ::cl::Buffer(mClRuntime->context(), CL_MEM_READ_WRITE,
                                     accumBytes, nullptr, &err));
    if (err != CL_SUCCESS) {
        mAccumBuf.reset();
        siam::log_tag("mnn",
                      "GPU accumulator alloc failed (%zu MB, CL %d); "
                      "falling back to host accumulation",
                      accumBytes >> 20, static_cast<int>(err));
        return false;
    }
    // Zero the accumulator (per fold) with a tiny kernel -- cl2.hpp's
    // enqueueFillBuffer is gated out at MNN's configured CL version.
    {
        auto zk = mClRuntime->buildKernelFromSource(
            kSiamAccumSource, "siam_zero", std::set<std::string>{}, 1);
        if (zk) {
            zk->get().setArg(0, *mAccumBuf);
            zk->get().setArg(1, static_cast<int>(accumFloats));
            mClRuntime->commandQueue().enqueueNDRangeKernel(
                zk->get(), ::cl::NullRange, ::cl::NDRange(accumFloats),
                ::cl::NullRange);
        }
    }

    mGaussBuf.reset(new ::cl::Buffer(mClRuntime->context(),
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     gaussBytes, const_cast<float*>(gauss), &err));
    if (err != CL_SUCCESS) {
        mAccumBuf.reset();
        mGaussBuf.reset();
        return false;
    }

    // Pinned (page-locked) host staging buffer for the per-fold readback.
    // Reading the device accumulator into pinned memory DMAs at full PCIe
    // bandwidth (~10x faster than into a pageable std::vector on the NVIDIA
    // ICD). Mapped once and kept resident for the fold.
    mPinnedBuf.reset(new ::cl::Buffer(mClRuntime->context(),
                                      CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,
                                      accumBytes, nullptr, &err));
    if (err == CL_SUCCESS) {
        mPinnedPtr = static_cast<float*>(mClRuntime->commandQueue().enqueueMapBuffer(
            *mPinnedBuf, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, accumBytes,
            nullptr, nullptr, &err));
    }
    if (err != CL_SUCCESS || mPinnedPtr == nullptr) {
        mAccumBuf.reset();
        mGaussBuf.reset();
        mPinnedBuf.reset();
        mPinnedPtr = nullptr;
        return false;
    }

    bmark("after alloc+zero+gauss+pinned");
    mAccumKernel = mClRuntime->buildKernelFromSource(
        kSiamAccumSource, "siam_accum", std::set<std::string>{}, 1);
    if (!mAccumKernel) {
        mAccumBuf.reset();
        mGaussBuf.reset();
        return false;
    }
    bmark("after build accum kernel");

    mAccumActive = true;
    return true;
}

void MnnEngine::gpu_accum_tile(const float* tile,
                               int64_t sz, int64_t sy, int64_t sx) {
    siam_mnn_clear_last_cl_error();

    std::memcpy(mHostInput->host<float>(), tile, mTileFloats * sizeof(float));
    if (!mInput->copyFromHostTensor(mHostInput.get())) {
        throw std::runtime_error("MNN: copyFromHostTensor failed (gpu_accum)");
    }

    MNN::ErrorCode rerr = mInterpreter->runSession(mSession);
    if (rerr != MNN::NO_ERROR) {
        std::ostringstream msg;
        msg << "MNN: runSession failed with code " << static_cast<int>(rerr);
        throw std::runtime_error(msg.str());
    }
    if (const int cl_err = siam_mnn_get_last_cl_error(); cl_err != 0) {
        std::ostringstream msg;
        msg << "MNN: OpenCL error during runSession (gpu_accum, CL " << cl_err
            << "). Try SIAMIZE_GPU_ACCUM=0, a smaller patch, or -c cpu.";
        throw std::runtime_error(msg.str());
    }

    // The session output device buffer is plain channel-major fp32; bind it
    // directly as the kernel's float* input.
    ::cl::Buffer* outBuf = reinterpret_cast<::cl::Buffer*>(mOutput->buffer().device);
    auto& k = mAccumKernel->get();
    int idx = 0;
    k.setArg(idx++, *outBuf);
    k.setArg(idx++, *mGaussBuf);
    k.setArg(idx++, *mAccumBuf);
    k.setArg(idx++, static_cast<int>(mNumClasses));  // C
    k.setArg(idx++, static_cast<int>(mPatch[0]));    // D = Z
    k.setArg(idx++, static_cast<int>(mPatch[1]));    // H = Y
    k.setArg(idx++, static_cast<int>(mPatch[2]));    // W = X
    k.setArg(idx++, static_cast<int>(mFull[0]));     // spatZ
    k.setArg(idx++, static_cast<int>(mFull[1]));     // spatY
    k.setArg(idx++, static_cast<int>(mFull[2]));     // spatX
    k.setArg(idx++, static_cast<int>(mChannelStride));
    k.setArg(idx++, static_cast<int>(sz));
    k.setArg(idx++, static_cast<int>(sy));
    k.setArg(idx++, static_cast<int>(sx));

    // Enqueue on the backend's own (in-order) queue, so this runs strictly
    // after the inference kernels just enqueued by runSession. Global range
    // is one element per (channel, voxel): (W, H, C*D).
    cl_int qerr = mClRuntime->commandQueue().enqueueNDRangeKernel(
        k, ::cl::NullRange,
        ::cl::NDRange(static_cast<size_t>(mPatch[2]),
                      static_cast<size_t>(mPatch[1]),
                      static_cast<size_t>(mPatch[0]) * mNumClasses),
        ::cl::NullRange);
    if (qerr != CL_SUCCESS) {
        std::ostringstream msg;
        msg << "MNN: siam_accum kernel enqueue failed (CL " << qerr << ")";
        throw std::runtime_error(msg.str());
    }
}

void MnnEngine::gpu_accum_finish(float* logits_out) {
    if (!mAccumActive) return;
    const bool tprof = std::getenv("SIAMIZE_GPU_ACCUM_TIME") != nullptr;
    auto _tf0 = std::chrono::steady_clock::now();
    mClRuntime->commandQueue().finish();
    if (tprof) {
        fprintf(stderr, "[gpu-accum-time] finish() queue drain: %.1f ms\n",
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - _tf0).count());
    }
    auto _tf1 = std::chrono::steady_clock::now();

    const size_t total =
        static_cast<size_t>(mNumClasses) * static_cast<size_t>(mChannelStride);
    const size_t bytes = total * sizeof(float);

    // DMA the channel-major device accumulator into pinned host memory
    // (full PCIe bandwidth), then add into the cross-fold accumulator (same
    // layout). Parallelized across the flat range.
    const float* mapped = mPinnedPtr;
    cl_int merr = mClRuntime->commandQueue().enqueueReadBuffer(
        *mAccumBuf, CL_TRUE, 0, bytes, mPinnedPtr);
    if (merr != CL_SUCCESS) {
        throw std::runtime_error("MNN: failed to read GPU accumulator back");
    }

    const auto add_range = [&](size_t i0, size_t i1) {
        for (size_t i = i0; i < i1; ++i) logits_out[i] += mapped[i];
    };
    int nthreads = static_cast<int>(std::thread::hardware_concurrency());
    nthreads = std::max(1, std::min(nthreads, 16));
    if (nthreads <= 1) {
        add_range(0, total);
    } else {
        std::vector<std::thread> pool;
        pool.reserve(nthreads - 1);
        const size_t per = (total + nthreads - 1) / nthreads;
        for (int t = 1; t < nthreads; ++t) {
            size_t i0 = std::min(total, static_cast<size_t>(t) * per);
            size_t i1 = std::min(total, i0 + per);
            if (i0 < i1) pool.emplace_back(add_range, i0, i1);
        }
        add_range(0, std::min(total, per));
        for (auto& th : pool) th.join();
    }

    if (tprof) {
        fprintf(stderr, "[gpu-accum-time] finish() read+add: %.1f ms\n",
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - _tf1).count());
    }

    if (mPinnedPtr) {
        mClRuntime->commandQueue().enqueueUnmapMemObject(*mPinnedBuf, mPinnedPtr);
        mClRuntime->commandQueue().finish();
        mPinnedPtr = nullptr;
    }
    mPinnedBuf.reset();
    mAccumBuf.reset();
    mGaussBuf.reset();
    mAccumKernel.reset();
    mAccumActive = false;
}

#else  // !SIAMIZE_MNN_GPU_ACCUM

bool MnnEngine::gpu_accum_supported() const { return false; }
bool MnnEngine::gpu_accum_begin(const std::array<int64_t, 3>&, const float*) {
    return false;
}
void MnnEngine::gpu_accum_tile(const float*, int64_t, int64_t, int64_t) {}
void MnnEngine::gpu_accum_finish(float*) {}

#endif  // SIAMIZE_MNN_GPU_ACCUM

}  // anonymous namespace

std::unique_ptr<Engine>
make_engine(const std::string& model_path,
            std::array<int64_t, 3> patch_size,
            int intra_threads,
            const std::string& device,
            const EngineTuning& tuning,
            bool verbose) {
    // EngineTuning's CUDA / CoreML knobs are inert for MNN; only the
    // gpuid is honored (mapped to MNNDeviceContext::deviceId for the
    // OpenCL / Vulkan backends). The signature stays the same as the
    // ORT impl so sliding.cpp doesn't need backend-specific branches.
    return std::unique_ptr<Engine>(
               new MnnEngine(model_path, patch_size, intra_threads,
                             device, tuning.gpu_platform, tuning.gpuid,
                             tuning.mnn_fp16, tuning.mnn_buffer, verbose));
}

const char* backend_name() {
    return "mnn";
}

}  // namespace siam

#endif  // SIAMIZE_HAS_MNN
