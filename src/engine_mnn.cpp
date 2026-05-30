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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
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
std::string default_mnn_cache_path(MNNForwardType type,
                                   MNN::BackendConfig::PrecisionMode prec) {
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
    return (dir / (std::string(dev_tag) + "-" + prec_tag + ".cache")).string();
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
    mInterpreter.reset(MNN::Interpreter::createFromFile(model_path.c_str()),
                       MNN::Interpreter::destroy);

    if (!mInterpreter) {
        throw std::runtime_error("MNN: failed to load model: " + model_path);
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
        // CPU path: numThread >= 4 segfaults in CPURaster::lambda#4 on
        // SIAM-class workloads even with the int64 patches. See
        // tools/mnn_probe/patches/README.md. Cap at 2 for safety.
        cfg.numThread = std::max(1, std::min(intra_threads, 2));
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
    mCachePath = default_mnn_cache_path(cfg.type, bcfg.precision);

    if (!mCachePath.empty()) {
        mInterpreter->setCacheFile(mCachePath.c_str());
        mInterpreter->setSessionMode(MNN::Interpreter::Session_Backend_Fix);

        if (verbose) {
            siam::log_tag("mnn", "tuning cache: %s", mCachePath.c_str());
        }
    }

    mSession = mInterpreter->createSession(cfg);

    if (!mSession) {
        throw std::runtime_error("MNN: createSession failed for " + model_path);
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
    mInterpreter->resizeTensor(mInput, tile_shape);
    mInterpreter->resizeSession(mSession);

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

void MnnEngine::run_tile(const float* tile, float* logits) {
    // Stage input via the host scratch Tensor. copyFromHostTensor
    // bridges to device memory for non-CPU backends; it's a memcpy
    // for CPU.
    std::memcpy(mHostInput->host<float>(), tile, mTileFloats * sizeof(float));

    if (!mInput->copyFromHostTensor(mHostInput.get())) {
        throw std::runtime_error("MNN: copyFromHostTensor failed");
    }

    MNN::ErrorCode err = mInterpreter->runSession(mSession);

    if (err != MNN::NO_ERROR) {
        std::ostringstream msg;
        msg << "MNN: runSession failed with code " << static_cast<int>(err);
        throw std::runtime_error(msg.str());
    }

    if (!mOutput->copyToHostTensor(mHostOutput.get())) {
        throw std::runtime_error("MNN: copyToHostTensor failed");
    }

    std::memcpy(logits, mHostOutput->host<float>(), mLogitFloats * sizeof(float));
}

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
