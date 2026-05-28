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

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/MNNForwardType.h>

#include <algorithm>
#include <cstring>
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

/// MNN-backed Engine. One Interpreter + one Session per fold. The
/// patch shape is fixed at construction; sliding.cpp always calls
/// run_tile() with that same shape.
class MnnEngine final : public Engine {
  public:
    MnnEngine(const std::string& model_path,
              std::array<int64_t, 3> patch_size,
              int intra_threads,
              const std::string& device,
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
};

MnnEngine::MnnEngine(const std::string& model_path,
                     std::array<int64_t, 3> patch_size,
                     int intra_threads,
                     const std::string& device,
                     bool verbose)
    : mPatch(patch_size) {
    mInterpreter.reset(MNN::Interpreter::createFromFile(model_path.c_str()),
                       MNN::Interpreter::destroy);

    if (!mInterpreter) {
        throw std::runtime_error("MNN: failed to load model: " + model_path);
    }

    MNN::ScheduleConfig cfg;
    cfg.type = resolve_forward_type(device, verbose);
    // Workaround: numThread >= 4 segfaults in CPURaster::lambda#4 on
    // SIAM-class workloads even with the int64 patches. See
    // tools/mnn_probe/patches/README.md. Cap at 2 for safety.
    cfg.numThread = std::max(1, std::min(intra_threads, 2));
    // "high" -> fp32 accumulators; "low" -> fp16. We default to high
    // because SIAM cascades fp16 noise across 76 layers and we already
    // see ~6 % logit drift even at high precision; low would compound.
    MNN::BackendConfig bcfg;
    bcfg.precision = MNN::BackendConfig::Precision_High;
    cfg.backendConfig = &bcfg;

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

        siam::log_tag("mnn", "%s device=%s numThread=%d num_classes=%lld",
                      model_path.c_str(), dev_name, cfg.numThread,
                      static_cast<long long>(mNumClasses));
    }
}

MnnEngine::~MnnEngine() {
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
            const EngineTuning& /*tuning*/,
            bool verbose) {
    // EngineTuning's CUDA / CoreML knobs are inert for MNN. The
    // signature stays the same as the ORT impl so sliding.cpp doesn't
    // need backend-specific branches.
    return std::unique_ptr<Engine>(
               new MnnEngine(model_path, patch_size, intra_threads, device, verbose));
}

const char* backend_name() {
    return "mnn";
}

}  // namespace siam

#endif  // SIAMIZE_HAS_MNN
