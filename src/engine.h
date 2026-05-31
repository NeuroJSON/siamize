/***************************************************************************//**
\file    engine.h

@brief   Backend-agnostic per-tile inference interface

Hides the choice of ONNX Runtime vs MNN (vs anything else later) behind
a small abstract interface. The sliding-window logic in sliding.cpp
constructs one Engine per fold via siam::make_engine() and then calls
run_tile() once per patch.

Build-time backend selection: exactly one of SIAMIZE_HAS_ORT,
SIAMIZE_HAS_MNN compiles. The respective engine_*.cpp implements
make_engine(). The CMake option SIAMIZE_BACKEND picks which.

Why one backend per build (not both at runtime):
  * Different runtime dependencies. ORT pulls in libonnxruntime.so /
    libonnxruntime_providers_cuda.so / cuDNN / CUDA. MNN pulls in
    libMNN.so. Bundling both balloons the deployable.
  * Both libraries register global state on load (allocators, thread
    pools). Coexisting them creates initialization-order issues and
    duplicates per-process state.
  * The interesting comparison is between siamize-ort and siamize-mnn
    as separate binaries -- when you care which runtime you're
    benchmarking, you don't want the other one linked in.
*******************************************************************************/

#ifndef SIAMIZE_ENGINE_H
#define SIAMIZE_ENGINE_H

#include "siam.h"
#include "sliding.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace siam {

/**
 * \class  Engine
 * \brief  Abstract per-fold inference engine
 *
 * One Engine instance corresponds to one loaded model (one fold of the
 * SIAM ensemble). The sliding-window driver instantiates Engines from
 * model paths via make_engine(), then calls run_tile() for each patch.
 *
 * Engines are NOT thread-safe between instances; each fold gets its
 * own Engine and runs sequentially in sliding_window(). Per-tile
 * intra-op parallelism happens INSIDE run_tile() via the backend's
 * own thread pool.
 *
 * Inputs:
 *   * `tile`  -- (1, 1, Z, Y, X) fp32 row-major (NCDHW), where (Z, Y, X)
 *                equals the patch_size passed to make_engine().
 *   * `logits` -- (1, num_classes, Z, Y, X) fp32 row-major output
 *                 buffer; the caller pre-allocates and owns it.
 *
 * Implementations must:
 *   - Not retain `tile` past the call (so the caller can reuse the
 *     buffer immediately).
 *   - Write into `logits` -- not reallocate it.
 *   - Throw std::runtime_error on session/run failures (the driver
 *     catches and surfaces as user-visible messages).
 */
class Engine {
  public:
    virtual ~Engine() = default;

    /// Run forward on one tile. Patch shape was fixed at construction.
    virtual void run_tile(const float* tile, float* logits) = 0;

    /// Returns the model's declared num_classes (= logits channel count).
    /// Useful for sanity-checking the caller's `logits` buffer size.
    virtual int64_t num_classes() const = 0;

    // ---- Optional GPU-side sliding accumulation -------------------------
    //
    // When gpu_accum_supported() is true, the driver uses this trio instead
    // of run_tile() + host accumulation: each tile's logits are accumulated
    // (Gaussian-weighted, at the tile's volume offset) directly into a
    // device-resident full-volume buffer, so the big per-tile device->host
    // readback (and the host accumulate loop) are replaced by a single
    // per-fold readback in gpu_accum_finish(). Only the MNN OpenCL backend
    // in fp32/BUFFER mode implements it; everything else keeps the host path.

    /// True if this engine can accumulate on the GPU for the current config.
    virtual bool gpu_accum_supported() const {
        return false;
    }

    /// Allocate the device accumulator for one fold. `full` = (Z,Y,X) of the
    /// padded volume; `gauss` = patch-sized (Z*Y*X, NCDHW row-major) weights.
    /// Returns false if device allocation failed -- the driver then falls
    /// back to the run_tile + host path for the whole run.
    virtual bool gpu_accum_begin(const std::array<int64_t, 3>& full,
                                 const float* gauss) {
        (void)full;
        (void)gauss;
        return false;
    }

    /// Forward one tile and accumulate its Gaussian-weighted logits into the
    /// device accumulator at offset (sz,sy,sx). No host readback.
    virtual void gpu_accum_tile(const float* tile,
                                int64_t sz, int64_t sy, int64_t sx) {
        (void)tile;
        (void)sz;
        (void)sy;
        (void)sx;
    }

    /// Add this fold's accumulated weighted logits into the host buffer
    /// `logits_out` (num_classes * Z*Y*X floats, channel-major), then release
    /// the device accumulator.
    virtual void gpu_accum_finish(float* logits_out) {
        (void)logits_out;
    }
};

/**
 * \brief Construct an Engine backed by the compiled-in backend
 *
 * The backend is fixed at build time (see SIAMIZE_BACKEND in CMake).
 * `device` is a hint accepted by each backend in its own vocabulary:
 *
 *  - ORT backend: "auto", "cpu", "cuda", "tensorrt", "coreml"
 *  - MNN backend: "auto", "cpu", "opencl", "vulkan", "metal"
 *
 * Backends silently ignore device strings they don't understand and
 * fall back to "cpu". This keeps the driver code in sliding.cpp
 * backend-agnostic.
 *
 * \param model_path     filesystem path to a .onnx (ORT) or .mnn (MNN)
 *                       model file
 * \param patch_size     (Z, Y, X) tile shape (must match the model's
 *                       declared / bindable input shape)
 * \param intra_threads  per-tile thread-pool size. ORT: SetIntraOpNumThreads.
 *                       MNN: ScheduleConfig::numThread. Both backends
 *                       accept 1 to N; the MNN backend additionally
 *                       caps at 2 on Linux x86_64 because numThread>=4
 *                       segfaults on SIAM-class workloads (see
 *                       tools/mnn_probe/patches/README.md).
 * \param device         backend-specific device-selection hint
 * \param tuning         per-EP knobs (CUDA/CoreML for ORT;
 *                       inert otherwise)
 * \param verbose        print backend choice to stderr
 */
std::unique_ptr<Engine>
make_engine(const std::string& model_path,
            std::array<int64_t, 3> patch_size,
            int intra_threads,
            const std::string& device,
            const EngineTuning& tuning,
            bool verbose);

/// Human-readable backend name baked in at compile time
/// ("onnxruntime" / "mnn"). For --version output and logging.
const char* backend_name();

}  // namespace siam

#endif  // SIAMIZE_ENGINE_H
