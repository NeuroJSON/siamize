/***************************************************************************//**
**  \mainpage siamize - native C++/ONNX port of SIAM v0.3 brain segmentation
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
\file    siamize_mex.cpp
\brief   MATLAB / GNU Octave MEX bridge for the siamize inference pipeline

Bridges the C++ inference core (siam::sliding_window + preprocess +
orient) to MATLAB / Octave. The pure-MATLAB dispatcher in
matlab/siamize.m massages user-friendly inputs (file paths, jnifti
structs, 3D arrays, default affines) into this MEX's strict
positional ABI, so end-users normally call siamize(), not the MEX
directly.

MEX signature (low-level ABI):

    labels = siamex(img, affine, models)
    labels = siamex(img, affine, models, opts)

Inputs:

  - img:     3D numeric array. Any standard MATLAB dtype
             (uint8 / int16 / int32 / uint16 / uint32 / single /
             double). Interpreted as a NIfTI voxel grid in MATLAB's
             natural (X, Y, Z) column-major layout.
  - affine:  3x4 or 4x4 double matrix in jsonlab's loadnifti
             convention (rows = world R/A/S, columns = voxel axes
             with last column = origin).
  - models:  char (one path) or cellstr / cell array of paths;
             logits are averaged across folds. Single-digit
             shortcuts ("0".."9") are expanded to
             "fold_<d>_fp16.onnx".
  - opts:    optional struct with any of `.device`, `.threads`,
             `.patch`, `.spacing`, `.classes`, `.tpm`, `.tpm_t`,
             `.trt_cache`, `.gpu`, `.arena`, `.lowmem`, `.verbose`.
             See read_opts() in this file for parsing details.
             The `.arena` toggle mirrors the CLI's `--no-arena` flag:
             pass `'arena', false` to keep peak RSS small at the
             cost of ~1.5x wall time. The `.lowmem` toggle mirrors
             the CLI's `--lowmem` flag: pass `'lowmem', true` to
             force the safe-defaults preset (smaller patch + no
             arena + smaller thread cap + tight VRAM knobs) on
             otherwise-large hosts. The same preset auto-applies
             when available RAM is < 14 GB or VRAM < 12 GB.

Output:

  - labels:  uint8 3D array of the SAME shape as `img`, in input
             axis order (suitable for round-trip via savenifti),
             OR float32 4D (X, Y, Z, C) when `opts.tpm` is true.

MEX and CLI share the same `siamize_core` static library, so MEX
predictions are bit-identical to `siamize -i ... -o ...` when run
with matching options.
*******************************************************************************/

#include "mex.h"

#include "siam.h"
#include "siam_log.h"
#include "orient.h"
#include "preprocess.h"
#include "sliding.h"
#include "engine.h"      // siam::backend_name() for the 'backend' query
#include "weights.h"

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

/*******************************************************************************/
/*! \fn    void die(const char* id, const std::string& msg)
    \brief Raise a MATLAB error and never return

    Thin wrapper around `mexErrMsgIdAndTxt` that lets us throw with a
    std::string message. The MEX runtime longjmps out of the calling
    function, so callers should not assume control returns.

    \param  id   MATLAB error identifier (`"siamize:..."`)
    \param  msg  human-readable message
*/
void die(const char* id, const std::string& msg) {
    mexErrMsgIdAndTxt(id, "%s", msg.c_str());
}

/*******************************************************************************/
/*! \fn    std::string mx_to_string(const mxArray* a)
    \brief Convert a MATLAB char array to a std::string
    \param  a  mxArray that must be of class `char` or `string`
    \return    its contents as a UTF-8 std::string
*/
std::string mx_to_string(const mxArray* a) {
    if (!mxIsChar(a)) {
        die("siamize:type", "expected a char array");
    }

    mwSize n = mxGetNumberOfElements(a) + 1;
    // Octave's mex.h declares mxGetString with a non-const char*; use a
    // std::vector buffer rather than std::string::data() to avoid a
    // const-correctness mismatch on older mex.h variants.
    std::vector<char> buf(n, '\0');

    if (mxGetString(a, buf.data(), n) != 0) {
        die("siamize:string", "failed to read string argument");
    }

    return std::string(buf.data());
}

// Convert any 3D mxArray to a siam::Volume in canonical (Z, Y, X) layout
// using affine-driven reorientation. MATLAB column-major (X-fastest) gives
// us the same memory order as our (Z, Y, X) Volume with X-fastest stride.
template <typename SrcT>
/*******************************************************************************/
/*! \fn    template <typename SrcT>
           void load_volume(const SrcT* src,
                            int64_t X, int64_t Y, int64_t Z,
                            const std::array<int, 3>& dst,
                            const std::array<int, 3>& sgn,
                            siam::Volume& canon)
    \brief Reorient a MATLAB column-major voxel buffer into canonical (Z, Y, X)

    Thin adapter over copy_reorient_to_canonical that exists so the
    MEX dispatch code can dispatch on the input mxClassID via a
    template instantiation without dragging the orient.h templates
    into the dispatch site.
*/
void load_volume(const SrcT* src, int64_t X, int64_t Y, int64_t Z,
                 const std::array<float, 16>& affine,
                 siam::Volume& canon_out,
                 std::array<int, 3>& dst, std::array<int, 3>& sgn,
                 std::array<float, 16>& affine_canon_out) {
    siam::axes_to_canonical(affine, dst, sgn);
    siam::copy_reorient_to_canonical<SrcT>(src, X, Y, Z, dst, sgn, canon_out);
    affine_canon_out = siam::canonicalize_affine(affine, dst, sgn, {X, Y, Z});
}

/*******************************************************************************/
/*! \fn    std::array<float, 16> read_affine(const mxArray* a)
    \brief Read a 3x4 or 4x4 MATLAB affine into siamize's row-major std::array

    MATLAB stores matrices column-major, so this routine transposes
    on the fly: `M(r, c)` from MATLAB ends up at `affine[r*4 + c]`
    in the returned array. Accepts both 3x4 (jsonlab loadnifti
    convention) and 4x4 (jnifti annotated form expanded) input;
    when 3x4, the missing fourth row is filled with `(0, 0, 0, 1)`.

    \param  a  the mxArray (must be class `double` or `single`)
    \return    row-major 4x4 affine as a 16-float std::array
*/
std::array<float, 16> read_affine(const mxArray* a) {
    if (mxIsComplex(a)) {
        die("siamize:affine", "affine must be real");
    }

    if (mxGetNumberOfDimensions(a) != 2) {
        die("siamize:affine", "affine must be a 2D matrix");
    }

    const mwSize* d = mxGetDimensions(a);

    // Read the (row, col) element from a column-major MATLAB matrix as float,
    // accepting either single or double storage. jsonlab's
    // nii.NIFTIHeader.Affine is single-precision; many other tools use double.
    auto get = [&](mwSize r, mwSize c, mwSize nrows) -> float {
        const size_t off = r + c * nrows;

        switch (mxGetClassID(a)) {
            case mxDOUBLE_CLASS:
                return static_cast<float>(static_cast<const double*>(mxGetData(a))[off]);

            case mxSINGLE_CLASS:
                return static_cast<const float*>(mxGetData(a))[off];

            default:
                die("siamize:affine", "affine must be single or double");
        }

        return 0.0f;
    };

    std::array<float, 16> aff{};
    aff[15] = 1.0f;

    if (d[0] == 3 && d[1] == 4) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j) {
                aff[i * 4 + j] = get(i, j, 3);
            }
    } else if (d[0] == 4 && d[1] == 4) {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                aff[i * 4 + j] = get(i, j, 4);
            }
    } else {
        die("siamize:affine",
            "affine must be 3x4 or 4x4 (got " + std::to_string(d[0]) +
            "x" + std::to_string(d[1]) + ")");
    }

    return aff;
}

/*******************************************************************************/
/*! \fn    std::vector<std::string> read_models(const mxArray* a)
    \brief Parse the `models` argument into a list of resolved fold paths

    Accepts a single char path, a cell array of char paths, or a
    cell array mixing paths and single-digit fold shortcuts. Each
    token is run through weights::resolve_model_path so missing
    weights auto-download from NeuroJSON.

    \param  a  the mxArray (char | cell-of-char | numeric scalar)
    \return    list of absolute .onnx paths
*/
/**
 * \brief Rewrite ORT-style basenames to the MNN equivalent when the
 *        MEX was built against the MNN backend.
 *
 * The shipped MATLAB wrapper (matlab/siamize.m) expands digit
 * shortcuts to `fold_<N>_fp16.onnx` regardless of which backend the
 * MEX was compiled against -- the .m file is backend-agnostic. When
 * SIAMIZE_HAS_MNN is defined, the MNN runtime can't consume .onnx
 * files, so rewrite any incoming `fold_<N>_fp16.onnx` basename to
 * the MNN equivalent `fold_<N>_int8.mnn` and let resolve_model_path
 * fetch the right thing from doc=mnn_i8a. Anything else (absolute
 * paths, custom basenames) passes through unchanged.
 *
 * Inert when SIAMIZE_HAS_MNN is not defined.
 */
std::string rewrite_for_backend(const std::string& m) {
#ifdef SIAMIZE_HAS_MNN
    static const std::string kOrtSuffix = "_fp16.onnx";
    static const std::string kMnnSuffix = "_int8.mnn";

    if (m.size() >= kOrtSuffix.size()
            && m.compare(m.size() - kOrtSuffix.size(),
                         kOrtSuffix.size(), kOrtSuffix) == 0) {
        // Only rewrite the basename form. Absolute paths
        // (containing '/' or '\\') were explicitly chosen by the
        // caller and we shouldn't second-guess them.
        if (m.find('/') == std::string::npos
                && m.find('\\') == std::string::npos) {
            return m.substr(0, m.size() - kOrtSuffix.size()) + kMnnSuffix;
        }
    }

#endif
    return m;
}

std::vector<std::string> read_models(const mxArray* a) {
    std::vector<std::string> out;

    if (mxIsEmpty(a)) {
        // empty -> caller wants the default single-fold
        return out;
    }

    if (mxIsChar(a)) {
        out.push_back(rewrite_for_backend(mx_to_string(a)));
    } else if (mxIsCell(a)) {
        mwSize n = mxGetNumberOfElements(a);
        out.reserve(n);

        for (mwSize i = 0; i < n; ++i) {
            const mxArray* c = mxGetCell(a, i);
            out.push_back(rewrite_for_backend(mx_to_string(c)));
        }
    } else {
        die("siamize:models", "models must be a char array, cellstr, or [] for default");
    }

    return out;
}

/**
 * \struct Opts
 * \brief  Parsed view of the MEX `opts` struct (defaults + per-field types)
 *
 * One C++ field per documented MATLAB option. Default values match
 * the CLI defaults so MEX and CLI produce identical results when
 * the user passes no overrides.
 */
struct Opts {
    std::string device = "auto";                       /**< "auto" | "cpu" | "cuda" | "tensorrt" */
    int threads = 0;                                   /**< 0 = std::thread::hardware_concurrency() */
    std::array<int64_t, 3> patch = {256, 256, 192};    /**< sliding-window patch (Z, Y, X) */
    float spacing = 0.75f;                             /**< target isotropic voxel size, mm */
    int64_t classes = 18;                              /**< inference output channels (network's logits) */
    int64_t classes_out = 18;                          /**< saved output channels (differs for SPM) */
    siam::ClassSet class_set = siam::ClassSet::CUSTOM_N; /**< 'spm' triggers SIAM-18 -> SPM-6 merge */
    std::string trt_cache;                             /**< TensorRT engine cache directory */
    bool verbose = true;                               /**< progress messages to the MATLAB / Octave
                                                            Command Window. Default ON to mirror the CLI
                                                            (see siamize.cpp commit 0d4c8a3); pass
                                                            opts.verbose = false to silence. */
    siam::EngineTuning engine_tuning;                      /**< CUDA EP knobs (gpuid, mem limit, ...) */
    bool  tpm = false;                                 /**< true -> emit 4D float32 TPM, false -> 3D labels */
    float tpm_temperature = 1.0f;                      /**< softmax temperature for the TPM */
    bool  upsample = false;                            /**< true -> return output at internal inference
                                                            resolution (0.75 mm by default), canonical RAS
                                                            orientation, full canonical extent (no
                                                            per-channel back-resample). Mirrors CLI
                                                            --upsample. The MEX's second output arg
                                                            carries the corresponding 4x4 affine; the
                                                            wrapper writes it to nii.NIFTIHeader.Affine. */
    bool  lowmem = false;                              /**< true -> force the lowmem preset
                                                            (smaller patch, no arena, smaller
                                                            thread cap, tight VRAM knobs).
                                                            Mirrors the CLI --lowmem flag. */
    // *_set trackers: true when the corresponding opts.* field was
    // explicitly set in the caller's struct. Used by the auto-lowmem
    // block in mexFunction to avoid stomping on user intent.
    bool  patch_set                = false;
    bool  threads_set              = false;
    bool  cpu_arena_set            = false;
    bool  cudnn_max_workspace_set  = false;
    bool  gpu_mem_limit_set        = false;
};

/*******************************************************************************/
/*! \fn    double opt_double(const mxArray* s, const char* name, double fallback)
    \brief Read an optional numeric scalar field from a MATLAB struct

    Convenience helper used by read_opts to pull `.threads`, `.spacing`,
    etc. out of the user-supplied opts struct without verbosely
    typing `mxGetField` + `mxGetScalar` everywhere.

    \param  s         pointer to the opts struct mxArray
    \param  name      field name to look up
    \param  fallback  value returned when the field is absent
    \return           field's numeric scalar, or \a fallback
*/
double opt_double(const mxArray* s, const char* name, double fallback) {
    const mxArray* f = mxGetField(s, 0, name);
    return (f && mxIsDouble(f) && !mxIsEmpty(f)) ? mxGetScalar(f) : fallback;
}

/*******************************************************************************/
/*! \fn    void read_opts(const mxArray* a, Opts& o)
    \brief Parse the optional `opts` struct argument into the Opts state object

    Walks every supported field in the input struct (device, threads,
    patch, spacing, classes, tpm, tpm_t, trt_cache, gpu, verbose),
    coercing each to its native C++ type and falling back to the
    defaults already initialized in \a o for any missing fields.
    Unknown fields are silently ignored so newer MATLAB-side
    wrappers can pass forward-compatible options.

    \param  a  the opts struct mxArray (must be class `struct`)
    \param  o  output: parsed options
*/
void read_opts(const mxArray* a, Opts& o) {
    if (!mxIsStruct(a)) {
        die("siamize:opts", "opts must be a struct");
    }

    const mxArray* f;

    if ((f = mxGetField(a, 0, "compute"))  && !mxIsEmpty(f)) {
        o.device = mx_to_string(f);

        if (o.device == "trt") {
            o.device = "tensorrt";
        }
    }

    if ((f = mxGetField(a, 0, "thread"))   && !mxIsEmpty(f)) {
        o.threads = static_cast<int>(mxGetScalar(f));
        o.threads_set = true;
    }

    if ((f = mxGetField(a, 0, "spacing"))  && !mxIsEmpty(f)) {
        o.spacing = static_cast<float>(mxGetScalar(f));
    }

    if ((f = mxGetField(a, 0, "classes"))  && !mxIsEmpty(f)) {
        // Accept either a numeric scalar (e.g. opts.classes = 18) or
        // the string 'spm' (opts.classes = 'spm') for the SIAM 18 ->
        // SPM 6 remap. Mirrors the CLI's --classes N|spm.
        if (mxIsChar(f)) {
            std::string s = mx_to_string(f);

            if (s == "spm" || s == "SPM") {
                o.class_set   = siam::ClassSet::SPM;
                o.classes     = 18;                          // still infer all 18 SIAM channels
                o.classes_out = siam::SPM6_NUM_CLASSES;      // merge to 6 SPM bins on output
            } else {
                die("siamize:opts",
                    "opts.classes string must be 'spm' (got '" + s + "')");
            }
        } else {
            o.classes     = static_cast<int64_t>(mxGetScalar(f));
            o.classes_out = o.classes;
        }
    }

    if ((f = mxGetField(a, 0, "trt_cache")) && !mxIsEmpty(f)) {
        o.trt_cache = mx_to_string(f);
    }

    if ((f = mxGetField(a, 0, "verbose"))  && !mxIsEmpty(f)) {
        o.verbose = mxGetScalar(f) != 0.0;
    }

    if ((f = mxGetField(a, 0, "patch")) && !mxIsEmpty(f)) {
        if (!mxIsDouble(f) || mxGetNumberOfElements(f) != 3) {
            die("siamize:patch", "opts.patch must be a [pz, py, px] double vector");
        }

        const double* p = mxGetPr(f);
        o.patch = {(int64_t)p[0], (int64_t)p[1], (int64_t)p[2]};
        o.patch_set = true;
    }

    // CUDA EP tuning knobs. All optional; absent fields keep EngineTuning
    // defaults (which are ORT defaults).
    if ((f = mxGetField(a, 0, "cudnn_max_workspace")) && !mxIsEmpty(f)) {
        o.engine_tuning.cudnn_max_workspace = static_cast<int>(mxGetScalar(f));
        o.cudnn_max_workspace_set = true;
    }

    if ((f = mxGetField(a, 0, "arena_extend")) && !mxIsEmpty(f)) {
        // Accept either the string ('same'/'power') or numeric (0/1).
        if (mxIsChar(f)) {
            std::string s = mx_to_string(f);
            o.engine_tuning.arena_same_as_req =
                (s == "same" || s == "kSameAsRequested") ? 1 : 0;
        } else {
            o.engine_tuning.arena_same_as_req = static_cast<int>(mxGetScalar(f)) ? 1 : 0;
        }
    }

    if ((f = mxGetField(a, 0, "cudnn_algo")) && !mxIsEmpty(f)) {
        std::string s = mx_to_string(f);

        // Canonicalize to ORT's uppercase string form.
        for (auto& c : s) {
            c = static_cast<char>(::toupper(static_cast<unsigned char>(c)));
        }

        if (s == "DEFAULT" || s == "HEURISTIC" || s == "EXHAUSTIVE") {
            o.engine_tuning.algo_search = s;
        } else {
            die("siamize:cudnn_algo",
                "opts.cudnn_algo must be 'default' | 'heuristic' | 'exhaustive'");
        }
    }

    if ((f = mxGetField(a, 0, "gpu_mem_limit")) && !mxIsEmpty(f)) {
        // Pass through as raw byte count. The .m wrapper is responsible for
        // K/M/G suffix interpretation if it wants -- here we just take a number.
        o.engine_tuning.gpu_mem_limit_bytes =
            static_cast<size_t>(mxGetScalar(f));
        o.gpu_mem_limit_set = true;
    }

    if ((f = mxGetField(a, 0, "gpu")) && !mxIsEmpty(f)) {
        // Accept either a scalar deviceId (legacy / CUDA / single-ICD
        // path) or a "P:D" string (MNN OpenCL platform P device D, for
        // multi-ICD hosts -- mirrors the CLI's -G P:D form).
        if (mxIsChar(f)) {
            std::string s = mx_to_string(f);
            size_t colon = s.find(':');

            if (colon == std::string::npos) {
                o.engine_tuning.gpuid = std::stoi(s);
            } else {
                o.engine_tuning.gpu_platform = std::stoi(s.substr(0, colon));
                o.engine_tuning.gpuid = std::stoi(s.substr(colon + 1));
            }
        } else {
            o.engine_tuning.gpuid = static_cast<int>(mxGetScalar(f));
        }
    }

    if ((f = mxGetField(a, 0, "gpu_platform")) && !mxIsEmpty(f)) {
        // MNN OpenCL/Vulkan platform index (inert for ORT). Same field
        // semantics as the P-half of "P:D" on opts.gpu, but settable
        // independently for callers that prefer a struct-shaped opts.
        o.engine_tuning.gpu_platform = static_cast<int>(mxGetScalar(f));
    }

    if ((f = mxGetField(a, 0, "arena")) && !mxIsEmpty(f)) {
        // Toggle for ORT's CPU memory arena + memory-pattern optimizer.
        // Default true (fast path). Pass false to match the CLI's
        // --no-arena flag when peak RSS matters more than wall time.
        o.engine_tuning.cpu_arena = (mxGetScalar(f) != 0.0);
        o.cpu_arena_set = true;
    }

    if ((f = mxGetField(a, 0, "lowmem")) && !mxIsEmpty(f)) {
        // Force the lowmem preset (mirrors CLI --lowmem). Same preset
        // is auto-applied when host RAM / GPU VRAM is tight; this
        // field opts in regardless of detection.
        o.lowmem = (mxGetScalar(f) != 0.0);
    }

    if ((f = mxGetField(a, 0, "tpm")) && !mxIsEmpty(f)) {
        // Accept numeric (0/1) or logical scalar; nonzero -> true.
        o.tpm = (mxGetScalar(f) != 0.0);
    }

    if ((f = mxGetField(a, 0, "upsample")) && !mxIsEmpty(f)) {
        // Mirrors CLI --upsample: return output at internal
        // inference resolution (canonical RAS, no back-resample to
        // input grid). When set, the MEX populates plhs[1] with the
        // 4x4 affine of the upsampled grid; the wrapper writes it
        // into nii.NIFTIHeader.Affine so downstream tools see the
        // correct spatial metadata.
        o.upsample = (mxGetScalar(f) != 0.0);
    }

    if ((f = mxGetField(a, 0, "tpm_t")) && !mxIsEmpty(f)) {
        o.tpm_temperature = static_cast<float>(mxGetScalar(f));

        if (o.tpm_temperature <= 0.0f) {
            die("siamize:tpm_t",
                "opts.tpm_t must be > 0");
        }
    }
}

}  // namespace

/*******************************************************************************/
/*! \fn    void mexFunction(int nlhs, mxArray* plhs[],
                            int nrhs, const mxArray* prhs[])
    \brief Entry point invoked by the MATLAB / Octave MEX runtime

    Parses the positional arguments (img, affine, models[, opts])
    through the helpers above, dispatches the input voxel buffer on
    mxClassID to load_volume<SrcT>(), runs the inference pipeline
    (preprocess -> sliding_window -> postprocess), and constructs
    either a 3D uint8 labelmap or a 4D float32 TPM mxArray
    depending on `opts.tpm`. Errors are surfaced to MATLAB via
    die() / mexErrMsgIdAndTxt.

    \param  nlhs  number of expected output mxArrays
    \param  plhs  output mxArray vector (length = nlhs)
    \param  nrhs  number of input mxArrays
    \param  prhs  input mxArray vector (length = nrhs)
*/
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    // See siamize.cpp's main() for the rationale -- align CUDA's GPU
    // enumeration with what `nvidia-smi -L` prints so that opts.gpu/-G N
    // selects the same card the user sees in nvidia-smi. Default
    // FASTEST_FIRST silently routes the request to whichever card CUDA
    // ranked highest, which on heterogeneous boxes (e.g. RTX 2080 +
    // A100) doesn't match nvidia-smi's PCI ordering. overwrite=0 keeps
    // an explicit user setting in the launching shell.
#ifndef _WIN32
    setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 0);
#else

    // Windows: setenv doesn't exist; use _putenv_s which UNCONDITIONALLY
    // overwrites, so check first.
    if (std::getenv("CUDA_DEVICE_ORDER") == nullptr) {
        _putenv_s("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
    }

#endif

    // Query mode: siamex('backend') returns the compiled-in backend
    // name ('ort' or 'mnn') as a MATLAB char array. The MATLAB
    // dispatcher (matlab/siamize.m) calls this once at startup so it
    // can build the right fold filename (_fp16.onnx vs _int8.mnn) and
    // pick the right WeightVariant for download.
    if (nrhs == 1 && mxIsChar(prhs[0])) {
        std::string q = mx_to_string(prhs[0]);

        if (q == "backend") {
            plhs[0] = mxCreateString(siam::backend_name());
            return;
        }
    }

    if (nrhs < 3) {
        die("siamize:nrhs",
            "Usage: labels = siamize(img, affine, models, [opts])\n"
            "       backend = siamize('backend')\n"
            "See `help siamize` for details.");
    }

    try {
        // ---- parse inputs ------------------------------------------------
        const mxArray* a_img    = prhs[0];
        const mxArray* a_aff    = prhs[1];
        const mxArray* a_models = prhs[2];

        if (mxGetNumberOfDimensions(a_img) != 3) {
            die("siamize:img", "img must be a 3D array");
        }

        const mwSize* dims = mxGetDimensions(a_img);
        const int64_t X = static_cast<int64_t>(dims[0]);
        const int64_t Y = static_cast<int64_t>(dims[1]);
        const int64_t Z = static_cast<int64_t>(dims[2]);

        std::array<float, 16> affine = read_affine(a_aff);
        std::vector<std::string> models = read_models(a_models);

        Opts opts;

        if (nrhs >= 4) {
            read_opts(prhs[3], opts);
        }

        // Mirror the CLI: hoist opts.verbose into the siam_log global so
        // shared modules (sliding/weights/etc.) emit their tagged log
        // lines via mexPrintf without each call site checking a flag.
        siam::set_verbose(opts.verbose);

        // Default to single-fold fold_0 if caller passed []/'' for models,
        // matching the CLI behavior. resolve_model_path then either uses
        // an existing local file, hits the shared cache, or curls the
        // weight in from the default URL.
        if (models.empty()) {
#ifdef SIAMIZE_HAS_MNN
            models.push_back("fold_0_int8.mnn");
#else
            models.push_back("fold_0_fp16.onnx");
#endif
        }

        // Same backend / EP gating as siamize.cpp. The MEX inherits
        // opts.device from the caller's struct, so the decision
        // propagates identically.
        siam::WeightVariant weight_variant_mex = siam::WeightVariant::DYNSHAPE;
#ifdef SIAMIZE_HAS_MNN
        // SIAMIZE_BACKEND=mnn: the MNN runtime cannot consume .onnx, so
        // the mnn_i8a bundle is the only valid pre-baked weight set.
        weight_variant_mex = siam::WeightVariant::MNN;
#elif defined(SIAMIZE_HAS_COREML)

        if (opts.device == "coreml" || opts.device == "auto") {
            weight_variant_mex = siam::WeightVariant::COREML;
        }

#else

        if (opts.device == "coreml") {
            weight_variant_mex = siam::WeightVariant::COREML;
        }

#endif

        for (auto& m : models) {
            m = siam::resolve_model_path(m, opts.verbose, weight_variant_mex);
        }

        // ----- Lowmem preset (manual or auto-detected) ---------------------
        // Mirrors the CLI auto-safe block. opts.lowmem forces the
        // preset; otherwise we detect tight host RAM via /proc/meminfo
        // and tight GPU VRAM via nvidia-smi.
#ifdef SIAMIZE_HAS_MNN
        bool gpu_active = (opts.device == "auto" ||
                           opts.device == "opencl" ||
                           opts.device == "vulkan" ||
                           opts.device == "metal");
#else
        bool gpu_active = (opts.device == "auto" ||
                           opts.device == "cuda" ||
                           opts.device == "tensorrt");
#endif
        long avail_ram_mb_l  = 0;
        long avail_vram_mb_l = 0;
#ifdef __linux__
        {
            FILE* mf = fopen("/proc/meminfo", "r");

            if (mf) {
                char ln[128];

                while (fgets(ln, sizeof(ln), mf)) {
                    if (std::strncmp(ln, "MemAvailable:", 13) == 0) {
                        long kb = 0;

                        if (sscanf(ln, "MemAvailable: %ld kB", &kb) == 1) {
                            avail_ram_mb_l = kb / 1024;
                        }

                        break;
                    }
                }

                fclose(mf);
            }
        }
#elif defined(_WIN32)
        {
            MEMORYSTATUSEX ms;
            ms.dwLength = sizeof(ms);

            if (GlobalMemoryStatusEx(&ms)) {
                avail_ram_mb_l = static_cast<long>(
                                     ms.ullAvailPhys / (1024ULL * 1024ULL));
            }
        }
#endif

        if (gpu_active) {
#ifdef _WIN32
            FILE* nv = _popen("nvidia-smi --query-gpu=memory.free "
                              "--format=csv,noheader,nounits 2>NUL", "r");
#else
            FILE* nv = popen("nvidia-smi --query-gpu=memory.free "
                             "--format=csv,noheader,nounits 2>/dev/null", "r");
#endif

            if (nv) {
                char ln[64] = {0};

                if (fgets(ln, sizeof(ln), nv)) {
                    long parsed = 0;

                    if (sscanf(ln, "%ld", &parsed) == 1 && parsed > 0) {
                        avail_vram_mb_l = parsed;
                    }
                }

#ifdef _WIN32
                _pclose(nv);
#else
                pclose(nv);
#endif
            }
        }

        bool ram_tight  = opts.lowmem
                          || (avail_ram_mb_l  > 0 && avail_ram_mb_l  < 14 * 1024);
        bool vram_tight = (opts.lowmem && gpu_active)
                          || (avail_vram_mb_l > 0 && avail_vram_mb_l < 12 * 1024);

        if (ram_tight) {
            // Patch shrink: for ORT, gated on EXPLICIT opts.lowmem
            // because old fixed-shape .onnx may not accept smaller
            // patches. For MNN, the shipped doc=mnn_i8a bundle is
            // always dyn-shape, so we auto-shrink without the opt-in.
            // The 128x128x96 tier kicks in when BOTH RAM and VRAM are
            // tight + a GPU device is active -- MNN-OpenCL allocates
            // the full forward-pass workspace ahead of the first tile.
            bool patch_shrink_allowed = opts.lowmem;
#ifdef SIAMIZE_HAS_MNN
            patch_shrink_allowed = true;
#endif

            if (patch_shrink_allowed && !opts.patch_set) {
                if (vram_tight && gpu_active) {
                    opts.patch = {128, 128, 96};
                } else {
                    opts.patch = {192, 192, 128};
                }
            }

            if (!opts.cpu_arena_set) {
                opts.engine_tuning.cpu_arena = false;
            }
        }

        if (vram_tight) {
            if (!opts.cudnn_max_workspace_set) {
                opts.engine_tuning.cudnn_max_workspace = 0;
            }

            if (!opts.gpu_mem_limit_set) {
                opts.engine_tuning.gpu_mem_limit_bytes = 6ull * 1024 * 1024 * 1024;
            }
        }

        // Match the CLI: auto -> min(hardware_concurrency, 16). When
        // ram_tight is in effect (manual --lowmem or detected tight
        // RAM), the auto-cap drops to 8 to halve thread-local scratch.
        bool threads_auto = (opts.threads <= 0);

        if (threads_auto) {
            unsigned hc = std::thread::hardware_concurrency();
            int detected = (hc > 0) ? static_cast<int>(hc) : 4;
            const int auto_cap = ram_tight ? 8 : 16;
            opts.threads = std::min(detected, auto_cap);
        }

        if (opts.verbose) {
            siam::log_tag("siamize",
                          "v0.1.0  device=%s  threads=%d%s%s%s%s",
                          opts.device.c_str(), opts.threads,
                          threads_auto ? " (auto)" : "",
                          opts.tpm ? "  tpm" : "",
                          opts.engine_tuning.cpu_arena ? "" : "  no-arena",
                          opts.lowmem ? "  lowmem" : "");

            if (ram_tight || vram_tight) {
                const char* source = opts.lowmem ? "opts.lowmem=true"
                                     : "memory-tight host detected";
                siam::log_hint("lowmem preset applied (%s)", source);
            }
        }

        // ---- reorient to canonical (Z, Y, X) -----------------------------
        siam::Volume canon;
        std::array<int, 3> dst{}, sgn{};
        std::array<float, 16> affine_canon{};

        switch (mxGetClassID(a_img)) {
            case mxDOUBLE_CLASS:
                load_volume<double>(mxGetPr(a_img),
                                    X, Y, Z, affine, canon, dst, sgn, affine_canon);
                break;

            case mxSINGLE_CLASS:
                load_volume<float>(static_cast<const float*>(mxGetData(a_img)),
                                   X, Y, Z, affine, canon, dst, sgn, affine_canon);
                break;

            case mxUINT8_CLASS:
                load_volume<uint8_t>(static_cast<const uint8_t*>(mxGetData(a_img)),
                                     X, Y, Z, affine, canon, dst, sgn, affine_canon);
                break;

            case mxINT16_CLASS:
                load_volume<int16_t>(static_cast<const int16_t*>(mxGetData(a_img)),
                                     X, Y, Z, affine, canon, dst, sgn, affine_canon);
                break;

            case mxUINT16_CLASS:
                load_volume<uint16_t>(static_cast<const uint16_t*>(mxGetData(a_img)),
                                      X, Y, Z, affine, canon, dst, sgn, affine_canon);
                break;

            case mxINT32_CLASS:
                load_volume<int32_t>(static_cast<const int32_t*>(mxGetData(a_img)),
                                     X, Y, Z, affine, canon, dst, sgn, affine_canon);
                break;

            case mxUINT32_CLASS:
                load_volume<uint32_t>(static_cast<const uint32_t*>(mxGetData(a_img)),
                                      X, Y, Z, affine, canon, dst, sgn, affine_canon);
                break;

            case mxINT8_CLASS:
                load_volume<int8_t>(static_cast<const int8_t*>(mxGetData(a_img)),
                                    X, Y, Z, affine, canon, dst, sgn, affine_canon);
                break;

            default:
                die("siamize:dtype", "img dtype not supported (try double, single, int16, etc.)");
        }

        // ---- pipeline (same as siamize.cpp main) -------------------------
        std::array<int64_t, 3> shape_canon = {
            canon.shape[0], canon.shape[1], canon.shape[2]
        };

        auto crop = siam::crop_to_nonzero(canon);
        canon = siam::Volume{};   // free

        siam::zscore_inplace(crop.cropped);

        std::array<float, 3> spacing_zyx_in = {
            std::fabs(affine_canon[10]),
            std::fabs(affine_canon[5]),
            std::fabs(affine_canon[0]),
        };
        std::array<float, 3> spacing_new = {opts.spacing, opts.spacing, opts.spacing};
        auto new_shape = siam::compute_new_shape(
        {crop.cropped.shape[0], crop.cropped.shape[1], crop.cropped.shape[2]},
        spacing_zyx_in, spacing_new);

        siam::Volume resampled = siam::resample_cubic(
                                     crop.cropped, new_shape[0], new_shape[1], new_shape[2]);
        crop.cropped = siam::Volume{};

        siam::LogitsVolume logits = siam::sliding_window(
                                        resampled, models, opts.patch, opts.classes,
                                        opts.threads, 0.5f, opts.verbose,
                                        opts.device, opts.trt_cache, opts.engine_tuning);
        resampled = siam::Volume{};

        const int64_t cZ = crop.bbox[0][1] - crop.bbox[0][0];
        const int64_t cY = crop.bbox[1][1] - crop.bbox[1][0];
        const int64_t cX = crop.bbox[2][1] - crop.bbox[2][0];

        // ---- Upsample branch: skip back-resample, return output at the
        //      internal inference resolution (canonical RAS, full canonical
        //      extent). Mirrors siamize.cpp --upsample. Output is in MATLAB
        //      column-major (X, Y, Z[, C]); the 4x4 affine for the new grid
        //      goes to plhs[1] for the wrapper to install in
        //      nii.NIFTIHeader.Affine.
        if (opts.upsample) {
            // Full canonical shape at the inference spacing. compute_new_shape
            // takes (Z, Y, X) shape + (Z, Y, X) input spacing + (Z, Y, X)
            // target spacing -- same convention used for the cropped resample
            // above.
            const std::array<float, 3> target_zyx = {
                opts.spacing, opts.spacing, opts.spacing
            };
            auto full_up_shape = siam::compute_new_shape(
                                     shape_canon, spacing_zyx_in, target_zyx);

            // bbox_lo_up: where the (cropped, network-resampled) region sits
            // inside the full upsampled canonical grid. Per-axis floor of
            // bbox_lo scaled by input_spacing / target_spacing.
            std::array<int64_t, 3> bbox_lo_up{};

            for (int i = 0; i < 3; ++i) {
                bbox_lo_up[i] = static_cast<int64_t>(std::llround(
                        static_cast<double>(crop.bbox[i][0]) *
                        spacing_zyx_in[i] / target_zyx[i]));
            }

            const std::array<int64_t, 3> net_extent = {
                logits.spat[0], logits.spat[1], logits.spat[2]
            };
            std::array<int64_t, 3> bbox_hi_up{};

            for (int i = 0; i < 3; ++i) {
                bbox_hi_up[i] = std::min(bbox_lo_up[i] + net_extent[i],
                                         full_up_shape[i]);
            }

            // Upsampled-grid affine. Columns indexed by world-axis (X, Y, Z =
            // i=0..2); zooms_canon there is in world-axis order, the REVERSE
            // of spacing_zyx_in (which is canonical voxel order Z, Y, X).
            const std::array<float, 3> zooms_canon = {
                spacing_zyx_in[2], spacing_zyx_in[1], spacing_zyx_in[0]
            };
            std::array<float, 16> A_up{};
            A_up[15] = 1.0f;

            for (int r = 0; r < 3; ++r) {
                float origin_shift = 0.0f;

                for (int i = 0; i < 3; ++i) {
                    const float scale = opts.spacing / zooms_canon[i];
                    A_up[r * 4 + i] = affine_canon[r * 4 + i] * scale;
                    origin_shift += affine_canon[r * 4 + i] * 0.5f * (scale - 1.0f);
                }

                A_up[r * 4 + 3] = affine_canon[r * 4 + 3] + origin_shift;
            }

            siam::log_tag("upsample",
                          "full canonical at %.3f mm  (Z,Y,X) (%" PRId64 ",%"
                          PRId64 ",%" PRId64 ")  bbox lo (%" PRId64 ",%"
                          PRId64 ",%" PRId64 ")",
                          opts.spacing,
                          full_up_shape[0], full_up_shape[1], full_up_shape[2],
                          bbox_lo_up[0], bbox_lo_up[1], bbox_lo_up[2]);

            const int64_t Zup = full_up_shape[0];
            const int64_t Yup = full_up_shape[1];
            const int64_t Xup = full_up_shape[2];
            const int64_t plane_up = Yup * Xup;

            const int64_t lzN = logits.spat[0];
            const int64_t lyN = logits.spat[1];
            const int64_t lxN = logits.spat[2];

            if (opts.tpm) {
                // softmax + optional SPM merge at network resolution.
                // See the non-upsample TPM branch below for the rationale on
                // the per-voxel scratch buffer pattern; same logic applies.
                const bool spm = (opts.class_set == siam::ClassSet::SPM);
                const float inv_T = 1.0f / opts.tpm_temperature;
                const int64_t N_net = lzN * lyN * lxN;

                #pragma omp parallel for schedule(static)

                for (int64_t i = 0; i < N_net; ++i) {
                    float m = logits.channel_ptr(0)[i] * inv_T;

                    for (int64_t c = 1; c < opts.classes; ++c) {
                        float v = logits.channel_ptr(c)[i] * inv_T;

                        if (v > m) {
                            m = v;
                        }
                    }

                    float s = 0.0f;

                    for (int64_t c = 0; c < opts.classes; ++c) {
                        float e = std::exp(logits.channel_ptr(c)[i] * inv_T - m);
                        logits.channel_ptr(c)[i] = e;
                        s += e;
                    }

                    float invs = 1.0f / s;

                    for (int64_t c = 0; c < opts.classes; ++c) {
                        logits.channel_ptr(c)[i] *= invs;
                    }

                    if (spm) {
                        float spm_bin[siam::SPM6_NUM_CLASSES] = {0, 0, 0, 0, 0, 0};

                        for (int64_t c = 0; c < opts.classes; ++c) {
                            spm_bin[siam::SIAM18_TO_SPM6[c]] += logits.channel_ptr(c)[i];
                        }

                        for (int64_t b = 0; b < siam::SPM6_NUM_CLASSES; ++b) {
                            logits.channel_ptr(b)[i] = spm_bin[b];
                        }
                    }
                }

                // Allocate MATLAB 4D output [Xup, Yup, Zup, C] (column-major
                // matches canonical X-fastest layout). Background channel
                // starts at 1.0 outside the bbox (channel 0 default, or
                // channel 5 = Air in SPM ordering).
                const int64_t out_classes = opts.classes_out;
                const int64_t bg_channel  = spm
                                            ? siam::SPM6_AIR_CHANNEL
                                            : 0;
                const mwSize tdims[4] = {
                    static_cast<mwSize>(Xup),
                    static_cast<mwSize>(Yup),
                    static_cast<mwSize>(Zup),
                    static_cast<mwSize>(out_classes),
                };
                plhs[0] = mxCreateNumericArray(4, tdims, mxSINGLE_CLASS, mxREAL);
                float* tpm_ptr = static_cast<float*>(mxGetData(plhs[0]));
                const int64_t per_chan_up = Xup * Yup * Zup;
                std::fill_n(tpm_ptr + bg_channel * per_chan_up, per_chan_up, 1.0f);

                for (int64_t c = 0; c < out_classes; ++c) {
                    const float* src = logits.channel_ptr(c);
                    float* dst_chan = tpm_ptr + c * per_chan_up;

                    for (int64_t z = bbox_lo_up[0]; z < bbox_hi_up[0]; ++z) {
                        const int64_t lz = z - bbox_lo_up[0];

                        for (int64_t y = bbox_lo_up[1]; y < bbox_hi_up[1]; ++y) {
                            const int64_t ly = y - bbox_lo_up[1];
                            const int64_t copy_n_x = bbox_hi_up[2] - bbox_lo_up[2];
                            std::copy_n(
                                &src[lz * lyN * lxN + ly * lxN],
                                copy_n_x,
                                &dst_chan[z * plane_up + y * Xup + bbox_lo_up[2]]);
                        }
                    }
                }
            } else {
                // Labels: argmax (SPM-aware) at network resolution, then
                // bbox-copy into the upsampled canonical buffer. Outside-
                // bbox default is label 0 (background) by default, label
                // 5 (Air) in SPM ordering.
                const bool spm = (opts.class_set == siam::ClassSet::SPM);
                const uint8_t bg_label = spm
                                         ? static_cast<uint8_t>(siam::SPM6_AIR_CHANNEL)
                                         : static_cast<uint8_t>(0);
                const mwSize odims[3] = {
                    static_cast<mwSize>(Xup),
                    static_cast<mwSize>(Yup),
                    static_cast<mwSize>(Zup),
                };
                plhs[0] = mxCreateNumericArray(3, odims, mxUINT8_CLASS, mxREAL);
                uint8_t* out_ptr = static_cast<uint8_t*>(mxGetData(plhs[0]));
                std::fill_n(out_ptr,
                            static_cast<size_t>(Xup) * Yup * Zup, bg_label);

                #pragma omp parallel for schedule(static) collapse(2)

                for (int64_t z = bbox_lo_up[0]; z < bbox_hi_up[0]; ++z) {
                    for (int64_t y = bbox_lo_up[1]; y < bbox_hi_up[1]; ++y) {
                        const int64_t lz = z - bbox_lo_up[0];
                        const int64_t ly = y - bbox_lo_up[1];

                        for (int64_t x = bbox_lo_up[2]; x < bbox_hi_up[2]; ++x) {
                            const int64_t lx = x - bbox_lo_up[2];
                            const int64_t i_net = lz * lyN * lxN + ly * lxN + lx;
                            int best = 0;

                            if (spm) {
                                float M = logits.channel_ptr(0)[i_net];

                                for (int64_t c = 1; c < opts.classes; ++c) {
                                    float v = logits.channel_ptr(c)[i_net];

                                    if (v > M) {
                                        M = v;
                                    }
                                }

                                float bin[siam::SPM6_NUM_CLASSES] = {0, 0, 0, 0, 0, 0};

                                for (int64_t c = 0; c < opts.classes; ++c) {
                                    bin[siam::SIAM18_TO_SPM6[c]] +=
                                        std::exp(logits.channel_ptr(c)[i_net] - M);
                                }

                                float bestv = bin[0];

                                for (int64_t b = 1; b < siam::SPM6_NUM_CLASSES; ++b) {
                                    if (bin[b] > bestv) {
                                        bestv = bin[b];
                                        best = static_cast<int>(b);
                                    }
                                }
                            } else {
                                float bestv = logits.channel_ptr(0)[i_net];

                                for (int64_t c = 1; c < opts.classes; ++c) {
                                    float v = logits.channel_ptr(c)[i_net];

                                    if (v > bestv) {
                                        bestv = v;
                                        best = static_cast<int>(c);
                                    }
                                }
                            }

                            out_ptr[z * plane_up + y * Xup + x] =
                                static_cast<uint8_t>(best);
                        }
                    }
                }
            }

            // plhs[1]: 4x4 affine for the upsampled grid, row-major, double.
            // MATLAB stores 4x4 column-major; mxCreateDoubleMatrix(4,4)
            // gives that. We write transposed so MATLAB's nii.NIFTIHeader.Affine
            // (which is row-major 3x4 or 4x4 in the JNIfTI convention) reads
            // correctly.
            if (nlhs >= 2) {
                plhs[1] = mxCreateDoubleMatrix(4, 4, mxREAL);
                double* a_ptr = mxGetPr(plhs[1]);

                for (int r = 0; r < 4; ++r) {
                    for (int c_ = 0; c_ < 4; ++c_) {
                        a_ptr[c_ * 4 + r] = static_cast<double>(A_up[r * 4 + c_]);
                    }
                }
            }

            return;
        }

        // ---- end upsample branch; non-upsample path below ----

        siam::LogitsVolume logits_back;
        logits_back.resize(opts.classes, cZ, cY, cX);
        {
            siam::Volume tmp_in;
            tmp_in.shape = {logits.spat[0], logits.spat[1], logits.spat[2]};
            tmp_in.data.resize(static_cast<size_t>(logits.cstride()));

            for (int64_t c = 0; c < opts.classes; ++c) {
                std::copy_n(logits.channel_ptr(c), logits.cstride(), tmp_in.data.data());
                siam::Volume v = siam::resample_trilinear(
                                     tmp_in, logits_back.spat[0], logits_back.spat[1], logits_back.spat[2]);
                std::copy_n(v.data.data(), v.numel(), logits_back.channel_ptr(c));
            }
        }
        logits = siam::LogitsVolume{};

        const int64_t Zc = shape_canon[0], Yc = shape_canon[1], Xc = shape_canon[2];

        if (opts.tpm) {
            // ---- TPM mode: softmax -> (optional SPM merge) ->
            //               un-crop -> reorient -> 4D float32 ----
            // See siamize.cpp for the rationale on the per-voxel
            // SPM merge running in-place inside the softmax loop.
            const bool spm = (opts.class_set == siam::ClassSet::SPM);
            const float inv_T = 1.0f / opts.tpm_temperature;
            const int64_t N = cZ * cY * cX;

            #pragma omp parallel for schedule(static)

            for (int64_t i = 0; i < N; ++i) {
                float m = logits_back.channel_ptr(0)[i] * inv_T;

                for (int64_t c = 1; c < opts.classes; ++c) {
                    float v = logits_back.channel_ptr(c)[i] * inv_T;

                    if (v > m) {
                        m = v;
                    }
                }

                float s = 0.0f;

                for (int64_t c = 0; c < opts.classes; ++c) {
                    float e = std::exp(logits_back.channel_ptr(c)[i] * inv_T - m);
                    logits_back.channel_ptr(c)[i] = e;
                    s += e;
                }

                float invs = 1.0f / s;

                for (int64_t c = 0; c < opts.classes; ++c) {
                    logits_back.channel_ptr(c)[i] *= invs;
                }

                if (spm) {
                    float spm_bin[siam::SPM6_NUM_CLASSES] = {0, 0, 0, 0, 0, 0};

                    for (int64_t c = 0; c < opts.classes; ++c) {
                        spm_bin[siam::SIAM18_TO_SPM6[c]] += logits_back.channel_ptr(c)[i];
                    }

                    for (int64_t b = 0; b < siam::SPM6_NUM_CLASSES; ++b) {
                        logits_back.channel_ptr(b)[i] = spm_bin[b];
                    }
                }
            }

            // Un-crop into canonical (C, Zc, Yc, Xc); outside bbox =
            // [1, 0, ...] in default (background-first) ordering, or
            // [0, 0, 0, 0, 0, 1] in SPM ordering (Air at channel 5).
            const int64_t out_classes  = opts.classes_out;
            const int64_t bg_chan_tpm  = spm ? siam::SPM6_AIR_CHANNEL : 0;
            std::vector<float> tpm_canon(
                static_cast<size_t>(out_classes) * Zc * Yc * Xc, 0.0f);
            std::fill_n(tpm_canon.data() + bg_chan_tpm * Zc * Yc * Xc,
                        Zc * Yc * Xc, 1.0f);

            for (int64_t c = 0; c < out_classes; ++c) {
                const float* src = logits_back.channel_ptr(c);
                float* dst_chan = tpm_canon.data() + c * Zc * Yc * Xc;

                for (int64_t z = 0; z < cZ; ++z) {
                    for (int64_t y = 0; y < cY; ++y) {
                        std::copy_n(
                            &src[z * cY * cX + y * cX],
                            cX,
                            &dst_chan[(z + crop.bbox[0][0]) * Yc * Xc +
                                                            (y + crop.bbox[1][0]) * Xc +
                                                            crop.bbox[2][0]]);
                    }
                }
            }

            // Allocate the MATLAB 4D output (single, [X, Y, Z, C]) and
            // reorient each channel to input-axis order.
            const mwSize tdims[4] = {
                static_cast<mwSize>(X),
                static_cast<mwSize>(Y),
                static_cast<mwSize>(Z),
                static_cast<mwSize>(out_classes),
            };
            plhs[0] = mxCreateNumericArray(4, tdims, mxSINGLE_CLASS, mxREAL);
            float* tpm_ptr = static_cast<float*>(mxGetData(plhs[0]));
            const int64_t per_chan_out = X * Y * Z;
            const int64_t per_chan_in  = Zc * Yc * Xc;

            for (int64_t c = 0; c < out_classes; ++c) {
                siam::copy_reorient_from_canonical<float, float>(
                    tpm_canon.data() + c * per_chan_in,
                    Zc, Yc, Xc,
                    X, Y, Z, dst, sgn,
                    tpm_ptr + c * per_chan_out);
            }
        } else {
            // ---- Labels mode: argmax -> un-crop -> reorient -> 3D uint8 ---
            // --classes spm: see siamize.cpp for why argmax-on-logits
            // doesn't commute with the SPM merge. Use exp-sum-per-bin
            // and argmax that; skip the divide-by-Z since argmax is
            // scale invariant.
            const bool spm_labels = (opts.class_set == siam::ClassSet::SPM);
            const uint8_t bg_label = spm_labels
                                     ? static_cast<uint8_t>(siam::SPM6_AIR_CHANNEL)
                                     : static_cast<uint8_t>(0);
            std::vector<uint8_t> labels_crop(static_cast<size_t>(cZ) * cY * cX, bg_label);

            #pragma omp parallel for schedule(static)

            for (int64_t z = 0; z < cZ; ++z) {
                for (int64_t y = 0; y < cY; ++y) {
                    for (int64_t x = 0; x < cX; ++x) {
                        int64_t i = z * cY * cX + y * cX + x;
                        int best = 0;

                        if (spm_labels) {
                            float M = logits_back.channel_ptr(0)[i];

                            for (int64_t c = 1; c < opts.classes; ++c) {
                                float v = logits_back.channel_ptr(c)[i];

                                if (v > M) {
                                    M = v;
                                }
                            }

                            float bin[siam::SPM6_NUM_CLASSES] = {0, 0, 0, 0, 0, 0};

                            for (int64_t c = 0; c < opts.classes; ++c) {
                                bin[siam::SIAM18_TO_SPM6[c]] +=
                                    std::exp(logits_back.channel_ptr(c)[i] - M);
                            }

                            float bestv = bin[0];

                            for (int64_t b = 1; b < siam::SPM6_NUM_CLASSES; ++b) {
                                if (bin[b] > bestv) {
                                    bestv = bin[b];
                                    best = static_cast<int>(b);
                                }
                            }
                        } else {
                            float bestv = logits_back.channel_ptr(0)[i];

                            for (int64_t c = 1; c < opts.classes; ++c) {
                                float v = logits_back.channel_ptr(c)[i];

                                if (v > bestv) {
                                    bestv = v;
                                    best = static_cast<int>(c);
                                }
                            }
                        }

                        labels_crop[i] = static_cast<uint8_t>(best);
                    }
                }
            }

            std::vector<uint8_t> labels_canon(
                static_cast<size_t>(Zc) * Yc * Xc, bg_label);

            for (int64_t z = 0; z < cZ; ++z) {
                for (int64_t y = 0; y < cY; ++y) {
                    std::copy_n(&labels_crop[z * cY * cX + y * cX],
                                cX,
                                &labels_canon[(z + crop.bbox[0][0]) * Yc * Xc
                                                                    + (y + crop.bbox[1][0]) * Xc
                                                                    + crop.bbox[2][0]]);
                }
            }

            const mwSize odims[3] = {
                static_cast<mwSize>(X),
                static_cast<mwSize>(Y),
                static_cast<mwSize>(Z),
            };
            plhs[0] = mxCreateNumericArray(3, odims, mxUINT8_CLASS, mxREAL);
            uint8_t* out_ptr = static_cast<uint8_t*>(mxGetData(plhs[0]));

            siam::copy_reorient_from_canonical<uint8_t>(
                labels_canon.data(),
                Zc, Yc, Xc,
                X, Y, Z,
                dst, sgn,
                out_ptr);
        }

    } catch (const std::exception& e) {
        die("siamize:runtime", e.what());
    }
}
