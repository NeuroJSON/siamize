// siamize MEX wrapper for MATLAB / Octave.
//
// Signature (MATLAB):
//   labels = siamize(img, affine, models)
//   labels = siamize(img, affine, models, opts)
//
// Inputs
//   img      -- 3D numeric array. Any standard MATLAB dtype (uint8/int16/
//               int32/uint16/uint32/single/double). Interpreted as a NIfTI
//               voxel grid in MATLAB's natural [X, Y, Z] column-major layout.
//   affine   -- 3x4 or 4x4 double matrix. Same convention as jsonlab's
//               loadnifti(): nii.NIFTIHeader.Affine, i.e. rows = (R, A, S)
//               world axes, columns = voxel axes (last column = origin).
//   models   -- char (single path) or cellstr (list of .onnx folds). Logits
//               are averaged across folds.
//   opts     -- optional struct with any of:
//                 .device       'auto' (default) | 'cpu' | 'cuda' | 'tensorrt'
//                 .threads      int (default 0 = all available cores)
//                 .patch        [pz, py, px] (default [256 256 192])
//                 .spacing      double (default 0.75 mm)
//                 .classes      int (default 18)
//                 .trt_cache    char (default ~/.cache/siamize/trt)
//                 .verbose      logical (default false)
//
// Output
//   labels   -- uint8 3D array of the SAME shape as `img`, in input axis
//               order. Suitable for direct round-trip via:
//                 nii = loadnifti('input.nii.gz');
//                 nii_out = nii;
//                 nii_out.NIFTIData = siamize(nii.NIFTIData, ...
//                                              nii.NIFTIHeader.Affine, ...
//                                              {'fold_0_fp16.onnx'});
//                 savenifti(nii_out, 'output.nii.gz');

#include "mex.h"

#include "siam.h"
#include "orient.h"
#include "preprocess.h"
#include "sliding.h"
#include "weights.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void die(const char* id, const std::string& msg) {
    mexErrMsgIdAndTxt(id, "%s", msg.c_str());
}

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
void load_volume(const SrcT* src, int64_t X, int64_t Y, int64_t Z,
                 const std::array<float, 16>& affine,
                 siam::Volume& canon_out,
                 std::array<int, 3>& dst, std::array<int, 3>& sgn,
                 std::array<float, 16>& affine_canon_out) {
    siam::axes_to_canonical(affine, dst, sgn);
    siam::copy_reorient_to_canonical<SrcT>(src, X, Y, Z, dst, sgn, canon_out);
    affine_canon_out = siam::canonicalize_affine(affine, dst, sgn, {X, Y, Z});
}

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

std::vector<std::string> read_models(const mxArray* a) {
    std::vector<std::string> out;

    if (mxIsEmpty(a)) {
        // empty -> caller wants the default single-fold
        return out;
    }

    if (mxIsChar(a)) {
        out.push_back(mx_to_string(a));
    } else if (mxIsCell(a)) {
        mwSize n = mxGetNumberOfElements(a);
        out.reserve(n);

        for (mwSize i = 0; i < n; ++i) {
            const mxArray* c = mxGetCell(a, i);
            out.push_back(mx_to_string(c));
        }
    } else {
        die("siamize:models", "models must be a char array, cellstr, or [] for default");
    }

    return out;
}

struct Opts {
    std::string device = "auto";
    int threads = 0;
    std::array<int64_t, 3> patch = {256, 256, 192};
    float spacing = 0.75f;
    int64_t classes = 18;
    std::string trt_cache;
    bool verbose = false;
    siam::CudaTuning cuda_tuning;
    bool  tpm = false;               // true -> output 4D float32 TPM instead of 3D labels
    float tpm_temperature = 1.0f;    // softmax temperature for the TPM
};

double opt_double(const mxArray* s, const char* name, double fallback) {
    const mxArray* f = mxGetField(s, 0, name);
    return (f && mxIsDouble(f) && !mxIsEmpty(f)) ? mxGetScalar(f) : fallback;
}

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
    }

    if ((f = mxGetField(a, 0, "spacing"))  && !mxIsEmpty(f)) {
        o.spacing = static_cast<float>(mxGetScalar(f));
    }

    if ((f = mxGetField(a, 0, "classes"))  && !mxIsEmpty(f)) {
        o.classes = static_cast<int64_t>(mxGetScalar(f));
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
    }

    // CUDA EP tuning knobs. All optional; absent fields keep CudaTuning
    // defaults (which are ORT defaults).
    if ((f = mxGetField(a, 0, "cudnn_max_workspace")) && !mxIsEmpty(f)) {
        o.cuda_tuning.cudnn_max_workspace = static_cast<int>(mxGetScalar(f));
    }

    if ((f = mxGetField(a, 0, "arena_extend")) && !mxIsEmpty(f)) {
        // Accept either the string ('same'/'power') or numeric (0/1).
        if (mxIsChar(f)) {
            std::string s = mx_to_string(f);
            o.cuda_tuning.arena_same_as_req =
                (s == "same" || s == "kSameAsRequested") ? 1 : 0;
        } else {
            o.cuda_tuning.arena_same_as_req = static_cast<int>(mxGetScalar(f)) ? 1 : 0;
        }
    }

    if ((f = mxGetField(a, 0, "cudnn_algo")) && !mxIsEmpty(f)) {
        std::string s = mx_to_string(f);

        // Canonicalize to ORT's uppercase string form.
        for (auto& c : s) {
            c = static_cast<char>(::toupper(static_cast<unsigned char>(c)));
        }

        if (s == "DEFAULT" || s == "HEURISTIC" || s == "EXHAUSTIVE") {
            o.cuda_tuning.algo_search = s;
        } else {
            die("siamize:cudnn_algo",
                "opts.cudnn_algo must be 'default' | 'heuristic' | 'exhaustive'");
        }
    }

    if ((f = mxGetField(a, 0, "gpu_mem_limit")) && !mxIsEmpty(f)) {
        // Pass through as raw byte count. The .m wrapper is responsible for
        // K/M/G suffix interpretation if it wants -- here we just take a number.
        o.cuda_tuning.gpu_mem_limit_bytes =
            static_cast<size_t>(mxGetScalar(f));
    }

    if ((f = mxGetField(a, 0, "gpu")) && !mxIsEmpty(f)) {
        o.cuda_tuning.gpuid = static_cast<int>(mxGetScalar(f));
    }

    if ((f = mxGetField(a, 0, "tpm")) && !mxIsEmpty(f)) {
        // Accept numeric (0/1) or logical scalar; nonzero -> true.
        o.tpm = (mxGetScalar(f) != 0.0);
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

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 3) {
        die("siamize:nrhs",
            "Usage: labels = siamize(img, affine, models, [opts])\n"
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

        // Default to single-fold fold_0 if caller passed []/'' for models,
        // matching the CLI behavior. resolve_model_path then either uses
        // an existing local file, hits the shared cache, or curls the
        // weight in from the default URL.
        if (models.empty()) {
            models.push_back("fold_0_fp16.onnx");
        }

        for (auto& m : models) {
            m = siam::resolve_model_path(m, opts.verbose);
        }

        if (opts.threads <= 0) {
            unsigned hc = std::thread::hardware_concurrency();
            opts.threads = (hc > 0) ? static_cast<int>(hc) : 4;
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
                                        opts.device, opts.trt_cache, opts.cuda_tuning);
        resampled = siam::Volume{};

        const int64_t cZ = crop.bbox[0][1] - crop.bbox[0][0];
        const int64_t cY = crop.bbox[1][1] - crop.bbox[1][0];
        const int64_t cX = crop.bbox[2][1] - crop.bbox[2][0];

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
            // ---- TPM mode: softmax -> un-crop -> reorient -> 4D float32 ---
            const float inv_T = 1.0f / opts.tpm_temperature;
            const int64_t N = cZ * cY * cX;

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
            }

            // Un-crop into canonical (C, Zc, Yc, Xc); outside bbox = [1,0,...]
            std::vector<float> tpm_canon(
                static_cast<size_t>(opts.classes) * Zc * Yc * Xc, 0.0f);
            std::fill_n(tpm_canon.data(), Zc * Yc * Xc, 1.0f);

            for (int64_t c = 0; c < opts.classes; ++c) {
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
                static_cast<mwSize>(opts.classes),
            };
            plhs[0] = mxCreateNumericArray(4, tdims, mxSINGLE_CLASS, mxREAL);
            float* tpm_ptr = static_cast<float*>(mxGetData(plhs[0]));
            const int64_t per_chan_out = X * Y * Z;
            const int64_t per_chan_in  = Zc * Yc * Xc;

            for (int64_t c = 0; c < opts.classes; ++c) {
                siam::copy_reorient_from_canonical<float, float>(
                    tpm_canon.data() + c * per_chan_in,
                    Zc, Yc, Xc,
                    X, Y, Z, dst, sgn,
                    tpm_ptr + c * per_chan_out);
            }
        } else {
            // ---- Labels mode: argmax -> un-crop -> reorient -> 3D uint8 ---
            std::vector<uint8_t> labels_crop(static_cast<size_t>(cZ) * cY * cX, 0);

            for (int64_t z = 0; z < cZ; ++z) {
                for (int64_t y = 0; y < cY; ++y) {
                    for (int64_t x = 0; x < cX; ++x) {
                        int64_t i = z * cY * cX + y * cX + x;
                        int best = 0;
                        float bestv = logits_back.channel_ptr(0)[i];

                        for (int64_t c = 1; c < opts.classes; ++c) {
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

            std::vector<uint8_t> labels_canon(
                static_cast<size_t>(Zc) * Yc * Xc, 0);

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
