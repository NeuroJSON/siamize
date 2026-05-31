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
\file    siamize.cpp
\brief   Command-line entry point: argument parsing + end-to-end pipeline

Implements the `siamize` CLI binary. The pipeline assembled here:

  -# Parse CLI flags (-i/-o/-M/-c/-G/-t/-P/-u/-C/-F/--tpm/--tpm-t/-v ...)
     into local state. Most flags have a long + short form; see
     siam::usage() for the canonical help text.
  -# Resolve model paths via weights::resolve_model_path (which
     auto-downloads from NeuroJSON on first use).
  -# Load the input volume:
     - `.nii` / `.nii.gz`  -> nifti_io::load_nifti_ras
     - `.jnii` / `.bnii`   -> jnifti_io::load_jnifti_ras
     Both produce a canonical (Z, Y, X) RAS NiftiImage.
  -# Pre-network preprocessing: nonzero crop -> spacing resample
     (Catmull-Rom cubic) -> z-score normalize.
  -# Sliding-window inference via siam::sliding_window with the
     selected Execution Provider (CPU / CUDA / TensorRT).
  -# Post-process:
     - Labels:   argmax -> un-resample -> un-crop -> uint8 (X,Y,Z)
     - TPM:      softmax (with optional temperature) -> un-resample
                 -> un-crop -> float32 4D (X,Y,Z,C)
  -# Save in the format selected by `-F/--format`:
     `nii` (NIfTI-1), `jnii` (JSON-text JNIfTI), or `bnii`
     (BJData binary JNIfTI).

The CLI mirrors the Python siam-pred reference but stays
single-binary with no Python runtime.
*******************************************************************************/

#include "siam.h"
#include "siam_log.h"
#include "nifti_io.h"
#include "jnifti_io.h"
#include "preprocess.h"
#include "sliding.h"
#include "weights.h"

#ifdef SIAMIZE_HAS_ORT
    #include <onnxruntime_cxx_api.h>
#endif

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#endif
#ifdef __APPLE__
    #include <sys/sysctl.h>
    #include <sys/types.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

using siam::ClassSet;
using siam::SIAM18_TO_SPM6;
using siam::SPM6_NUM_CLASSES;
using siam::SPM6_AIR_CHANNEL;


/*******************************************************************************/
/*! \fn    std::vector<std::string> split_csv(const std::string& s)
    \brief Split a comma-separated string into a list of non-empty tokens

    Used to parse the `-M / --models` argument, which accepts a
    comma-separated list of fold specs (each spec is either a path,
    a basename, or a one-character fold-index shortcut).

    \param  s  the raw input string
    \return    list of non-empty tokens
*/
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

/*******************************************************************************/
/*! \fn    long available_ram_mb()
    \brief Best-effort host-available-RAM probe (Linux only; 0 elsewhere)

    Reads `MemAvailable` from /proc/meminfo (the kernel's own estimate
    of memory available to start new applications without swapping).
    Used to drive the [hint] pre-flight memory-pressure warning. On
    non-Linux platforms returns 0 (caller skips the hint).

    \return  available RAM in MiB, or 0 if unknown
*/
long available_ram_mb() {
#ifdef __linux__
    std::ifstream f("/proc/meminfo");

    if (!f) {
        return 0;
    }

    std::string line;

    while (std::getline(f, line)) {
        if (line.rfind("MemAvailable:", 0) == 0) {
            long kb = 0;

            if (std::sscanf(line.c_str(), "MemAvailable: %ld kB", &kb) == 1) {
                return kb / 1024;
            }
        }
    }

    return 0;
#elif defined(_WIN32)
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);

    if (GlobalMemoryStatusEx(&ms)) {
        return static_cast<long>(ms.ullAvailPhys / (1024ULL * 1024ULL));
    }

    return 0;
#elif defined(__APPLE__)
    // sysctl HW_MEMSIZE gives total physical -- not "available", but
    // close enough for the auto-lowmem heuristic; the user can always
    // pass flags explicitly.
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    int64_t bytes = 0;
    size_t len = sizeof(bytes);

    if (sysctl(mib, 2, &bytes, &len, NULL, 0) == 0 && bytes > 0) {
        return static_cast<long>(bytes / (1024 * 1024));
    }

    return 0;
#else
    return 0;
#endif
}

/*******************************************************************************/
/*! \fn    long available_vram_mb(int gpuid)
    \brief Best-effort GPU-free-memory probe via nvidia-smi (returns 0 if unknown)

    Spawns `nvidia-smi --id=N --query-gpu=memory.free --format=csv,
    noheader,nounits` via popen and parses the first integer on its
    first line. Works on any host with the NVIDIA driver/userland
    installed. Returns 0 on any failure -- caller treats 0 as
    "unknown, skip the VRAM-tight auto-safe path".

    Targets the specific GPU the user selected via -G/--gpu instead
    of always GPU 0: on a multi-GPU box the auto-lowmem heuristic
    must look at the device that will actually run the inference,
    otherwise (e.g.) the user picks a 24 GB 3090 but the probe sees
    an 8 GB 2080 at index 0 and caps gpu_mem_limit at 6 GB.

    \param  gpuid  0-based GPU index to query (default 0)
    \return  free VRAM in MiB on the requested GPU, or 0 if unknown
*/
long available_vram_mb(int gpuid = 0) {
    char cmd[256];
#ifdef _WIN32
    std::snprintf(cmd, sizeof(cmd),
                  "nvidia-smi --id=%d --query-gpu=memory.free "
                  "--format=csv,noheader,nounits 2>NUL", gpuid);
    FILE* pipe = _popen(cmd, "r");
#else
    std::snprintf(cmd, sizeof(cmd),
                  "nvidia-smi --id=%d --query-gpu=memory.free "
                  "--format=csv,noheader,nounits 2>/dev/null", gpuid);
    FILE* pipe = popen(cmd, "r");
#endif

    if (!pipe) {
        return 0;
    }

    long mb = 0;
    char buf[64] = {0};

    if (std::fgets(buf, sizeof(buf), pipe) != nullptr) {
        // First line carries the free MB for the requested GPU as a plain int.
        long parsed = 0;

        if (std::sscanf(buf, "%ld", &parsed) == 1 && parsed > 0) {
            mb = parsed;
        }
    }

#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    return mb;
}

/*******************************************************************************/
/*! \fn    std::string expand_fold_shortcut(const std::string& tok)
    \brief Expand a single-digit fold shortcut to the canonical filename

    `"0".."9"` becomes `"fold_<d>_fp32.mnn"` (MNN build) or
    `"fold_<d>_fp16.onnx"` (ONNX build); anything else passes through
    unchanged. Lets users write `-M 0,1,2,3,4` instead of spelling
    out the full filenames.

    \param  tok  one model spec token
    \return      expanded filename if \a tok was a digit, else \a tok verbatim
*/
std::string expand_fold_shortcut(const std::string& tok) {
    if (tok.size() == 1 && tok[0] >= '0' && tok[0] <= '9') {
#ifdef SIAMIZE_HAS_MNN
        // MNN backend: native-Conv3D .mnn binaries with fp32 weights
        // (see WeightVariant::MNN, doc=mnn_n3d). Future fp16/int8
        // variants will live under the same doc as fold_<N>_fp16.mnn
        // / fold_<N>_int8.mnn.
        return "fold_" + tok + "_fp32.mnn";
#else
        return "fold_" + tok + "_fp16.onnx";
#endif
    }

    return tok;
}

/*******************************************************************************/
/*! \fn    void usage(const char* exe)
    \brief Print the full CLI help text to stderr

    Documents every flag (short + long), the input/output extensions
    handled by extension dispatch, the -F/--format options, the
    execution-provider choices on `-c/--compute`, and a worked
    examples block. Kept in sync with the README's CLI section.

    \param  exe  argv[0] (typically `siamize`)
*/
void usage(const char* exe) {
    std::fprintf(stderr,
                 "[siamize]  v0.1.0  native C++/ONNX port of SIAM v0.3 brain segmentation\n"
                 "           https://github.com/NeuroJSON/siamize\n"
                 "\n"
                 "Usage: %s -i input.nii(.gz) -o output.nii.gz [-M 0,1,2,3,4] [-c auto] [-G 0] [-t N] [-v]\n"
                 "\n"
                 "  -i, --input         input volume (.nii / .nii.gz NIfTI-1/2, or .jnii / .bnii\n"
                 "                      JNIfTI; format inferred from extension). 3D scalar.\n"
                 "  -o, --output        output file. By default a 3D uint8 labelmap; pass `--tpm`\n"
                 "                      to write a 4D float32 tissue probability map instead.\n"
                 "                      Container format follows -F/--format (default nii).\n"
                 "                      JNIfTI spec: https://neurojson.org/jnifti\n"
                 "  -F, --format        output container: nii (NIfTI-1, gzipped if .gz) | jnii\n"
                 "                      (JSON-text JNIfTI, zlib+base64 payload) | bnii (BJData\n"
                 "                      binary JNIfTI, zlib payload). If -F is omitted, the\n"
                 "                      format is inferred from the -o extension: `.jnii` ->\n"
                 "                      jnii, `.bnii` -> bnii, anything else -> nii.\n"
                 "                      .nii.gz has the broadest tool support, but the\n"
                 "                      compressed payload is not directly searchable; .bnii\n"
                 "                      interoperates with jsonlab / the NeuroJSON ecosystem\n"
                 "                      and indexers can walk its structured keys; .jnii is\n"
                 "                      human-readable JSON.\n"
#ifdef SIAMIZE_HAS_MNN
                 "  -M, --models        comma-separated .mnn files (one per fold), logits are averaged.\n"
                 "                      Defaults to single-fold 'fold_0_fp32.mnn'. Each entry can be\n"
                 "                      a full path, a basename, or a single digit shortcut (e.g.\n"
                 "                      -M 0,1,2,3,4 expands to fold_<N>_fp32.mnn). Bare basenames\n"
                 "                      are looked up under $SIAMIZE_CACHE_DIR/mnn_n3d/ (default\n"
                 "                      $HOME/.cache/siamize/models/mnn_n3d/) and auto-downloaded from\n"
                 "                      $SIAMIZE_WEIGHTS_BASE_URL on miss. The shipped .mnn files are\n"
                 "                      dynamic-shape native-Conv3D fp32 weights (~540 MB / fold).\n"
#else
                 "  -M, --models        comma-separated .onnx files (one per fold), logits are averaged.\n"
                 "                      Defaults to single-fold 'fold_0_fp16.onnx'. Each entry can be\n"
                 "                      a full path, a basename, or a single digit shortcut (e.g.\n"
                 "                      -M 0,1,2,3,4 expands to fold_<N>_fp16.onnx). Bare\n"
                 "                      basenames are looked up under $SIAMIZE_CACHE_DIR\n"
                 "                      (default $HOME/.cache/siamize/models/) and auto-downloaded\n"
                 "                      from $SIAMIZE_WEIGHTS_BASE_URL on miss.\n"
#endif
#ifdef SIAMIZE_HAS_MNN
                 "  -c, --compute D     MNN forward type: auto|cpu|opencl|vulkan|metal (default auto).\n"
                 "                      auto -> OpenCL when MNN was built with MNN_OPENCL=ON, else CPU.\n"
                 "                      opencl uses MNN's OpenCL backend (NVIDIA via ICD, AMD, Intel iGPU).\n"
                 "                      vulkan/metal require MNN_VULKAN / MNN_METAL at MNN build time.\n"
                 "                      No CUDA/TensorRT/CoreML in this build (SIAMIZE_BACKEND=mnn).\n"
                 "                      Per-session kernel-tuning data auto-cached at\n"
                 "                      $SIAMIZE_CACHE_DIR/mnn-tune/<dev>-<prec>.cache;\n"
                 "                      first session is slow, subsequent runs reuse the cache.\n"
                 "      --mnn-fp16        run MNN GPU compute in fp16 (Precision_Low) instead\n"
                 "                        of fp32 (Precision_High). Engages Tensor Cores on Volta+\n"
                 "                        NVIDIA via OpenCL, half-throughput SIMD on AMD / Intel\n"
                 "                        GPUs; silently no-op on devices reporting fp16:0\n"
                 "                        (PoCL CPU-OpenCL, older GPUs without native fp16).\n"
                 "                        Typical: 1.5-2x speedup with small accuracy cost.\n"
                 "      --mnn-buffer      force MNN's OpenCL BUFFER memory mode. Default ON\n"
                 "                        because the shipped doc=mnn_n3d / fold_X_fp32.mnn\n"
                 "                        weights need BUFFER for the native Conv3DBufExecution\n"
                 "                        path; without it OpenCL falls through to a ~3300-op\n"
                 "                        geometry decomposition with multi-minute first-run\n"
                 "                        compile time. Redundant when the flag is already on.\n"
                 "      --mnn-image       opt OUT of BUFFER mode, force MNN's OpenCL IMAGE\n"
                 "                        memory mode. Useful only on driver/weight combos\n"
                 "                        where BUFFER's `conv_2d_buf` int kernel fails to\n"
                 "                        JIT-compile (older NVIDIA + legacy mnn_i8a int8\n"
                 "                        weights, CL build error -9999). Inference will be\n"
                 "                        slow on the shipped mnn_n3d weights.\n"
#else
                 "  -c, --compute D     execution provider: auto|cpu|cuda|tensorrt (default auto).\n"
                 "                      auto tries CUDA / CoreML (if compiled in) then falls back to CPU.\n"
                 "                      tensorrt tries TRT > CUDA > CPU (first run builds engines).\n"
                 "                      coreml uses Apple Silicon CPU + Metal GPU + ANE via the ORT\n"
                 "                      CoreML EP (macOS only; requires -DSIAMIZE_GPU=coreml).\n"
#endif
                 "      --trt-cache-dir P  TensorRT engine cache dir (default ~/.cache/siamize/trt).\n"
                 "                         Engines are GPU- and TRT-version-specific; cached on first run.\n"
                 "      --coreml-units U   CoreML compute target: all (default; CPU+GPU+ANE, Core ML\n"
                 "                         routes per op), cpune (CPU + Neural Engine only),\n"
                 "                         cpugpu (CPU + Metal GPU only), cpu (CPU only, debug).\n"
                 "      --coreml-cache-dir P  CoreML model-compile cache (default ~/.cache/siamize/coreml).\n"
                 "                         First-run compile of ONNX -> .mlmodelc takes ~10-30 s; cached after.\n"
                 "      --coreml-static-shapes 0|1  RequireStaticInputShapes for CoreML EP. Default 1;\n"
                 "                         pair with the fixed-shape ONNX bundle for best fusion. Set 0\n"
                 "                         when using the dynamic-shape ONNX (e.g. with --lowmem).\n"
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
                 "  -G, --gpu N|P:D     GPU device index (default 0 = first visible GPU). Accepts:\n"
                 "                         N    : device N within platform 0 (legacy / CUDA / single-ICD).\n"
                 "                                Honors CUDA_VISIBLE_DEVICES; `nvidia-smi --query-gpu=\n"
                 "                                index,name --format=csv,noheader` shows the mapping.\n"
                 "                         P:D  : OpenCL/Vulkan platform P, device D (MNN backend only).\n"
                 "                                Needed on multi-ICD hosts (e.g. PoCL + NVIDIA, AMD +\n"
                 "                                Intel) where the real GPU is not on platform 0. List\n"
                 "                                available platforms with `clinfo -l`. ORT EPs ignore P.\n"
                 "  -t, --thread N      ORT intra-op threads. Default 0 = auto, which selects\n"
                 "                      min(hardware_concurrency, 16). The 16-thread cap is\n"
                 "                      a heuristic: on huge-core hosts (e.g. AMD Zen2 64-core)\n"
                 "                      ORT's CPU EP plateaus past ~16 threads on this workload\n"
                 "                      and starts regressing at 32+ due to memory-bandwidth\n"
                 "                      and L3 contention. Pass -t 0 -t <hc> explicitly to\n"
                 "                      override; ignored for GPU EPs.\n"
                 "      --no-arena      disable ORT's CPU memory arena + memory-pattern\n"
                 "                      optimizer. Default off (arena enabled). Saves ~16 GB\n"
                 "                      peak RSS on the 18-class network but adds ~1.5x to\n"
                 "                      wall time; use only on RAM-constrained hosts.\n"
                 "      --shuffle       enable JData _ArrayShuffle_=4 (byte-shuffle before\n"
                 "                      zlib) on TPM .jnii/.bnii output. Default OFF for\n"
                 "                      interop with readers that don't yet understand\n"
                 "                      _ArrayShuffle_ (current jsonlab). When set, yields\n"
                 "                      1.5-2.5x smaller TPM files at no decode cost for\n"
                 "                      spec-compliant readers (siamize itself, future\n"
                 "                      jsonlab).\n"
                 "      --lowmem        force the full low-memory preset (--no-arena +\n"
                 "                      -t auto-cap=8 + --cudnn-max-workspace 0 +\n"
                 "                      --gpu-mem-limit 6G + -P 192x192x128). The -P\n"
                 "                      shrink REQUIRES an ONNX exported with dynamic\n"
                 "                      spatial axes (tools/onnx_export/ default since\n"
                 "                      the dynamic_axes change); pass --lowmem to opt\n"
                 "                      in only when your weights support it. Without\n"
                 "                      --lowmem, auto-detect applies the SAFE SUBSET\n"
                 "                      (everything except -P) whenever available host\n"
                 "                      RAM is < 14 GB or free GPU VRAM is < 12 GB, so\n"
                 "                      out-of-the-box behavior never breaks inference.\n"
                 "                      Flags passed alongside --lowmem are NOT overridden.\n"
                 "      --tpm [0|1]     toggle TPM-mode output (default off). When on, the\n"
                 "                      file at `-o` is a 4D float32 tissue probability map of\n"
                 "                      shape (X, Y, Z, num_classes) -- softmax over the 18 fold-\n"
                 "                      averaged class logits -- INSTEAD of the discrete labelmap.\n"
                 "                      Use .nii.gz to gzip on save. Raw size ~3 GB; gzipped ~250 MB.\n"
                 "                      `--tpm` alone is equivalent to `--tpm 1`.\n"
                 "      --tpm-t T       softmax temperature for the TPM output (default 1.0).\n"
                 "                      T > 1 softens the distribution (more graded boundaries,\n"
                 "                      better-calibrated probabilities); T < 1 sharpens. Only\n"
                 "                      meaningful when --tpm is on.\n"
                 "      --upsample      save the output at the internal inference resolution\n"
                 "                      (target spacing, default 0.75 mm isotropic) instead\n"
                 "                      of the input file's grid. Skips the per-channel\n"
                 "                      trilinear back-resample, so a low-resolution input\n"
                 "                      (e.g. 64^3 @ 4 mm) yields a super-resolved\n"
                 "                      segmentation at ~341^3 @ 0.75 mm matching the network's\n"
                 "                      learned detail. Output is in canonical (Z,Y,X) RAS\n"
                 "                      orientation; the affine is rebuilt accordingly and the\n"
                 "                      result is un-cropped to the full canonical extent\n"
                 "                      (regions outside the nonzero head bbox are\n"
                 "                      background-filled). Works with --tpm and all -F formats.\n"
                 "  -P, --patch ZxYxX   sliding-window patch size, default 256x256x192 (matches\n"
                 "                      SIAM v0.3 plans). siamize tiles the resampled-to-target-\n"
                 "                      spacing volume with overlapping windows of this size;\n"
                 "                      each tile is exactly one network forward pass. Peak\n"
                 "                      working memory scales with this size cubed -- on tight\n"
                 "                      RAM (e.g. 32 GB hosts) try 192x192x128 or 128x128x96.\n"
                 "  -u, --spacing v     target isotropic spacing in mm, default 0.75 (SIAM v0.3 training).\n"
                 "  -C, --classes N|spm number of output classes (default 18 = SIAM v0.3).\n"
                 "                      Pass an integer to drive a non-SIAM model with N\n"
                 "                      output channels. Pass 'spm' to remap SIAM v0.3's\n"
                 "                      18 classes into SPM12's 6 TPM channels (GM, WM,\n"
                 "                      CSF, Bone, Soft, Air). Works with both labelmap\n"
                 "                      (argmax -> 0..5 in that order) and --tpm output\n"
                 "                      (4D float32 in spm12/tpm/TPM.nii channel order).\n"
                 "                      Anomalies (SIAM class 17) fold into GM.\n"
                 "  -v, --verbose       print progress (default ON; flag kept for backward\n"
                 "                      compat with older scripts -- already-verbose runs).\n"
                 "  -q, --quiet         suppress progress output. Warnings ([warn] ...) and\n"
                 "                      runtime errors still print. Useful for batch jobs that\n"
                 "                      capture stderr selectively.\n"
                 "  -h, --help\n"
                 "\n"
                 "Examples\n"
                 "  # Default: single-fold (fold_0), CPU auto-detect, write 3D uint8 labels.\n"
                 "  siamize -i input.nii.gz -o labels.nii.gz\n"
                 "\n"
                 "  # Full 5-fold ensemble with auto-download of any missing weights.\n"
                 "  siamize -i input.nii.gz -o labels.nii.gz -M 0,1,2,3,4 -v\n"
                 "\n"
                 "  # Same on GPU (auto -> CUDA, falls back to CPU if it can't load).\n"
                 "  siamize -i input.nii.gz -o labels.nii.gz -M 0,1,2,3,4 -c cuda -v\n"
                 "\n"
                 "  # Multi-GPU box: pick GPU 1 of N (check indices via `nvidia-smi`).\n"
                 "  siamize -i input.nii.gz -o labels.nii.gz -M 0:4 -c cuda -G 1\n"
                 "\n"
                 "  # Tight-VRAM 8 GB consumer card: smaller patch + lean cuDNN options.\n"
                 "  siamize -i input.nii.gz -o labels.nii.gz -M 0:4 -c cuda \\\n"
                 "          -P 192x192x128 --cudnn-max-workspace 0 --arena-extend same\n"
                 "\n"
                 "  # Throttle CPU usage to 4 intra-op threads.\n"
                 "  siamize -i input.nii.gz -o labels.nii.gz -M 0:4 -c cpu -t 4\n"
                 "\n"
                 "  # 4D tissue probability map output (float32 (X,Y,Z,18) instead of labels):\n"
                 "  siamize -i input.nii.gz -o tpm.nii.gz   -M 0:4 -c cuda --tpm\n"
                 "\n"
                 "  # Same, with a softer (temperature-scaled) softmax for graded boundaries:\n"
                 "  siamize -i input.nii.gz -o tpm.nii.gz   -M 0:4 -c cuda --tpm --tpm-t 1.5\n"
                 "\n"
                 "  # JNIfTI output (smaller on disk than .nii.gz; readable by jsonlab):\n"
                 "  siamize -i input.nii.gz   -o labels.jnii -M 0:4 -F jnii\n"
                 "  siamize -i input.nii.gz   -o labels.bnii -M 0:4 -F bnii\n"
                 "\n"
                 "  # JNIfTI input round-trip (use jsonlab/savejnifti to make one first):\n"
                 "  siamize -i preproc.bnii   -o labels.bnii -M 0:4 -F bnii\n"
                 "\n"
                 "  # TensorRT engine path (opt-in: heavy first-run engine build, fast thereafter):\n"
                 "  siamize -i input.nii.gz -o labels.nii.gz -M 0 -c tensorrt \\\n"
                 "          --trt-cache-dir $HOME/.cache/siamize/trt\n"
                 "\n"
                 "  # Mix shortcuts and explicit paths in --models:\n"
                 "  siamize -i input.nii.gz -o labels.nii.gz \\\n"
                 "          -M 0,2,/abs/path/fold_4_fp16.onnx\n",
                 exe);
}

}  // namespace

/*******************************************************************************/
/*! \fn    int main(int argc, char** argv)
    \brief Program entry point: parse args, run the pipeline, write output

    Walks argv with manual flag handling (no getopt) so the same
    short+long flag parsing can be lifted into the MEX bridge. The
    main steps mirror the pipeline outlined in this file's @brief.

    \param  argc  argument count (from the C runtime)
    \param  argv  argument vector
    \return       0 on success, 1 on a runtime failure (std::exception),
                  2 on a bad-CLI error
*/
int main(int argc, char** argv) {
    // Default CUDA_DEVICE_ORDER=PCI_BUS_ID so siamize's -G N matches the
    // index a user sees in `nvidia-smi -L`. The CUDA runtime's default
    // is FASTEST_FIRST, which sorts GPUs by perf class (compute capability,
    // mem bandwidth) -- meaning on a heterogeneous box like a workstation
    // with both an RTX 2080 (faster-ranked in CUDA's heuristic because of
    // tensor cores) AND an A100, the A100 may NOT be CUDA device 1 the way
    // it appears in `nvidia-smi -L`. The default flipped indices silently
    // route `-G 1` to whichever card CUDA ranked second, not the card the
    // user expected. PCI_BUS_ID is the same enumeration nvidia-smi prints,
    // so users can map directly between the two.
    //
    // We use setenv with overwrite=0 so an explicit user choice
    // (CUDA_DEVICE_ORDER=FASTEST_FIRST in the environment) still wins.
    // Windows lacks POSIX setenv; use _putenv_s after an explicit
    // "is it already set?" check so we match the overwrite=0 semantics.
#ifndef _WIN32
    setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 0);
#else

    if (std::getenv("CUDA_DEVICE_ORDER") == nullptr) {
        _putenv_s("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
    }

#endif

    std::string input_path, output_path, models_csv;
    std::string device = "auto";       // auto | cpu | cuda | tensorrt
    std::string trt_cache_dir;         // empty => $HOME/.cache/siamize/trt
    bool        tpm_mode        = false; // true => write 4D TPM to output, not labels
    float       tpm_temperature = 1.0f;  // softmax temperature (>1 = softer)
    bool        tpm_shuffle     = false; // apply JData _ArrayShuffle_=4 before zlib
    // on TPM jnii/bnii output. Default OFF for
    // interop with readers that don't yet
    // understand _ArrayShuffle_ (e.g. current
    // jsonlab); pass --shuffle to opt in to
    // the 1.5-2.5x size win for spec-compliant
    // readers (siamize itself, future jsonlab).
    bool        upsample_mode   = false; // true => save at internal 0.75 mm resolution
    //         (skip per-channel back-resample),
    //         un-cropped to full canonical extent,
    //         using canonical RAS orientation.
    bool        lowmem_mode     = false; // true => force the lowmem default preset
    //         (smaller patch, no arena, smaller
    //         thread cap, tight VRAM knobs). Same
    //         preset that auto-detection applies on
    //         a memory-tight host -- this flag just
    //         opts in regardless of detection.
    std::string out_format      = "nii"; // nii | jnii | bnii
    int threads = 0;     // 0 = auto: std::thread::hardware_concurrency()
    // Verbose progress is ON by default in the CLI; a multi-minute job
    // should not look hung. Pass --quiet / -q to silence everything
    // except warnings ([warn] ...) and errors. The legacy -v / --verbose
    // flag is kept as a no-op for backward compatibility with old
    // scripts.
    bool verbose = true;
    std::array<int64_t, 3> patch = {256, 256, 192};
    float target_spacing = 0.75f;
    int64_t num_classes     = 18;   // inference output channels (network's logits)
    int64_t num_classes_out = 18;   // saved output channels (may differ for --classes spm)
    ClassSet class_set = ClassSet::CUSTOM_N;
    siam::EngineTuning engine_tuning;

    // Per-flag "user explicitly set this" trackers. Used by the post-
    // parse auto-safe block to avoid stomping on flags the user passed
    // intentionally. Anything not listed here is auto-applied
    // unconditionally when tight memory is detected.
    bool patch_set                = false;
    bool threads_set              = false;
    bool cpu_arena_set            = false;
    bool cudnn_max_workspace_set  = false;
    bool gpu_mem_limit_set        = false;
    bool format_set               = false;
    bool coreml_units_set         = false;

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
        } else if (a == "-M" || a == "--models") {
            models_csv = need();
        } else if (a == "-c" || a == "--compute") {
            device = need();

            // Allow `trt` as an alias for `tensorrt`.
            if (device == "trt") {
                device = "tensorrt";
            }

#ifdef SIAMIZE_HAS_MNN

            // MNN-backend build: device vocabulary is auto|cpu|opencl|vulkan|metal.
            // The MNN runtime picks the forward type at runtime via
            // MNNForwardType (no CUDA/CoreML/TensorRT EPs to register).
            if (device != "auto" && device != "cpu" && device != "opencl"
                    && device != "vulkan" && device != "metal") {
                std::fprintf(stderr,
                             "-c/--compute must be auto|cpu|opencl|vulkan|metal "
                             "for the MNN backend (got '%s')\n",
                             device.c_str());
                return 2;
            }

#else

            if (device != "auto" && device != "cpu" && device != "cuda"
                    && device != "tensorrt" && device != "coreml") {
                std::fprintf(stderr,
                             "-c/--compute must be auto|cpu|cuda|tensorrt|coreml (got '%s')\n",
                             device.c_str());
                return 2;
            }

#endif
        } else if (a == "--trt-cache-dir") {
            trt_cache_dir = need();
        } else if (a == "--coreml-units") {
            std::string s = need();

            if (s == "all") {
                engine_tuning.coreml_units = siam::CoreMLUnits::ALL;
            } else if (s == "cpune" || s == "ane") {
                engine_tuning.coreml_units = siam::CoreMLUnits::CPU_AND_ANE;
            } else if (s == "cpugpu" || s == "gpu") {
                engine_tuning.coreml_units = siam::CoreMLUnits::CPU_AND_GPU;
            } else if (s == "cpu") {
                engine_tuning.coreml_units = siam::CoreMLUnits::CPU_ONLY;
            } else {
                std::fprintf(stderr,
                             "--coreml-units must be all|cpune|cpugpu|cpu (got '%s')\n",
                             s.c_str());
                return 2;
            }

            coreml_units_set = true;
        } else if (a == "--coreml-cache-dir") {
            engine_tuning.coreml_cache_dir = need();
        } else if (a == "--coreml-static-shapes") {
            std::string s = need();
            engine_tuning.coreml_static_shapes = (s != "0" && s != "false");
        } else if (a == "-t" || a == "--thread") {
            threads = std::stoi(need());
            threads_set = true;
        } else if (a == "-u" || a == "--spacing") {
            target_spacing = std::stof(need());
        } else if (a == "-C" || a == "--classes") {
            std::string s = need();

            if (s == "spm" || s == "SPM") {
                class_set       = ClassSet::SPM;
                num_classes     = 18;                  // still infer all 18 SIAM channels
                num_classes_out = SPM6_NUM_CLASSES;    // merge to 6 SPM bins on output
            } else {
                try {
                    num_classes = std::stoll(s);
                } catch (...) {
                    std::fprintf(stderr,
                                 "--classes must be a positive integer or 'spm' (got '%s')\n",
                                 s.c_str());
                    return 2;
                }

                num_classes_out = num_classes;
            }
        } else if (a == "-P" || a == "--patch") {
            std::string p = need();
            char x;
            std::stringstream ss(p);
            ss >> patch[0] >> x >> patch[1] >> x >> patch[2];
            patch_set = true;
        } else if (a == "--cudnn-max-workspace") {
            engine_tuning.cudnn_max_workspace = std::stoi(need());
            cudnn_max_workspace_set = true;
        } else if (a == "--arena-extend") {
            std::string s = need();

            if      (s == "same"  || s == "kSameAsRequested") {
                engine_tuning.arena_same_as_req = 1;
            } else if (s == "power" || s == "kNextPowerOfTwo") {
                engine_tuning.arena_same_as_req = 0;
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
                engine_tuning.algo_search = "DEFAULT";
            } else if (s == "heuristic"  || s == "HEURISTIC")  {
                engine_tuning.algo_search = "HEURISTIC";
            } else if (s == "exhaustive" || s == "EXHAUSTIVE") {
                engine_tuning.algo_search = "EXHAUSTIVE";
            } else {
                std::fprintf(stderr,
                             "--cudnn-algo must be default|heuristic|exhaustive (got '%s')\n",
                             s.c_str());
                return 2;
            }
        } else if (a == "--no-arena") {
            engine_tuning.cpu_arena = false;
            cpu_arena_set = true;
        } else if (a == "--shuffle") {
            tpm_shuffle = true;
        } else if (a == "--upsample") {
            upsample_mode = true;
        } else if (a == "--lowmem") {
            lowmem_mode = true;
        } else if (a == "-G" || a == "--gpu") {
            // Two accepted forms:
            //   -G N        device N (legacy / CUDA path, platform 0)
            //   -G P:D      OpenCL/Vulkan platform P, device D (MNN)
            // The ":D" form lets users on multi-ICD hosts (e.g. PoCL +
            // NVIDIA, AMD + Intel) reach a GPU that lives on a non-zero
            // platform. ORT EPs ignore the platform half.
            std::string s = need();
            size_t colon = s.find(':');

            if (colon == std::string::npos) {
                engine_tuning.gpuid = std::stoi(s);
            } else {
                engine_tuning.gpu_platform = std::stoi(s.substr(0, colon));
                engine_tuning.gpuid = std::stoi(s.substr(colon + 1));
            }
        } else if (a == "--mnn-fp16") {
            // MNN BackendConfig::Precision_Low. On Volta+ NVIDIA via
            // OpenCL, this engages Tensor Cores for fp16 conv; on AMD
            // / Intel GPUs, half-throughput SIMD; on devices reporting
            // fp16:0 (PoCL CPU-OpenCL, older GPUs), MNN silently falls
            // back to fp32. Inert for the ORT backend.
            engine_tuning.mnn_fp16 = true;
        } else if (a == "--mnn-buffer") {
            // MNN_GPU_MEMORY_BUFFER. Default ON because the shipped
            // doc=mnn_n3d fp32 weights need BUFFER for the native
            // Conv3DBufExecution path; without it the OpenCL backend
            // has no Conv3D creator and decomposes into ~3300 small
            // 2D ops. Accepting the flag explicitly is a no-op.
            engine_tuning.mnn_buffer = true;
        } else if (a == "--mnn-image") {
            // Opt OUT of BUFFER. Force MNN's OpenCL IMAGE memory mode.
            // Useful for the legacy mnn_i8a int weights on NVIDIA
            // drivers whose `conv_2d_buf` int kernel fails to JIT-
            // compile (CL build error -9999). Inference is slow on
            // the shipped mnn_n3d weights because Conv3D falls
            // through to geometry decomposition.
            engine_tuning.mnn_buffer = false;
        } else if (a == "-F" || a == "--format") {
            std::string s = need();

            if (s != "nii" && s != "jnii" && s != "bnii") {
                std::fprintf(stderr,
                             "-F/--format must be nii|jnii|bnii (got '%s')\n",
                             s.c_str());
                return 2;
            }

            out_format = s;
            format_set = true;
        } else if (a == "--tpm") {
            // Bare flag OR optional 0|1|true|false. If the next arg
            // looks like a bool value, consume it; otherwise default to
            // true so `--tpm` alone toggles TPM mode on.
            tpm_mode = true;

            if (i + 1 < argc) {
                std::string nxt = argv[i + 1];

                if (nxt == "0" || nxt == "false") {
                    ++i;
                    tpm_mode = false;
                } else if (nxt == "1" || nxt == "true") {
                    ++i;
                    tpm_mode = true;
                }
            }
        } else if (a == "--tpm-t") {
            tpm_temperature = std::stof(need());

            if (tpm_temperature <= 0.0f) {
                std::fprintf(stderr,
                             "--tpm-t must be > 0 (got '%g')\n",
                             tpm_temperature);
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

            engine_tuning.gpu_mem_limit_bytes =
                static_cast<size_t>(std::stoull(s)) * mult;
            gpu_mem_limit_set = true;
        } else if (a == "-v" || a == "--verbose") {
            // Verbose is on by default now; -v is kept as a no-op.
            verbose = true;
            siam::set_verbose(true);
        } else if (a == "-q" || a == "--quiet") {
            verbose = false;
            siam::set_verbose(false);
        } else if (a == "-h" || a == "--help")   {
            usage(argv[0]);
            return 0;
        } else                                   {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            usage(argv[0]);
            return 2;
        }
    }

    // After parse completes, push the resolved verbose state into the
    // siam_log global so log_tag/log_cont gate correctly. Verbose
    // defaults to true (set above); -q / --quiet flips it off.
    siam::set_verbose(verbose);

    if (input_path.empty() || output_path.empty()) {
        usage(argv[0]);
        return 2;
    }

    // ----- Infer output format from extension --------------------------
    // -F/--format wins if the user passed it; otherwise we sniff the
    // output path. `.jnii` -> text-JSON JNIfTI, `.bnii` -> BJData binary
    // JNIfTI, anything else (`.nii`, `.nii.gz`, ...) -> NIfTI-1. This
    // mirrors the input-side extension dispatch already in place, and
    // matches what users expect from a `.jnii` filename: a JSON-text
    // file per https://neurojson.org/jnifti , not an opaque binary
    // blob with the wrong extension.
    if (!format_set) {
        auto path_ends_with_ci = [](const std::string & s, const std::string & suf) {
            if (s.size() < suf.size()) {
                return false;
            }

            for (size_t i = 0; i < suf.size(); ++i) {
                char a = s[s.size() - suf.size() + i], b = suf[i];

                if (std::tolower(static_cast<unsigned char>(a))
                        != std::tolower(static_cast<unsigned char>(b))) {
                    return false;
                }
            }

            return true;
        };

        if (path_ends_with_ci(output_path, ".jnii")) {
            out_format = "jnii";
        } else if (path_ends_with_ci(output_path, ".bnii")) {
            out_format = "bnii";
        } else {
            out_format = "nii";
        }
    }

    // ----- Lowmem preset (manual or auto-detected) ----------------------
    // Detect available host RAM (via /proc/meminfo) and GPU VRAM (via
    // nvidia-smi when the user picked a CUDA-capable device) and apply
    // memory-saving flag defaults when either is tight. The user can
    // force this preset on otherwise-large hosts by passing --lowmem.
    // Flags the user explicitly passed alongside --lowmem are NOT
    // overridden -- the `*_set` trackers in the parse loop guard
    // against stomping on intent.
    //
    // Thresholds (auto-detect):
    //   RAM  < 14 GB available  -> CPU-side lowmem defaults
    //   VRAM < 12 GB available  -> GPU-side lowmem defaults
    //
    // CPU lowmem set:
    //   -P 192x192x128, --no-arena, -t auto-cap = 8
    //   targets ~5-7 GB peak RSS (fits 16 GB host with OS overhead)
    //
    // GPU lowmem set (ORT EPs):
    //   --cudnn-max-workspace 0, --gpu-mem-limit 6G
    //   targets ~3-4 GB peak VRAM (fits 8 GB GPU)
    //
    // MNN backend (-DSIAMIZE_BACKEND=mnn) -- the cudnn / arena knobs
    // above are inert. Memory pressure shows up via MNN's own
    // workspace allocation at first run_tile; we mitigate by:
    //   - auto-applying -P 192x192x128 on a RAM-tight host (no
    //     --lowmem opt-in required, since the shipped doc=mnn_n3d
    //     bundle is always dyn-shape).
    //   - auto-applying -P 128x128x96 when BOTH RAM and VRAM are
    //     tight on a GPU device (MNN-OpenCL allocates the full
    //     forward-pass workspace ahead of the first tile, so a
    //     192x192x128 patch on a 10 GB GPU + 12 GB host can still
    //     OOM-kill the process).
    const long avail_ram_mb  = available_ram_mb();
#ifdef SIAMIZE_HAS_MNN
    // MNN backend: GPU devices are opencl/vulkan/metal. "auto" goes
    // through engine_mnn.cpp's resolve_forward_type which picks OpenCL
    // if MNN_OPENCL was on at build time, else CPU -- treat it as
    // GPU-active so the VRAM check below uses the right thresholds.
    const bool gpu_active    = (device == "auto" ||
                                device == "opencl" ||
                                device == "vulkan" ||
                                device == "metal");
#else
    const bool gpu_active    = (device == "auto" ||
                                device == "cuda" ||
                                device == "tensorrt");
#endif
    const long avail_vram_mb = gpu_active
                               ? available_vram_mb(engine_tuning.gpuid)
                               : 0;
    const bool ram_tight     = lowmem_mode
                               || (avail_ram_mb  > 0 && avail_ram_mb  < 14 * 1024);
    const bool vram_tight    = (lowmem_mode && gpu_active)
                               || (avail_vram_mb > 0 && avail_vram_mb < 12 * 1024);

    std::vector<std::string> auto_applied;

    // Patch shrink. Lives outside ram_tight so it also fires on
    // RAM-comfortable but VRAM-tight hosts (common case: 16 GB+
    // workstation with a 10-12 GB GPU). For ORT, still gated on
    // EXPLICIT --lowmem (old fixed-shape .onnx rejects smaller
    // patches). For MNN, the shipped doc=mnn_n3d is dyn-shape so
    // auto-shrink is safe without the opt-in.
    //
    // Tier choice (workspace estimate is for MNN-OpenCL):
    //   patch       workspace    tiles on 340x340x213    tile count
    //   256x256x192 ~6-10 GB     default                   8
    //   192x192x128 ~3-5  GB                              27
    //   128x128x96  ~1.5-2 GB                            100
    //
    // We avoid the 128x128x96 tier for vanilla ram_tight / vram_tight
    // because going below 192x192x128 triples the tile count and
    // shifts the Gaussian-blend pattern enough to cost ~0.05 Dice
    // vs the canonical patch -- only worth it on really tight GPUs.
    // Threshold for the very-tight branch: VRAM < 8 GB AND a GPU
    // device is actually being used.
    {
        bool patch_shrink_allowed = lowmem_mode;
#ifdef SIAMIZE_HAS_MNN
        patch_shrink_allowed = true;
#endif
        const bool vram_very_tight = gpu_active
                                     && avail_vram_mb > 0
                                     && avail_vram_mb < 8 * 1024;

        if (patch_shrink_allowed && !patch_set && (ram_tight || vram_tight)) {
            if (vram_very_tight) {
                patch = {128, 128, 96};
                auto_applied.push_back("-P 128x128x96");
            } else {
                patch = {192, 192, 128};
                auto_applied.push_back("-P 192x192x128");
            }
        }
    }

    if (ram_tight) {
        if (!cpu_arena_set) {
            engine_tuning.cpu_arena = false;
            auto_applied.push_back("--no-arena");
        }

        // CoreML EP compile-memory mitigation. Apple's mlcompilerd
        // peaks ~6-8 GB compiling SIAM v0.3's MLProgram for all
        // three compute units. Empirically (e2337c6 CI run on the
        // macos-14 free runner, 7 GB RAM): both `ALL` and `cpugpu`
        // are killed by the OS OOM at exit 137 before the compile
        // returns -- the 283 MB ONNX with 75 InstanceNorm + 200+
        // Conv3D nodes blows past the budget even when we skip ANE
        // codegen and ask only for Metal GPU. Only CPU_ONLY's
        // ~1-2 GB compile peak fits. Auto-fallback there.
        //
        // The trade-off is real: -c coreml --coreml-units cpu is
        // typically NO faster than -c cpu (Core ML CPU backend vs
        // ORT MLAS), and pays the ~10 s ONNX -> .mlmodelc compile
        // on top. The right user move on tight-RAM hosts is just
        // -c cpu. We keep the CoreML+CPU path here so the wiring
        // is exercised and so users with >7 GB free will hit the
        // ALL / cpugpu branch when their host has the headroom.
        //
        // Future tiering: when avail RAM is in the 8-14 GB range,
        // CPU_AND_GPU may fit (GPU codegen alone is the median
        // peak). Untested. Leaving CPU_ONLY as the conservative
        // auto-pick until we have a confirmed-fits target.
        //
        // Same opt-out as the other auto-lowmem knobs: don't stomp
        // on an explicit --coreml-units choice.
        if (!coreml_units_set
                && (device == "coreml" || device == "auto")) {
            engine_tuning.coreml_units = siam::CoreMLUnits::CPU_ONLY;
            auto_applied.push_back("--coreml-units cpu");
        }

        // -t auto-cap reduction is handled in the threads-resolution
        // block below (uses ram_tight to pick 8 instead of 16).
    }

    if (vram_tight) {
        if (!cudnn_max_workspace_set) {
            engine_tuning.cudnn_max_workspace = 0;
            auto_applied.push_back("--cudnn-max-workspace 0");
        }

        if (!gpu_mem_limit_set) {
            engine_tuning.gpu_mem_limit_bytes = 6ull * 1024 * 1024 * 1024;
            auto_applied.push_back("--gpu-mem-limit 6G");
        }
    }

    // Resolve threads before the header so it can show the actual count.
    // Auto path: min(hardware_concurrency, AUTO_CAP). AUTO_CAP is 8 when
    // RAM is tight (per the safe-defaults block above) and 16 otherwise.
    // The 16 cap was measured on a Threadripper 3990X (Zen2, 64C/128T)
    // where ORT's CPU EP hits its wall-time minimum at -t 16 and
    // regresses at higher counts due to memory bandwidth and cross-CCD
    // L3 contention. Users with unusual workloads or hardware can pass
    // -t <N> explicitly.
    bool threads_auto = (threads <= 0);

    if (threads_auto) {
        unsigned hc = std::thread::hardware_concurrency();
        int detected = (hc > 0) ? static_cast<int>(hc) : 4;
        const int auto_cap = ram_tight ? 8 : 16;
        threads = std::min(detected, auto_cap);

        if (ram_tight && !threads_set) {
            auto_applied.push_back("-t cap=8 (auto-tight)");
        }
    }

    if (!auto_applied.empty()) {
        const char* source = lowmem_mode ? "--lowmem requested"
                             : "memory-tight host detected";
        siam::log_hint("lowmem preset applied (%s; RAM %ld MB%s%s%s):",
                       source,
                       avail_ram_mb,
                       gpu_active && avail_vram_mb > 0 ? ", VRAM " : "",
                       gpu_active && avail_vram_mb > 0 ?
                       std::to_string(avail_vram_mb).c_str() : "",
                       gpu_active && avail_vram_mb > 0 ? " MB" : "");

        for (const auto& s : auto_applied) {
            siam::log_hint("  %s", s.c_str());
        }

        siam::log_hint("(override by passing the flags explicitly; see --help)");
    }

    // Verbose header line: prints once at the start of the run so the
    // following [tag] lines have an unmistakable anchor at the top.
    siam::log_tag("siamize",
                  "v0.1.0  device=%s  threads=%d%s  format=%s%s%s%s%s%s%s",
                  device.c_str(), threads,
                  threads_auto ? " (auto)" : "",
                  out_format.c_str(),
                  tpm_mode ? "  --tpm" : "",
                  upsample_mode ? "  --upsample" : "",
                  engine_tuning.cpu_arena ? "" : "  --no-arena",
                  lowmem_mode ? "  --lowmem" : "",
                  class_set == ClassSet::SPM ? "  --classes spm" : "",
                  (tpm_mode && tpm_shuffle) ? "  --shuffle" : "");

    // Pre-flight memory check: warn if available RAM is below the
    // expected peak for the current config. We can't trap SIGKILL
    // from the kernel OOM killer, so the only way to be useful is to
    // tell the user UP FRONT what flags would shrink the working
    // set. The 20 GB / 8 GB thresholds are empirical for the default
    // 256x256x192 patch with arena ON / OFF respectively.
    {
        long avail_mb = available_ram_mb();
        bool memory_saver = !engine_tuning.cpu_arena
                            || threads < 16
                            || patch[0] < 256 || patch[1] < 256 || patch[2] < 192;

        if (avail_mb > 0 && avail_mb < 20 * 1024 && !memory_saver) {
            siam::log_hint("only %ld MB RAM available; default config may OOM. Consider:",
                           avail_mb);
            siam::log_hint("  --no-arena -t 8 -P 192x192x128   (peak ~6-8 GB)");
            siam::log_hint("  --no-arena -t 4 -P 128x128x96    (peak ~3-4 GB)");
        }
    }

    if (models_csv.empty()) {
#ifdef SIAMIZE_HAS_MNN
        models_csv = "fold_0_fp32.mnn";
        siam::log_tag("models",
                      "-M/--models not given; defaulting to single-fold "
                      "fold_0_fp32.mnn (auto-downloaded if missing).");
#else
        models_csv = "fold_0_fp16.onnx";
        siam::log_tag("models",
                      "-M/--models not given; defaulting to single-fold "
                      "fold_0_fp16.onnx (auto-downloaded if missing).");
#endif
    }

    auto model_paths = split_csv(models_csv);

    for (auto& m : model_paths) {
        m = expand_fold_shortcut(m);
    }

    auto t_start = std::chrono::steady_clock::now();

    siam::log_tag("input", "%s", input_path.c_str());

    // Input dispatch: .jnii / .bnii go through the JNIfTI reader, anything
    // else (typically .nii / .nii.gz) through the NIfTI-1 reader.
    auto ends_with_ci = [](const std::string & s, const std::string & suf) {
        if (s.size() < suf.size()) {
            return false;
        }

        for (size_t i = 0; i < suf.size(); ++i) {
            char a = s[s.size() - suf.size() + i], b = suf[i];

            if (std::tolower(static_cast<unsigned char>(a))
                    != std::tolower(static_cast<unsigned char>(b))) {
                return false;
            }
        }

        return true;
    };
    NiftiImage img = (ends_with_ci(input_path, ".jnii") || ends_with_ci(input_path, ".bnii"))
                     ? siam::load_jnifti_ras(input_path)
                     : siam::load_nifti_ras(input_path);

    siam::log_cont("canon shape (Z,Y,X) (%" PRId64 ",%" PRId64 ",%" PRId64
                   ")  voxel size (X,Y,Z) (%.3f,%.3f,%.3f) mm",
                   img.volume.shape[0], img.volume.shape[1], img.volume.shape[2],
                   img.zooms_canon[0], img.zooms_canon[1], img.zooms_canon[2]);

    std::array<int64_t, 3> shape_canon = {img.volume.shape[0], img.volume.shape[1], img.volume.shape[2]};

    // crop to nonzero
    auto crop = siam::crop_to_nonzero(img.volume);

    siam::log_tag("crop", "nonzero bbox (Z,Y,X) (%" PRId64 ",%" PRId64 ",%" PRId64 ")",
                  crop.cropped.shape[0], crop.cropped.shape[1], crop.cropped.shape[2]);

    // z-score
    siam::zscore_inplace(crop.cropped);

    // resample to (target_spacing)^3 mm
    // zooms_canon is in (X, Y, Z); for our (Z, Y, X) layout we reverse.
    std::array<float, 3> spacing_zyx = {img.zooms_canon[2], img.zooms_canon[1], img.zooms_canon[0]};
    std::array<float, 3> new_spacing = {target_spacing, target_spacing, target_spacing};
    auto new_shape = siam::compute_new_shape(
    {crop.cropped.shape[0], crop.cropped.shape[1], crop.cropped.shape[2]},
    spacing_zyx, new_spacing);

    siam::log_tag("resample", "(Z,Y,X) (%" PRId64 ",%" PRId64 ",%" PRId64 ") @ %.3f mm",
                  new_shape[0], new_shape[1], new_shape[2], target_spacing);

    // Forward path: cubic Catmull-Rom (matches scipy order=3 to within ~0.1% Dice).
    Volume resampled = siam::resample_cubic(crop.cropped, new_shape[0], new_shape[1], new_shape[2]);
    crop.cropped = Volume{};  // free

    // Resolve every model spec: existing path, cache lookup, or auto-fetch.
    // Done here (after preprocessing) so [weights] lines land between
    // [resample] and [infer] in the verbose log -- a user typo on the
    // input path still fails fast above, and the network download (if
    // any) happens just before it's needed.
    // Pick which on-server weight bundle to pull. CoreML EP needs the
    // 'coreml' variant (rank-3 InstanceNorm rewrite); other EPs use
    // dynshape so --lowmem's -P shrink keeps working. On non-CoreML
    // builds the `auto` branch never targets CoreML, so we always
    // fall back to the dynshape default.
    siam::WeightVariant weight_variant = siam::WeightVariant::DYNSHAPE;
#ifdef SIAMIZE_HAS_MNN
    // SIAMIZE_BACKEND=mnn: always pull the `mnn` variant. The MNN
    // backend cannot consume .onnx directly, so this is the only
    // valid pre-baked weight bundle for this build.
    weight_variant = siam::WeightVariant::MNN;
    (void)device;
#elif defined(SIAMIZE_HAS_COREML)

    if (device == "coreml" || device == "auto") {
        weight_variant = siam::WeightVariant::COREML;
    }

#else

    if (device == "coreml") {
        // User passed -c coreml on a non-CoreML build. sliding.cpp
        // will throw a clearer error later; we still pick the matching
        // weight variant for consistency in case they switch the build.
        weight_variant = siam::WeightVariant::COREML;
    }

#endif

    for (auto& m : model_paths) {
        try {
            m = siam::resolve_model_path(m, verbose, weight_variant);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "siamize: %s\n", e.what());
            return 3;
        }
    }

    // sliding window. If -c auto chose CUDA/TensorRT and the
    // session aborts mid-Run with an allocation failure (tight VRAM,
    // contention with other GPU clients, etc.), fall back to CPU and
    // retry. -c cuda / -c tensorrt stay strict -- the
    // exception re-raises so the user sees the explicit failure they
    // asked for.
    LogitsVolume logits;

#ifdef SIAMIZE_HAS_ORT

    try {
        logits = siam::sliding_window(
                     resampled, model_paths, patch, num_classes,
                     threads, 0.5f, verbose, device, trt_cache_dir, engine_tuning);
    } catch (const Ort::Exception& e) {
        std::string msg = e.what();
        const bool oom = msg.find("allocate") != std::string::npos
                         || msg.find("Allocate") != std::string::npos
                         || msg.find("out of memory") != std::string::npos
                         || msg.find("CUDA_ERROR_OUT_OF_MEMORY") != std::string::npos;
        // cuDNN/cuBLAS handle creation failures or other CUDA-runtime
        // errors that fire AFTER the CUDA EP successfully appends but
        // BEFORE/DURING session creation. The library loaded; the
        // runtime state is bad (driver/cuDNN version mismatch, GPU in
        // compute-prohibited mode, another process holding the device,
        // etc.). Distinct from OOM because the remedy is different
        // -- adjusting workspace knobs won't help; only switching off
        // GPU does.
        const bool cuda_init = msg.find("CUDNN")        != std::string::npos
                               || msg.find("cudnn")     != std::string::npos
                               || msg.find("CUBLAS")    != std::string::npos
                               || msg.find("cublas")    != std::string::npos
                               || msg.find("CUDA failure") != std::string::npos
                               || msg.find("Exception during initialization") != std::string::npos;
        // CoreML compile failures: ORT successfully appended the
        // CoreML EP but Apple's mlcompilerd rejected the generated
        // .mlpackage. The common cause is op-coverage gaps -- e.g.
        // rank-5 InstanceNormalization, which CoreML's MLProgram
        // format supports only at ranks 3-4 (1D/2D). ORT 1.26's
        // CoreML EP doesn't always partition these out, so the
        // compile blows up on the whole graph instead of just
        // pushing the unsupported op to CPU.
        const bool coreml_compile =
            msg.find("Error compiling model") != std::string::npos
            || msg.find("Failed to parse the model specification") != std::string::npos
            || msg.find("Unable to parse ML Program") != std::string::npos
            || msg.find("CoreML") != std::string::npos;

        if (device == "auto" && oom) {
            siam::log_warn("GPU allocation failed: %s", e.what());
            siam::log_hint("-c auto falling back to CPU. To force CPU from the");
            siam::log_cont("start, pass `-c cpu`; for tight-VRAM GPUs try also");
            siam::log_cont("`--cudnn-max-workspace 0 --arena-extend same` before");
            siam::log_cont("giving up on GPU.");
            logits = siam::sliding_window(
                         resampled, model_paths, patch, num_classes,
                         threads, 0.5f, verbose, std::string("cpu"), trt_cache_dir, engine_tuning);
        } else if (device == "auto" && cuda_init) {
            siam::log_warn("CUDA/cuDNN session init failed: %s", e.what());
            siam::log_hint("-c auto falling back to CPU. Common root causes:");
            siam::log_cont("  - cuDNN library version mismatch (ORT 1.26 expects cuDNN 9.x)");
            siam::log_cont("  - NVIDIA driver too old for installed CUDA runtime (check `nvidia-smi`)");
            siam::log_cont("  - GPU in compute-prohibited mode (`nvidia-smi -q | grep \"Compute Mode\"`)");
            siam::log_cont("  - another process holding the GPU exclusively");
            siam::log_cont("  - pass `-c cpu` to skip GPU entirely");
            logits = siam::sliding_window(
                         resampled, model_paths, patch, num_classes,
                         threads, 0.5f, verbose, std::string("cpu"), trt_cache_dir, engine_tuning);
        } else if (cuda_init) {
            // User explicitly asked for cuda / tensorrt and the
            // session init failed. Print a useful diagnostic and exit
            // cleanly instead of letting the exception abort the
            // process via std::terminate.
            siam::log_warn("CUDA/cuDNN session init failed: %s", e.what());
            siam::log_hint("you passed `-c %s`; common root causes:", device.c_str());
            siam::log_cont("  - cuDNN library version mismatch (ORT 1.26 expects cuDNN 9.x)");
            siam::log_cont("  - NVIDIA driver too old for installed CUDA runtime (check `nvidia-smi`)");
            siam::log_cont("  - GPU in compute-prohibited mode (`nvidia-smi -q | grep \"Compute Mode\"`)");
            siam::log_cont("  - another process holding the GPU exclusively");
            siam::log_cont("  - pass `-c cpu` or `-c auto` to switch off GPU");
            return 4;
        } else if (device == "auto" && coreml_compile) {
            siam::log_warn("CoreML model compile failed: %s", e.what());
            siam::log_hint("-c auto falling back to CPU. Common cause: op-coverage gap.");
            siam::log_cont("  - rank-5 InstanceNormalization (SIAM v0.3 uses 3D InstanceNorm,");
            siam::log_cont("    which CoreML's MLProgram format supports only at ranks 3-4)");
            siam::log_cont("  - 3D ConvTranspose with non-standard stride/output_padding");
            siam::log_cont("workaround: re-export the ONNX with rank-5 InstanceNorm rewritten");
            siam::log_cont("as Reshape -> InstanceNorm2D -> Reshape (see tools/onnx_export/)");
            logits = siam::sliding_window(
                         resampled, model_paths, patch, num_classes,
                         threads, 0.5f, verbose, std::string("cpu"), trt_cache_dir, engine_tuning);
        } else if (coreml_compile) {
            siam::log_warn("CoreML model compile failed: %s", e.what());
            siam::log_hint("you passed `-c %s`. The SIAM v0.3 ONNX has ops Core ML's", device.c_str());
            siam::log_cont("MLProgram format can't currently handle (typically rank-5");
            siam::log_cont("InstanceNormalization in the encoder stem). Pass `-c cpu` or");
            siam::log_cont("`-c auto` (auto falls back to CPU after this point).");
            return 4;
        } else {
            siam::log_warn("ORT exception: %s", e.what());
            return 4;
        }
    }

#else
    // MNN backend: no EP-specific recovery (no CUDA/CoreML/TRT to fall
    // back from). MnnEngine throws std::runtime_error on failure; let
    // it propagate up to main()'s top-level catch, which prints a clean
    // diagnostic and exits 1. If we ever add an "auto fall back to cpu
    // on OpenCL session failure" path, this is where it belongs.
    logits = siam::sliding_window(
                 resampled, model_paths, patch, num_classes,
                 threads, 0.5f, verbose, device, trt_cache_dir, engine_tuning);
#endif

    resampled = Volume{};  // free

    // ===== Upsample branch: save the network's output at its native     ====
    // ===== inference resolution (0.75 mm by default), un-cropped to     ====
    // ===== the full canonical extent, with canonical RAS orientation.   ====
    // ===== Skips the per-channel trilinear back-resample below, so a    ====
    // ===== low-res input yields a super-resolved segmentation matching  ====
    // ===== the network's learned detail.                                ====
    if (upsample_mode) {
        // Shape of the FULL canonical grid at the target spacing.
        auto full_up_shape = siam::compute_new_shape(
                                 shape_canon, spacing_zyx, new_spacing);

        // Where the cropped-and-network-resampled region falls inside
        // the full upsampled grid (per-axis floor of bbox_lo scaled by
        // input_spacing / target_spacing).
        std::array<int64_t, 3> bbox_lo_up{};

        for (int i = 0; i < 3; ++i) {
            bbox_lo_up[i] = static_cast<int64_t>(std::llround(
                    static_cast<double>(crop.bbox[i][0]) *
                    spacing_zyx[i] / new_spacing[i]));
        }

        // Clamp the high end so round-off from compute_new_shape never
        // drives writes past the end of the buffer.
        const std::array<int64_t, 3> net_extent = {
            logits.spat[0], logits.spat[1], logits.spat[2]
        };
        std::array<int64_t, 3> bbox_hi_up{};

        for (int i = 0; i < 3; ++i) {
            bbox_hi_up[i] = std::min(bbox_lo_up[i] + net_extent[i],
                                     full_up_shape[i]);
        }

        siam::log_tag("upsample",
                      "full canonical at %.3f mm  (Z,Y,X) (%" PRId64 ",%" PRId64 ",%" PRId64
                      ")  bbox lo (%" PRId64 ",%" PRId64 ",%" PRId64 ")",
                      target_spacing,
                      full_up_shape[0], full_up_shape[1], full_up_shape[2],
                      bbox_lo_up[0], bbox_lo_up[1], bbox_lo_up[2]);

        // Build the canonical-RAS affine for the high-res grid:
        //   col_i_new = col_i_canon * (target_spacing / canonical_zoom[i])
        //   origin_new = origin_canon + half-pixel shift from spacing change
        std::array<float, 16> A_up{};
        A_up[15] = 1.0f;

        for (int r = 0; r < 3; ++r) {
            float origin_shift = 0.0f;

            for (int i = 0; i < 3; ++i) {
                const float scale = target_spacing / img.zooms_canon[i];
                A_up[r * 4 + i] = img.affine_canon[r * 4 + i] * scale;
                origin_shift += img.affine_canon[r * 4 + i] * 0.5f * (scale - 1.0f);
            }

            A_up[r * 4 + 3] = img.affine_canon[r * 4 + 3] + origin_shift;
        }

        // Synthetic NiftiImage in canonical RAS coords. perm/flip identity
        // so save_nifti_* / save_jnifti_* do a memcpy through
        // copy_reorient_from_canonical instead of a real reorient.
        NiftiImage img_up;
        img_up.affine_orig = A_up;
        img_up.affine_canon = A_up;
        img_up.zooms_canon = {target_spacing, target_spacing, target_spacing};
        img_up.perm_canon_to_orig = {0, 1, 2};
        img_up.flip_canon = {1, 1, 1};
        img_up.shape_orig = {full_up_shape[2], full_up_shape[1], full_up_shape[0]};
        img_up.volume.shape = {full_up_shape[0], full_up_shape[1], full_up_shape[2]};

        const int64_t Zup = full_up_shape[0];
        const int64_t Yup = full_up_shape[1];
        const int64_t Xup = full_up_shape[2];
        const int64_t plane_up = Yup * Xup;
        const int64_t voxel_count_up = Zup * Yup * Xup;

        const int64_t lzN = logits.spat[0];
        const int64_t lyN = logits.spat[1];
        const int64_t lxN = logits.spat[2];

        if (tpm_mode) {
            // ---- TPM branch (upsample): softmax in place on logits, then
            //      pad each channel into a full-canonical-extent buffer
            //      with background channel = 1.0 outside the bbox.
            if (tpm_temperature != 1.0f) {
                siam::log_tag("tpm",
                              "softmax over %" PRId64 " classes  T=%.3f",
                              num_classes, tpm_temperature);
            } else {
                siam::log_tag("tpm",
                              "softmax over %" PRId64 " classes",
                              num_classes);
            }

            if (class_set == ClassSet::SPM) {
                siam::log_cont("merge -> SPM6 (GM, WM, CSF, Bone, Soft, Air)");
            }

            const float inv_T = 1.0f / tpm_temperature;
            const int64_t N_net = lzN * lyN * lxN;

            // Per-voxel softmax is fully independent across i, so the
            // outer loop parallelizes cleanly. On a 4-class/4-core CI
            // runner this drops the softmax phase ~3-4x; on huo's
            // 64-core box it tops out around the 10-12x mark before
            // memory bandwidth saturates (the channel-major layout
            // forces num_classes scattered loads per voxel).
            //
            // --classes spm: after the standard 18-channel softmax,
            // merge into SPM12's 6 TPM bins in-place. We gather all
            // 18 probabilities into a stack-local spm[6] scratch
            // first, THEN overwrite logits channels 0..5 -- doing it
            // in two passes avoids the channel-aliasing trap where
            // e.g. SPM bin 0 (GM) reads logits[5] but SPM bin 5 (Air)
            // already overwrote logits[5] with logits[0]'s prior
            // value. Channels 6..17 of logits become stale after the
            // merge; the bbox-copy loop below only iterates over
            // num_classes_out (6 in SPM mode) so they're never read.
            const bool spm = (class_set == ClassSet::SPM);
            #pragma omp parallel for schedule(static)

            for (int64_t i = 0; i < N_net; ++i) {
                float m = logits.channel_ptr(0)[i] * inv_T;

                for (int64_t c = 1; c < num_classes; ++c) {
                    float v = logits.channel_ptr(c)[i] * inv_T;

                    if (v > m) {
                        m = v;
                    }
                }

                float s = 0.0f;

                for (int64_t c = 0; c < num_classes; ++c) {
                    float e = std::exp(logits.channel_ptr(c)[i] * inv_T - m);
                    logits.channel_ptr(c)[i] = e;
                    s += e;
                }

                float invs = 1.0f / s;

                for (int64_t c = 0; c < num_classes; ++c) {
                    logits.channel_ptr(c)[i] *= invs;
                }

                if (spm) {
                    float spm_bin[SPM6_NUM_CLASSES] = {0, 0, 0, 0, 0, 0};

                    for (int64_t c = 0; c < num_classes; ++c) {
                        spm_bin[SIAM18_TO_SPM6[c]] += logits.channel_ptr(c)[i];
                    }

                    for (int64_t b = 0; b < SPM6_NUM_CLASSES; ++b) {
                        logits.channel_ptr(b)[i] = spm_bin[b];
                    }
                }
            }

            std::vector<float> tpm_up(static_cast<size_t>(num_classes_out) * voxel_count_up, 0.0f);
            // Background channel starts at 1.0 outside the bbox so
            // voxels we never write keep p(background)=1, p(other)=0.
            // SPM ordering puts Air (background) at index 5; the
            // default 18-class ordering keeps background at index 0.
            const int64_t bg_channel = (class_set == ClassSet::SPM) ? SPM6_AIR_CHANNEL : 0;
            std::fill_n(tpm_up.data() + bg_channel * voxel_count_up,
                        voxel_count_up, 1.0f);

            for (int64_t c = 0; c < num_classes_out; ++c) {
                const float* src = logits.channel_ptr(c);
                float* dst_chan = tpm_up.data() + c * voxel_count_up;

                for (int64_t z = bbox_lo_up[0]; z < bbox_hi_up[0]; ++z) {
                    const int64_t lz = z - bbox_lo_up[0];

                    for (int64_t y = bbox_lo_up[1]; y < bbox_hi_up[1]; ++y) {
                        const int64_t ly = y - bbox_lo_up[1];
                        const int64_t copy_n_x = bbox_hi_up[2] - bbox_lo_up[2];
                        std::copy_n(&src[lz * lyN * lxN + ly * lxN],
                                    copy_n_x,
                                    &dst_chan[z * plane_up + y * Xup + bbox_lo_up[2]]);
                    }
                }
            }

            logits = LogitsVolume{};

            if (out_format == "nii") {
                siam::save_nifti_tpm(output_path, img_up,
                                     tpm_up.data(), num_classes_out);
            } else {
                siam::save_jnifti_tpm(output_path, img_up,
                                      tpm_up.data(), num_classes_out,
                                      out_format, class_set, tpm_shuffle);
            }
        } else {
            // ---- Labels branch (upsample): argmax over logits at
            //      network resolution, then pad into a full canonical
            //      buffer (zeros outside the bbox = background class
            //      = label 0 = Air in SPM order, also background).
            //
            // --classes spm: argmax-on-logits is invalid because
            // merging probabilities across SIAM bins doesn't commute
            // with argmax on raw logits (e.g. if Thal(6) has the
            // highest logit but cortical GM(1) + cerGM(5) + Hippo(12)
            // jointly sum to higher GM probability, the right answer
            // is GM). Compute per-bin score = sum_{c in bin} exp(logit_c
            // - max_logit) and argmax that. Skip the divide-by-Z step
            // since argmax is invariant under per-voxel scaling.
            // Outside-bbox default: 0 in the standard 18-class ordering
            // (label 0 = background), but in SPM ordering label 0 = GM,
            // so outside-bbox voxels need label 5 (Air) instead.
            const uint8_t bg_label =
                (class_set == ClassSet::SPM)
                ? static_cast<uint8_t>(SPM6_AIR_CHANNEL)
                : static_cast<uint8_t>(0);
            std::vector<uint8_t> labels_up(static_cast<size_t>(voxel_count_up), bg_label);
            const bool spm = (class_set == ClassSet::SPM);

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

                            for (int64_t c = 1; c < num_classes; ++c) {
                                float v = logits.channel_ptr(c)[i_net];

                                if (v > M) {
                                    M = v;
                                }
                            }

                            float bin[SPM6_NUM_CLASSES] = {0, 0, 0, 0, 0, 0};

                            for (int64_t c = 0; c < num_classes; ++c) {
                                bin[SIAM18_TO_SPM6[c]] +=
                                    std::exp(logits.channel_ptr(c)[i_net] - M);
                            }

                            float bestv = bin[0];

                            for (int64_t b = 1; b < SPM6_NUM_CLASSES; ++b) {
                                if (bin[b] > bestv) {
                                    bestv = bin[b];
                                    best = static_cast<int>(b);
                                }
                            }
                        } else {
                            float bestv = logits.channel_ptr(0)[i_net];

                            for (int64_t c = 1; c < num_classes; ++c) {
                                float v = logits.channel_ptr(c)[i_net];

                                if (v > bestv) {
                                    bestv = v;
                                    best = static_cast<int>(c);
                                }
                            }
                        }

                        labels_up[z * plane_up + y * Xup + x] =
                            static_cast<uint8_t>(best);
                    }
                }
            }

            logits = LogitsVolume{};

            if (out_format == "nii") {
                siam::save_nifti_labels(output_path, img_up, labels_up.data());
            } else {
                siam::save_jnifti_labels(output_path, img_up, labels_up.data(),
                                         out_format, class_set,
                                         static_cast<int>(num_classes));
            }
        }

        auto t_end_up = std::chrono::steady_clock::now();
        double seconds_up = std::chrono::duration<double>(t_end_up - t_start).count();
        siam::log_tag("saved", "%s  in %.1fs", output_path.c_str(), seconds_up);
        return 0;
    }

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

    const int64_t cZ = logits_back.spat[0];
    const int64_t cY = logits_back.spat[1];
    const int64_t cX = logits_back.spat[2];
    const int64_t Zc = shape_canon[0], Yc = shape_canon[1], Xc = shape_canon[2];

    if (tpm_mode) {
        // ---- TPM branch: softmax logits -> un-crop -> save 4D float32 ----
        if (tpm_temperature != 1.0f) {
            siam::log_tag("tpm", "softmax over %" PRId64 " classes  T=%.3f",
                          num_classes, tpm_temperature);
        } else {
            siam::log_tag("tpm", "softmax over %" PRId64 " classes",
                          num_classes);
        }

        if (class_set == ClassSet::SPM) {
            siam::log_cont("merge -> SPM6 (GM, WM, CSF, Bone, Soft, Air)");
        }

        const float inv_T = 1.0f / tpm_temperature;
        const int64_t N = cZ * cY * cX;

        // See the parallel-softmax + SPM merge comments in the
        // upsample branch above; the loop body is identical and
        // same parallelization + merge logic applies.
        const bool spm = (class_set == ClassSet::SPM);
        #pragma omp parallel for schedule(static)

        for (int64_t i = 0; i < N; ++i) {
            // numerically stable softmax over the C channel logits
            float m = logits_back.channel_ptr(0)[i] * inv_T;

            for (int64_t c = 1; c < num_classes; ++c) {
                float v = logits_back.channel_ptr(c)[i] * inv_T;

                if (v > m) {
                    m = v;
                }
            }

            float s = 0.0f;

            for (int64_t c = 0; c < num_classes; ++c) {
                float e = std::exp(logits_back.channel_ptr(c)[i] * inv_T - m);
                logits_back.channel_ptr(c)[i] = e;
                s += e;
            }

            float invs = 1.0f / s;

            for (int64_t c = 0; c < num_classes; ++c) {
                logits_back.channel_ptr(c)[i] *= invs;
            }

            if (spm) {
                float spm_bin[SPM6_NUM_CLASSES] = {0, 0, 0, 0, 0, 0};

                for (int64_t c = 0; c < num_classes; ++c) {
                    spm_bin[SIAM18_TO_SPM6[c]] += logits_back.channel_ptr(c)[i];
                }

                for (int64_t b = 0; b < SPM6_NUM_CLASSES; ++b) {
                    logits_back.channel_ptr(b)[i] = spm_bin[b];
                }
            }
        }

        // Un-crop to canonical shape. Outside the bbox we want
        // p(background)=1 and p(other)=0. SPM ordering puts Air
        // (background) at channel 5; default 18-class ordering keeps
        // background at channel 0.
        std::vector<float> tpm_canon(static_cast<size_t>(num_classes_out) * Zc * Yc * Xc, 0.0f);
        const int64_t bg_channel_canon = (class_set == ClassSet::SPM) ? SPM6_AIR_CHANNEL : 0;
        std::fill_n(tpm_canon.data() + bg_channel_canon * Zc * Yc * Xc,
                    Zc * Yc * Xc, 1.0f);

        for (int64_t c = 0; c < num_classes_out; ++c) {
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

        logits_back = LogitsVolume{};

        if (out_format == "nii") {
            siam::save_nifti_tpm(output_path, img, tpm_canon.data(), num_classes_out);
        } else {
            siam::save_jnifti_tpm(output_path, img, tpm_canon.data(),
                                  num_classes_out, out_format, class_set,
                                  tpm_shuffle);
        }
    } else {
        // ---- Labels branch: argmax -> un-crop -> save 3D uint8 ----------
        // --classes spm: see the SPM-aware labels path in the upsample
        // branch above for why argmax-on-logits is insufficient and we
        // use exp-sum-per-bin (skip divide-by-Z since argmax is scale
        // invariant).
        std::vector<uint8_t> labels_crop(static_cast<size_t>(cZ) * cY * cX, 0);
        const bool spm_labels = (class_set == ClassSet::SPM);

        #pragma omp parallel for schedule(static)

        for (int64_t z = 0; z < cZ; ++z) {
            for (int64_t y = 0; y < cY; ++y) {
                for (int64_t x = 0; x < cX; ++x) {
                    int64_t i = z * cY * cX + y * cX + x;
                    int best = 0;

                    if (spm_labels) {
                        float M = logits_back.channel_ptr(0)[i];

                        for (int64_t c = 1; c < num_classes; ++c) {
                            float v = logits_back.channel_ptr(c)[i];

                            if (v > M) {
                                M = v;
                            }
                        }

                        float bin[SPM6_NUM_CLASSES] = {0, 0, 0, 0, 0, 0};

                        for (int64_t c = 0; c < num_classes; ++c) {
                            bin[SIAM18_TO_SPM6[c]] +=
                                std::exp(logits_back.channel_ptr(c)[i] - M);
                        }

                        float bestv = bin[0];

                        for (int64_t b = 1; b < SPM6_NUM_CLASSES; ++b) {
                            if (bin[b] > bestv) {
                                bestv = bin[b];
                                best = static_cast<int>(b);
                            }
                        }
                    } else {
                        float bestv = logits_back.channel_ptr(0)[i];

                        for (int64_t c = 1; c < num_classes; ++c) {
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

        logits_back = LogitsVolume{};

        // See note in upsample labels branch: SPM ordering puts Air
        // at label 5, so outside-bbox needs label 5 (not 0 = GM).
        const uint8_t bg_label_canon =
            (class_set == ClassSet::SPM)
            ? static_cast<uint8_t>(SPM6_AIR_CHANNEL)
            : static_cast<uint8_t>(0);
        std::vector<uint8_t> labels_canon(static_cast<size_t>(Zc) * Yc * Xc, bg_label_canon);

        for (int64_t z = 0; z < cZ; ++z) {
            for (int64_t y = 0; y < cY; ++y) {
                std::copy_n(&labels_crop[z * cY * cX + y * cX],
                            cX,
                            &labels_canon[(z + crop.bbox[0][0]) * Yc * Xc +
                                                                (y + crop.bbox[1][0]) * Xc +
                                                                crop.bbox[2][0]]);
            }
        }

        if (out_format == "nii") {
            siam::save_nifti_labels(output_path, img, labels_canon.data());
        } else {
            siam::save_jnifti_labels(output_path, img, labels_canon.data(),
                                     out_format, class_set,
                                     static_cast<int>(num_classes));
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(t_end - t_start).count();

    siam::log_tag("saved", "%s  in %.1fs", output_path.c_str(), seconds);

    return 0;
}
